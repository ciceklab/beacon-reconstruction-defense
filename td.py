import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from device_util import announce_device, get_torch_device
device = get_torch_device()
announce_device(device)

class ReplayBuffer:

    def __init__(self, seq_len, token_dim, action_dim, max_size, dvc):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.dvc = dvc
        self.seq_len = seq_len
        self.token_dim = token_dim
        self.s = torch.zeros((max_size, seq_len, token_dim), dtype=torch.float, device=self.dvc)
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float, device=self.dvc)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)
        self.s_next = torch.zeros((max_size, seq_len, token_dim), dtype=torch.float, device=self.dvc)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    def store(self, s, a, r, s_next, dw):
        self.s[self.ptr] = torch.as_tensor(s, dtype=torch.float32, device=self.dvc)
        self.a[self.ptr] = torch.as_tensor(a, dtype=torch.float32, device=self.dvc)
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.as_tensor(s_next, dtype=torch.float32, device=self.dvc)
        self.dw[self.ptr] = dw
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return (self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind])

class SequenceEncoder(nn.Module):

    def __init__(self, seq_len, token_dim, d_model=64, nhead=4, num_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(token_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    @staticmethod
    def padding_mask_from_state(state_seq):
        
        padding = state_seq.abs().sum(dim=-1) <= 1e-12
        all_padding = padding.all(dim=1)
        if all_padding.any():
            padding = padding.clone()
            padding[all_padding, -1] = False
        return padding

    def forward(self, state_seq):
        x = self.input_proj(state_seq) + self.pos_embedding
        padding_mask = self.padding_mask_from_state(state_seq)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return x[:, -1, :]

class Actor(nn.Module):

    def __init__(self, seq_len, token_dim, action_dim, net_width, maxaction, d_model=64):
        super().__init__()
        self.encoder = SequenceEncoder(seq_len, token_dim, d_model=d_model)
        self.head = nn.Sequential(nn.Linear(d_model, net_width), nn.Tanh(), nn.Linear(net_width, action_dim), nn.Tanh())
        self.maxaction = maxaction

    def forward(self, state_seq):
        pooled = self.encoder(state_seq)
        return self.head(pooled) * self.maxaction

class Double_Q_Critic(nn.Module):

    def __init__(self, seq_len, token_dim, action_dim, net_width, d_model=64):
        super().__init__()
        self.encoder = SequenceEncoder(seq_len, token_dim, d_model=d_model)
        sa_dim = d_model + action_dim
        self.q1 = nn.Sequential(nn.Linear(sa_dim, net_width), nn.ReLU(), nn.Linear(net_width, net_width), nn.ReLU(), nn.Linear(net_width, 1))
        self.q2 = nn.Sequential(nn.Linear(sa_dim, net_width), nn.ReLU(), nn.Linear(net_width, net_width), nn.ReLU(), nn.Linear(net_width, 1))

    def _encode(self, state_seq, action):
        pooled = self.encoder(state_seq)
        sa = torch.cat([pooled, action], dim=1)
        return sa

    def Q1(self, state_seq, action):
        return self.q1(self._encode(state_seq, action))

    def forward(self, state_seq, action):
        sa = self._encode(state_seq, action)
        return (self.q1(sa), self.q2(sa))

class TD3:

    def __init__(self, seq_len=20, token_dim=19, action_dim=1, max_action=1.0, lr_actor=0.0001, lr_critic=0.0001, gamma=0.99, tau=0.005, K_epochs=50, net_width=256, d_model=64, reward_clip=50.0, policy_delay=2, critic_loss_type='huber', huber_beta=1.0, explore_noise_min=0.04, inference_only=False, device_override=None):
        self.max_action = max_action
        self.policy_noise = 0.2 * self.max_action
        self.noise_clip = 0.5 * self.max_action
        self.tau = tau
        self.delay_counter = 0
        self.gamma = gamma
        self.explore_noise = 0.15
        self.action_dim = action_dim
        self.batch_size = 256
        self.K_epochs = K_epochs
        self.delay_freq = policy_delay
        self.reward_clip = reward_clip
        self.critic_loss_type = critic_loss_type
        self.huber_beta = huber_beta
        self.explore_noise_min = explore_noise_min
        self.seq_len = seq_len
        self.token_dim = token_dim
        self.inference_only = inference_only
        run_device = device if device_override is None else torch.device(device_override)
        self.actor = Actor(seq_len, token_dim, action_dim, net_width, self.max_action, d_model=d_model).to(run_device)
        if inference_only:
            self.actor_optimizer = None
            self.actor_target = None
            self.q_critic = None
            self.q_critic_optimizer = None
            self.q_critic_target = None
            self.buffer = None
            return
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.actor_target = copy.deepcopy(self.actor)
        self.q_critic = Double_Q_Critic(seq_len, token_dim, action_dim, net_width, d_model=d_model).to(run_device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=lr_critic)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        self.buffer = ReplayBuffer(seq_len, token_dim, action_dim, max_size=int(100000.0), dvc=run_device)

    def _to_tensor(self, state):
        state = np.asarray(state, dtype=np.float32)
        run_device = next(self.actor.parameters()).device
        if state.ndim == 2:
            t = torch.as_tensor(state, dtype=torch.float32, device=run_device).unsqueeze(0)
        else:
            t = torch.as_tensor(state, dtype=torch.float32, device=run_device)
        return t

    def select_action(self, state, deterministic):
        with torch.no_grad():
            t = self._to_tensor(state)
            a = self.actor(t).cpu().numpy()[0]
            if deterministic:
                return a
            noise = np.random.normal(0, self.max_action * self.explore_noise, size=self.action_dim)
            return (a + noise).clip(-self.max_action, self.max_action)

    def update(self):
        critic_losses = []
        actor_losses = []
        for _ in range(self.K_epochs):
            self.delay_counter += 1
            with torch.no_grad():
                s, a, r, s_next, dw = self.buffer.sample(self.batch_size)
                target_a_noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                smoothed_target_a = (self.actor_target(s_next) + target_a_noise).clamp(-self.max_action, self.max_action)
                target_Q1, target_Q2 = self.q_critic_target(s_next, smoothed_target_a)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = r + ~dw * self.gamma * target_Q
                if self.reward_clip is not None and self.reward_clip > 0:
                    target_Q = torch.clamp(target_Q, -self.reward_clip, self.reward_clip)
            current_Q1, current_Q2 = self.q_critic(s, a)
            if self.critic_loss_type == 'huber':
                q_loss = F.smooth_l1_loss(current_Q1, target_Q, beta=self.huber_beta) + F.smooth_l1_loss(current_Q2, target_Q, beta=self.huber_beta)
            else:
                q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            critic_losses.append(float(q_loss.item()))
            self.q_critic_optimizer.zero_grad()
            q_loss.backward()
            self.q_critic_optimizer.step()
            if self.delay_counter > self.delay_freq:
                a_loss = -self.q_critic.Q1(s, self.actor(s)).mean()
                actor_losses.append(float(a_loss.item()))
                self.actor_optimizer.zero_grad()
                a_loss.backward()
                self.actor_optimizer.step()
                with torch.no_grad():
                    for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                self.delay_counter = 0
        return {'critic_loss': float(np.mean(critic_losses)) if critic_losses else 0.0, 'actor_loss': float(np.mean(actor_losses)) if actor_losses else 0.0}

    def save(self, path):
        torch.save(self.actor.state_dict(), '{}/actor.pth'.format(path))
        torch.save(self.q_critic.state_dict(), '{}/q_critic.pth'.format(path))

    def load(self, path):
        run_device = next(self.actor.parameters()).device
        load_kw = {"map_location": run_device, "weights_only": True}
        self.actor.load_state_dict(torch.load('{}/actor.pth'.format(path), **load_kw))
        if self.q_critic is None:
            return
        self.q_critic.load_state_dict(torch.load('{}/q_critic.pth'.format(path), **load_kw))
