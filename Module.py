import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import hamming
from scipy.optimize import linear_sum_assignment
from device_util import get_torch_device

class ReconstructionTool(nn.Module):

    def __init__(self, beacon_shape, correlation_matrix, initial_guess=None, device=None):
        super().__init__()
        self.device = device if device else str(get_torch_device())
        self.num_snps, self.num_indiv = beacon_shape
        self.C_target = torch.tensor(correlation_matrix, dtype=torch.float32, device=self.device)
        if initial_guess is not None:
            guess_tensor = torch.tensor(initial_guess, dtype=torch.float32, device=self.device)
            init_logits = guess_tensor * 4 - 2
        else:
            init_logits = torch.randn(self.num_snps, self.num_indiv, device=self.device)
        self.B_logits = nn.Parameter(init_logits.requires_grad_(True))
        use_fused = self.device == 'cuda'
        self.opt_corr = torch.optim.Adam([self.B_logits], lr=0.001, fused=use_fused)
        self.opt_freq = torch.optim.Adam([self.B_logits], lr=0.01, fused=use_fused)

    def calculate_current_correlations(self, B_soft):
        p11 = torch.matmul(B_soft, B_soft.T)
        row_sums = torch.sum(B_soft, dim=1, keepdim=True)
        total_matches = 2 * p11 + self.num_indiv - row_sums - row_sums.T
        return total_matches / self.num_indiv

    def _train_corr_step(self):
        self.opt_corr.zero_grad(set_to_none=True)
        self.B_logits.data.clamp_(-5.0, 5.0)
        B_soft = torch.sigmoid(self.B_logits)
        current_corr = self.calculate_current_correlations(B_soft)
        loss_corr = torch.norm(current_corr - self.C_target, p='fro')
        loss_corr.backward()
        self.opt_corr.step()
        return loss_corr

    def _train_freq_step(self, target_freqs):
        self.opt_freq.zero_grad(set_to_none=True)
        self.B_logits.data.clamp_(-5.0, 5.0)
        B_soft = torch.sigmoid(self.B_logits)
        freq_estimated = torch.mean(B_soft, dim=1)
        loss_freq = nn.functional.mse_loss(freq_estimated, target_freqs)
        loss_freq.backward()
        self.opt_freq.step()
        return loss_freq

    def optimize(self, target_frequencies, cycles=1, corr_steps=1000, freq_steps=500):
        if not torch.is_tensor(target_frequencies):
            target_freqs = torch.tensor(target_frequencies, dtype=torch.float32, device=self.device)
        else:
            target_freqs = target_frequencies.clone().detach().to(self.device)
        final_corr_loss = 0.0
        final_freq_loss = 0.0
        for _ in range(cycles):
            for _ in range(corr_steps):
                final_corr_loss = self._train_corr_step().item()
            for _ in range(freq_steps):
                final_freq_loss = self._train_freq_step(target_freqs).item()
        return (final_corr_loss, final_freq_loss)

    def get_reconstruction(self):
        with torch.no_grad():
            return (torch.sigmoid(self.B_logits) > 0.5).int().cpu().numpy()

    @staticmethod
    def calculate_accuracy(beacon, new_beacon):
        return np.mean(beacon == new_beacon)

    @staticmethod
    def calculate_f1(beacon, new_beacon):
        tp = np.sum((beacon == 1) & (new_beacon == 1))
        fp = np.sum((beacon == 0) & (new_beacon == 1))
        fn = np.sum((beacon == 1) & (new_beacon == 0))
        if tp == 0:
            return 0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def compare_and_sort_columns(beacon, reconstructed_beacon):
        num_columns = beacon.shape[1]
        cost_matrix = np.zeros((num_columns, num_columns))
        for i in range(num_columns):
            for j in range(num_columns):
                cost_matrix[i, j] = hamming(beacon[:, i], reconstructed_beacon[:, j])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        sorted_beacon = np.zeros_like(beacon)
        for i in range(num_columns):
            sorted_beacon[:, i] = reconstructed_beacon[:, col_ind[i]]
        return sorted_beacon

    @staticmethod
    def greedy_initialization(beacon_freqs, N):
        M = len(beacon_freqs)
        guess = np.zeros((M, N))
        for i in range(M):
            count = int(beacon_freqs[i]) if beacon_freqs[i] > 1 else int(round(beacon_freqs[i] * N))
            indices = np.random.choice(N, count, replace=False)
            guess[i, indices] = 1.0
        return guess

def evaluate_defense_f1(queried_indices, noisy_freqs, true_db, full_corr_matrix, device=None, fast_mode=True, return_reconstruction=False):
    if device is None:
        device = str(get_torch_device())
    q_idx = queried_indices.clone().detach().cpu().numpy().flatten()
    db_np = true_db.cpu().numpy() if torch.is_tensor(true_db) else np.array(true_db)
    corr_np = full_corr_matrix.cpu().numpy() if torch.is_tensor(full_corr_matrix) else np.array(full_corr_matrix)
    if corr_np.shape[0] <= max(q_idx):
        raise ValueError(f'\nCRITICAL SHAPE MISMATCH:\nYour correlation matrix has size {corr_np.shape[0]}x{corr_np.shape[1]},\nbut the PPO Agent queried SNP index {max(q_idx)}.\nPlease ensure C_target is sliced correctly in main.py!')
    true_db_subset = db_np[q_idx, :]
    C_subset = corr_np[np.ix_(q_idx, q_idx)]
    num_snps, num_indiv = true_db_subset.shape
    baseline_guess = ReconstructionTool.greedy_initialization(noisy_freqs.clone().detach().cpu().numpy(), num_indiv)
    optimizer_tool = ReconstructionTool((num_snps, num_indiv), C_subset, initial_guess=baseline_guess, device=device)
    if fast_mode:
        corr_steps = 201
        freq_steps = 101
        max_iterations = 15
        grace_period = 3
    else:
        corr_steps = 1001
        freq_steps = 501
        max_iterations = 100
        grace_period = 5
    previous_binary = optimizer_tool.get_reconstruction()
    for i in range(max_iterations):
        optimizer_tool.optimize(noisy_freqs, corr_steps=corr_steps, freq_steps=freq_steps)
        current_binary = optimizer_tool.get_reconstruction()
        flips = np.sum(previous_binary != current_binary)
        previous_binary = current_binary
        if i > grace_period and flips < 10 and fast_mode:
            break
        elif i > grace_period and flips < 3 and (not fast_mode):
            break
    reconstructed_beacon = optimizer_tool.get_reconstruction()
    reconstructed_sorted = ReconstructionTool.compare_and_sort_columns(true_db_subset, reconstructed_beacon)
    f1 = ReconstructionTool.calculate_f1(true_db_subset, reconstructed_sorted)
    if return_reconstruction:
        return f1, reconstructed_sorted
    return f1
