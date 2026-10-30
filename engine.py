import csv
import json
import wandb
from datetime import datetime
import os
import numpy as np
from beacon import IDX_ROLLING_UTIL
from utility_metrics import build_wandb_log_dict, evaluate_episode_utility

def _csv_optional(value):
    if value is None:
        return ''
    return value


def state_rolling_util(state, fallback: float) -> float:
    
    array = np.asarray(state)
    if array.ndim >= 1 and array.shape[-1] > IDX_ROLLING_UTIL:
        return float(array[-1, IDX_ROLLING_UTIL])
    return float(fallback)

def _run_eval_episodes(env, beacon_agent, n_episodes, attacker_type='mixed'):
    
    original_eval = getattr(env.args, 'evaluate', False)
    original_bra_fast = getattr(env.args, 'bra_fast', False)
    original_baseline = env.args.privacy_f1_baseline
    original_attacker = getattr(env.args, 'attacker_type', 'mixed')
    original_force = getattr(env.args, 'force_attacker_type', False)
    full_baseline = getattr(env.args, 'privacy_f1_baseline_full', None)
    env.args.evaluate = True
    env.args.bra_fast = False
    env.args.attacker_type = attacker_type
    env.args.force_attacker_type = True
    if full_baseline is not None:
        env.args.privacy_f1_baseline = full_baseline
    f1_scores = []
    mae_scores = []
    cer_scores = []
    drop_scores = []
    for _ in range(n_episodes):
        env.reset()
        done = False
        info = {}
        while not done:
            _, _, done, info = env.step(beacon_agent=beacon_agent)
        f1_scores.append(float(info.get('f1', 1.0)))
        mae_scores.append(float(info.get('mae', 0.0)))
        cer_scores.append(float(info.get('Clinical_Error_Ratio', 0.0)))
        drop_scores.append(float(info.get('relative_f1_drop', 0.0)))
    env.args.evaluate = original_eval
    env.args.bra_fast = original_bra_fast
    env.args.privacy_f1_baseline = original_baseline
    env.args.attacker_type = original_attacker
    env.args.force_attacker_type = original_force
    return (f1_scores, mae_scores, cer_scores, drop_scores)

def train_TD_beacon(args, env, beacon_agent, *extra_args):
    print('============================================================================================')
    start_time = datetime.now().replace(microsecond=0)
    mode_str = 'EVALUATION (Weights Frozen)' if getattr(args, 'evaluate', False) else 'TRAINING'
    print(f'Started {mode_str} at (GMT) : ', start_time)
    print('============================================================================================')
    if not getattr(args, 'evaluate', False):
        directory = args.results_dir + '/weights'
        if not os.path.exists(directory):
            os.makedirs(directory)
        checkpoint_path = directory
    best_reward = -float('inf')
    best_episode = 0
    best_train_f1 = float('inf')
    best_train_episode = 0
    best_eval_f1 = float('inf')
    best_eval_episode = 0
    best_eval_honest_mae = None
    best_eval_honest_cer = None
    best_eval_attack_drop = None
    eval_no_improve = 0
    eval_freq = int(getattr(args, 'eval_freq', 10))
    eval_episodes = int(getattr(args, 'eval_episodes', 3))
    early_stop_patience = int(getattr(args, 'early_stop_patience', 0))
    explore_decay = float(getattr(args, 'explore_decay', 0.998))
    explore_decay_start = int(getattr(args, 'explore_decay_start', 100))
    explore_min = float(getattr(args, 'explore_noise_min', 0.04))
    beacon_agent.explore_noise_min = explore_min
    metrics_path = os.path.join(args.results_dir, 'convergence_metrics.csv')
    metrics_csv = open(metrics_path, 'w', newline='')
    metrics_writer = csv.DictWriter(metrics_csv, fieldnames=['episode', 'total_reward', 'f1', 'rolling_mae_utilization', 'avg_state_rolling_util', 'budget_penalty_count', 'critic_loss', 'actor_loss', 'explore_noise', 'mae', 'privacy_reward', 'utility_reward', 'f1_delta', 'eval_f1', 'best_eval_f1', 'episode_suspicion_mean', 'episode_suspicion_peak', 'episode_attacker_type', 'noise_maf_spearman', 'rarity_preservation', 'rare_to_common_noise', 'clinical_error_ratio', 'hub_targeting_score', 'noise_hub_spearman', 'frac_lowhub_noised', 'binary_flip_rate', 'binary_false_no_rate', 'binary_false_yes_rate', 'binary_honest_yes_rate'])
    metrics_writer.writeheader()
    i_episode = 1
    stopped_early = False
    while i_episode <= args.episodes:
        env.reset()
        current_ep_reward = 0
        ep_rolling_util = 0.0
        binfo, rewards, done, info = env.step(beacon_agent=beacon_agent)
        state, action, reward, dw = binfo
        current_ep_reward += reward
        ep_rolling_util += state_rolling_util(state, env.rolling_mae_utilization)
        final_info = info
        ep_steps = max(getattr(env, "max_steps", args.max_queries), 1)
        for t in range(ep_steps - 1):
            binfo, rewards, done, info = env.step(beacon_agent=beacon_agent)
            next_state = binfo[0]
            ep_rolling_util += state_rolling_util(next_state, env.rolling_mae_utilization)
            if not getattr(args, 'evaluate', False):
                beacon_agent.buffer.store(state, action, reward, next_state, dw)
            state, action, reward, dw = binfo
            current_ep_reward += reward
            if done:
                final_info = info
                break
        if not getattr(args, 'evaluate', False):
            beacon_agent.buffer.store(state, action, reward, state, True)
        if getattr(args, 'eval_utility', False) and 'episode_mafs' in final_info:
            mafs = final_info['episode_mafs']
            noises = final_info['episode_noises']
            if len(mafs) > 0:
                hub_scores = final_info.get('episode_hub_scores', [])
                utility_metrics = evaluate_episode_utility(mafs, noises, hub_scores if hub_scores else None, rare_threshold=float(getattr(args, 'rare_maf_threshold', 0.05)), hub_threshold=float(getattr(args, 'hub_noise_min', 0.25)))
                print('\n' + '-' * 65)
                print('Utility preservation analysis')
                print(f"Minimum_Perturbed_MAF   : {utility_metrics['Minimum_Perturbed_MAF']:.3f}")
                print(f"Noise_MAF_Spearman_Corr : {utility_metrics['Noise_MAF_Spearman_Corr']:.3f}")
                print(f"True Clinical Error (CER): {utility_metrics['Clinical_Error_Ratio']:.3f}")
                print(f"Avg Noise on Rare SNPs  : {utility_metrics['Avg_Clinical_Noise']:.4f}")
                print(f"Rarity preservation     : {utility_metrics['Rarity_Preservation_Score']:.3f}")
                print(f"Rare/Common noise ratio : {utility_metrics['Rare_to_Common_Noise_Ratio']:.3f}")
                if 'Hub_Targeting_Score' in utility_metrics:
                    print(f"Hub targeting score     : {utility_metrics['Hub_Targeting_Score']:.3f}")
                    print(f"Noise-Hub Spearman      : {utility_metrics['Noise_Hub_Spearman_Corr']:.3f}")
                    print(f"Low-hub queries noised  : {utility_metrics['Frac_LowHub_Queries_Noised']:.3f}")
                print('-' * 65 + '\n')
                final_info.update(utility_metrics)
        if current_ep_reward > best_reward:
            best_reward = current_ep_reward
            best_episode = i_episode
        train_f1 = float(final_info.get('f1', 1.0))
        if train_f1 < best_train_f1:
            best_train_f1 = train_f1
            best_train_episode = i_episode
        update_metrics = {'critic_loss': 0.0, 'actor_loss': 0.0}
        if not getattr(args, 'evaluate', False):
            warmup = getattr(args, 'buffer_warmup', 500)
            if beacon_agent.buffer.size >= warmup:
                update_metrics = beacon_agent.update()
        eval_f1_val = None
        if not getattr(args, 'evaluate', False) and eval_freq < 10 ** 8 and (i_episode % eval_freq == 0):
            print(f'\n[BRA-full checkpoint] {eval_episodes} frozen-policy MIXED eval episodes (bra_fast=False)...')
            eval_f1s, eval_maes, eval_cers, eval_drops = _run_eval_episodes(env, beacon_agent, eval_episodes, attacker_type='mixed')
            eval_f1_val = sum(eval_f1s) / len(eval_f1s)
            eval_mae_avg = sum(eval_maes) / len(eval_maes)
            eval_drop_avg = sum(eval_drops) / len(eval_drops)
            full_anchor = float(getattr(args, 'privacy_f1_baseline_full', args.privacy_f1_baseline))
            lift = full_anchor - eval_f1_val
            print(f'[BRA-full checkpoint] MIXED Eval F1: {eval_f1_val:.4f} | anchor: {full_anchor:.4f} | lift: {lift:+.4f} | Eval MAE: {eval_mae_avg:.4f}')

            
            _hcfg = getattr(args, 'honest_eval_episodes', None)
            honest_eval_n = max(1, int(_hcfg) if _hcfg else max(2, eval_episodes // 2))
            honest_f1s, honest_maes, honest_cers, _honest_drops = _run_eval_episodes(env, beacon_agent, honest_eval_n, attacker_type='honest')
            honest_mae_avg = sum(honest_maes) / len(honest_maes)
            honest_f1_avg = sum(honest_f1s) / len(honest_f1s)
            honest_cer_avg = sum(honest_cers) / len(honest_cers)
            print(f'[Honest utility] {honest_eval_n} episodes | mean|noise| (MAE): {honest_mae_avg:.4f} | CER: {honest_cer_avg:.4f} | honest F1: {honest_f1_avg:.4f} (lower MAE = quieter = better)')

            eval_best_path = os.path.join(args.results_dir, 'eval_checkpoints.json')
            history = []
            if os.path.isfile(eval_best_path):
                with open(eval_best_path) as f:
                    history = json.load(f)
            history.append({'episode': i_episode, 'eval_f1_mixed': eval_f1_val, 'eval_mae_mixed': eval_mae_avg,
                            'honest_mae': honest_mae_avg, 'honest_f1': honest_f1_avg,
                            'bra_full_anchor': full_anchor, 'lift_vs_anchor': lift})
            with open(eval_best_path, 'w') as f:
                json.dump(history, f, indent=2)
            if eval_f1_val < best_eval_f1:
                best_eval_f1 = eval_f1_val
                best_eval_episode = i_episode
                eval_no_improve = 0
                best_eval_honest_mae = honest_mae_avg
                best_eval_honest_cer = honest_cer_avg
                best_eval_attack_drop = eval_drop_avg
                print('New MIXED eval best — saving weights_best_eval/')
                eval_dir = checkpoint_path + '_best_eval'
                os.makedirs(eval_dir, exist_ok=True)
                beacon_agent.save(eval_dir)
                with open(os.path.join(eval_dir, 'best_metrics.txt'), 'w') as f:
                    f.write(f'Eval after episode {i_episode}\n')
                    f.write(f'MIXED Eval F1: {eval_f1_val:.4f}\n')
                    f.write(f'BRA-full anchor: {full_anchor:.4f}\n')
                    f.write(f'Lift vs anchor: {lift:+.4f}\n')
                    f.write(f'MIXED Eval MAE: {eval_mae_avg:.4f}\n')
                    f.write(f'Honest MAE: {honest_mae_avg:.4f}\n')
                    f.write(f'Honest F1: {honest_f1_avg:.4f}\n')
            else:
                eval_no_improve += 1
                if early_stop_patience > 0 and eval_no_improve >= early_stop_patience:
                    print(f'Early stop (MIXED eval): no eval F1 improvement for {early_stop_patience} checks.')
                    stopped_early = True
            print('--------------------------------------------------------------------------------------------\n')
        if metrics_writer is not None:
            row = {'episode': i_episode, 'total_reward': current_ep_reward, 'f1': final_info.get('f1', 0.0), 'rolling_mae_utilization': final_info.get('rolling_mae_utilization', 0.0), 'avg_state_rolling_util': ep_rolling_util / ep_steps, 'budget_penalty_count': final_info.get('episode_budget_penalties', 0), 'critic_loss': update_metrics.get('critic_loss', 0.0), 'actor_loss': update_metrics.get('actor_loss', 0.0), 'explore_noise': beacon_agent.explore_noise, 'mae': final_info.get('mae', 0.0), 'privacy_reward': final_info.get('privacy_reward', 0.0), 'utility_reward': final_info.get('utility_reward', 0.0), 'f1_delta': final_info.get('f1_delta', 0.0), 'eval_f1': '' if eval_f1_val is None else eval_f1_val, 'best_eval_f1': best_eval_f1 if best_eval_f1 < float('inf') else '', 'episode_suspicion_mean': final_info.get('episode_suspicion_mean', ''), 'episode_suspicion_peak': final_info.get('episode_suspicion_peak', ''), 'episode_attacker_type': final_info.get('episode_attacker_type', ''), 'noise_maf_spearman': final_info.get('Noise_MAF_Spearman_Corr', ''), 'rarity_preservation': final_info.get('Rarity_Preservation_Score', ''), 'rare_to_common_noise': final_info.get('Rare_to_Common_Noise_Ratio', ''), 'clinical_error_ratio': final_info.get('Clinical_Error_Ratio', ''), 'hub_targeting_score': final_info.get('Hub_Targeting_Score', ''), 'noise_hub_spearman': final_info.get('Noise_Hub_Spearman_Corr', ''), 'frac_lowhub_noised': final_info.get('Frac_LowHub_Queries_Noised', ''), 'binary_flip_rate': _csv_optional(final_info.get('binary_flip_rate')), 'binary_false_no_rate': _csv_optional(final_info.get('binary_false_no_rate')), 'binary_false_yes_rate': _csv_optional(final_info.get('binary_false_yes_rate')), 'binary_honest_yes_rate': _csv_optional(final_info.get('binary_honest_yes_rate'))}
            metrics_writer.writerow(row)
            metrics_csv.flush()
        if hasattr(args, 'use_wandb') and args.use_wandb:
            minimal = getattr(args, 'wandb_minimal', False)
            log_dict = build_wandb_log_dict(episode=i_episode, final_info=final_info, update_metrics=update_metrics, best_eval_f1=best_eval_f1, eval_f1_val=eval_f1_val, minimal=minimal, total_episode_reward=current_ep_reward if not minimal else None, explore_noise=beacon_agent.explore_noise if not minimal else None)
            if eval_f1_val is not None:
                log_dict['Eval/Attack F1'] = float(eval_f1_val)
                log_dict['Eval/Attack MAE'] = float(eval_mae_avg)
                log_dict['Eval/Attack Relative Drop'] = float(eval_drop_avg)
                log_dict['Eval/Honest MAE'] = float(honest_mae_avg)
                log_dict['Eval/Honest CER'] = float(honest_cer_avg)
                log_dict['Eval/Honest F1'] = float(honest_f1_avg)
            log_dict['Suspicion/P(adv)'] = float(final_info.get('episode_suspicion_probability', 0.0))
            if not minimal:
                log_dict['State/Avg Rolling MAE Utilization'] = ep_rolling_util / ep_steps
                log_dict['Suspicion/Episode Mean'] = float(final_info.get('episode_suspicion_mean', 0.0))
                log_dict['Reward Components/Utility Reward'] = float(final_info.get('utility_reward', 0.0))
            wandb.log(log_dict)
        print(f"--> Episode {i_episode} finished. Total Reward: {current_ep_reward:.4f} | Attacker F1: {final_info.get('f1', 0.0):.4f}")
        if not getattr(args, 'evaluate', False):
            warmup = getattr(args, 'buffer_warmup', 500)
            print('--------------------------------------------------------------------------------------------')
            if beacon_agent.buffer.size >= warmup:
                print(f"TD3 update | Critic: {update_metrics.get('critic_loss', 0.0):.4f} | Caps: {final_info.get('episode_budget_penalties', 0)} | Util: {final_info.get('rolling_mae_utilization', 0.0):.3f} | Proxy: {final_info.get('bra_proxy_loss', 0.0):.2f}")
            else:
                print(f'Skipping TD3 update (buffer {beacon_agent.buffer.size}/{warmup})')
            if i_episode >= explore_decay_start:
                beacon_agent.explore_noise = max(explore_min, beacon_agent.explore_noise * explore_decay)
            else:
                beacon_agent.explore_noise = max(explore_min, beacon_agent.explore_noise * 0.995)
            print(f'Current Exploration Noise: {beacon_agent.explore_noise:.4f}')
            beacon_agent.save(checkpoint_path)
        i_episode += 1
        if stopped_early:
            break
    metrics_csv.close()
    print(f'Convergence metrics saved to {metrics_path}')
    if getattr(args, 'evaluate', False):
        import statistics as _st
        with open(metrics_path, newline='') as f:
            rows = list(csv.DictReader(f))
        f1s = [float(r['f1']) for r in rows if r.get('f1')]
        baseline_path = os.path.join(args.results_dir, 'baseline_no_defense.json')
        no_def_mean = None
        if os.path.isfile(baseline_path):
            with open(baseline_path) as bf:
                no_def_mean = json.load(bf).get('f1_mean')
        if f1s:
            last_n = min(50, len(f1s))
            tail = f1s[-last_n:]
            print(f'Cohort-B eval summary ({len(f1s)} episodes):')
            print(f'  Min F1 (best privacy): {min(f1s):.4f}')
            print(f'  Mean F1 (all):         {_st.mean(f1s):.4f} ± {_st.pstdev(f1s):.4f}')
            print(f'  Mean F1 (last {last_n}):  {_st.mean(tail):.4f}')
            print(f"  BRA-full anchor (startup): {getattr(args, 'privacy_f1_baseline_full', args.privacy_f1_baseline):.4f}")
            if getattr(args, 'bra_fast', False):
                print(f'  BRA-fast anchor (train):   {args.privacy_f1_baseline:.4f}')
            if no_def_mean is not None:
                print(f'  No-defense mean F1:    {no_def_mean:.4f}')
                print(f'  Defense lift vs no-def: {no_def_mean - _st.mean(tail):+.4f} (positive = helps)')
    else:
        if best_train_f1 < float('inf'):
            print(f'Best train F1: {best_train_f1:.4f} @ episode {best_train_episode}')
        if best_eval_f1 < float('inf'):
            print(f'Best eval F1: {best_eval_f1:.4f} @ episode {best_eval_episode}')
        if hasattr(args, 'use_wandb') and args.use_wandb and best_eval_f1 < float('inf'):
            wandb.summary['Best/Attack F1'] = float(best_eval_f1)
            if best_eval_attack_drop is not None:
                wandb.summary['Best/Attack Relative Drop'] = float(best_eval_attack_drop)
            if best_eval_honest_mae is not None:
                wandb.summary['Best/Honest MAE'] = float(best_eval_honest_mae)
            if best_eval_honest_cer is not None:
                wandb.summary['Best/Honest CER'] = float(best_eval_honest_cer)
