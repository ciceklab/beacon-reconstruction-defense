from __future__ import annotations
import numpy as np
import scipy.stats as stats

def evaluate_utility_preservation(maf_values, injected_noise, rare_threshold: float=0.05, common_threshold: float=0.2) -> dict[str, float]:
    maf_array = np.asarray(maf_values, dtype=np.float64)
    noise_array = np.abs(np.asarray(injected_noise, dtype=np.float64))
    n = len(maf_array)
    if n == 0:
        return _empty_metrics()
    if np.std(noise_array) > 0 and np.std(maf_array) > 0:
        spearman, _ = stats.spearmanr(maf_array, noise_array)
    else:
        spearman = 0.0
    meaningful_noise_mask = noise_array > 0.01
    if np.any(meaningful_noise_mask):
        min_perturbed_maf = float(np.min(maf_array[meaningful_noise_mask]))
    else:
        min_perturbed_maf = float(np.min(maf_array))
    clinical_mask = maf_array <= rare_threshold
    clinical_mafs = maf_array[clinical_mask]
    clinical_noises = noise_array[clinical_mask]
    if len(clinical_mafs) > 0:
        ruined_queries = int(np.sum(clinical_noises > clinical_mafs))
        cer_ratio = float(ruined_queries / len(clinical_mafs))
        avg_clinical_noise = float(np.mean(clinical_noises))
        max_rare_noise = float(np.max(clinical_noises))
        mean_rare_noise = float(np.mean(clinical_noises))
        frac_rare_touched = float(np.mean(clinical_noises > 0.01))
    else:
        cer_ratio = 0.0
        avg_clinical_noise = 0.0
        max_rare_noise = 0.0
        mean_rare_noise = 0.0
        frac_rare_touched = 0.0
    common_mask = maf_array >= common_threshold
    if np.any(common_mask):
        mean_common_noise = float(np.mean(noise_array[common_mask]))
    else:
        mean_common_noise = float(np.mean(noise_array))
    total_noise = float(np.sum(noise_array))
    if total_noise > 1e-12:
        rare_mass_frac = float(np.sum(noise_array[clinical_mask]) / total_noise) if np.any(clinical_mask) else 0.0
        common_mass_frac = float(np.sum(noise_array[common_mask]) / total_noise) if np.any(common_mask) else 0.0
    else:
        rare_mass_frac = 0.0
        common_mass_frac = 0.0
    if total_noise > 1e-12 and np.std(maf_array) > 0:
        maf_weighted_mean_noise = float(np.average(noise_array, weights=maf_array + 1e-06))
    else:
        maf_weighted_mean_noise = 0.0
    if mean_rare_noise > 1e-09 and mean_common_noise > 1e-09:
        rare_to_common_noise_ratio = mean_rare_noise / mean_common_noise
    elif mean_rare_noise <= 1e-09 and mean_common_noise > 1e-09:
        rare_to_common_noise_ratio = 0.0
    else:
        rare_to_common_noise_ratio = 1.0
    rarity_preservation = float(np.clip(1.0 - rare_to_common_noise_ratio, 0.0, 1.0))
    quintile_noise = []
    if n >= 5:
        bins = np.percentile(maf_array, [0, 20, 40, 60, 80, 100])
        for i in range(5):
            lo, hi = (bins[i], bins[i + 1])
            mask = (maf_array >= lo) & (maf_array <= hi if i == 4 else maf_array < hi)
            if np.any(mask):
                quintile_noise.append(float(np.mean(noise_array[mask])))
            else:
                quintile_noise.append(0.0)
    else:
        quintile_noise = [0.0] * 5
    queried = {'Absolute_Min_Targeted': float(np.min(maf_array)), 'Absolute_Max_Targeted': float(np.max(maf_array)), 'Mean_Targeted': float(np.mean(maf_array)), 'Median_Targeted': float(np.median(maf_array))}
    return {'Minimum_Perturbed_MAF': min_perturbed_maf, 'Noise_MAF_Spearman_Corr': float(spearman), 'Clinical_Error_Ratio': cer_ratio, 'Avg_Clinical_Noise': avg_clinical_noise, 'Max_Rare_Noise': max_rare_noise, 'Mean_Rare_Noise': mean_rare_noise, 'Mean_Common_Noise': mean_common_noise, 'Frac_Rare_Variants_Noised': frac_rare_touched, 'Rare_Noise_Mass_Fraction': rare_mass_frac, 'Common_Noise_Mass_Fraction': common_mass_frac, 'Rare_to_Common_Noise_Ratio': rare_to_common_noise_ratio, 'Rarity_Preservation_Score': rarity_preservation, 'MAF_Weighted_Mean_Noise': maf_weighted_mean_noise, 'Quintile1_Rare_Mean_Noise': quintile_noise[0], 'Quintile5_Common_Mean_Noise': quintile_noise[4], 'Queried_MAF': queried}

def evaluate_ld_targeting(hub_scores, injected_noise, hub_threshold: float=0.25) -> dict[str, float]:
    hub_array = np.asarray(hub_scores, dtype=np.float64)
    noise_array = np.abs(np.asarray(injected_noise, dtype=np.float64))
    n = len(hub_array)
    if n == 0 or len(noise_array) != n:
        return _empty_ld_metrics()
    if np.std(noise_array) > 0 and np.std(hub_array) > 0:
        spearman, _ = stats.spearmanr(hub_array, noise_array)
    else:
        spearman = 0.0
    hub_mask = hub_array >= hub_threshold
    low_hub_mask = ~hub_mask
    total_noise = float(np.sum(noise_array))
    if total_noise > 1e-12:
        hub_mass_frac = float(np.sum(noise_array[hub_mask]) / total_noise) if np.any(hub_mask) else 0.0
        low_hub_mass_frac = float(np.sum(noise_array[low_hub_mask]) / total_noise) if np.any(low_hub_mask) else 0.0
    else:
        hub_mass_frac = 0.0
        low_hub_mass_frac = 0.0
    mean_hub_noise = float(np.mean(noise_array[hub_mask])) if np.any(hub_mask) else 0.0
    mean_low_hub_noise = float(np.mean(noise_array[low_hub_mask])) if np.any(low_hub_mask) else 0.0
    if mean_low_hub_noise > 1e-09 and mean_hub_noise > 1e-09:
        hub_to_low_ratio = mean_hub_noise / mean_low_hub_noise
    elif mean_low_hub_noise <= 1e-09 and mean_hub_noise > 1e-09:
        hub_to_low_ratio = 0.0
    else:
        hub_to_low_ratio = 1.0
    hub_targeting_score = float(np.clip(hub_mass_frac, 0.0, 1.0))
    frac_low_hub_noised = float(np.mean(noise_array[low_hub_mask] > 0.01)) if np.any(low_hub_mask) else 0.0
    frac_hub_noised = float(np.mean(noise_array[hub_mask] > 0.01)) if np.any(hub_mask) else 0.0
    return {'Noise_Hub_Spearman_Corr': float(spearman), 'Hub_Noise_Mass_Fraction': hub_mass_frac, 'LowHub_Noise_Mass_Fraction': low_hub_mass_frac, 'Mean_Hub_Noise': mean_hub_noise, 'Mean_LowHub_Noise': mean_low_hub_noise, 'Hub_to_LowHub_Noise_Ratio': hub_to_low_ratio, 'Hub_Targeting_Score': hub_targeting_score, 'Frac_LowHub_Queries_Noised': frac_low_hub_noised, 'Frac_Hub_Queries_Noised': frac_hub_noised}

def _empty_ld_metrics() -> dict[str, float]:
    return {'Noise_Hub_Spearman_Corr': 0.0, 'Hub_Noise_Mass_Fraction': 0.0, 'LowHub_Noise_Mass_Fraction': 0.0, 'Mean_Hub_Noise': 0.0, 'Mean_LowHub_Noise': 0.0, 'Hub_to_LowHub_Noise_Ratio': 1.0, 'Hub_Targeting_Score': 0.0, 'Frac_LowHub_Queries_Noised': 0.0, 'Frac_Hub_Queries_Noised': 0.0}

def evaluate_episode_utility(maf_values, injected_noise, hub_scores=None, *, rare_threshold: float=0.05, common_threshold: float=0.2, hub_threshold: float=0.25) -> dict[str, float]:
    metrics = evaluate_utility_preservation(maf_values, injected_noise, rare_threshold=rare_threshold, common_threshold=common_threshold)
    if hub_scores is not None and len(hub_scores) > 0:
        metrics.update(evaluate_ld_targeting(hub_scores, injected_noise, hub_threshold=hub_threshold))
    return metrics

def _empty_metrics() -> dict[str, float]:
    return {'Minimum_Perturbed_MAF': 1.0, 'Noise_MAF_Spearman_Corr': 0.0, 'Clinical_Error_Ratio': 0.0, 'Avg_Clinical_Noise': 0.0, 'Max_Rare_Noise': 0.0, 'Mean_Rare_Noise': 0.0, 'Mean_Common_Noise': 0.0, 'Frac_Rare_Variants_Noised': 0.0, 'Rare_Noise_Mass_Fraction': 0.0, 'Common_Noise_Mass_Fraction': 0.0, 'Rare_to_Common_Noise_Ratio': 1.0, 'Rarity_Preservation_Score': 0.0, 'MAF_Weighted_Mean_Noise': 0.0, 'Quintile1_Rare_Mean_Noise': 0.0, 'Quintile5_Common_Mean_Noise': 0.0, 'Queried_MAF': {'Absolute_Min_Targeted': 0.0, 'Absolute_Max_Targeted': 0.0, 'Mean_Targeted': 0.0, 'Median_Targeted': 0.0}}

def wandb_utility_log_dict(metrics: dict) -> dict[str, float]:
    utility_keys = tuple(_empty_metrics().keys()) + tuple(_empty_ld_metrics().keys())
    out: dict[str, float] = {}
    for key in utility_keys:
        if key not in metrics:
            continue
        val = metrics[key]
        if key == 'Queried_MAF' and isinstance(val, dict):
            for subk, subv in val.items():
                out[f'Queried_MAF/{subk}'] = float(subv)
        else:
            out[f'Utility/{key}'] = float(val)
    return out
UTILITY_KEYS = ('Rarity_Preservation_Score', 'Rare_to_Common_Noise_Ratio', 'Clinical_Error_Ratio', 'Frac_Rare_Variants_Noised', 'Noise_MAF_Spearman_Corr', 'Common_Noise_Mass_Fraction', 'Noise_Hub_Spearman_Corr', 'Hub_Noise_Mass_Fraction', 'Hub_Targeting_Score', 'Frac_LowHub_Queries_Noised')

def build_wandb_log_dict(*, episode: int, final_info: dict, update_metrics: dict, best_eval_f1: float, eval_f1_val: float | None, minimal: bool, total_episode_reward: float | None=None, explore_noise: float | None=None) -> dict[str, float]:
    log_dict: dict[str, float] = {'Episode': episode, 'Attacker F1 Score': float(final_info.get('f1', 0.0)), 'rolling_mae_utilization': float(final_info.get('rolling_mae_utilization', 0.0)), 'Agent Noise (MAE)': float(final_info.get('mae', 0.0)), 'Budget/Cap Events': float(final_info.get('episode_budget_penalties', 0)), 'Loss/Critic': float(update_metrics.get('critic_loss', 0.0))}
    if best_eval_f1 < float('inf'):
        log_dict['best_eval_f1'] = float(best_eval_f1)
    if eval_f1_val is not None:
        log_dict['eval_f1'] = float(eval_f1_val)
    if not minimal:
        if total_episode_reward is not None:
            log_dict['Total Episode Reward'] = float(total_episode_reward)
        log_dict['Reward Components/Privacy Reward'] = float(final_info.get('privacy_reward', 0.0))
        log_dict['Loss/Actor'] = float(update_metrics.get('actor_loss', 0.0))
        if explore_noise is not None:
            log_dict['Exploration Noise'] = float(explore_noise)
    if 'Noise_MAF_Spearman_Corr' in final_info:
        log_dict.update(wandb_utility_log_dict(final_info))
    return log_dict
