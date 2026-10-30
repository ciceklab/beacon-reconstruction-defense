import numpy as np
import torch

def calculate_genotype_correlations(B_soft: torch.Tensor, num_indiv: int) -> np.ndarray:
    p11 = torch.matmul(B_soft, B_soft.T)
    row_sums = torch.sum(B_soft, dim=1, keepdim=True)
    total_matches = 2 * p11 + num_indiv - row_sums - row_sums.T
    corr = total_matches / num_indiv
    return corr.detach().cpu().numpy()

def apply_noisy_frequencies_to_genotypes(beacon_case: torch.Tensor, target_freqs: np.ndarray) -> torch.Tensor:
    case = beacon_case.clone().float()
    n_ind, k = case.shape
    for j in range(k):
        target = float(np.clip(target_freqs[j], 0.0, 1.0))
        col = case[:, j]
        current = col.mean().item()
        target_count = int(np.round(target * n_ind))
        current_count = int(col.sum().item())
        delta = target_count - current_count
        if delta > 0:
            zeros = (col < 0.5).nonzero(as_tuple=True)[0]
            if len(zeros) > 0:
                flip_idx = zeros[torch.randperm(len(zeros))[:delta]]
                case[flip_idx, j] = 1.0
        elif delta < 0:
            ones = (col > 0.5).nonzero(as_tuple=True)[0]
            if len(ones) > 0:
                flip_idx = ones[torch.randperm(len(ones))[:-delta]]
                case[flip_idx, j] = 0.0
    return case

def subset_corr_mismatch(beacon_case: torch.Tensor, queried_indices: np.ndarray, noisy_freqs: np.ndarray, target_corr: np.ndarray, device: str='cpu') -> float:
    q_idx = np.asarray(queried_indices, dtype=int)
    if len(q_idx) < 2:
        return 0.0
    freqs = np.asarray(noisy_freqs, dtype=np.float64).flatten()
    B_pert = apply_noisy_frequencies_to_genotypes(beacon_case[:, q_idx], freqs)
    n_ind = beacon_case.shape[0]
    est = calculate_genotype_correlations(B_pert.T.to(device), n_ind)
    C_target = target_corr[np.ix_(q_idx, q_idx)]
    return float(np.linalg.norm(est - C_target, ord='fro'))
