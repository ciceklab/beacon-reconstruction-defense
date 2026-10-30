from __future__ import annotations
import numpy as np

def precompute_panel_ld_anchors(corr_matrix: np.ndarray, k: int=4, noise_threshold: float=0.3486) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(corr_matrix, dtype=np.float64)
    n = c.shape[0]
    k = min(k, max(n - 1, 0))
    if k == 0:
        return (np.zeros((n, 0), dtype=np.int32), np.zeros((n, 0), dtype=np.float32))
    idx_out = np.zeros((n, k), dtype=np.int32)
    val_out = np.zeros((n, k), dtype=np.float32)
    for i in range(n):
        row = c[i].copy()
        row[i] = 0.0
        abs_row = np.abs(row)
        order = np.argsort(-abs_row)
        chosen = 0
        for j in order:
            if chosen >= k:
                break
            if abs_row[j] < 1e-09:
                break
            idx_out[i, chosen] = j
            val_out[i, chosen] = float(row[j])
            chosen += 1
    return (idx_out, val_out)

def ld_polar_topk(signed_r: np.ndarray, k: int) -> np.ndarray:
    r = np.asarray(signed_r, dtype=np.float64).flatten()
    out = np.zeros(2 * k, dtype=np.float32)
    if r.size == 0:
        return out
    order = np.argsort(-np.abs(r))
    take = order[:k]
    for i, j in enumerate(take):
        out[2 * i] = float(np.clip(r[j], -1.0, 1.0))
        out[2 * i + 1] = float(np.clip(abs(r[j]), 0.0, 1.0))
    return out

def ld_history_features(corr_matrix: np.ndarray, snp_idx: int, history_indices: np.ndarray, k: int=4) -> np.ndarray:
    if len(history_indices) == 0:
        return np.zeros(2 * k, dtype=np.float32)
    h = np.asarray(history_indices, dtype=int)
    row = np.asarray(corr_matrix[snp_idx, h], dtype=np.float64)
    return ld_polar_topk(row, k)

def ld_panel_anchor_features(panel_idx: np.ndarray, panel_vals: np.ndarray, snp_idx: int, k: int=4) -> np.ndarray:
    out = np.zeros(2 * k, dtype=np.float32)
    if panel_idx.shape[1] == 0:
        return out
    idx = panel_idx[snp_idx]
    vals = panel_vals[snp_idx]
    for i in range(min(k, len(vals))):
        r = float(vals[i])
        out[2 * i] = float(np.clip(r, -1.0, 1.0))
        out[2 * i + 1] = float(np.clip(abs(r), 0.0, 1.0))
    return out

def compute_attacker_knowledge(corr_matrix: np.ndarray, history_indices: np.ndarray, true_freqs: np.ndarray, noisy_responses: np.ndarray, query_progress: float, rolling_util: float, hub_scores_history: np.ndarray | None=None, noise_threshold: float=0.3486) -> np.ndarray:
    c = np.asarray(corr_matrix, dtype=np.float64)
    n = c.shape[0]
    h = np.asarray(history_indices, dtype=int)
    out = np.zeros(8, dtype=np.float32)
    out[0] = float(np.clip(query_progress, 0.0, 1.0))
    out[6] = float(np.clip(1.0 - rolling_util, 0.0, 1.0))
    if len(h) == 0:
        out[4] = float(np.max(np.abs(c))) if n else 0.0
        out[7] = 0.0
        return out
    sub = c[np.ix_(h, h)]
    abs_sub = np.abs(sub)
    out[1] = float(np.mean(abs_sub ** 2))
    try:
        evals = np.linalg.eigvalsh(sub)
        lam_max = float(np.max(evals)) if evals.size else 0.0
    except np.linalg.LinAlgError:
        lam_max = 0.0
    out[2] = float(lam_max / max(len(h), 1))
    tf = np.asarray(true_freqs, dtype=np.float64).flatten()
    nr = np.asarray(noisy_responses, dtype=np.float64).flatten()
    if tf.size == len(h) and nr.size == len(h):
        out[3] = float(np.mean(np.abs(nr - tf)))
    queried_set = set(h.tolist())
    unqueried = [i for i in range(n) if i not in queried_set]
    if unqueried:
        risks = [float(np.max(np.abs(c[q, unqueried]))) for q in h]
        out[4] = float(np.max(risks))
    if hub_scores_history is not None and len(hub_scores_history) == len(h):
        hub_min = 0.25
        out[5] = float(np.mean(hub_scores_history >= hub_min))
    out[7] = float(np.clip(0.35 * out[1] + 0.25 * out[2] + 0.25 * out[3] + 0.15 * out[0], 0.0, 1.0))
    return out
