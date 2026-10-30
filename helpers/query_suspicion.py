from __future__ import annotations

import numpy as np

AGENT_SUSPICION_DIM = 5
SUSPICION_FEATURE_DIM = 7
LD_THRESHOLD_PERCENTILE = 90.0
DETECTOR_FEATURE_NAMES = (
    "top_decile_frac",
    "attack_composition",
    "ld_density",
    "mean_hub",
    "maf_concentration",
    "mean_abs_ld",
    "query_proximity",
)

_DEFAULT_WEIGHTS = np.array([0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float64)


def panel_ld_threshold(
    corr_matrix: np.ndarray,
    percentile: float = LD_THRESHOLD_PERCENTILE,
) -> float:
    
    corr = np.abs(np.asarray(corr_matrix, dtype=np.float64))
    if corr.ndim != 2 or min(corr.shape) < 2:
        return 0.0
    triu = corr[np.triu_indices(corr.shape[0], k=1)]
    finite = triu[np.isfinite(triu)]
    if finite.size == 0:
        return 0.0
    return float(np.clip(np.percentile(finite, percentile), 0.0, 1.0))


def compute_suspicion_features(
    *,
    queried_indices: np.ndarray,
    maf_values: np.ndarray,
    corr_matrix: np.ndarray,
    hub_scores_history: list[float],
    current_hub: float,
    ld_noise_threshold: float | None = None,
) -> np.ndarray:
    
    queried = np.asarray(queried_indices, dtype=int).reshape(-1)
    maf = np.asarray(maf_values, dtype=np.float64).reshape(-1)
    n_snps = len(maf)
    if n_snps == 0 or queried.size == 0:
        return np.zeros(SUSPICION_FEATURE_DIM, dtype=np.float32)

    queried = np.clip(queried, 0, n_snps - 1)
    query_mafs = maf[queried]
    if ld_noise_threshold is None:
        ld_noise_threshold = panel_ld_threshold(corr_matrix)

    
    top_decile_cutoff = float(np.percentile(maf, 90))
    top_decile_frac = float(np.mean(query_mafs >= top_decile_cutoff))

    
    
    bottom_decile_cutoff = float(np.percentile(maf, 10))
    low_decile_frac = float(np.mean(query_mafs <= bottom_decile_cutoff))
    dual_tail = 4.0 * top_decile_frac * low_decile_frac
    high_only_mining = top_decile_frac * (1.0 - low_decile_frac)
    attack_composition = float(np.clip(max(dual_tail, high_only_mining * 0.85), 0.0, 1.0))

    
    
    
    if queried.size >= 2:
        sub = np.abs(np.asarray(corr_matrix, dtype=np.float64)[np.ix_(queried, queried)])
        triu = sub[np.triu_indices(len(queried), k=1)]
        if triu.size > 0:
            ld_density = float(np.mean(triu > ld_noise_threshold))
            mean_abs_ld = float(np.clip(np.mean(triu), 0.0, 1.0))
        else:
            ld_density = 0.0
            mean_abs_ld = 0.0
        mean_step = float(np.mean(np.abs(np.diff(queried.astype(np.float64)))))
        query_proximity = float(np.clip(1.0 - mean_step / max(n_snps - 1, 1), 0.0, 1.0))
    else:
        ld_density = 0.0
        mean_abs_ld = 0.0
        query_proximity = 0.0

    
    hubs = list(hub_scores_history) + [float(current_hub)]
    mean_hub = float(np.clip(np.mean(hubs), 0.0, 1.0))

    
    if queried.size >= 2:
        hist, _ = np.histogram(query_mafs, bins=5, range=(0.0, 1.0), density=False)
        probs = hist.astype(np.float64)
        total = probs.sum()
        if total > 0:
            probs = probs / total
            nonzero = probs[probs > 0]
            entropy = float(-np.sum(nonzero * np.log(nonzero + 1e-12)))
            max_entropy = float(np.log(5))
            concentration = 1.0 - entropy / max(max_entropy, 1e-12)
        else:
            concentration = 0.0
    else:
        concentration = 0.0

    return np.array(
        [
            float(np.clip(top_decile_frac, 0.0, 1.0)),
            attack_composition,
            float(np.clip(ld_density, 0.0, 1.0)),
            mean_hub,
            float(np.clip(concentration, 0.0, 1.0)),
            mean_abs_ld,
            query_proximity,
        ],
        dtype=np.float32,
    )


def aggregate_suspicion(features: np.ndarray, weights: np.ndarray | None = None) -> float:
    
    vec = np.asarray(features, dtype=np.float64).reshape(-1)
    if vec.size < AGENT_SUSPICION_DIM:
        raise ValueError(f"Expected at least {AGENT_SUSPICION_DIM} suspicion features, got {vec.size}")
    scored = vec[:AGENT_SUSPICION_DIM]
    w = _DEFAULT_WEIGHTS if weights is None else np.asarray(weights, dtype=np.float64)
    if w.size != scored.size:
        raise ValueError(f"Expected {scored.size} weights, got {w.size}")
    return float(np.clip(np.dot(w, scored), 0.0, 1.0))


def compute_suspicion_gate(
    suspicion_score: float,
    *,
    mode: str = "authorization",
    threshold: float = 0.45,
    power: float = 1.25,
    honest_ceiling: float = 0.42,
    attack_floor: float = 0.65,
) -> float:
    
    score = float(np.clip(suspicion_score, 0.0, 1.0))
    if mode == "hard":
        return 1.0 if score >= threshold else 0.0
    if mode == "authorization":
        
        
        if score <= honest_ceiling:
            return 0.0
        if score >= threshold:
            span = max(1.0 - threshold, 1e-6)
            t = (score - threshold) / span
            return float(np.clip(attack_floor + (1.0 - attack_floor) * t, 0.0, 1.0))
        border = max(threshold - honest_ceiling, 1e-6)
        return float(np.clip(0.2 * (score - honest_ceiling) / border, 0.0, 0.2))
    if mode == "soft":
        numer = max(0.0, score - threshold)
        scaled = (numer / max(1.0 - threshold, 1e-6)) ** power
        return float(np.clip(scaled, 0.0, 1.0))
    raise ValueError(f"Unknown suspicion gate mode {mode!r}; use 'authorization', 'soft', or 'hard'.")
