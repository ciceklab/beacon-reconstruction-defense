from __future__ import annotations

import numpy as np

HEURISTIC_POLICIES = ("uniform", "full", "common", "proportional")

RARE_MAF_MAX = 0.05


def heuristic_action(
    policy: str,
    *,
    maf: float,
    maf_percentile: float,
    rng: np.random.Generator,
    common_maf_threshold: float = 0.30,
) -> float:
    
    if policy not in HEURISTIC_POLICIES:
        raise ValueError(f"Unknown heuristic policy {policy!r}; use one of {HEURISTIC_POLICIES}")
    sign = 1.0 if rng.random() < 0.5 else -1.0
    if policy == "uniform":
        return float(rng.uniform(-1.0, 1.0))
    if policy == "full":
        return sign
    if policy == "common":
        return sign if float(maf) > float(common_maf_threshold) else 0.0
    return sign * float(np.clip(maf_percentile, 0.0, 1.0))


def gated_heuristic_action(maf: float, uniform: float) -> float:
    
    if float(maf) <= RARE_MAF_MAX:
        return 0.0
    return 1.0 if float(uniform) < 0.5 else -1.0
