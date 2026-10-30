from __future__ import annotations

import numpy as np
import pandas as pd

MAF_CATEGORIES = ["Ultra-Rare", "Rare", "Low-Freq", "Common", "Highly-Common"]
STRATA_COMMON_TO_RARE = ["Highly-Common", "Common", "Low-Freq", "Rare", "Ultra-Rare"]
STRATUM_LABELS = {
    "Highly-Common": "Highly common\nMAF > 0.30",
    "Common": "Common\n0.10 < MAF ≤ 0.30",
    "Low-Freq": "Low frequency\n0.05 < MAF ≤ 0.10",
    "Rare": "Rare\n0.01 < MAF ≤ 0.05",
    "Ultra-Rare": "Ultra-rare\nMAF ≤ 0.01",
}


def assign_strata(mafs: np.ndarray) -> np.ndarray:
    
    values = np.asarray(mafs, dtype=np.float64)
    strata = np.full(values.shape, "Highly-Common", dtype=object)
    strata[values <= 0.3] = "Common"
    strata[values <= 0.1] = "Low-Freq"
    strata[values <= 0.05] = "Rare"
    strata[values <= 0.01] = "Ultra-Rare"
    return strata


def select_stratified_indices(
    mafs: np.ndarray,
    pool_indices: np.ndarray,
    gene_size: int,
    rng,
) -> np.ndarray:
    
    pool_indices = np.asarray(pool_indices, dtype=int)
    if gene_size >= len(pool_indices):
        return np.sort(pool_indices.copy())

    local_mafs = np.asarray(mafs)[pool_indices]
    categories = pd.Series(assign_strata(local_mafs))
    fractions = categories.value_counts() / len(pool_indices)
    allocations = (fractions * gene_size).round().astype(int)
    difference = gene_size - int(allocations.sum())
    if difference:
        allocations[allocations.idxmax()] += difference

    selected: list[int] = []
    for category, count in allocations.items():
        if count <= 0:
            continue
        local_candidates = categories[categories == category].index.values
        global_candidates = pool_indices[local_candidates]
        if count > len(global_candidates):
            raise ValueError(
                f"Need {count} {category} SNPs but pool has "
                f"{len(global_candidates)}"
            )
        selected.extend(
            rng.choice(global_candidates, size=count, replace=False).tolist()
        )
    return np.sort(np.asarray(selected, dtype=int))
