from __future__ import annotations

import numpy as np

MATCH_ORDER = ("shuffled", "rare-first", "common-first")


def _ld_scores(
    corr_matrix: np.ndarray,
    candidates: np.ndarray,
    selected: list[int],
) -> np.ndarray:
    
    if len(selected) == 0:
        return np.abs(corr_matrix[candidates, :]).mean(axis=1)
    return np.abs(corr_matrix[np.ix_(candidates, np.asarray(selected, dtype=int))]).mean(axis=1)


def maf_matched_queries(
    target_mafs: np.ndarray,
    panel_mafs: np.ndarray,
    corr_matrix: np.ndarray,
    *,
    rng: np.random.Generator,
    ld_greedy: bool = True,
    tolerance: float = 0.01,
    candidate_floor: int = 8,
    match_order: str = "shuffled",
) -> np.ndarray:
    
    targets = np.asarray(target_mafs, dtype=np.float64).reshape(-1)
    maf = np.asarray(panel_mafs, dtype=np.float64).reshape(-1)
    corr = np.asarray(corr_matrix, dtype=np.float64)
    n_panel = maf.size
    if targets.size == 0:
        raise ValueError("camouflage needs at least one target MAF")
    if targets.size > n_panel:
        raise ValueError(
            f"camouflage needs {targets.size} queries but the panel has {n_panel} variants"
        )
    if corr.shape[0] < n_panel or corr.shape[1] < n_panel:
        raise ValueError(
            f"agreement matrix {corr.shape} is smaller than the {n_panel}-variant panel"
        )
    if match_order not in MATCH_ORDER:
        raise ValueError(f"Unknown match_order {match_order!r}; use one of {MATCH_ORDER}")
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    if candidate_floor < 1:
        raise ValueError(f"candidate_floor must be at least 1, got {candidate_floor}")

    if match_order == "rare-first":
        order = np.argsort(targets, kind="stable")
    elif match_order == "common-first":
        order = np.argsort(-targets, kind="stable")
    else:
        order = rng.permutation(targets.size)

    available = np.ones(n_panel, dtype=bool)
    selected: list[int] = []
    for position in order:
        target = float(targets[position])
        band = float(tolerance)
        while True:
            candidates = np.flatnonzero(available & (np.abs(maf - target) <= band))
            if candidates.size >= candidate_floor or band > 1.0:
                break
            band *= 2.0
        if candidates.size == 0:
            
            remaining = np.flatnonzero(available)
            candidates = remaining[np.argsort(np.abs(maf[remaining] - target))[:candidate_floor]]
        if ld_greedy:
            pick = int(candidates[int(np.argmax(_ld_scores(corr, candidates, selected)))])
        else:
            pick = int(rng.choice(candidates))
        selected.append(pick)
        available[pick] = False

    return np.array(selected, dtype=int)


def histogram_edges(n_bins: int = 10) -> np.ndarray:
    
    if int(n_bins) < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    return np.linspace(0.0, 1.0, int(n_bins) + 1)


def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    
    index = np.digitize(np.asarray(values, dtype=np.float64).reshape(-1), edges, right=False) - 1
    return np.clip(index, 0, edges.size - 2)


def _counts_for_draw(target_mafs: np.ndarray, n_draw: int, edges: np.ndarray) -> np.ndarray:
    
    counts, _ = np.histogram(np.asarray(target_mafs, dtype=np.float64).reshape(-1), bins=edges)
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("honest MAF histogram is empty")
    if n_draw < 1:
        raise ValueError(f"n_draw must be positive, got {n_draw}")
    raw = counts.astype(np.float64) / float(total) * float(n_draw)
    assigned = np.floor(raw).astype(int)
    shortfall = int(n_draw - assigned.sum())
    remainder = np.argsort(-(raw - assigned))
    for bin_id in remainder[:shortfall]:
        assigned[int(bin_id)] += 1
    return assigned


def histogram_matched_queries(
    target_mafs: np.ndarray,
    panel_mafs: np.ndarray,
    corr_matrix: np.ndarray,
    *,
    n_draw: int,
    rng: np.random.Generator,
    n_bins: int = 10,
    ld_greedy: bool = True,
) -> np.ndarray:
    
    maf = np.asarray(panel_mafs, dtype=np.float64).reshape(-1)
    corr = np.asarray(corr_matrix, dtype=np.float64)
    if maf.size == 0:
        raise ValueError("camouflage panel is empty")
    if n_draw > maf.size:
        raise ValueError(f"camouflage needs {n_draw} queries but the panel has {maf.size}")
    if corr.shape[0] < maf.size or corr.shape[1] < maf.size:
        raise ValueError(f"agreement matrix {corr.shape} is smaller than the {maf.size}-variant panel")
    edges = histogram_edges(n_bins)
    need = _counts_for_draw(target_mafs, n_draw, edges)
    panel_bins = _bin_index(maf, edges)
    available = np.ones(maf.size, dtype=bool)
    selected: list[int] = []
    bin_order = rng.permutation(need.size)
    for bin_id in bin_order:
        quota = int(need[int(bin_id)])
        for _ in range(quota):
            candidates = _bin_candidates(int(bin_id), panel_bins, available, need.size)
            if candidates.size == 0:
                raise ValueError("camouflage ran out of unused variants")
            if ld_greedy:
                pick = int(candidates[int(np.argmax(_ld_scores(corr, candidates, selected)))])
            else:
                pick = int(rng.choice(candidates))
            selected.append(pick)
            available[pick] = False
    order = rng.permutation(len(selected))
    return np.asarray(selected, dtype=int)[order]


def _bin_candidates(
    bin_id: int,
    panel_bins: np.ndarray,
    available: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    
    radius = 0
    while radius <= n_bins:
        wanted = [
            bin_id + offset
            for offset in range(-radius, radius + 1)
            if 0 <= bin_id + offset < n_bins
        ]
        candidates = np.flatnonzero(available & np.isin(panel_bins, wanted))
        if candidates.size > 0:
            return candidates
        radius += 1
    return np.flatnonzero(available)


def maf_profile(values: np.ndarray, edges: np.ndarray | None = None) -> np.ndarray:
    
    if edges is None:
        edges = np.array([0.0, 0.01, 0.05, 0.1, 0.3, 1.0], dtype=np.float64)
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64).reshape(-1), bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros(counts.size, dtype=np.float64)
    return counts.astype(np.float64) / float(total)
