from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SnpManifest:
    prefix: str
    chromosome: int
    n_individuals: int
    positions: np.ndarray  
    rsids: np.ndarray  
    alt_counts: np.ndarray  

    def lookup_nearest(self, position: int, *, window_bp: int = 200_000) -> int | None:
        pos = int(position)
        idx = np.searchsorted(self.positions, pos)
        candidates: list[tuple[int, int]] = []
        for i in (idx - 1, idx, idx + 1):
            if 0 <= i < len(self.positions):
                dist = abs(int(self.positions[i]) - pos)
                if dist <= window_bp:
                    candidates.append((dist, i))
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[0])[1]

    def lookup_rsid(self, rsid: str) -> int | None:
        target = rsid if rsid.startswith("rs") else f"rs{rsid}"
        hits = np.where(self.rsids == target)[0]
        if hits.size:
            return int(hits[0])
        return None


def manifest_path(data_dir: str, prefix: str) -> str:
    return os.path.join(data_dir, f"{prefix}_snp_manifest.npz")


def infer_chromosome(prefix: str) -> int:
    match = re.search(r"chr(\d+)", prefix, re.I)
    if match:
        return int(match.group(1))
    return 0


def load_manifest(data_dir: str, prefix: str) -> SnpManifest | None:
    path = manifest_path(data_dir, prefix)
    if not os.path.isfile(path):
        return None
    blob = np.load(path, allow_pickle=True)
    return SnpManifest(
        prefix=str(blob["prefix"]),
        chromosome=int(blob["chromosome"]),
        n_individuals=int(blob["n_individuals"]),
        positions=np.asarray(blob["positions"], dtype=np.int64),
        rsids=np.asarray(blob["rsids"], dtype=object),
        alt_counts=np.asarray(blob["alt_counts"], dtype=np.int32),
    )


def save_manifest(manifest: SnpManifest, data_dir: str) -> str:
    path = manifest_path(data_dir, manifest.prefix)
    np.savez_compressed(
        path,
        prefix=manifest.prefix,
        chromosome=manifest.chromosome,
        n_individuals=manifest.n_individuals,
        positions=manifest.positions,
        rsids=manifest.rsids,
        alt_counts=manifest.alt_counts,
    )
    return path


def disease_window_start(manifest: SnpManifest, center_bp: int, window_size: int) -> int:
    
    if manifest.positions.size == 0:
        return 0
    center_idx = int(np.searchsorted(manifest.positions, int(center_bp)))
    center_idx = int(np.clip(center_idx, 0, manifest.positions.size - 1))
    return max(0, center_idx - window_size // 2)



HONEST_BP_WINDOWS: dict[int, tuple[int, int]] = {
    6: (31_800_000, 33_050_000),
    22: (18_800_000, 20_100_000),
    1: (54_000_000, 57_000_000),
}


def honest_bp_window_indices(manifest: SnpManifest, prefix: str) -> np.ndarray | None:
    
    chrom = infer_chromosome(prefix)
    span = HONEST_BP_WINDOWS.get(chrom)
    if span is None:
        return None
    lo_bp, hi_bp = span
    lo = int(np.searchsorted(manifest.positions, lo_bp))
    hi = int(np.searchsorted(manifest.positions, hi_bp, side="right"))
    if hi <= lo:
        return None
    return np.arange(lo, hi, dtype=int)


def default_honest_window_start(prefix: str, manifest: SnpManifest, window_size: int) -> int | None:
    
    chrom = infer_chromosome(prefix)
    if chrom == 6:
        return disease_window_start(manifest, 32_600_000, window_size)
    if chrom == 22:
        return disease_window_start(manifest, 19_700_000, window_size)
    if chrom == 1:
        return disease_window_start(manifest, 55_500_000, window_size)
    return None
