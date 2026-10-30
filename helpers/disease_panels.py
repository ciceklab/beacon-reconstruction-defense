from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from helpers.disease_catalog import DISEASE_BY_ID, DISEASE_CATALOG, DISEASE_IDS, DiseaseDefinition
from helpers.snp_manifest import SnpManifest, infer_chromosome, load_manifest


def list_diseases_for_chromosome(chromosome: int) -> list[DiseaseDefinition]:
    return [d for d in DISEASE_CATALOG if any(l.chromosome == chromosome for l in d.loci)]


def catalog_locus_global_indices(
    manifest: SnpManifest,
    *,
    gene_window_bp: int = 200_000,
) -> set[int]:
    
    chrom = manifest.chromosome
    pinned: set[int] = set()
    for disease in list_diseases_for_chromosome(chrom):
        for locus in disease.loci:
            if locus.chromosome != chrom:
                continue
            idx = manifest.lookup_rsid(str(locus.rsid))
            if idx is None:
                idx = manifest.lookup_nearest(int(locus.position), window_bp=gene_window_bp)
            if idx is not None:
                pinned.add(int(idx))
    return pinned


def ensure_disease_loci_in_pool(
    stratified_indices: np.ndarray,
    window_indices: np.ndarray,
    manifest: SnpManifest,
    gene_size: int,
    *,
    gene_window_bp: int = 200_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    
    rng = rng if rng is not None else np.random.default_rng()
    window_set = {int(x) for x in np.asarray(window_indices, dtype=int).reshape(-1)}
    pinned = {idx for idx in catalog_locus_global_indices(manifest, gene_window_bp=gene_window_bp) if idx in window_set}
    stratified = {int(x) for x in np.asarray(stratified_indices, dtype=int).reshape(-1)}
    merged = pinned | stratified
    if len(merged) > gene_size:
        extras = sorted(merged - pinned)
        keep_n = max(0, gene_size - len(pinned))
        if keep_n > 0 and extras:
            keep_extra = set(rng.choice(np.array(extras, dtype=int), size=min(keep_n, len(extras)), replace=False).tolist())
        else:
            keep_extra = set()
        merged = pinned | keep_extra
    elif len(merged) < gene_size:
        remaining = [int(x) for x in window_set if int(x) not in merged]
        need = gene_size - len(merged)
        if remaining and need > 0:
            add = rng.choice(np.array(remaining, dtype=int), size=min(need, len(remaining)), replace=False)
            merged |= {int(x) for x in np.atleast_1d(add).tolist()}
    return np.sort(np.array(sorted(merged), dtype=int))


def get_disease_panel(
    panel_id: str | None = None,
    *,
    num_queries: int | None = None,
    chromosome: int | None = None,
) -> dict[str, Any]:
    if panel_id is not None:
        disease = DISEASE_BY_ID[panel_id]
    elif chromosome is not None:
        candidates = list_diseases_for_chromosome(chromosome)
        if not candidates:
            raise KeyError(f"No disease catalog entries for chromosome {chromosome}")
        disease = random.choice(candidates)
    else:
        disease = random.choice(DISEASE_CATALOG)

    loci = [
        {
            "chromosome": l.chromosome,
            "position": l.position,
            "rsid": l.rsid,
            "gene": l.gene,
            "core": l.core,
        }
        for l in disease.loci
    ]
    return {
        "id": disease.disease_id,
        "name": disease.display_name,
        "num_queries": num_queries,
        "loci": loci,
        "filler_tiers": disease.filler_tiers,
        "rare_maf_max": disease.rare_maf_max,
        "low_freq_maf_max": disease.low_freq_maf_max,
    }


def _tier_fill_indices(
    panel_mafs: np.ndarray,
    exclude: set[int],
    tier: str,
    panel: dict[str, Any],
    n_need: int,
    rng: np.random.Generator,
) -> list[int]:
    if n_need <= 0:
        return []
    available = np.array([i for i in range(panel_mafs.size) if i not in exclude], dtype=int)
    if available.size == 0:
        return []
    maf = panel_mafs
    rare_max = float(panel["rare_maf_max"])
    low_max = float(panel["low_freq_maf_max"])
    if tier == "rare":
        mask = maf[available] <= rare_max
    elif tier == "low":
        mask = (maf[available] > rare_max) & (maf[available] <= low_max)
    else:
        mask = maf[available] > low_max
    pool = available[mask] if np.any(mask) else available
    n = min(n_need, pool.size)
    return rng.choice(pool, size=n, replace=False).tolist()


def resolve_panel_indices(
    mafs: torch.Tensor,
    panel: dict[str, Any],
    *,
    manifest: SnpManifest | None = None,
    pool_global_indices: np.ndarray | None = None,
    gene_window_bp: int = 200_000,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    
    mafs_flat = np.asarray(mafs.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    gene_size = mafs_flat.size
    if gene_size == 0:
        raise ValueError("Cannot resolve disease panel on empty MAF vector")

    num_queries = panel.get("num_queries")
    if num_queries is None:
        raise ValueError("panel['num_queries'] must be set before resolving indices")
    num_queries = int(num_queries)

    rng = rng if rng is not None else np.random.default_rng()

    global_to_local: dict[int, int] = {}
    if pool_global_indices is not None:
        for local_i, global_i in enumerate(np.asarray(pool_global_indices, dtype=int).reshape(-1)):
            global_to_local[int(global_i)] = int(local_i)

    def _global_to_local(global_idx: int | None) -> int | None:
        if global_idx is None:
            return None
        if pool_global_indices is None:
            return int(global_idx) if 0 <= int(global_idx) < gene_size else None
        return global_to_local.get(int(global_idx))

    selected: list[int] = []
    core_flags: list[bool] = []
    used: set[int] = set()
    resolved_loci: list[dict[str, Any]] = []

    if manifest is not None:
        for locus in panel.get("loci", []):
            if int(locus["chromosome"]) != manifest.chromosome:
                continue
            global_idx = manifest.lookup_rsid(str(locus["rsid"]))
            if global_idx is None:
                global_idx = manifest.lookup_nearest(int(locus["position"]), window_bp=gene_window_bp)
            local_idx = _global_to_local(global_idx)
            if local_idx is None or local_idx in used:
                continue
            selected.append(int(local_idx))
            core_flags.append(bool(locus.get("core", True)))
            used.add(int(local_idx))
            resolved_loci.append(
                {
                    **locus,
                    "global_index": int(global_idx),
                    "panel_index": int(local_idx),
                    "matched_rsid": str(manifest.rsids[global_idx]),
                    "matched_position": int(manifest.positions[global_idx]),
                }
            )
            if len(selected) >= num_queries:
                break

    if len(selected) < num_queries:
        n_fill = num_queries - len(selected)
        tiers = panel.get("filler_tiers", (0.35, 0.35, 0.30))
        rare_n = int(round(n_fill * tiers[0]))
        low_n = int(round(n_fill * tiers[1]))
        common_n = n_fill - rare_n - low_n
        for tier, n_need in (("rare", rare_n), ("low", low_n), ("common", common_n)):
            for idx in _tier_fill_indices(mafs_flat, used, tier, panel, n_need, rng):
                if len(selected) >= num_queries:
                    break
                selected.append(int(idx))
                core_flags.append(False)
                used.add(int(idx))

    if len(selected) < num_queries:
        remaining = [i for i in range(gene_size) if i not in used]
        if remaining:
            extra = rng.choice(
                np.array(remaining, dtype=int),
                size=min(num_queries - len(selected), len(remaining)),
                replace=False,
            )
            for idx in extra.tolist():
                selected.append(int(idx))
                core_flags.append(False)

    if len(selected) != num_queries:
        raise ValueError(
            f"Panel {panel['id']} resolved {len(selected)} queries, expected {num_queries}."
        )

    order = rng.permutation(len(selected))
    selected = [selected[i] for i in order]
    panel["resolved_loci"] = [resolved_loci[i] for i in order if i < len(resolved_loci)]
    panel["gene_core_mask"] = [core_flags[i] for i in order]
    return torch.tensor(selected, dtype=torch.long)


def resolve_honest_queries(
    mafs: torch.Tensor,
    *,
    data_dir: str | None,
    pop_prefix: str | None,
    disease_id: str | None,
    max_queries: int,
    pool_global_indices: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    gene_window_bp: int = 200_000,
) -> tuple[torch.Tensor, dict[str, Any]]:
    chrom = infer_chromosome(pop_prefix or "")
    panel = get_disease_panel(disease_id, num_queries=max_queries, chromosome=chrom or None)
    manifest = load_manifest(data_dir, pop_prefix) if data_dir and pop_prefix else None
    if manifest is None and chrom > 0:
        raise FileNotFoundError(
            f"No SNP manifest for prefix '{pop_prefix}' under {data_dir}. "
            f"Run: python scripts/build_snp_manifest.py --data-dir {data_dir} --prefix {pop_prefix} --download-vcf"
        )
    queries = resolve_panel_indices(
        mafs,
        panel,
        manifest=manifest,
        pool_global_indices=pool_global_indices,
        gene_window_bp=gene_window_bp,
        rng=rng,
    )
    return queries, panel



DISEASE_PANELS = [
    {"id": d.disease_id, "name": d.display_name, "num_queries": None, "loci": []}
    for d in DISEASE_CATALOG
]
