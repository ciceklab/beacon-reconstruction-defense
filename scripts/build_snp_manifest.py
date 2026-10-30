from __future__ import annotations

import argparse
import gzip
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from helpers.snp_manifest import SnpManifest, infer_chromosome, save_manifest

DEFAULT_PANEL = os.path.join(ROOT, "data/chr22/integrated_call_samples_v3.20130502.ALL.panel")
DEFAULT_VCF_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
    "ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
)


def _load_super_pop_samples(panel_path: str, super_pop: str) -> set[str]:
    samples: set[str] = set()
    with open(panel_path) as handle:
        handle.readline()
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[2] == super_pop:
                samples.add(parts[0])
    if not samples:
        raise ValueError(f"No {super_pop} samples found in {panel_path}")
    return samples


def _parse_vcf_alt_counts(
    vcf_path: str,
    super_pop_samples: set[str],
) -> tuple[list[int], list[str], list[int]]:
    positions: list[int] = []
    rsids: list[str] = []
    alt_counts: list[int] = []

    with gzip.open(vcf_path, "rt") as handle:
        eur_idx: list[int] | None = None
        for line in handle:
            if line.startswith("#CHROM"):
                vcf_samples = line.strip().split("\t")[9:]
                eur_idx = [i for i, sample in enumerate(vcf_samples) if sample in super_pop_samples]
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            pos = int(parts[1])
            rsid = parts[2]
            genotypes = parts[9:]
            alt = 0
            for col in eur_idx or []:
                gt = genotypes[col].replace("|", "/").split("/")
                if gt[0] == "." or gt[1] == ".":
                    continue
                if gt[0] != "0" or gt[1] != "0":
                    alt += 1
            positions.append(pos)
            rsids.append(rsid if rsid != "." else f"chr{parts[0]}:{pos}")
            alt_counts.append(alt)
    return positions, rsids, alt_counts


def _align_panel_to_vcf(
    panel_mafs: np.ndarray,
    panel_alt_counts: np.ndarray,
    vcf_positions: list[int],
    vcf_rsids: list[str],
    vcf_alt_counts: list[int],
    n_individuals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_panel = len(panel_mafs)
    out_pos = np.zeros(n_panel, dtype=np.int64)
    out_rsid = np.empty(n_panel, dtype=object)
    out_alt = np.zeros(n_panel, dtype=np.int32)

    vi = 0
    for j in range(n_panel):
        need = int(panel_alt_counts[j])
        target_maf = float(panel_mafs[j])
        while vi < len(vcf_positions):
            pos = vcf_positions[vi]
            rsid = vcf_rsids[vi]
            alt503 = vcf_alt_counts[vi]
            if alt503 >= need and abs(target_maf - need / n_individuals) < 1e-8:
                out_pos[j] = pos
                out_rsid[j] = rsid
                out_alt[j] = need
                vi += 1
                break
            vi += 1
        else:
            raise RuntimeError(
                f"Failed to align panel SNP index {j} (alt_count={need}, maf={target_maf:.6f}). "
                "VCF exhausted before panel end."
            )
    return out_pos, out_rsid, out_alt


def build_manifest(
    *,
    data_dir: str,
    prefix: str,
    vcf_path: str,
    panel_path: str,
    super_pop: str,
) -> SnpManifest:
    mafs_path = os.path.join(data_dir, f"{prefix}_cleaned_mafs.npy")
    binary_path = os.path.join(data_dir, f"{prefix}_cleaned_binary.npy")
    if not os.path.isfile(mafs_path) or not os.path.isfile(binary_path):
        raise FileNotFoundError(f"Missing panel arrays for prefix '{prefix}' in {data_dir}")

    panel_mafs = np.load(mafs_path)
    binary = np.load(binary_path)
    if binary.ndim != 2 or binary.shape[1] != panel_mafs.size:
        raise ValueError(f"Shape mismatch: binary {binary.shape}, mafs {panel_mafs.shape}")

    maf_from_binary = binary.mean(axis=0)
    if np.max(np.abs(maf_from_binary - panel_mafs)) > 1e-6:
        raise ValueError("Stored MAFs do not match binary matrix column means.")

    super_samples = _load_super_pop_samples(panel_path, super_pop)
    vcf_positions, vcf_rsids, vcf_alt_counts = _parse_vcf_alt_counts(vcf_path, super_samples)
    panel_alt_counts = binary.sum(axis=0).astype(int)

    positions, rsids, alt_counts = _align_panel_to_vcf(
        panel_mafs,
        panel_alt_counts,
        vcf_positions,
        vcf_rsids,
        vcf_alt_counts,
        n_individuals=binary.shape[0],
    )

    chrom = infer_chromosome(prefix)
    return SnpManifest(
        prefix=prefix,
        chromosome=chrom,
        n_individuals=int(binary.shape[0]),
        positions=positions,
        rsids=rsids,
        alt_counts=alt_counts,
    )


def _default_vcf_for_prefix(prefix: str) -> str | None:
    chrom = infer_chromosome(prefix)
    if chrom <= 0:
        return None
    local = os.path.join(ROOT, "data", f"chr{chrom}", os.path.basename(DEFAULT_VCF_URL.format(chrom=chrom)))
    return local if os.path.isfile(local) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SNP manifest for a 1KGP super-cohort panel.")
    parser.add_argument("--data-dir", required=True, help="Directory containing *_cleaned_{mafs,binary}.npy")
    parser.add_argument("--prefix", required=True, help="Panel prefix, e.g. eur_chr6")
    parser.add_argument("--vcf", default=None, help="Path to phased chr VCF (.vcf.gz)")
    parser.add_argument("--panel", default=DEFAULT_PANEL, help="1000G integrated_call_samples panel file")
    parser.add_argument("--super-pop", default=None, choices=["EUR", "AFR", "AMR"], help="Super population code")
    parser.add_argument("--download-vcf", action="store_true", help="Download chr VCF from 1000 Genomes if missing")
    args = parser.parse_args()

    super_pop = args.super_pop
    if super_pop is None:
        if args.prefix.lower().startswith("eur"):
            super_pop = "EUR"
        elif args.prefix.lower().startswith("afr"):
            super_pop = "AFR"
        elif args.prefix.lower().startswith("amr"):
            super_pop = "AMR"
        else:
            raise ValueError("Could not infer --super-pop from prefix; pass explicitly.")

    vcf_path = args.vcf or _default_vcf_for_prefix(args.prefix)
    if vcf_path is None or not os.path.isfile(vcf_path):
        if args.download_vcf:
            chrom = infer_chromosome(args.prefix)
            url = DEFAULT_VCF_URL.format(chrom=chrom)
            dest_dir = os.path.join(ROOT, "data", f"chr{chrom}")
            os.makedirs(dest_dir, exist_ok=True)
            vcf_path = os.path.join(dest_dir, os.path.basename(url))
            if not os.path.isfile(vcf_path):
                import shutil
                import subprocess

                print(f"Downloading {url} ...")
                if shutil.which("curl"):
                    subprocess.run(
                        ["curl", "-L", "-C", "-", "-o", vcf_path, url],
                        check=True,
                    )
                else:
                    import urllib.request

                    urllib.request.urlretrieve(url, vcf_path)
        else:
            raise FileNotFoundError(
                f"No VCF for {args.prefix}. Pass --vcf or --download-vcf "
                f"(expected under data/chr<N>/ALL.chr<N>....vcf.gz)."
            )

    print(f"Building manifest for {args.prefix} ({super_pop}) using {vcf_path} ...")
    manifest = build_manifest(
        data_dir=args.data_dir,
        prefix=args.prefix,
        vcf_path=vcf_path,
        panel_path=args.panel,
        super_pop=super_pop,
    )
    out_path = save_manifest(manifest, args.data_dir)
    rsid_frac = float(np.mean(np.asarray(manifest.rsids, dtype=str) != ""))
    print(
        f"Saved {out_path}\n"
        f"  SNPs: {manifest.positions.size:,}\n"
        f"  Chr: {manifest.chromosome}\n"
        f"  Individuals: {manifest.n_individuals}\n"
        f"  Position range: {manifest.positions[0]:,} – {manifest.positions[-1]:,}\n"
        f"  Named rsIDs: {rsid_frac:.1%}"
    )


if __name__ == "__main__":
    main()
