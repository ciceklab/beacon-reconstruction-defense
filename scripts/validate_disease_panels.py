from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from helpers.disease_catalog import DISEASE_CATALOG
from helpers.disease_panels import list_diseases_for_chromosome
from helpers.snp_manifest import infer_chromosome, load_manifest

PANELS = [
    ("EUR", "eur_chr1"),
    ("EUR", "eur_chr6"),
    ("EUR", "eur_chr22"),
    ("AFR", "afr_chr1"),
    ("AFR", "afr_chr6"),
    ("AFR", "afr_chr22"),
    ("AMR", "amr_chr1"),
    ("AMR", "amr_chr6"),
    ("AMR", "amr_chr22"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate disease SNP mapping for 1KGP panels.")
    parser.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    args = parser.parse_args()

    for pop, prefix in PANELS:
        data_dir = os.path.join(args.data_root, pop)
        manifest = load_manifest(data_dir, prefix)
        chrom = infer_chromosome(prefix)
        print(f"\n=== {prefix} ({pop}) ===")
        if manifest is None:
            print("  manifest: MISSING")
            continue
        print(
            f"  manifest: {manifest.positions.size:,} SNPs | "
            f"chr{manifest.chromosome} | n={manifest.n_individuals}"
        )
        diseases = list_diseases_for_chromosome(chrom)
        for disease in diseases:
            hits = 0
            for locus in disease.loci:
                if locus.chromosome != chrom:
                    continue
                idx = manifest.lookup_rsid(locus.rsid)
                if idx is None:
                    idx = manifest.lookup_nearest(locus.position, window_bp=200_000)
                if idx is not None:
                    hits += 1
                    dist = abs(int(manifest.positions[idx]) - locus.position)
                    print(
                        f"  OK  {disease.disease_id:22} {locus.rsid:14} "
                        f"catalog={locus.position:,} panel={manifest.positions[idx]:,} "
                        f"Δbp={dist:,} idx={idx}"
                    )
                else:
                    print(
                        f"  MISS {disease.disease_id:22} {locus.rsid:14} "
                        f"catalog={locus.position:,}"
                    )
            print(f"       -> {hits}/{sum(1 for l in disease.loci if l.chromosome == chrom)} loci on chr{chrom}")


if __name__ == "__main__":
    main()
