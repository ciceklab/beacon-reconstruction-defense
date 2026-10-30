from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "build_snp_manifest.py"

PANELS = [
    ("data/EUR", "eur_chr1"),
    ("data/EUR", "eur_chr6"),
    ("data/EUR", "eur_chr22"),
    ("data/AFR", "afr_chr1"),
    ("data/AFR", "afr_chr6"),
    ("data/AFR", "afr_chr22"),
    ("data/AMR", "amr_chr1"),
    ("data/AMR", "amr_chr6"),
    ("data/AMR", "amr_chr22"),
]


def main() -> None:
    download = "--download-vcf" in sys.argv
    for rel_dir, prefix in PANELS:
        data_dir = ROOT / rel_dir
        cmd = [
            sys.executable,
            str(BUILD),
            "--data-dir",
            str(data_dir),
            "--prefix",
            prefix,
        ]
        if download:
            cmd.append("--download-vcf")
        print("\n>>>", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
