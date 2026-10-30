from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_PANELS = (
    "EUR:eur_chr1",
    "EUR:eur_chr6",
    "EUR:eur_chr22",
    "AFR:afr_chr1",
    "AFR:afr_chr6",
    "AFR:afr_chr22",
    "AMR:amr_chr1",
    "AMR:amr_chr6",
    "AMR:amr_chr22",
)


def parse_panel(value: str) -> tuple[str, str]:
    try:
        population, prefix = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Panel must be POPULATION:prefix, got {value!r}"
        ) from exc
    return population.upper(), prefix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", nargs="+", default=DEFAULT_PANELS)
    parser.add_argument("--results-root", default="results/detectors-v2")
    parser.add_argument("--episodes-per-class", type=int, default=200)
    parser.add_argument("--fpr-budget", type=float, default=0.10)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--gene-size", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    for panel_value in args.panels:
        population, prefix = parse_panel(panel_value)
        data_dir = Path("data") / population
        for suffix in ("cleaned_mafs.npy", "cleaned_binary.npy"):
            required = data_dir / f"{prefix}_{suffix}"
            if not required.exists():
                raise SystemExit(f"Missing panel input: {required}")

        output_dir = results_root / prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "main.py",
            "--calibrate-detector",
            "--calibration-episodes",
            str(args.episodes_per_class),
            "--detector-fpr-budget",
            str(args.fpr_budget),
            "--detector-train-fraction",
            str(args.train_fraction),
            "--detector-calibration-fraction",
            str(args.calibration_fraction),
            "--detector-split-seed",
            str(args.split_seed),
            "--data",
            str(data_dir),
            "--pop_prefix",
            prefix,
            "--gene_size",
            str(args.gene_size),
            "--max_queries",
            str(args.max_queries),
            "--shuffle-queries",
            "--no-svd-embeddings",
            "--seed",
            str(args.seed),
            "--cuda-device",
            str(args.cuda_device),
            "--results-dir",
            str(output_dir),
        ]
        log_path = output_dir / "calibration.log"
        print(f"\nCalibrating {prefix}: {' '.join(command)}", flush=True)
        with log_path.open("w") as log_file:
            subprocess.run(
                command,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

        detector_path = output_dir / "suspicion_detector.json"
        detector = json.loads(detector_path.read_text())
        manifest[prefix] = {
            "population": population,
            "detector_path": str(detector_path),
            "step_auc": detector["auc"],
            "episode_auc": detector.get("episode_auc"),
            "test_honest_fpr": detector["honest_fpr"],
            "test_mixed_tpr": detector["mixed_tpr"],
            "threshold": detector["threshold"],
        }
        print(
            f"{prefix}: episode AUC={detector.get('episode_auc', float('nan')):.4f} "
            f"test FPR={detector['honest_fpr']:.3f} "
            f"test TPR={detector['mixed_tpr']:.3f}"
        )

    manifest_path = results_root / "calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved {manifest_path}")


if __name__ == "__main__":
    main()
