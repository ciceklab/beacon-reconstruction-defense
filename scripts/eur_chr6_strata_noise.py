from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers.strata_noise import figure_title, plot_from_summary


def checklist_command(args: argparse.Namespace) -> list[str]:
    prefix = args.prefix
    population = prefix.split("_")[0].upper()
    command = [
        sys.executable,
        "-u",
        "main.py",
        "--data",
        f"data/{population}",
        "--pop_prefix",
        prefix,
        "--gene_size",
        "1000",
        "--beacon_size",
        "50",
        "--corr_size",
        "50",
        "--max_queries",
        "50",
        "--seq-len",
        "50",
        "--mae-window",
        "50",
        "--mae-limit",
        "0.126",
        "--no-mae-limit-ramp",
        "--hard-mae-budget",
        "--seed",
        str(args.seed),
        "--shuffle-queries",
        "--sparse-reward",
        "--f1-reward-scale",
        "400",
        "--budget-inefficiency-tax",
        "1.0",
        "--f1-overbaseline-penalty",
        "0",
        "--proxy-reward-scale",
        "0",
        "--reward-normalize-scale",
        "1.0",
        "--reward-clip",
        "200",
        "--cer-penalty-weight",
        "500",
        "--rarity-token-features",
        "--ld-token-features",
        "--ld-history-k",
        "4",
        "--ld-panel-k",
        "4",
        "--attacker-knowledge-features",
        "--attacker-knowledge-dim",
        "8",
        "--no-svd-embeddings",
        "--eval_utility",
        "--suspicion-detector-path",
        str(args.detector),
        "--trace-strata-noise",
        "--strata-episodes",
        str(args.episodes),
        "--strata-out-dir",
        str(args.out_dir),
        "--eval_cohort",
        "B",
        "--force-attacker-type",
        "--attacker_type",
        "mixed",
        "--resume-beacon",
        str(args.checkpoint),
        "--episodes",
        str(args.episodes),
        "--results-dir",
        str(args.results_dir),
        "--run_name",
        f"{prefix}-strata-noise",
    ]
    if args.cuda_device is not None:
        command.extend(["--cuda-device", str(args.cuda_device)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot checklist-v3 noise by MAF stratum")
    parser.add_argument("--prefix", default="eur_chr6")
    parser.add_argument(
        "--checkpoint",
        default="results/mae-knee-v2/eur_chr6/mae-0.125926/run2/weights_best_eval",
    )
    parser.add_argument(
        "--detector",
        default="results/panel-detectors-v3/eur_chr6/suspicion_detector.json",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--out-dir", default="figures/checklist-v3")
    parser.add_argument("--results-dir", default="results/checklist-v3/eur_chr6/strata-trace-v2")
    parser.add_argument("--from-summary", default=None, help="Replot an existing summary CSV")
    args = parser.parse_args()

    if args.from_summary:
        out = Path(args.out_dir) / f"{args.prefix}_maf_strata_noise.png"
        plot_from_summary(
            Path(args.from_summary),
            out,
            title=figure_title(args.prefix),
        )
        print(f"Saved {out}")
        return

    checkpoint = Path(args.checkpoint)
    actor = checkpoint / "actor.pth"
    if not actor.is_file():
        raise SystemExit(
            f"No checklist-v3 actor at {actor}. "
            "The eur_chr6 weights from eval live at "
            "results/checklist-v3/eur_chr6/run2/weights_best_eval on the GPU host."
        )
    detector = Path(args.detector)
    if not detector.is_file():
        raise SystemExit(f"No suspicion detector at {detector}")
    command = checklist_command(args)
    print(" ".join(command), flush=True)
    subprocess.check_call(command, cwd=ROOT)


if __name__ == "__main__":
    main()
