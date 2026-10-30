from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for entry in (str(ROOT), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import run_eur_chr6_robustness as sweep


BEACON_SIZE = 75
ATTACKER_REFERENCE = 40
CONDITION = sweep.Condition(
    "beacon-75",
    "beacon",
    beacon_size=BEACON_SIZE,
    corr_size=ATTACKER_REFERENCE,
    note=(
        "75-person beacon. Attacker reference is 40 because a 50-person "
        "reference plus two 75-person beacons needs 200 people."
    ),
)
WANDB_GROUP = "eur-chr6-beacon-size"
RESULTS_ROOT = "results/eur-chr6-beacon-size"


def training_pool(n_people: int, beacon_size: int, attacker_reference: int) -> int:
    return int(n_people) - int(beacon_size) - int(attacker_reference)


def assert_beacon_fits(n_people: int, beacon_size: int, attacker_reference: int) -> None:
    
    pool = training_pool(n_people, beacon_size, attacker_reference)
    if pool < beacon_size or attacker_reference < 1:
        largest = (int(n_people) - 50) // 2
        raise ValueError(
            f"A {beacon_size}-person beacon with a {attacker_reference}-person attacker "
            f"reference leaves a training pool of {pool}, which is smaller than the beacon. "
            f"EUR chr6 has {n_people} people. With the 50-person attacker "
            f"reference the largest disjoint beacon is {largest}."
        )


def retag(command: list[str], run_name: str) -> list[str]:
    
    rewritten: list[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        if token in ("--run_name", "--wandb-group") and index + 1 < len(command):
            rewritten.append(token)
            rewritten.append(run_name if token == "--run_name" else WANDB_GROUP)
            index += 2
            continue
        if token == "robustness":
            rewritten.append("beacon-size")
            index += 1
            continue
        rewritten.append(token)
        index += 1
    return rewritten


def train_command(condition_dir: Path, detector: Path, args: argparse.Namespace) -> list[str]:
    command = sweep.train_command(CONDITION, condition_dir, detector, args)
    return retag(command, f"eur-chr6-{CONDITION.name}")


def eval_command(
    attacker: str,
    checkpoint: Path,
    output_dir: Path,
    detector: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = sweep.eval_command(CONDITION, attacker, checkpoint, output_dir, detector, args)
    return retag(command, f"eur-chr6-{CONDITION.name}-eval-{attacker}")


def run_condition(args: argparse.Namespace) -> None:
    detector = Path(args.detector)
    if not detector.is_file():
        raise SystemExit(f"Missing detector: {detector}")
    n_people, _snps, _honest = sweep.panel_shapes()
    assert_beacon_fits(n_people, CONDITION.beacon_size, CONDITION.corr_size)
    condition_dir = Path(args.results_root) / CONDITION.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    (condition_dir / "condition.json").write_text(
        json.dumps(
            {
                "name": CONDITION.name,
                "axis": CONDITION.axis,
                "note": CONDITION.note,
                "mae_cap": sweep.MAE_CAP,
                "beacon_size": CONDITION.beacon_size,
                "gene_size": CONDITION.gene_size,
                "max_queries": CONDITION.max_queries,
                "corr_size": CONDITION.corr_size,
                "seed": CONDITION.seed,
                "shuffle": CONDITION.shuffle,
                "training_pool": training_pool(n_people, CONDITION.beacon_size, CONDITION.corr_size),
                "held_out_beacon": CONDITION.beacon_size,
                "attacker_reference": CONDITION.corr_size,
            },
            indent=2,
        )
        + "\n"
    )
    train_log = condition_dir / "train.log"
    checkpoint = sweep.find_checkpoint(condition_dir)
    training_finished = (
        checkpoint is not None and sweep.log_finished(train_log) and (checkpoint / "actor.pth").is_file()
    )
    if training_finished and not args.retrain:
        print(f"Reusing finished training in {checkpoint}", flush=True)
    else:
        sweep.stream(train_command(condition_dir, detector, args), train_log)
        checkpoint = sweep.find_checkpoint(condition_dir)
    if checkpoint is None:
        raise SystemExit(f"No saved defender under {condition_dir}")
    for attacker, folder in (
        ("mixed", "eval-attack"),
        ("optimal", "eval-optimal"),
        ("honest", "eval-honest"),
    ):
        output_dir = condition_dir / folder
        log_path = condition_dir / f"{folder}.log"
        if not args.retrain and sweep.eval_ready(output_dir, log_path, args.eval_episodes, checkpoint):
            print(f"Reusing finished {attacker} eval in {output_dir}", flush=True)
            continue
        sweep.stream(
            eval_command(attacker, checkpoint, output_dir, detector, args),
            log_path,
        )


def condition_ready(root: Path, episodes: int) -> bool:
    condition_dir = root / CONDITION.name
    checkpoint = sweep.find_checkpoint(condition_dir)
    training_finished = (
        checkpoint is not None
        and sweep.log_finished(condition_dir / "train.log")
        and (checkpoint / "actor.pth").is_file()
    )
    if not training_finished or checkpoint is None:
        return False
    return all(
        sweep.eval_ready(condition_dir / folder, condition_dir / f"{folder}.log", episodes, checkpoint)
        for folder in ("eval-attack", "eval-optimal", "eval-honest")
    )


def summarize(root: Path) -> None:
    condition_dir = root / CONDITION.name
    attack_dir = condition_dir / "eval-attack"
    optimal_dir = condition_dir / "eval-optimal"
    honest_dir = condition_dir / "eval-honest"
    checkpoint = sweep.find_checkpoint(condition_dir)
    if checkpoint is None or not sweep.eval_ready(attack_dir, condition_dir / "eval-attack.log", 1, checkpoint):
        print("No finished beacon-75 eval to summarize yet.")
        return
    with (attack_dir / "convergence_metrics.csv").open(newline="") as handle:
        attack = list(csv.DictReader(handle))
    with (attack_dir / "baseline_no_defense.json").open() as handle:
        no_defense_f1 = float(json.load(handle)["f1_mean"])
    with (honest_dir / "convergence_metrics.csv").open(newline="") as handle:
        honest = list(csv.DictReader(handle))
    attack_f1 = sweep.column_mean(attack, "f1")
    row = {
        "name": CONDITION.name,
        "beacon_size": CONDITION.beacon_size,
        "corr_size": CONDITION.corr_size,
        "no_defense_f1": no_defense_f1,
        "attack_f1": attack_f1,
        "relative_drop": (no_defense_f1 - attack_f1) / no_defense_f1 if no_defense_f1 else 0.0,
        "attack_mae": sweep.column_mean(attack, "mae"),
        "attack_cer": sweep.column_mean(attack, "clinical_error_ratio"),
        "honest_mae": sweep.column_mean(honest, "mae"),
        "honest_cer": sweep.column_mean(honest, "clinical_error_ratio"),
        "optimal_f1": "",
    }
    if sweep.eval_ready(optimal_dir, condition_dir / "eval-optimal.log", 1, checkpoint):
        with (optimal_dir / "convergence_metrics.csv").open(newline="") as handle:
            optimal = list(csv.DictReader(handle))
        row["optimal_f1"] = sweep.column_mean(optimal, "f1")
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps([row], indent=2) + "\n")
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    optimal = f"{float(row['optimal_f1']):.3f}" if row["optimal_f1"] != "" else ""
    print("\nEUR chr6 beacon size (held-out cohort B)")
    print(
        f"{CONDITION.name}: attack F1 {row['attack_f1']:.3f}  drop {row['relative_drop']:.1%}  "
        f"opt F1 {optimal}  attack MAE {row['attack_mae']:.4f}  attack CER {row['attack_cer']:.3f}  "
        f"honest MAE {row['honest_mae']:.4f}  honest CER {row['honest_cer']:.3f}"
    )
    print(f"Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default=None, help="CUDA index, for example 0 or 0,1")
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--detector", default=str(sweep.DETECTOR))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    args.wandb = not args.no_wandb
    return args


def main() -> None:
    args = parse_args()
    print(
        f"Beacon {BEACON_SIZE} with attacker reference {ATTACKER_REFERENCE}: "
        f"{BEACON_SIZE} train + {BEACON_SIZE} held-out + {ATTACKER_REFERENCE} attacker = 190. "
        "N=25 and N=50 are already in results/eur-chr6-robustness.",
        flush=True,
    )
    if args.worker:
        run_condition(args)
        summarize(Path(args.results_root))
        return
    if not args.gpus:
        raise SystemExit("Pass --gpus with the CUDA index to use, for example --gpus 0")
    gpus = sweep.parse_gpus(args.gpus)
    if not args.retrain and condition_ready(Path(args.results_root), args.eval_episodes):
        print("Skipping finished condition: beacon-75", flush=True)
        summarize(Path(args.results_root))
        return
    gpu = gpus[0]
    log_path = Path(args.results_root) / CONDITION.name / "worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--cuda-device",
        "0",
        "--results-root",
        args.results_root,
        "--detector",
        args.detector,
        "--episodes",
        str(args.episodes),
        "--eval-episodes",
        str(args.eval_episodes),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--eval-seed",
        str(args.eval_seed),
    ]
    if args.retrain:
        command.append("--retrain")
    if not args.wandb:
        command.append("--no-wandb")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"Starting {CONDITION.name} on CUDA {gpu} -> {log_path}", flush=True)
    started = time.time()
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(ROOT),
        )
        code = proc.wait()
    elapsed = time.time() - started
    if code != 0:
        raise SystemExit(f"Failed {CONDITION.name} on CUDA {gpu} (exit {code}) after {elapsed:.0f}s. See {log_path}")
    print(f"Finished {CONDITION.name} on CUDA {gpu}", flush=True)
    summarize(Path(args.results_root))


if __name__ == "__main__":
    main()
