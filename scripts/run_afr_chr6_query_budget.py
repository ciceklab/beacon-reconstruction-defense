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

import run_afr_chr6_robustness as sweep

QUERY_BUDGETS = (30, 100, 200)
EVALS = (
    ("mixed", "eval-attack"),
    ("honest", "eval-honest"),
)
TRAINED_SEQ_LEN = sweep.TRAINED_SEQ_LEN
TRAINED_MAE_WINDOW = 50
CHECKPOINT = sweep.CHECKPOINT
WANDB_GROUP = "afr-chr6-query-budget"
RESULTS_ROOT = "results/afr-chr6-query-budget"

CONDITIONS: tuple[sweep.Condition, ...] = tuple(
    sweep.Condition(
        f"queries-{budget}",
        "queries",
        max_queries=budget,
        note=f"{budget} queries on the frozen sequence-length-{TRAINED_SEQ_LEN} actor",
    )
    for budget in QUERY_BUDGETS
)


def published_checkpoint(path: Path | None = None) -> Path:
    return sweep.published_checkpoint(CHECKPOINT if path is None else Path(path))


def common_args(condition: sweep.Condition, detector: Path, cuda_device: int) -> list[str]:
    command = sweep.common_args(condition, detector, cuda_device)
    command[command.index("--mae-window") + 1] = str(TRAINED_MAE_WINDOW)
    command[command.index("--seq-len") + 1] = str(TRAINED_SEQ_LEN)
    return command


def eval_command(
    condition: sweep.Condition,
    attacker: str,
    checkpoint: Path,
    output_dir: Path,
    detector: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "main.py",
        *common_args(condition, detector, args.cuda_device),
        "--evaluate",
        "--eval_cohort",
        "B",
        "--force-attacker-type",
        "--attacker_type",
        attacker,
        "--resume-beacon",
        str(checkpoint),
        "--episodes",
        str(args.eval_episodes),
        "--seed",
        str(args.eval_seed),
        "--results-dir",
        str(output_dir),
        "--run_name",
        f"afr-chr6-{condition.name}-eval-{attacker}",
        "--tags",
        "query-budget",
        "eval",
        sweep.PREFIX,
        condition.name,
        attacker,
        "--wandb-group",
        f"{WANDB_GROUP}-eval",
    ]
    if args.wandb:
        command.append("--use_wandb")
    return command


def assert_eval_fits(condition: sweep.Condition, n_people: int, honest_snps: int) -> None:
    if condition.beacon_size + condition.corr_size > n_people:
        raise ValueError(
            f"{condition.name} needs a held-out beacon of {condition.beacon_size} and an "
            f"attacker reference of {condition.corr_size}, which does not fit in {n_people} people"
        )
    if condition.gene_size < condition.max_queries:
        raise ValueError(
            f"{condition.name} draws {condition.gene_size} SNPs for {condition.max_queries} queries"
        )
    if condition.gene_size > honest_snps:
        raise ValueError(
            f"{condition.name} asks for {condition.gene_size} SNPs but the HLA window has {honest_snps}"
        )


def run_condition(condition: sweep.Condition, args: argparse.Namespace) -> None:
    detector = Path(args.detector)
    if not detector.is_file():
        raise SystemExit(f"Missing detector: {detector}")
    checkpoint = published_checkpoint(Path(args.checkpoint))
    n_people, _snps, honest_snps = sweep.panel_shapes()
    assert_eval_fits(condition, n_people, honest_snps)
    condition_dir = Path(args.results_root) / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    (condition_dir / "condition.json").write_text(
        json.dumps(
            {
                "name": condition.name,
                "note": condition.note,
                "checkpoint": str(checkpoint),
                "mae_cap": sweep.MAE_CAP,
                "seq_len": TRAINED_SEQ_LEN,
                "mae_window": TRAINED_MAE_WINDOW,
                "max_queries": condition.max_queries,
                "gene_size": condition.gene_size,
                "beacon_size": condition.beacon_size,
                "evals": [attacker for attacker, _folder in EVALS],
                "eval_seed": args.eval_seed,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Evaluating {checkpoint} at {condition.max_queries} queries", flush=True)
    for attacker, folder in EVALS:
        output_dir = condition_dir / folder
        log_path = condition_dir / f"{folder}.log"
        if not args.reeval and sweep.eval_ready(output_dir, log_path, args.eval_episodes, checkpoint):
            print(f"Reusing finished {attacker} eval in {output_dir}", flush=True)
            continue
        sweep.stream(
            eval_command(condition, attacker, checkpoint, output_dir, detector, args),
            log_path,
        )


def selected_conditions(args: argparse.Namespace) -> list[sweep.Condition]:
    if not args.only:
        return list(CONDITIONS)
    wanted = set(args.only)
    return [condition for condition in CONDITIONS if condition.name in wanted]


def condition_ready(condition: sweep.Condition, root: Path, episodes: int, checkpoint: Path) -> bool:
    condition_dir = root / condition.name
    return all(
        sweep.eval_ready(
            condition_dir / folder,
            condition_dir / f"{folder}.log",
            episodes,
            checkpoint,
        )
        for _attacker, folder in EVALS
    )


def pending_conditions(args: argparse.Namespace, checkpoint: Path) -> list[sweep.Condition]:
    chosen = selected_conditions(args)
    if args.reeval:
        return chosen
    root = Path(args.results_root)
    pending: list[sweep.Condition] = []
    skipped: list[str] = []
    for condition in chosen:
        if condition_ready(condition, root, args.eval_episodes, checkpoint):
            skipped.append(condition.name)
        else:
            pending.append(condition)
    if skipped:
        print(f"Skipping finished conditions: {', '.join(skipped)}", flush=True)
    return pending


def child_command(condition: sweep.Condition, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--only",
        condition.name,
        "--cuda-device",
        "0",
        "--checkpoint",
        args.checkpoint,
        "--results-root",
        args.results_root,
        "--detector",
        args.detector,
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-seed",
        str(args.eval_seed),
    ]
    if args.reeval:
        command.append("--reeval")
    if not args.wandb:
        command.append("--no-wandb")
    return command


def schedule(args: argparse.Namespace, gpus: list[int], pending: list[sweep.Condition]) -> list[str]:
    print(
        f"Scheduling {len(pending)} AFR chr6 query-budget evals on CUDA "
        f"{', '.join(str(gpu) for gpu in gpus)}. "
        f"Checkpoint {args.checkpoint}. Evals: mixed, honest. "
        f"WandB {'on' if args.wandb else 'off'}.",
        flush=True,
    )
    for condition in pending:
        print(f"  {condition.name:<14} queries {condition.max_queries}", flush=True)
    active: list[tuple[subprocess.Popen, str, int, Path]] = []
    failed: list[str] = []
    spawn_delay = 8.0
    last_spawn = 0.0
    while pending or active:
        still_running: list[tuple[subprocess.Popen, str, int, Path]] = []
        for proc, name, gpu, log_path in active:
            code = proc.poll()
            if code is None:
                still_running.append((proc, name, gpu, log_path))
                continue
            if code == 0:
                print(f"Finished {name} on CUDA {gpu}", flush=True)
            else:
                print(f"Failed {name} on CUDA {gpu} (exit {code}). See {log_path}", flush=True)
                failed.append(name)
        active = still_running
        busy = {gpu for _, _, gpu, _ in active}
        while pending and len(active) < len(gpus):
            free = [gpu for gpu in gpus if gpu not in busy]
            if not free:
                break
            if active and time.time() - last_spawn < spawn_delay:
                break
            condition = pending.pop(0)
            gpu = free[0]
            busy.add(gpu)
            log_path = Path(args.results_root) / condition.name / "worker.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"Starting {condition.name} on CUDA {gpu} -> {log_path}", flush=True)
            with log_path.open("w") as log_file:
                proc = subprocess.Popen(
                    child_command(condition, args),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(ROOT),
                )
            active.append((proc, condition.name, gpu, log_path))
            last_spawn = time.time()
        if pending or active:
            time.sleep(2)
    return failed


def summarize(root: Path, episodes: int, checkpoint: Path) -> None:
    rows = []
    for condition in CONDITIONS:
        condition_dir = root / condition.name
        attack_dir = condition_dir / "eval-attack"
        honest_dir = condition_dir / "eval-honest"
        if not sweep.eval_ready(attack_dir, condition_dir / "eval-attack.log", episodes, checkpoint):
            continue
        if not sweep.eval_ready(honest_dir, condition_dir / "eval-honest.log", episodes, checkpoint):
            continue
        with (attack_dir / "convergence_metrics.csv").open(newline="") as handle:
            attack = list(csv.DictReader(handle))
        with (attack_dir / "baseline_no_defense.json").open() as handle:
            no_defense_f1 = float(json.load(handle)["f1_mean"])
        with (honest_dir / "convergence_metrics.csv").open(newline="") as handle:
            honest = list(csv.DictReader(handle))
        attack_f1 = sweep.column_mean(attack, "f1")
        rows.append(
            {
                "name": condition.name,
                "max_queries": condition.max_queries,
                "checkpoint": str(checkpoint),
                "no_defense_f1": no_defense_f1,
                "attack_f1": attack_f1,
                "relative_drop": (no_defense_f1 - attack_f1) / no_defense_f1 if no_defense_f1 else 0.0,
                "attack_mae": sweep.column_mean(attack, "mae"),
                "honest_f1": sweep.column_mean(honest, "f1"),
                "honest_mae": sweep.column_mean(honest, "mae"),
                "honest_cer": sweep.column_mean(honest, "clinical_error_ratio"),
            }
        )
    summary_path = root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with (root / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("\nAFR chr6 query budget (frozen actor, held-out cohort B)")
    print(
        f"{'condition':<14} {'queries':>7} {'atk F1':>8} {'drop':>8} "
        f"{'hon F1':>8} {'hon MAE':>8} {'hon CER':>8}"
    )
    for row in rows:
        print(
            f"{row['name']:<14} {row['max_queries']:7d} {row['attack_f1']:8.3f} "
            f"{row['relative_drop']:8.1%} {row['honest_f1']:8.3f} "
            f"{row['honest_mae']:8.4f} {row['honest_cer']:8.3f}"
        )
    print(f"Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated CUDA device indices. One budget per device. Example: 0,1,2,3",
    )
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--detector", default=str(sweep.DETECTOR))
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[condition.name for condition in CONDITIONS],
        help="Run these budgets instead of the full list",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reeval", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    args.wandb = not args.no_wandb
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        chosen = selected_conditions(args)
        if len(chosen) != 1:
            raise SystemExit("A worker runs exactly one condition")
        run_condition(chosen[0], args)
        return
    if not args.gpus:
        raise SystemExit("Pass --gpus with the CUDA indices to use, for example --gpus 0,1,2,3")
    detector = Path(args.detector)
    if not detector.is_file():
        raise SystemExit(f"Missing detector: {detector}")
    checkpoint = published_checkpoint(Path(args.checkpoint))
    n_people, _snps, honest_snps = sweep.panel_shapes()
    for condition in selected_conditions(args):
        assert_eval_fits(condition, n_people, honest_snps)
    gpus = sweep.parse_gpus(args.gpus)
    pending = pending_conditions(args, checkpoint)
    if not pending:
        print("Every selected condition is already finished.", flush=True)
        summarize(Path(args.results_root), args.eval_episodes, checkpoint)
        return
    failed = schedule(args, gpus, pending)
    summarize(Path(args.results_root), args.eval_episodes, checkpoint)
    if failed:
        raise SystemExit(f"Failed conditions: {', '.join(failed)}")


if __name__ == "__main__":
    main()
