from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for entry in (str(ROOT), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import run_afr_chr6_robustness as sweep

CHECKPOINT = sweep.CHECKPOINT
WANDB_GROUP = "afr-chr6-tests"
RESULTS_ROOT = "results/afr-chr6-tests"
POLICY_EVALS = (
    ("mixed", "eval-attack"),
    ("honest", "eval-honest"),
)


@dataclass(frozen=True)
class Condition:
    name: str
    heuristic: str | None = None
    camouflage: bool = False
    camouflage_ld: bool = True
    note: str = ""

    def evals(self) -> tuple[tuple[str, str], ...]:
        if self.camouflage:
            return (("camouflaged", "eval-camouflage"),)
        return POLICY_EVALS


TD3 = Condition("td3", note="reference actor, re-scored on the shared eval seed")
GATED_UNIFORM = Condition("gated-uniform", heuristic="uniform", note="uniform action in [-1, 1]")
GATED_FULL = Condition("gated-full", heuristic="full", note="full budget, random sign")
GATED_COMMON = Condition(
    "gated-common",
    heuristic="common",
    note="full action on MAF > 0.30, silence below",
)
GATED_PROPORTIONAL = Condition(
    "gated-proportional",
    heuristic="proportional",
    note="action scaled by MAF percentile",
)
CAMOUFLAGE_LD = Condition(
    "camouflaged-ld",
    camouflage=True,
    camouflage_ld=True,
    note="MAF histogram match, LD-greedy inside each band",
)
CAMOUFLAGE_MAF = Condition(
    "camouflaged-maf",
    camouflage=True,
    camouflage_ld=False,
    note="MAF histogram match, uniform draw inside each band",
)

CONDITIONS: tuple[Condition, ...] = (
    TD3,
    GATED_UNIFORM,
    GATED_FULL,
    GATED_COMMON,
    CAMOUFLAGE_LD,
    CAMOUFLAGE_MAF,
)
ALL_CONDITIONS: tuple[Condition, ...] = CONDITIONS + (GATED_PROPORTIONAL,)


def published_checkpoint(path: Path | None = None) -> Path:
    return sweep.published_checkpoint(CHECKPOINT if path is None else Path(path))


def eval_command(
    condition: Condition,
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
        *sweep.common_args(sweep.REFERENCE, detector, args.cuda_device),
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
        f"afr-chr6-tests-{condition.name}-eval-{attacker}",
        "--tags",
        "gatedhcamo",
        "eval",
        sweep.PREFIX,
        condition.name,
        attacker,
        "--wandb-group",
        f"{WANDB_GROUP}-eval",
    ]
    if condition.heuristic:
        command.extend(["--heuristic-policy", condition.heuristic])
    if condition.camouflage and not condition.camouflage_ld:
        command.append("--no-camouflage-ld")
    if args.wandb:
        command.append("--use_wandb")
    return command


def run_condition(condition: Condition, args: argparse.Namespace) -> None:
    detector = Path(args.detector)
    if not detector.is_file():
        raise SystemExit(f"Missing detector: {detector}")
    checkpoint = published_checkpoint(Path(args.checkpoint))
    n_people, n_snps, honest_snps = sweep.panel_shapes()
    sweep.assert_fits(sweep.REFERENCE, n_people, n_snps, honest_snps)
    condition_dir = Path(args.results_root) / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    (condition_dir / "condition.json").write_text(
        json.dumps(
            {
                "name": condition.name,
                "note": condition.note,
                "checkpoint": str(checkpoint),
                "heuristic": condition.heuristic,
                "camouflage_ld": condition.camouflage_ld if condition.camouflage else None,
                "mae_cap": sweep.MAE_CAP,
                "eval_seed": args.eval_seed,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Evaluating {condition.name} from {checkpoint}", flush=True)
    for attacker, folder in condition.evals():
        output_dir = condition_dir / folder
        log_path = condition_dir / f"{folder}.log"
        if not args.reeval and sweep.eval_ready(output_dir, log_path, args.eval_episodes, checkpoint):
            print(f"Reusing finished {attacker} eval in {output_dir}", flush=True)
            continue
        sweep.stream(
            eval_command(condition, attacker, checkpoint, output_dir, detector, args),
            log_path,
        )


def selected_conditions(args: argparse.Namespace) -> list[Condition]:
    catalog = {condition.name: condition for condition in ALL_CONDITIONS}
    if not args.only:
        return list(CONDITIONS)
    return [catalog[name] for name in args.only]


def condition_ready(condition: Condition, root: Path, episodes: int, checkpoint: Path) -> bool:
    condition_dir = root / condition.name
    return all(
        sweep.eval_ready(condition_dir / folder, condition_dir / f"{folder}.log", episodes, checkpoint)
        for _attacker, folder in condition.evals()
    )


def pending_conditions(args: argparse.Namespace, checkpoint: Path) -> list[Condition]:
    chosen = selected_conditions(args)
    if args.reeval:
        return chosen
    root = Path(args.results_root)
    pending: list[Condition] = []
    skipped: list[str] = []
    for condition in chosen:
        if condition_ready(condition, root, args.eval_episodes, checkpoint):
            skipped.append(condition.name)
        else:
            pending.append(condition)
    if skipped:
        print(f"Skipping finished conditions: {', '.join(skipped)}", flush=True)
    return pending


def child_command(condition: Condition, args: argparse.Namespace) -> list[str]:
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


def schedule(args: argparse.Namespace, gpus: list[int], pending: list[Condition]) -> list[str]:
    print(
        f"Scheduling {len(pending)} AFR chr6 tests evals on CUDA "
        f"{', '.join(str(gpu) for gpu in gpus)}. "
        f"Checkpoint {args.checkpoint}. WandB {'on' if args.wandb else 'off'}.",
        flush=True,
    )
    for condition in pending:
        print(f"  {condition.name:<20} {condition.note}", flush=True)
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


def _row(condition: Condition, episodes: int, checkpoint: Path, root: Path) -> dict | None:
    condition_dir = root / condition.name
    if not condition_ready(condition, root, episodes, checkpoint):
        return None
    row: dict = {
        "name": condition.name,
        "checkpoint": str(checkpoint),
        "heuristic": condition.heuristic or "",
    }
    for attacker, folder in condition.evals():
        output = condition_dir / folder
        with (output / "convergence_metrics.csv").open(newline="") as handle:
            metrics = list(csv.DictReader(handle))
        with (output / "baseline_no_defense.json").open() as handle:
            no_defense_f1 = float(json.load(handle)["f1_mean"])
        f1 = sweep.column_mean(metrics, "f1")
        prefix = "camo" if condition.camouflage else ("attack" if attacker == "mixed" else "honest")
        row[f"{prefix}_f1"] = f1
        row[f"{prefix}_mae"] = sweep.column_mean(metrics, "mae")
        row[f"{prefix}_cer"] = sweep.column_mean(metrics, "clinical_error_ratio")
        if attacker != "honest":
            row[f"{prefix}_drop"] = (no_defense_f1 - f1) / no_defense_f1 if no_defense_f1 else 0.0
            row[f"{prefix}_no_defense_f1"] = no_defense_f1
    return row


def summarize(root: Path, episodes: int, checkpoint: Path) -> None:
    rows = []
    for condition in ALL_CONDITIONS:
        row = _row(condition, episodes, checkpoint, root)
        if row is not None:
            rows.append(row)
    summary_path = root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (root / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print("\nAFR chr6 tests controls (frozen actor, held-out cohort B)")
    print(f"{'condition':<20} {'atk F1':>8} {'drop':>8} {'hon MAE':>8} {'camo F1':>8}")
    for row in rows:
        attack = f"{row['attack_f1']:8.3f}" if "attack_f1" in row else f"{'':>8}"
        drop = f"{row['attack_drop']:8.1%}" if "attack_drop" in row else f"{'':>8}"
        honest = f"{row['honest_mae']:8.4f}" if "honest_mae" in row else f"{'':>8}"
        camo = f"{row['camo_f1']:8.3f}" if "camo_f1" in row else f"{'':>8}"
        print(f"{row['name']:<20} {attack} {drop} {honest} {camo}")
    print(f"Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated CUDA device indices. One condition per device. Example: 0,1,2,3",
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
        choices=[condition.name for condition in ALL_CONDITIONS],
        help="Run these rows instead of the default grid",
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
