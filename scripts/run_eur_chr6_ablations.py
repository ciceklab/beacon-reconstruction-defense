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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POPULATION = "EUR"
PREFIX = "eur_chr6"
MAE_CAP = 0.125926
DETECTOR = Path("results/panel-detectors-v3/eur_chr6/suspicion_detector.json")
WANDB_GROUP = "eur-chr6-ablations"
CER_WEIGHT = 500.0
HONEST_FRACTION = 0.20
MIXED_FRACTION = 0.80


@dataclass(frozen=True)
class Condition:
    name: str
    axis: str
    suspicion_gate: bool = True
    cer_penalty_weight: float = CER_WEIGHT
    hard_mae_budget: bool = True
    attacker_knowledge: bool = True
    honest_episode_fraction: float = HONEST_FRACTION
    reward_only_budget: bool = False
    maf_detector_state: bool = False
    note: str = ""


REFERENCE = Condition("reference", "reference", note=" EUR chr6 recipe")

CONDITIONS: tuple[Condition, ...] = (
    REFERENCE,
    Condition(
        "no-suspicion-gate",
        "suspicion_gate",
        suspicion_gate=False,
        note="actor noise is not multiplied by the detector gate",
    ),
    Condition(
        "no-cer-penalty",
        "cer_penalty_weight",
        cer_penalty_weight=0.0,
        note="finished null: F1 reward already avoids rare alleles on attack episodes",
    ),
    Condition(
        "reward-only-budget",
        "reward_only_budget",
        reward_only_budget=True,
        note="over-cap noise is kept and taxed, not clipped or zeroed",
    ),
    Condition(
        "maf-detector-only",
        "maf_detector_state",
        maf_detector_state=True,
        note="token is query MAF plus detector features",
    ),
    Condition(
        "honest-fraction-0",
        "honest_episode_fraction",
        honest_episode_fraction=0.0,
        note="training is 80% mixed and 20% optimal; eval still includes honest",
    ),
)

CONTROLS = (
    "suspicion_gate",
    "cer_penalty_weight",
    "hard_mae_budget",
    "attacker_knowledge",
    "honest_episode_fraction",
    "reward_only_budget",
    "maf_detector_state",
)


def control_tuple(condition: Condition) -> tuple:
    return tuple(getattr(condition, name) for name in CONTROLS)


def assert_one_control(conditions: tuple[Condition, ...] = CONDITIONS) -> None:
    
    names = [condition.name for condition in conditions]
    if len(names) != len(set(names)):
        raise ValueError("Condition names must be unique")
    reference = control_tuple(REFERENCE)
    for condition in conditions:
        if condition.name == REFERENCE.name:
            if control_tuple(condition) != reference:
                raise ValueError("reference does not match the controls")
            continue
        if condition.axis not in CONTROLS:
            raise ValueError(f"{condition.name} has unknown axis {condition.axis}")
        changed = [
            name
            for name, left, right in zip(CONTROLS, control_tuple(condition), reference)
            if left != right
        ]
        if changed != [condition.axis]:
            raise ValueError(f"{condition.name} changes {changed}, expected [{condition.axis}]")
        if condition.honest_episode_fraction < 0 or condition.honest_episode_fraction + MIXED_FRACTION > 1.0:
            raise ValueError(f"{condition.name} honest fraction does not leave room for the mixed share")


def stream(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nRunning: {' '.join(command)}", flush=True)
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        return_code = proc.wait()
    if return_code != 0:
        raise SystemExit(f"Command failed ({return_code}). See {log_path}")


def log_finished(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    with log_path.open(errors="replace") as handle:
        return any("Convergence metrics saved to" in line for line in handle)


def csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def column_mean(rows: list[dict[str, str]], name: str) -> float:
    values = [float(row[name]) for row in rows if row.get(name) not in ("", None)]
    if not values:
        raise ValueError(f"No values for {name}")
    return sum(values) / len(values)


def find_checkpoint(condition_dir: Path) -> Path | None:
    found = list(condition_dir.glob("run*/weights_best_eval/actor.pth"))
    if not found:
        return None
    found.sort(key=lambda path: path.stat().st_mtime)
    return found[-1].parent


def common_args(condition: Condition, detector: Path, cuda_device: int) -> list[str]:
    command = [
        "--data",
        f"data/{POPULATION}",
        "--pop_prefix",
        PREFIX,
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
        f"{MAE_CAP:.6f}",
        "--no-mae-limit-ramp",
        (
            "--reward-only-mae-budget"
            if condition.reward_only_budget
            else "--hard-mae-budget"
            if condition.hard_mae_budget
            else "--no-hard-mae-budget"
        ),
        "--seed",
        "3",
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
        f"{condition.cer_penalty_weight:g}",
        "--honest-episode-fraction",
        f"{condition.honest_episode_fraction:g}",
        "--mixed-episode-fraction",
        f"{MIXED_FRACTION:g}",
        "--no-svd-embeddings",
        "--eval_utility",
        "--bra-fast",
        "--cuda-device",
        str(cuda_device),
        "--suspicion-detector-path",
        str(detector),
    ]
    if condition.maf_detector_state:
        command.append("--maf-detector-state")
    else:
        command.extend(
            [
                "--rarity-token-features",
                "--ld-token-features",
                "--ld-history-k",
                "4",
                "--ld-panel-k",
                "4",
            ]
        )
        if condition.attacker_knowledge:
            command.extend(["--attacker-knowledge-features", "--attacker-knowledge-dim", "8"])
    if not condition.suspicion_gate:
        command.append("--no-suspicion-gate")
    return command


def train_command(condition: Condition, condition_dir: Path, detector: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "main.py",
        *common_args(condition, detector, args.cuda_device),
        "--episodes",
        str(args.episodes),
        "--eval_cohort",
        "A",
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--eval-freq",
        "10",
        "--eval-episodes",
        "5",
        "--honest-eval-episodes",
        "3",
        "--results-dir",
        str(condition_dir),
        "--run_name",
        f"eur-chr6-ablate-{condition.name}",
        "--tags",
        "ablation",
        PREFIX,
        condition.axis,
        condition.name,
        "--wandb-group",
        WANDB_GROUP,
    ]
    if args.wandb:
        command.append("--use_wandb")
    return command


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
        f"eur-chr6-ablate-{condition.name}-eval-{attacker}",
        "--tags",
        "ablation",
        "eval",
        PREFIX,
        condition.axis,
        condition.name,
        attacker,
        "--wandb-group",
        f"{WANDB_GROUP}-eval",
    ]
    if args.wandb:
        command.append("--use_wandb")
    return command


def eval_ready(output_dir: Path, log_path: Path, episodes: int, checkpoint: Path) -> bool:
    metrics = output_dir / "convergence_metrics.csv"
    actor = checkpoint / "actor.pth"
    if not log_finished(log_path) or csv_rows(metrics) < episodes:
        return False
    if not (output_dir / "baseline_no_defense.json").is_file():
        return False
    if not actor.is_file() or not metrics.is_file():
        return False
    return metrics.stat().st_mtime >= actor.stat().st_mtime


def run_condition(condition: Condition, args: argparse.Namespace) -> None:
    detector = Path(args.detector)
    if not detector.is_file():
        raise SystemExit(f"Missing detector: {detector}")
    condition_dir = Path(args.results_root) / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    (condition_dir / "condition.json").write_text(
        json.dumps(
            {
                "name": condition.name,
                "axis": condition.axis,
                "note": condition.note,
                "mae_cap": MAE_CAP,
                "suspicion_gate": condition.suspicion_gate,
                "cer_penalty_weight": condition.cer_penalty_weight,
                "hard_mae_budget": condition.hard_mae_budget,
                "reward_only_budget": condition.reward_only_budget,
                "attacker_knowledge": condition.attacker_knowledge,
                "maf_detector_state": condition.maf_detector_state,
                "honest_episode_fraction": condition.honest_episode_fraction,
                "mixed_episode_fraction": MIXED_FRACTION,
            },
            indent=2,
        )
        + "\n"
    )
    train_log = condition_dir / "train.log"
    checkpoint = find_checkpoint(condition_dir)
    training_finished = (
        checkpoint is not None and log_finished(train_log) and (checkpoint / "actor.pth").is_file()
    )
    if training_finished and not args.retrain:
        print(f"Reusing finished training in {checkpoint}", flush=True)
    else:
        stream(train_command(condition, condition_dir, detector, args), train_log)
        checkpoint = find_checkpoint(condition_dir)
    if checkpoint is None:
        raise SystemExit(f"No saved defender under {condition_dir}")
    for attacker, folder in (
        ("mixed", "eval-attack"),
        ("optimal", "eval-optimal"),
        ("honest", "eval-honest"),
    ):
        output_dir = condition_dir / folder
        log_path = condition_dir / f"{folder}.log"
        if not args.retrain and eval_ready(output_dir, log_path, args.eval_episodes, checkpoint):
            print(f"Reusing finished {attacker} eval in {output_dir}", flush=True)
            continue
        stream(
            eval_command(condition, attacker, checkpoint, output_dir, detector, args),
            log_path,
        )


def parse_gpus(spec: str) -> list[int]:
    try:
        gpu_ids = [int(part.strip()) for part in spec.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"--gpus must be comma-separated device ids, got {spec!r}") from exc
    if not gpu_ids:
        raise SystemExit("--gpus needs at least one device, for example 0,1,2,3")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise SystemExit(f"--gpus repeats a device: {spec}")
    return gpu_ids


def selected_conditions(args: argparse.Namespace) -> list[Condition]:
    if not args.only:
        return list(CONDITIONS)
    wanted = set(args.only)
    return [condition for condition in CONDITIONS if condition.name in wanted]


def condition_ready(condition: Condition, root: Path, episodes: int) -> bool:
    condition_dir = root / condition.name
    checkpoint = find_checkpoint(condition_dir)
    training_finished = (
        checkpoint is not None
        and log_finished(condition_dir / "train.log")
        and (checkpoint / "actor.pth").is_file()
    )
    if not training_finished or checkpoint is None:
        return False
    return all(
        eval_ready(condition_dir / folder, condition_dir / f"{folder}.log", episodes, checkpoint)
        for folder in ("eval-attack", "eval-optimal", "eval-honest")
    )


def pending_conditions(args: argparse.Namespace) -> list[Condition]:
    chosen = selected_conditions(args)
    if args.retrain:
        return chosen
    root = Path(args.results_root)
    pending: list[Condition] = []
    skipped: list[str] = []
    for condition in chosen:
        if condition_ready(condition, root, args.eval_episodes):
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
    return command


def schedule(args: argparse.Namespace, gpus: list[int], pending: list[Condition]) -> list[str]:
    print(
        f"Scheduling {len(pending)} EUR chr6 ablations on CUDA "
        f"{', '.join(str(gpu) for gpu in gpus)}. "
        f"MAE cap {MAE_CAP:.6f}. WandB {'on' if args.wandb else 'off'}.",
        flush=True,
    )
    for condition in pending:
        print(f"  {condition.name:<24} {condition.note}", flush=True)
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


def summarize(root: Path, episodes: int) -> None:
    rows = []
    for condition in CONDITIONS:
        condition_dir = root / condition.name
        attack_dir = condition_dir / "eval-attack"
        optimal_dir = condition_dir / "eval-optimal"
        honest_dir = condition_dir / "eval-honest"
        checkpoint = find_checkpoint(condition_dir)
        if checkpoint is None:
            continue
        if not eval_ready(attack_dir, condition_dir / "eval-attack.log", episodes, checkpoint):
            continue
        if not eval_ready(honest_dir, condition_dir / "eval-honest.log", episodes, checkpoint):
            continue
        with (attack_dir / "convergence_metrics.csv").open(newline="") as handle:
            attack = list(csv.DictReader(handle))
        with (attack_dir / "baseline_no_defense.json").open() as handle:
            no_defense_f1 = float(json.load(handle)["f1_mean"])
        with (honest_dir / "convergence_metrics.csv").open(newline="") as handle:
            honest = list(csv.DictReader(handle))
        attack_f1 = column_mean(attack, "f1")
        row = {
            "name": condition.name,
            "axis": condition.axis,
            "note": condition.note,
            "suspicion_gate": int(condition.suspicion_gate),
            "cer_penalty_weight": condition.cer_penalty_weight,
            "hard_mae_budget": int(condition.hard_mae_budget),
            "attacker_knowledge": int(condition.attacker_knowledge),
            "honest_episode_fraction": condition.honest_episode_fraction,
            "mae_cap": MAE_CAP,
            "no_defense_f1": no_defense_f1,
            "attack_f1": attack_f1,
            "relative_drop": (no_defense_f1 - attack_f1) / no_defense_f1 if no_defense_f1 else 0.0,
            "attack_mae": column_mean(attack, "mae"),
            "attack_cer": column_mean(attack, "clinical_error_ratio"),
            "attack_budget_penalties": column_mean(attack, "budget_penalty_count"),
            "honest_f1": column_mean(honest, "f1"),
            "honest_mae": column_mean(honest, "mae"),
            "honest_cer": column_mean(honest, "clinical_error_ratio"),
            "optimal_f1": "",
        }
        if eval_ready(optimal_dir, condition_dir / "eval-optimal.log", episodes, checkpoint):
            with (optimal_dir / "convergence_metrics.csv").open(newline="") as handle:
                optimal = list(csv.DictReader(handle))
            row["optimal_f1"] = column_mean(optimal, "f1")
        rows.append(row)
    summary_path = root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2) + "\n")
    csv_path = root / "summary.csv"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("\nEUR chr6 ablations (held-out cohort B)")
    print(
        f"{'condition':<24} {'atk F1':>8} {'drop':>8} {'atk CER':>8} "
        f"{'opt F1':>8} {'hon MAE':>8} {'hon CER':>8}"
    )
    for row in rows:
        optimal = f"{float(row['optimal_f1']):8.3f}" if row["optimal_f1"] != "" else f"{'':>8}"
        print(
            f"{row['name']:<24} {row['attack_f1']:8.3f} {row['relative_drop']:8.1%} "
            f"{row['attack_cer']:8.3f} {optimal} {row['honest_mae']:8.4f} {row['honest_cer']:8.3f}"
        )
    print(f"Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated CUDA device indices. One condition per device. Example: 0,1,2,3",
    )
    parser.add_argument("--results-root", default="results/eur-chr6-ablations")
    parser.add_argument("--detector", default=str(DETECTOR))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[condition.name for condition in CONDITIONS],
        help="Run these conditions instead of the full grid",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    args.wandb = not args.no_wandb
    return args


def main() -> None:
    assert_one_control()
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
    gpus = parse_gpus(args.gpus)
    pending = pending_conditions(args)
    if not pending:
        print("Every selected condition is already finished.", flush=True)
        summarize(Path(args.results_root), args.eval_episodes)
        return
    failed = schedule(args, gpus, pending)
    summarize(Path(args.results_root), args.eval_episodes)
    if failed:
        raise SystemExit(f"Failed conditions: {', '.join(failed)}")


if __name__ == "__main__":
    main()
