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

from helpers.snp_manifest import honest_bp_window_indices, load_manifest

POPULATION = "EUR"
PREFIX = "eur_chr6"

MAE_CAP = 0.125926

WINDOW_SNPS = 5000

TRAIN_BEACON_POOL = 64

REGION_PARM = 0  
REGION_MID = 380_847  
REGION_QARM = 761_694  
DETECTOR = Path("results/panel-detectors-v3/eur_chr6/suspicion_detector.json")
WANDB_GROUP = "eur-chr6-robustness"


@dataclass(frozen=True)
class Condition:
    name: str
    axis: str
    beacon_size: int = 50
    gene_size: int = 1000
    max_queries: int = 50
    corr_size: int = 50
    seed: int = 3
    fixed_window_start: int | None = None
    shuffle: bool = True
    note: str = ""


REFERENCE = Condition("reference", "reference", note="HLA window")

CONDITIONS: tuple[Condition, ...] = (
    REFERENCE,
    Condition("beacon-25", "beacon", beacon_size=25, note="25-person beacon, 50-person attacker reference"),
    Condition("snps-250", "snps", gene_size=250, note="250 SNPs from the HLA window"),
    Condition("snps-500", "snps", gene_size=500, note="500 SNPs from the HLA window"),
    Condition("snps-2000", "snps", gene_size=2000, note="2,000 SNPs from the HLA window"),
    Condition(
        "region-6p",
        "region",
        fixed_window_start=REGION_PARM,
        note="6p, 63,979–1,114,974 bp",
    ),
    Condition(
        "region-mid",
        "region",
        fixed_window_start=REGION_MID,
        note="mid-chromosome, 78.3–79.3 Mb",
    ),
    Condition(
        "region-6q",
        "region",
        fixed_window_start=REGION_QARM,
        note="6q telomere, 169.9–171.1 Mb",
    ),
    Condition("queries-25", "queries", max_queries=25, note="25 queries; MAE window covers the episode"),
    Condition("queries-100", "queries", max_queries=100, note="100 queries; MAE window covers the episode"),
    Condition("attacker-25", "attacker", corr_size=25, note="25-person attacker reference"),
    Condition("attacker-100", "attacker", corr_size=100, note="100-person attacker reference, 40-person training pool"),
    Condition("seed-11", "seed", seed=11, note="training seed 11; eval seed stays 123"),
    Condition("seed-29", "seed", seed=29, note="training seed 29; eval seed stays 123"),
    Condition("order-fixed", "order", shuffle=False, note="attacker query order is not shuffled"),
)

FACTORS = (
    "beacon_size",
    "gene_size",
    "max_queries",
    "corr_size",
    "seed",
    "fixed_window_start",
    "shuffle",
)
AXIS_FIELD = {
    "beacon": "beacon_size",
    "snps": "gene_size",
    "queries": "max_queries",
    "attacker": "corr_size",
    "seed": "seed",
    "region": "fixed_window_start",
    "order": "shuffle",
}


def factor_tuple(condition: Condition) -> tuple:
    return tuple(getattr(condition, name) for name in FACTORS)


def assert_one_factor(conditions: tuple[Condition, ...] = CONDITIONS) -> None:
    
    names = [condition.name for condition in conditions]
    if len(names) != len(set(names)):
        raise ValueError("Condition names must be unique")
    reference = factor_tuple(REFERENCE)
    for condition in conditions:
        if condition.name == REFERENCE.name:
            if factor_tuple(condition) != reference:
                raise ValueError("reference does not match the setting")
            continue
        field = AXIS_FIELD.get(condition.axis)
        if field is None:
            raise ValueError(f"{condition.name} has unknown axis {condition.axis}")
        changed = [
            name
            for name, left, right in zip(FACTORS, factor_tuple(condition), reference)
            if left != right
        ]
        if changed != [field]:
            raise ValueError(f"{condition.name} changes {changed}, expected [{field}]")


def assert_fits(condition: Condition, n_people: int, n_snps: int, honest_snps: int) -> None:
    
    training_pool = n_people - condition.beacon_size - condition.corr_size
    if condition.beacon_size < 1 or condition.corr_size < 1 or training_pool < 1:
        raise ValueError(
            f"{condition.name} needs beacon {condition.beacon_size} + attacker reference "
            f"{condition.corr_size} + a training pool, but EUR chr6 has {n_people} people "
            f"(training pool would be {training_pool})"
        )
    if condition.beacon_size > TRAIN_BEACON_POOL:
        raise ValueError(
            f"{condition.name} beacon {condition.beacon_size} exceeds the cohort-A "
            f"draw of {TRAIN_BEACON_POOL} people"
        )
    if condition.gene_size < condition.max_queries:
        raise ValueError(
            f"{condition.name} draws {condition.gene_size} SNPs for {condition.max_queries} queries"
        )
    if condition.fixed_window_start is None:
        if condition.gene_size > honest_snps:
            raise ValueError(
                f"{condition.name} asks for {condition.gene_size} SNPs but the HLA window has {honest_snps}"
            )
        return
    start = condition.fixed_window_start
    if start < 0 or start + WINDOW_SNPS > n_snps:
        raise ValueError(
            f"{condition.name} window start {start} + {WINDOW_SNPS} exceeds {n_snps} SNPs"
        )
    if condition.gene_size > WINDOW_SNPS:
        raise ValueError(
            f"{condition.name} asks for {condition.gene_size} SNPs from a {WINDOW_SNPS}-SNP window"
        )


def panel_shapes() -> tuple[int, int, int]:
    binary = ROOT / "data" / POPULATION / f"{PREFIX}_cleaned_binary.npy"
    if not binary.is_file():
        raise SystemExit(f"Missing EUR chr6 genotypes: {binary}")
    import numpy as np

    people, snps = np.load(binary, mmap_mode="r").shape
    manifest = load_manifest(str(ROOT / "data" / POPULATION), PREFIX)
    if manifest is None:
        raise SystemExit(f"Missing SNP manifest for {PREFIX}")
    honest = honest_bp_window_indices(manifest, PREFIX)
    if honest is None or honest.size < 1:
        raise SystemExit("EUR chr6 has no HLA window in the manifest")
    return int(people), int(snps), int(honest.size)


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
        str(condition.gene_size),
        "--beacon_size",
        str(condition.beacon_size),
        "--corr_size",
        str(condition.corr_size),
        "--max_queries",
        str(condition.max_queries),
        "--seq-len",
        str(condition.max_queries),
        "--mae-window",
        str(condition.max_queries),
        "--mae-limit",
        f"{MAE_CAP:.6f}",
        "--no-mae-limit-ramp",
        "--hard-mae-budget",
        "--seed",
        str(condition.seed),
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
        "--bra-fast",
        "--cuda-device",
        str(cuda_device),
        "--suspicion-detector-path",
        str(detector),
    ]
    if condition.shuffle:
        command.append("--shuffle-queries")
    if condition.fixed_window_start is not None:
        command.extend(["--fixed-window-start", str(condition.fixed_window_start)])
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
        f"eur-chr6-robust-{condition.name}",
        "--tags",
        "robustness",
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
        f"eur-chr6-robust-{condition.name}-eval-{attacker}",
        "--tags",
        "robustness",
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
    n_people, n_snps, honest_snps = panel_shapes()
    assert_fits(condition, n_people, n_snps, honest_snps)
    condition_dir = Path(args.results_root) / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    (condition_dir / "condition.json").write_text(
        json.dumps(
            {
                "name": condition.name,
                "axis": condition.axis,
                "note": condition.note,
                "mae_cap": MAE_CAP,
                "beacon_size": condition.beacon_size,
                "gene_size": condition.gene_size,
                "max_queries": condition.max_queries,
                "corr_size": condition.corr_size,
                "seed": condition.seed,
                "fixed_window_start": condition.fixed_window_start,
                "shuffle": condition.shuffle,
                "training_pool": n_people - condition.beacon_size - condition.corr_size,
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
        eval_ready(
            condition_dir / folder,
            condition_dir / f"{folder}.log",
            episodes,
            checkpoint,
        )
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
        f"Scheduling {len(pending)} EUR chr6 conditions on CUDA "
        f"{', '.join(str(gpu) for gpu in gpus)}. "
        f"MAE cap {MAE_CAP:.6f}. WandB {'on' if args.wandb else 'off'}.",
        flush=True,
    )
    for condition in pending:
        window = "HLA" if condition.fixed_window_start is None else f"start {condition.fixed_window_start}"
        print(
            f"  {condition.name:<14} {condition.axis:<10} "
            f"beacon {condition.beacon_size:<3} snps {condition.gene_size:<4} "
            f"queries {condition.max_queries:<3} attacker {condition.corr_size:<3} "
            f"seed {condition.seed:<3} {window} "
            f"{'shuffled' if condition.shuffle else 'ordered'}",
            flush=True,
        )
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
            "beacon_size": condition.beacon_size,
            "gene_size": condition.gene_size,
            "max_queries": condition.max_queries,
            "corr_size": condition.corr_size,
            "seed": condition.seed,
            "fixed_window_start": "" if condition.fixed_window_start is None else condition.fixed_window_start,
            "shuffle": int(condition.shuffle),
            "mae_cap": MAE_CAP,
            "no_defense_f1": no_defense_f1,
            "attack_f1": attack_f1,
            "relative_drop": (no_defense_f1 - attack_f1) / no_defense_f1 if no_defense_f1 else 0.0,
            "attack_mae": column_mean(attack, "mae"),
            "attack_cer": column_mean(attack, "clinical_error_ratio"),
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
    print("\nEUR chr6 robustness (held-out cohort B, mixed attacker)")
    print(
        f"{'condition':<14} {'atk F1':>8} {'drop':>8} {'opt F1':>8} "
        f"{'hon MAE':>8} {'hon CER':>8}"
    )
    for row in rows:
        optimal = f"{float(row['optimal_f1']):8.3f}" if row["optimal_f1"] != "" else f"{'':>8}"
        print(
            f"{row['name']:<14} {row['attack_f1']:8.3f} {row['relative_drop']:8.1%} "
            f"{optimal} {row['honest_mae']:8.4f} {row['honest_cer']:8.3f}"
        )
    print(f"Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated CUDA device indices. One condition per device. Example: 0,1,2,3",
    )
    parser.add_argument("--results-root", default="results/eur-chr6-robustness")
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
    assert_one_factor()
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
    n_people, n_snps, honest_snps = panel_shapes()
    for condition in selected_conditions(args):
        assert_fits(condition, n_people, n_snps, honest_snps)
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
