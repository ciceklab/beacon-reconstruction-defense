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
for path in (SCRIPTS, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibrate_panel_detectors import DEFAULT_PANELS, parse_panel

ATTACKS = (("mixed", "eval-attack"), ("honest", "eval-honest"))


def cap_label(mae: float) -> str:
    text = f"{float(mae):.6f}".rstrip("0").rstrip(".")
    return f"mae-{text}"


def find_checkpoint(panel_dir: Path) -> Path | None:
    found = list(panel_dir.glob("run*/weights_best_eval/actor.pth"))
    if not found:
        return None
    found.sort(key=lambda path: path.stat().st_mtime)
    return found[-1].parent


def resolve_panel(panel: str, knee_root: Path) -> dict:
    population, prefix = parse_panel(panel)
    knee_path = knee_root / prefix / "knee.json"
    if not knee_path.is_file():
        raise SystemExit(f"Missing knee file: {knee_path}")
    knee = json.loads(knee_path.read_text())
    return {
        "panel": panel,
        "population": population,
        "prefix": prefix,
        "mae": round(float(knee["selected_cap"]), 6),
        "knee_inside_grid": bool(knee.get("knee_inside_grid")),
    }


def shared_args(job: dict, detector: Path, seed: int) -> list[str]:
    prefix = job["prefix"]
    mae = float(job["mae"])
    return [
        "--data",
        f"data/{job['population']}",
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
        f"{mae:.6f}",
        "--no-mae-limit-ramp",
        "--hard-mae-budget",
        "--seed",
        str(seed),
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
        "--bra-fast",
        "--binary-beacon",
        "--cuda-device",
        "0",
        "--suspicion-detector-path",
        str(detector),
    ]


def train_command(job: dict, panel_dir: Path, detector: Path, args: argparse.Namespace) -> list[str]:
    prefix = job["prefix"]
    label = cap_label(job["mae"])
    command = [
        sys.executable,
        "-u",
        "main.py",
        *shared_args(job, detector, args.seed),
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
        str(panel_dir),
        "--run_name",
        f"{prefix}-{label}-yesno",
        "--tags",
        "yes-no-beacon",
        "train",
        prefix,
        label,
        "--wandb-group",
        "yes-no-beacon-train",
    ]
    if args.wandb:
        command.append("--use_wandb")
    return command


def eval_command(
    job: dict,
    attacker: str,
    checkpoint: Path,
    output_dir: Path,
    detector: Path,
    args: argparse.Namespace,
) -> list[str]:
    prefix = job["prefix"]
    label = cap_label(job["mae"])
    command = [
        sys.executable,
        "-u",
        "main.py",
        *shared_args(job, detector, args.eval_seed),
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
        "--results-dir",
        str(output_dir),
        "--run_name",
        f"{prefix}-{label}-yesno-{attacker}",
        "--tags",
        "yes-no-beacon",
        "eval",
        prefix,
        label,
        attacker,
        "--wandb-group",
        "yes-no-beacon",
    ]
    if args.wandb:
        command.append("--use_wandb")
    return command


def csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def log_finished(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    with log_path.open(errors="replace") as handle:
        return any("Convergence metrics saved to" in line for line in handle)


def eval_done(output_dir: Path, log_path: Path, episodes: int) -> bool:
    if not log_finished(log_path):
        return False
    if csv_rows(output_dir / "convergence_metrics.csv") < episodes:
        return False
    return (output_dir / "baseline_no_defense.json").is_file()


def column_mean(rows: list[dict[str, str]], name: str) -> float | None:
    values = [float(row[name]) for row in rows if row.get(name) not in ("", None)]
    if not values:
        return None
    return sum(values) / len(values)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(results_root: Path, jobs: list[dict], episodes: int) -> list[dict]:
    rows = []
    for job in jobs:
        prefix = job["prefix"]
        panel_dir = results_root / prefix
        attack_dir = panel_dir / "eval-attack"
        honest_dir = panel_dir / "eval-honest"
        attack_csv = attack_dir / "convergence_metrics.csv"
        honest_csv = honest_dir / "convergence_metrics.csv"
        baseline_path = attack_dir / "baseline_no_defense.json"
        if csv_rows(attack_csv) < episodes or csv_rows(honest_csv) < episodes:
            print(f"{prefix}: eval is incomplete", flush=True)
            continue
        if not baseline_path.is_file():
            print(f"{prefix}: missing {baseline_path}", flush=True)
            continue
        attack = _read_csv(attack_csv)
        honest = _read_csv(honest_csv)
        baseline = json.loads(baseline_path.read_text())
        no_defense = float(baseline["f1_mean"])
        attack_f1 = column_mean(attack, "f1")
        drop = (no_defense - attack_f1) / no_defense if no_defense and attack_f1 is not None else None
        rows.append(
            {
                "panel": prefix,
                "mae_cap": job["mae"],
                "checkpoint": str(find_checkpoint(panel_dir) or ""),
                "no_defense_f1": no_defense,
                "attack_f1": attack_f1,
                "relative_drop": drop,
                "attack_mae": column_mean(attack, "mae"),
                "attack_flip_rate": column_mean(attack, "binary_flip_rate"),
                "attack_false_no_rate": column_mean(attack, "binary_false_no_rate"),
                "attack_false_yes_rate": column_mean(attack, "binary_false_yes_rate"),
                "honest_mae": column_mean(honest, "mae"),
                "honest_flip_rate": column_mean(honest, "binary_flip_rate"),
                "honest_false_no_rate": column_mean(honest, "binary_false_no_rate"),
                "honest_cer": column_mean(honest, "clinical_error_ratio"),
            }
        )
    path = results_root / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    _print_summary(rows)
    print(f"Wrote {path}", flush=True)
    return rows


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _print_summary(rows: list[dict]) -> None:
    if not rows:
        print("No finished yes/no evals to summarize.", flush=True)
        return
    header = (
        f"{'panel':<12} {'cap':>8} {'atk F1':>8} {'base F1':>8} {'drop':>8} "
        f"{'atk flip':>8} {'hon flip':>8} {'atk MAE':>8} {'hon MAE':>8}"
    )
    print("\n" + header, flush=True)
    for row in rows:
        print(
            f"{row['panel']:<12} {row['mae_cap']:8.6f} {_fmt(row['attack_f1'], 4):>8} "
            f"{_fmt(row['no_defense_f1'], 4):>8} {_fmt(row['relative_drop'], 3):>8} "
            f"{_fmt(row['attack_flip_rate']):>8} {_fmt(row['honest_flip_rate']):>8} "
            f"{_fmt(row['attack_mae'], 4):>8} {_fmt(row['honest_mae'], 4):>8}",
            flush=True,
        )


def missing_inputs(jobs: list[dict], detectors_root: Path) -> list[str]:
    missing = []
    for job in jobs:
        detector = detectors_root / job["prefix"] / "suspicion_detector.json"
        if not detector.is_file():
            missing.append(str(detector))
        for suffix in ("cleaned_binary.npy", "cleaned_mafs.npy"):
            data = ROOT / "data" / job["population"] / f"{job['prefix']}_{suffix}"
            if not data.is_file():
                missing.append(str(data))
    return missing


def parse_gpus(spec: str) -> list[int]:
    try:
        gpu_ids = [int(part.strip()) for part in spec.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"--gpus must be comma-separated device ids, got {spec!r}") from exc
    if not gpu_ids:
        raise SystemExit("--gpus needs at least one device")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise SystemExit(f"--gpus repeats a device: {spec}")
    return gpu_ids


def train_done(panel_dir: Path, episodes_log: Path) -> bool:
    return log_finished(episodes_log) and find_checkpoint(panel_dir) is not None


def eval_is_current(output_dir: Path, log_path: Path, checkpoint: Path, episodes: int) -> bool:
    if not eval_done(output_dir, log_path, episodes):
        return False
    metrics = output_dir / "convergence_metrics.csv"
    actor = checkpoint / "actor.pth"
    return metrics.stat().st_mtime >= actor.stat().st_mtime


def launch(args: argparse.Namespace, jobs: list[dict], gpus: list[int]) -> list[str]:
    results_root = Path(args.results_root)
    
    pending: list[tuple[str, dict]] = []
    for job in jobs:
        panel_dir = results_root / job["prefix"]
        train_log = panel_dir / "train.log"
        if train_done(panel_dir, train_log) and not args.rerun:
            print(f"Reusing finished {job['prefix']} training", flush=True)
            pending.extend((f"eval-{attacker}", job) for attacker, _folder in ATTACKS)
            continue
        pending.append(("train", job))

    active: list[tuple[subprocess.Popen, str, str, dict, int, Path]] = []
    failed: list[str] = []
    spawn_delay = 8.0
    last_spawn = 0.0

    def spawn(phase: str, job: dict, gpu: int) -> tuple[subprocess.Popen, str, str, dict, int, Path] | None:
        panel_dir = results_root / job["prefix"]
        detector = Path(args.detectors_root) / job["prefix"] / "suspicion_detector.json"
        panel_dir.mkdir(parents=True, exist_ok=True)
        if phase == "train":
            log_path = panel_dir / "train.log"
            command = train_command(job, panel_dir, detector, args)
            name = f"{job['prefix']} train"
        else:
            checkpoint = find_checkpoint(panel_dir)
            if checkpoint is None:
                print(f"No checkpoint for {job['prefix']}; skipping eval", flush=True)
                failed.append(f"{job['prefix']} eval")
                return None
            
            attacker = phase.split("-", 1)[1]
            folder = dict(ATTACKS)[attacker]
            output_dir = panel_dir / folder
            log_path = panel_dir / f"{folder}.log"
            if not args.rerun and eval_is_current(output_dir, log_path, checkpoint, args.eval_episodes):
                print(f"Reusing finished {job['prefix']} {attacker} eval", flush=True)
                return None
            command = eval_command(job, attacker, checkpoint, output_dir, detector, args)
            name = f"{job['prefix']} {attacker}"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"Starting {name} on CUDA {gpu} -> {log_path}", flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            proc = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(ROOT),
            )
        return proc, phase, name, job, gpu, log_path

    while pending or active:
        still_running = []
        for proc, phase, name, job, gpu, log_path in active:
            code = proc.poll()
            if code is None:
                still_running.append((proc, phase, name, job, gpu, log_path))
                continue
            if code != 0:
                print(f"Failed {name} on CUDA {gpu} (exit {code}). See {log_path}", flush=True)
                failed.append(name)
                continue
            print(f"Finished {name} on CUDA {gpu}", flush=True)
            if phase == "train":
                pending.extend((f"eval-{attacker}", job) for attacker, _folder in ATTACKS)
        active = still_running
        busy = {gpu for _, _, _, _, gpu, _ in active}
        while pending and len(active) < len(gpus):
            free = [gpu for gpu in gpus if gpu not in busy]
            if not free:
                break
            if active and time.time() - last_spawn < spawn_delay:
                break
            phase, job = pending.pop(0)
            started = spawn(phase, job, free[0])
            if started is None:
                continue
            busy.add(free[0])
            active.append(started)
            last_spawn = time.time()
        if pending or active:
            time.sleep(2)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", default=list(DEFAULT_PANELS))
    parser.add_argument("--knee-root", default="results/mae-knee-v2")
    parser.add_argument("--results-root", default="results/binary-beacon-train")
    parser.add_argument("--detectors-root", default="results/panel-detectors-v3")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Train and evaluate again even if a run already finished")
    parser.add_argument("--dry-run", action="store_true", help="Print the nine caps and exit")
    parser.add_argument("--summarize", action="store_true", help="Write summary.json from finished evals and exit")
    args = parser.parse_args()

    knee_root = Path(args.knee_root)
    jobs = [resolve_panel(panel, knee_root) for panel in args.panels]
    print(
        f"{'panel':<12} {'MAE cap':>10} {'knee':>8}  next",
        flush=True,
    )
    for job in jobs:
        where = "inside" if job["knee_inside_grid"] else "top"
        print(
            f"{job['prefix']:<12} {job['mae']:10.6f} {where:>8}  train on cohort A, then cohort B eval",
            flush=True,
        )

    if args.summarize:
        summarize(Path(args.results_root), jobs, args.eval_episodes)
        return
    if args.dry_run:
        missing = missing_inputs(jobs, Path(args.detectors_root))
        if missing:
            print("\nMissing inputs (needed for a real run):", flush=True)
            for path in missing:
                print(f"  {path}", flush=True)
        return

    missing = missing_inputs(jobs, Path(args.detectors_root))
    if missing:
        listed = "\n  ".join(missing)
        raise SystemExit(f"Missing inputs:\n  {listed}")

    manifest = Path(args.results_root) / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            [
                {
                    "panel": job["prefix"],
                    "population": job["population"],
                    "mae_cap": job["mae"],
                    "lie_rule": "flip the presence bit when the gated action magnitude is above 0.5",
                }
                for job in jobs
            ],
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {manifest}", flush=True)
    failed = launch(args, jobs, parse_gpus(args.gpus))
    summarize(Path(args.results_root), jobs, args.eval_episodes)
    if failed:
        raise SystemExit(f"Failed jobs: {', '.join(failed)}")


if __name__ == "__main__":
    main()
