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


def stream(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nRunning: {' '.join(command)}", flush=True)
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        return_code = proc.wait()
    if return_code != 0:
        raise SystemExit(f"Command failed ({return_code}). See {log_path}")


def common_args(population: str, prefix: str, args: argparse.Namespace) -> list[str]:
    return [
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
        "0.15",
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
        "--cuda-device",
        str(args.cuda_device),
        "--suspicion-detector-path",
        str(Path(args.detectors_root) / prefix / "suspicion_detector.json"),
    ]


def train_command(population: str, prefix: str, panel_dir: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "main.py",
        *common_args(population, prefix, args),
        "--episodes",
        str(args.episodes),
        "--eval_cohort",
        "A",
        "--bra-fast",
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
        f"{prefix}-checklist-v3",
        "--tags",
        "checklist-v3",
        "train",
        prefix,
    ]
    if args.wandb:
        command.extend(
            [
                "--use_wandb",
                "--wandb-group",
                "checklist-v3-train",
            ]
        )
    return command


def eval_command(
    population: str,
    prefix: str,
    attacker: str,
    checkpoint: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "main.py",
        *common_args(population, prefix, args),
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
        f"{prefix}-eval-{attacker}",
        "--tags",
        "checklist-v3",
        "eval",
        "cohort-B",
        prefix,
        attacker,
    ]
    if args.wandb:
        command.extend(["--use_wandb", "--wandb-group", "checklist-v3-eval"])
    return command


def find_checkpoint(panel_dir: Path) -> Path:
    found = list(panel_dir.glob("run*/weights_best_eval/actor.pth"))
    if not found:
        raise SystemExit(f"No saved defender under {panel_dir}")
    found.sort(key=lambda path: path.stat().st_mtime)
    return found[-1].parent


def _column_mean(rows: list[dict[str, str]], name: str) -> float:
    values = []
    for row in rows:
        raw = row.get(name, "")
        if raw in ("", None):
            continue
        values.append(float(raw))
    if not values:
        raise SystemExit(f"No values for {name}")
    return sum(values) / len(values)


def summarize_panel(prefix: str, mixed_dir: Path, honest_dir: Path) -> None:
    import wandb

    with (mixed_dir / "convergence_metrics.csv").open(newline="") as handle:
        mixed_rows = list(csv.DictReader(handle))
    with (honest_dir / "convergence_metrics.csv").open(newline="") as handle:
        honest_rows = list(csv.DictReader(handle))
    with (mixed_dir / "baseline_no_defense.json").open() as handle:
        no_defense_f1 = float(json.load(handle)["f1_mean"])
    attack_f1 = _column_mean(mixed_rows, "f1")
    attack_mae = _column_mean(mixed_rows, "mae")
    honest_mae = _column_mean(honest_rows, "mae")
    honest_cer = _column_mean(honest_rows, "clinical_error_ratio")
    relative_drop = (no_defense_f1 - attack_f1) / no_defense_f1 if no_defense_f1 else 0.0
    wandb.init(
        project="BRA-RL-Defense",
        name=f"{prefix}",
        group="checklist-v3",
        tags=["checklist-v3", prefix],
        config={"prefix": prefix},
    )
    wandb.log(
        {
            "Episode": 0,
            "Attack/F1": attack_f1,
            "Attack/MAE": attack_mae,
            "Attack/Relative Drop": relative_drop,
            "Attack/No-defense F1": no_defense_f1,
            "Honest/MAE": honest_mae,
            "Honest/CER": honest_cer,
        }
    )
    table = wandb.Table(
        columns=["series", "value"],
        data=[
            ["Attack F1", attack_f1],
            ["No-defense attack F1", no_defense_f1],
            ["Honest noise", honest_mae],
            ["Honest clinical error", honest_cer],
        ],
    )
    wandb.log(
        {
            "Attack vs honest": wandb.plot.bar(
                table,
                "series",
                "value",
                title=f"{prefix}: attackers damaged, researchers quiet",
            )
        }
    )
    wandb.finish()
    print(
        f"{prefix} held-out: attack F1 {attack_f1:.3f} "
        f"(no defense {no_defense_f1:.3f}, drop {relative_drop:.1%}) | "
        f"honest MAE {honest_mae:.4f} CER {honest_cer:.4f}"
    )


def _csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _log_finished(log_path: Path) -> bool:
    
    if not log_path.is_file():
        return False
    with log_path.open(errors="replace") as handle:
        return any("Convergence metrics saved to" in line for line in handle)


def _eval_complete(output_dir: Path, episodes: int, need_baseline: bool) -> bool:
    if _csv_rows(output_dir / "convergence_metrics.csv") < episodes:
        return False
    if need_baseline and not (output_dir / "baseline_no_defense.json").is_file():
        return False
    return True


def _eval_reusable(output_dir: Path, log_path: Path, episodes: int, checkpoint: Path, need_baseline: bool) -> bool:
    metrics = output_dir / "convergence_metrics.csv"
    actor = checkpoint / "actor.pth"
    if not _log_finished(log_path) or not _eval_complete(output_dir, episodes, need_baseline):
        return False
    if not actor.is_file() or not metrics.is_file():
        return False
    return metrics.stat().st_mtime >= actor.stat().st_mtime


def run_panel(population: str, prefix: str, args: argparse.Namespace) -> None:
    physical = os.environ.get("CHECKLIST_V3_PHYSICAL_GPU")
    if physical:
        print(f"{prefix}: physical CUDA {physical} (visible as cuda:0)", flush=True)
    detector = Path(args.detectors_root) / prefix / "suspicion_detector.json"
    if not detector.is_file():
        raise SystemExit(f"Missing checklist: {detector}")
    panel_dir = Path(args.results_root) / prefix
    panel_dir.mkdir(parents=True, exist_ok=True)
    train_log = panel_dir / "train.log"
    checkpoint = next(panel_dir.glob("run*/weights_best_eval/actor.pth"), None)
    training_finished = checkpoint is not None and _log_finished(train_log)
    if training_finished and not args.retrain:
        print(f"Reusing finished training in {checkpoint.parent}", flush=True)
    else:
        if checkpoint is not None and not args.retrain:
            print(
                f"Ignoring {checkpoint.parent}: training did not finish. Starting {prefix}-checklist-v3.",
                flush=True,
            )
        stream(
            train_command(population, prefix, panel_dir, args),
            train_log,
        )
    if args.skip_eval:
        return
    weights = find_checkpoint(panel_dir)
    mixed_dir = panel_dir / "eval-attack"
    honest_dir = panel_dir / "eval-honest"
    if args.retrain or not _eval_reusable(
        mixed_dir, panel_dir / "eval-attack.log", args.eval_episodes, weights, need_baseline=True
    ):
        stream(
            eval_command(population, prefix, "mixed", weights, mixed_dir, args),
            panel_dir / "eval-attack.log",
        )
    else:
        print(f"Reusing finished mixed eval in {mixed_dir}", flush=True)
    if args.retrain or not _eval_reusable(
        honest_dir, panel_dir / "eval-honest.log", args.eval_episodes, weights, need_baseline=False
    ):
        stream(
            eval_command(population, prefix, "honest", weights, honest_dir, args),
            panel_dir / "eval-honest.log",
        )
    else:
        print(f"Reusing finished honest eval in {honest_dir}", flush=True)
    if not args.wandb:
        return

    summarize_panel(prefix, mixed_dir, honest_dir)
    (panel_dir / "summary.json").write_text(
        json.dumps({"prefix": prefix, "mixed_dir": str(mixed_dir), "honest_dir": str(honest_dir)})
    )


def _parse_gpus(spec: str) -> list[int]:
    try:
        gpu_ids = [int(part.strip()) for part in spec.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"--gpus must be comma-separated device ids, got {spec!r}") from exc
    if not gpu_ids:
        raise SystemExit("--gpus needs at least one device, for example 4,5,6,7")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise SystemExit(f"--gpus repeats a device: {spec}")
    return gpu_ids


def _child_command(panel: str, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--panels",
        panel,
        "--cuda-device",
        "0",
        "--detectors-root",
        args.detectors_root,
        "--results-root",
        args.results_root,
        "--episodes",
        str(args.episodes),
        "--eval-episodes",
        str(args.eval_episodes),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--seed",
        str(args.seed),
    ]
    if args.retrain:
        command.append("--retrain")
    if args.skip_eval:
        command.append("--skip-eval")
    if args.no_wandb:
        command.append("--no-wandb")
    return command


def run_parallel(args: argparse.Namespace) -> None:
    gpu_ids = _parse_gpus(args.gpus)
    panels = list(args.panels)
    for panel_value in panels:
        _, prefix = parse_panel(panel_value)
        detector = Path(args.detectors_root) / prefix / "suspicion_detector.json"
        if not detector.is_file():
            raise SystemExit(f"Missing checklist: {detector}")
    print(
        f"Scheduling {len(panels)} chromosomes on CUDA {', '.join(str(gpu) for gpu in gpu_ids)}. "
        "One chromosome per GPU, through train, eval mixed, eval honest.",
        flush=True,
    )
    print(
        "A chromosome trains, then runs both evals. "
        "Only a stage whose log shows a normal exit is reused. "
        f"Per-chromosome logs: {args.results_root}/<prefix>/worker.log",
        flush=True,
    )
    pending = list(panels)
    active: list[tuple[subprocess.Popen, str, int, Path]] = []
    failed: list[str] = []
    spawn_delay = 8.0
    last_spawn = 0.0
    while pending or active:
        still_running: list[tuple[subprocess.Popen, str, int, Path]] = []
        for proc, prefix, gpu, log_path in active:
            code = proc.poll()
            if code is None:
                still_running.append((proc, prefix, gpu, log_path))
                continue
            if code == 0:
                print(f"Finished {prefix} on CUDA {gpu}", flush=True)
            else:
                print(f"Failed {prefix} on CUDA {gpu} (exit {code}). See {log_path}", flush=True)
                failed.append(prefix)
        active = still_running
        busy = {gpu for _, _, gpu, _ in active}
        while pending and len(active) < len(gpu_ids):
            free = [gpu for gpu in gpu_ids if gpu not in busy]
            if not free:
                break
            if active and time.time() - last_spawn < spawn_delay:
                break
            panel_value = pending.pop(0)
            _, prefix = parse_panel(panel_value)
            gpu = free[0]
            busy.add(gpu)
            log_path = Path(args.results_root) / prefix / "worker.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["CHECKLIST_V3_PHYSICAL_GPU"] = str(gpu)
            print(f"Starting {prefix} on CUDA {gpu} -> {log_path}", flush=True)
            with log_path.open("w") as log_file:
                proc = subprocess.Popen(
                    _child_command(panel_value, args),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(ROOT),
                )
            active.append((proc, prefix, gpu, log_path))
            last_spawn = time.time()
        if pending or active:
            time.sleep(2)
    if failed:
        raise SystemExit(f"Failed chromosomes: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", nargs="+", default=list(DEFAULT_PANELS))
    parser.add_argument("--detectors-root", default="results/panel-detectors-v3")
    parser.add_argument("--results-root", default="results/checklist-v3")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated CUDA devices. One chromosome per GPU, e.g. 4,5,6,7.",
    )
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    args.wandb = not args.no_wandb

    if args.gpus:
        run_parallel(args)
    else:
        for panel_value in args.panels:
            population, prefix = parse_panel(panel_value)
            run_panel(population, prefix, args)

    print(
        "\nW&B project BRA-RL-Defense, groups checklist-v3-train, "
        "checklist-v3-eval, and checklist-v3."
    )


if __name__ == "__main__":
    main()
