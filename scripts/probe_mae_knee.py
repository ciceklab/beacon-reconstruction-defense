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
from helpers.mae_knee import replay_search, select_mae_knee


CAPS: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25)
REFINE_COUNT = 5


def cap_label(mae: float) -> str:
    text = f"{float(mae):.6f}".rstrip("0").rstrip(".")
    return f"mae-{text}"


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
        raise SystemExit(f"No values for {name} in the eval table")
    return sum(values) / len(values)


def find_checkpoint(cap_dir: Path) -> Path:
    found = list(cap_dir.glob("run*/weights_best_eval/actor.pth"))
    if not found:
        raise SystemExit(f"No saved defender under {cap_dir}")
    found.sort(key=lambda path: path.stat().st_mtime)
    return found[-1].parent


def common_args(
    population: str,
    prefix: str,
    mae: float,
    detector: Path,
    args: argparse.Namespace,
) -> list[str]:
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
        f"{mae:.6f}",
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
        "--bra-fast",
        "--cuda-device",
        str(args.cuda_device),
        "--suspicion-detector-path",
        str(detector),
    ]


def train_command(
    population: str,
    prefix: str,
    mae: float,
    cap_dir: Path,
    detector: Path,
    args: argparse.Namespace,
) -> list[str]:
    label = cap_label(mae)
    command = [
        sys.executable,
        "-u",
        "main.py",
        *common_args(population, prefix, mae, detector, args),
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
        str(cap_dir),
        "--run_name",
        f"{prefix}-{label}",
        "--tags",
        "mae-knee",
        prefix,
        label,
        "--wandb-group",
        f"mae-knee-{prefix}",
    ]
    if args.wandb:
        command.append("--use_wandb")
    return command


def eval_command(
    population: str,
    prefix: str,
    mae: float,
    attacker: str,
    cohort: str,
    seed: int,
    checkpoint: Path,
    output_dir: Path,
    detector: Path,
    args: argparse.Namespace,
) -> list[str]:
    label = cap_label(mae)
    command = [
        sys.executable,
        "-u",
        "main.py",
        *common_args(population, prefix, mae, detector, args),
        "--evaluate",
        "--eval_cohort",
        cohort,
        "--force-attacker-type",
        "--attacker_type",
        attacker,
        "--resume-beacon",
        str(checkpoint),
        "--episodes",
        str(args.eval_episodes),
        "--seed",
        str(seed),
        "--results-dir",
        str(output_dir),
        "--run_name",
        f"{prefix}-{label}-eval-{cohort}-{attacker}",
        "--tags",
        "mae-knee",
        "eval",
        prefix,
        label,
        cohort,
        attacker,
        "--wandb-group",
        f"mae-knee-{prefix}-eval",
    ]
    if args.wandb:
        command.append("--use_wandb")
    return command


def eval_done(output_dir: Path, episodes: int, need_baseline: bool) -> bool:
    if csv_rows(output_dir / "convergence_metrics.csv") < episodes:
        return False
    if need_baseline and not (output_dir / "baseline_no_defense.json").is_file():
        return False
    return True


def attack_drop(output_dir: Path) -> dict[str, float]:
    mixed_path = output_dir / "convergence_metrics.csv"
    baseline_path = output_dir / "baseline_no_defense.json"
    with mixed_path.open(newline="") as handle:
        mixed = list(csv.DictReader(handle))
    with baseline_path.open() as handle:
        no_defense_f1 = float(json.load(handle)["f1_mean"])
    attack_f1 = column_mean(mixed, "f1")
    drop = (no_defense_f1 - attack_f1) / no_defense_f1 if no_defense_f1 else 0.0
    return {
        "no_defense_f1": no_defense_f1,
        "attack_f1": attack_f1,
        "relative_drop": drop,
        "attack_mae": column_mean(mixed, "mae"),
        "attack_cer": column_mean(mixed, "clinical_error_ratio"),
    }


def write_knee(panel_dir: Path, prefix: str, measured: list[dict]) -> dict:
    chosen = select_mae_knee(
        [row["mae_limit"] for row in measured],
        [row["relative_drop"] for row in measured],
    )
    payload = {
        "panel": prefix,
        "cohort": "A",
        "rule": (
            "Interior cap where marginal relative attacker-F1 drop falls the most. "
            "A tie keeps the smaller cap. If every step still accelerates, keep the largest cap. "
            "Clinical error is recorded and is not an input."
        ),
        "selected_cap": chosen["selected_cap"],
        "knee_inside_grid": chosen["knee_inside_grid"],
        "selection_curve": measured,
        "interior": chosen["interior"],
    }
    path = panel_dir / "knee.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    where = "inside the grid" if chosen["knee_inside_grid"] else "at the top of the grid (curve still rising)"
    print(
        f"\n{prefix}: selected MAE cap {chosen['selected_cap']:.6f} ({where}). Wrote {path}",
        flush=True,
    )
    return payload


def summarize_heldout(panel_dir: Path, knee: dict, episodes: int) -> None:
    rows = []
    missing = []
    for point in knee["selection_curve"]:
        label = cap_label(point["mae_limit"])
        attack_dir = panel_dir / label / "eval-attack"
        honest_dir = panel_dir / label / "eval-honest"
        if not eval_done(attack_dir, episodes, True) or not eval_done(honest_dir, episodes, False):
            missing.append(label)
            continue
        attack = attack_drop(attack_dir)
        with (honest_dir / "convergence_metrics.csv").open(newline="") as handle:
            honest = list(csv.DictReader(handle))
        rows.append(
            {
                "label": label,
                "mae_limit": point["mae_limit"],
                "selected": abs(float(point["mae_limit"]) - float(knee["selected_cap"])) < 1e-6,
                "cohort_a_relative_drop": point["relative_drop"],
                "cohort_b_relative_drop": attack["relative_drop"],
                "cohort_b_attack_f1": attack["attack_f1"],
                "cohort_b_attack_mae": attack["attack_mae"],
                "cohort_b_attack_cer": attack["attack_cer"],
                "cohort_b_honest_mae": column_mean(honest, "mae"),
                "cohort_b_honest_cer": column_mean(honest, "clinical_error_ratio"),
            }
        )
    if missing:
        raise SystemExit(
            f"Cohort B is missing for {panel_dir.name}: {', '.join(missing)}"
        )
    path = panel_dir / "heldout_summary.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {path} ({len(rows)} caps)")


def detector_for(prefix: str, args: argparse.Namespace) -> Path:
    path = Path(args.detectors_root) / prefix / "suspicion_detector.json"
    if not path.is_file():
        raise SystemExit(f"Missing checklist: {path}")
    return path


def run_cap(args: argparse.Namespace) -> None:
    population, prefix = parse_panel(args.panel)
    mae = float(args.cap)
    detector = detector_for(prefix, args)
    panel_dir = Path(args.results_root) / prefix
    cap_dir = panel_dir / cap_label(mae)
    cap_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "train":
        train_log = cap_dir / "train.log"
        checkpoint = next(cap_dir.glob("run*/weights_best_eval/actor.pth"), None)
        if checkpoint is not None and log_finished(train_log) and not args.retrain:
            print(f"Reusing finished training in {checkpoint.parent}", flush=True)
            return
        stream(train_command(population, prefix, mae, cap_dir, detector, args), train_log)
        return
    weights = find_checkpoint(cap_dir)
    if args.phase == "select":
        output_dir = cap_dir / "eval-select"
        log_path = cap_dir / "eval-select.log"
        if (
            not args.retrain
            and log_finished(log_path)
            and eval_done(output_dir, args.eval_episodes, True)
        ):
            print(f"Reusing finished cohort-A eval in {output_dir}", flush=True)
            return
        stream(
            eval_command(
                population, prefix, mae, "mixed", "A", args.seed, weights, output_dir, detector, args
            ),
            log_path,
        )
        return
    if args.phase != "test":
        raise SystemExit(f"Unknown phase {args.phase}")
    for attacker, folder, need_baseline in (
        ("mixed", "eval-attack", True),
        ("honest", "eval-honest", False),
    ):
        output_dir = cap_dir / folder
        log_path = cap_dir / f"{folder}.log"
        if (
            not args.retrain
            and log_finished(log_path)
            and eval_done(output_dir, args.eval_episodes, need_baseline)
        ):
            print(f"Reusing finished {attacker} eval in {output_dir}", flush=True)
            continue
        stream(
            eval_command(
                population,
                prefix,
                mae,
                attacker,
                "B",
                args.eval_seed,
                weights,
                output_dir,
                detector,
                args,
            ),
            log_path,
        )


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


def child_command(mae: float, panel: str, phase: str, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--phase",
        phase,
        "--panel",
        panel,
        "--cap",
        f"{mae:.6f}",
        "--cuda-device",
        "0",
        "--results-root",
        args.results_root,
        "--detectors-root",
        args.detectors_root,
        "--episodes",
        str(args.episodes),
        "--eval-episodes",
        str(args.eval_episodes),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--seed",
        str(args.seed),
        "--eval-seed",
        str(args.eval_seed),
    ]
    if args.retrain:
        command.append("--retrain")
    if args.no_wandb:
        command.append("--no-wandb")
    return command


def cap_catalog(panel_dir: Path, wide: list[float]) -> dict[float, Path]:
    
    found: dict[float, Path] = {}
    for mae in wide:
        key = round(float(mae), 6)
        found[key] = panel_dir / cap_label(key)
    if panel_dir.is_dir():
        for cap_dir in sorted(path for path in panel_dir.glob("mae-*") if path.is_dir()):
            key = round(float(cap_dir.name.split("-", 1)[1]), 6)
            found[key] = cap_dir
    return dict(sorted(found.items()))


def measured_rows(paths: dict[float, Path], episodes: int) -> list[dict]:
    rows = []
    for mae, cap_dir in paths.items():
        if not cap_phase_done(cap_dir, "select", episodes):
            continue
        row = attack_drop(cap_dir / "eval-select")
        row["mae_limit"] = mae
        rows.append(row)
    return rows


def cap_phase_done(cap_dir: Path, phase: str, episodes: int) -> bool:
    
    if phase == "train":
        checkpoint = next(cap_dir.glob("run*/weights_best_eval/actor.pth"), None)
        return checkpoint is not None and log_finished(cap_dir / "train.log")
    if phase == "select":
        return log_finished(cap_dir / "eval-select.log") and eval_done(
            cap_dir / "eval-select", episodes, True
        )
    if phase == "test":
        attack = cap_dir / "eval-attack"
        honest = cap_dir / "eval-honest"
        return (
            log_finished(cap_dir / "eval-attack.log")
            and log_finished(cap_dir / "eval-honest.log")
            and eval_done(attack, episodes, True)
            and eval_done(honest, episodes, False)
        )
    return False


def schedule(args: argparse.Namespace, gpus: list[int]) -> list[str]:
    
    panels = []

    def refill(panel: dict, announce: bool) -> None:
        if panel["stage"] == "done" or panel["running"] or panel["ready"]:
            return
        prefix = panel["prefix"]
        panel_dir = Path(args.results_root) / prefix
        episodes = args.eval_episodes
        paths = cap_catalog(panel_dir, list(args.caps))
        rows = measured_rows(paths, episodes)
        measured = [row["mae_limit"] for row in rows]
        plan = replay_search(
            list(args.caps),
            measured,
            {row["mae_limit"]: row["relative_drop"] for row in rows},
            refine_count=REFINE_COUNT,
            max_waves=args.max_waves,
            resolution=args.resolution,
            ceiling=args.cap_ceiling,
        )
        missing = set(plan["missing"])
        for mae in paths:
            if mae not in set(measured) and mae not in missing:
                missing.add(mae)
        missing_caps = sorted(missing)
        if announce:
            for mae, cap_dir in paths.items():
                label = cap_label(mae)
                if cap_phase_done(cap_dir, "select", episodes):
                    print(f"Reusing finished cohort-A eval for {prefix} {label}", flush=True)
                elif cap_phase_done(cap_dir, "train", episodes):
                    print(f"Reusing finished training for {prefix} {label}", flush=True)
                if cap_phase_done(cap_dir, "test", episodes):
                    print(f"Reusing finished cohort-B eval for {prefix} {label}", flush=True)
        if missing_caps or not plan["closed"]:
            jobs: list[tuple[str, float]] = []
            for mae in missing_caps:
                cap_dir = paths.get(mae, panel_dir / cap_label(mae))
                if cap_phase_done(cap_dir, "select", episodes):
                    continue
                phase = "select" if cap_phase_done(cap_dir, "train", episodes) else "train"
                jobs.append((phase, mae))
            if not jobs:
                raise SystemExit(f"{prefix} still has an open search but nothing was queued")
            panel["stage"] = "search"
            panel["wave"] = plan["wave"]
            panel["ready"] = jobs
            print(
                f"{prefix} wave {plan['wave']} still needs: "
                f"{', '.join(f'{phase} {cap_label(mae)}' for phase, mae in jobs)}",
                flush=True,
            )
            return
        if len(rows) < 3:
            raise SystemExit(f"{prefix} needs at least three cohort-A results before cohort B")
        knee = write_knee(panel_dir, prefix, rows)
        panel["knee"] = knee
        tests = [
            row["mae_limit"]
            for row in rows
            if not cap_phase_done(paths[row["mae_limit"]], "test", episodes)
        ]
        if tests:
            panel["stage"] = "test"
            panel["ready"] = [("test", mae) for mae in tests]
            print(
                f"{prefix}: cohort B for all {len(rows)} caps "
                f"(selected {knee['selected_cap']:.4f}); still needs "
                f"{', '.join(cap_label(mae) for mae in tests)}",
                flush=True,
            )
            return
        summarize_heldout(panel_dir, knee, episodes)
        panel["stage"] = "done"
        print(f"{prefix}: cohort B already finished for all {len(rows)} caps", flush=True)

    for panel_name in args.panels:
        _, prefix = parse_panel(panel_name)
        panel = {
            "panel": panel_name,
            "prefix": prefix,
            "wave": 1,
            "stage": "search",
            "ready": [],
            "running": 0,
            "knee": None,
        }
        panels.append(panel)
        refill(panel, announce=True)
    print(
        f"Wide scan is {len(args.caps)} caps, so chromosomes share "
        f"CUDA {', '.join(str(gpu) for gpu in gpus)}.",
        flush=True,
    )

    active: list[tuple[subprocess.Popen, dict, str, float, int, Path]] = []
    failed: list[str] = []
    spawn_delay = 8.0
    last_spawn = 0.0
    cursor = 0

    def pop_job() -> tuple[dict, str, float] | None:
        nonlocal cursor
        for stage in ("search", "test"):
            for offset in range(len(panels)):
                panel = panels[(cursor + offset) % len(panels)]
                if panel["stage"] != stage or not panel["ready"]:
                    continue
                phase, mae = panel["ready"].pop(0)
                cursor = (cursor + offset + 1) % len(panels)
                return panel, phase, mae
        return None

    while any(panel["stage"] != "done" for panel in panels) or active:
        still_running = []
        for proc, panel, phase, mae, gpu, log_path in active:
            code = proc.poll()
            if code is None:
                still_running.append((proc, panel, phase, mae, gpu, log_path))
                continue
            panel["running"] -= 1
            if code != 0:
                print(
                    f"Failed {panel['prefix']} {cap_label(mae)} {phase} on CUDA {gpu} "
                    f"(exit {code}). See {log_path}",
                    flush=True,
                )
                failed.append(f"{panel['prefix']}:{cap_label(mae)}:{phase}")
                continue
            print(
                f"Finished {panel['prefix']} {cap_label(mae)} {phase} on CUDA {gpu}",
                flush=True,
            )
            if phase == "train":
                panel["ready"].append(("select", mae))
        active = still_running
        if failed:
            if not active:
                break
            time.sleep(2)
            continue
        for panel in panels:
            if panel["stage"] != "done" and panel["running"] == 0 and not panel["ready"]:
                refill(panel, announce=False)
        busy = {gpu for _, _, _, _, gpu, _ in active}
        while len(active) < len(gpus):
            free = [gpu for gpu in gpus if gpu not in busy]
            if not free:
                break
            if active and time.time() - last_spawn < spawn_delay:
                break
            job = pop_job()
            if job is None:
                break
            panel, phase, mae = job
            gpu = free[0]
            busy.add(gpu)
            log_path = Path(args.results_root) / panel["prefix"] / cap_label(mae) / f"worker-{phase}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(
                f"Starting {panel['prefix']} {cap_label(mae)} {phase} on CUDA {gpu} -> {log_path}",
                flush=True,
            )
            with log_path.open("w") as log_file:
                proc = subprocess.Popen(
                    child_command(mae, panel["panel"], phase, args),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(ROOT),
                )
            panel["running"] += 1
            active.append((proc, panel, phase, mae, gpu, log_path))
            last_spawn = time.time()
        if any(panel["stage"] != "done" for panel in panels) or active:
            idle = [
                panel
                for panel in panels
                if panel["stage"] != "done" and panel["running"] == 0 and not panel["ready"]
            ]
            if not active and idle:
                names = ", ".join(panel["prefix"] for panel in idle)
                raise SystemExit(f"Stopped with unfinished panels: {names}")
            time.sleep(2)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", default=list(DEFAULT_PANELS))
    parser.add_argument("--panel", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--caps", nargs="+", type=float, default=list(CAPS))
    parser.add_argument("--cap", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("train", "select", "test"), default="train")
    parser.add_argument("--results-root", default="results/mae-knee")
    parser.add_argument("--detectors-root", default="results/panel-detectors-v3")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.005,
        help="Stop zooming once the unresolved bracket is at most this wide",
    )
    parser.add_argument("--max-waves", type=int, default=4)
    parser.add_argument("--cap-ceiling", type=float, default=0.25)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    args.wandb = not args.no_wandb
    if len(args.caps) < 3:
        raise SystemExit("--caps needs at least three values")
    if args.worker:
        if args.panel is None or args.cap is None:
            raise SystemExit("A worker needs --panel and --cap")
        run_cap(args)
        return

    gpus = parse_gpus(args.gpus)
    for panel in args.panels:
        _, prefix = parse_panel(panel)
        detector_for(prefix, args)
    failed = schedule(args, gpus)
    if failed:
        raise SystemExit(f"Failed jobs: {', '.join(failed)}")


if __name__ == "__main__":
    main()
