from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import json


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_GENE_SIZE = 1000
BEACON_SIZE = 50
TRAINED_SEQ_LEN = 50
TRAINED_MAE_WINDOW = 50
DEFAULT_BEACON_SIZE = 50
DEFAULT_GENE_SIZE = 1000
DEFAULT_QUERIES = 50


PANELS = (
    ("EUR", "eur_chr1"),
    ("EUR", "eur_chr6"),
    ("EUR", "eur_chr22"),
    ("AFR", "afr_chr1"),
    ("AFR", "afr_chr6"),
    ("AFR", "afr_chr22"),
    ("AMR", "amr_chr1"),
    ("AMR", "amr_chr6"),
    ("AMR", "amr_chr22"),
)


def ld_matrix(genotypes: np.ndarray) -> np.ndarray:
    values = np.asarray(genotypes, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError(f"Linkage needs at least 3 people and 2 SNPs, got {values.shape}.")
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(values, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation

def cap_label(mae: float) -> str:
    text = f"{float(mae):.6f}".rstrip("0").rstrip(".")
    return f"mae-{text}"

def checkpoint_for(prefix: str, knee_root: Path) -> tuple[float, Path]:
    knee_path = knee_root / prefix / "knee.json"
    if not knee_path.is_file():
        raise SystemExit(f"Missing knee file: {knee_path}")
    knee = json.loads(knee_path.read_text())
    cap = float(knee["selected_cap"])
    cap_dir = knee_root / prefix / cap_label(cap)
    found = list(cap_dir.glob("run*/weights_best_eval/actor.pth"))
    if not found:
        raise SystemExit(f"No defender weights under {cap_dir}")
    found.sort(key=lambda path: path.stat().st_mtime)
    return cap, found[-1].parent

def command_for(
    population: str,
    prefix: str,
    cap: float,
    weights: Path,
    detector: Path,
    out_dir: Path,
    seed: int,
    beacon_size: int,
) -> list[str]:
    return [
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
        str(beacon_size),
        "--corr_size",
        "50",
        "--max_queries",
        "50",
        "--seq-len",
        "50",
        "--mae-window",
        "50",
        "--mae-limit",
        f"{cap:.6f}",
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
        "--suspicion-detector-path",
        str(detector),
        "--evaluate",
        "--force-attacker-type",
        "--attacker_type",
        "mixed",
        "--eval_cohort",
        "A",
        "--resume-beacon",
        str(weights),
        str(out_dir),
        "--results-dir",
        str(out_dir / "run"),
        "--episodes",
        "1",
    ]

def imputation_command(
    population,
    prefix,
    cap,
    weights,
    detector,
    out_dir,
    seed,
    beacon_size,
    queries,
    gene_size,
    cuda_device,
    run_beagle,
    jar,
    java,
    full_pool: bool = False,
) -> list[str]:
    command = command_for(population, prefix, cap, weights, detector, out_dir, seed, beacon_size)
    if not Path(detector).is_file():
        flag = command.index("--suspicion-detector-path")
        del command[flag : flag + 2]
    _replace(command, "--gene_size", gene_size)
    _replace(command, "--max_queries", queries)
    _replace(command, "--seq-len", TRAINED_SEQ_LEN)
    _replace(command, "--mae-window", TRAINED_MAE_WINDOW)
    command.extend(
        [
            "--cuda-device",
            str(cuda_device),
            "--beagle-jar",
            str(jar),
        ]
    )
    if java:
        command.extend(["--java", java])
    if full_pool:
        command.append("--full-pool")
    if run_beagle:
        command.append("--run-beagle")
    return command

def offdiagonal_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise ValueError(f"Linkage matrices must be square and match, got {left.shape} and {right.shape}.")
    index = np.triu_indices(left.shape[0], k=1)
    x = np.asarray(left[index], dtype=np.float64)
    y = np.asarray(right[index], dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def queried_genotypes(
    truth: np.ndarray,
    reconstruction: np.ndarray,
    query_indices: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries = np.asarray(query_indices, dtype=int).reshape(-1)
    panel = np.asarray(truth)
    guess = np.asarray(reconstruction)
    sites = np.asarray(positions)
    if guess.ndim != 2 or guess.shape[1] != queries.size:
        raise ValueError(f"Reconstruction {guess.shape} does not match {queries.size} queries.")
    if queries.size == 0 or int(queries.min()) < 0 or int(queries.max()) >= panel.shape[1]:
        raise ValueError("A query index falls outside the truth panel.")
    if sites.shape[0] != panel.shape[1]:
        raise ValueError("Positions must cover every column of the truth panel.")
    order = np.argsort(sites[queries], kind="mergesort")
    ordered_queries = queries[order]
    return panel[:, ordered_queries], guess[:, order], sites[ordered_queries]


def score_genotypes(truth: np.ndarray, unguarded: np.ndarray, defended: np.ndarray) -> dict[str, float]:
    true_ld = ld_matrix(truth)
    unguarded_ld = ld_matrix(unguarded)
    defended_ld = ld_matrix(defended)
    off = np.triu_indices(true_ld.shape[0], k=1)
    return {
        "unguarded": offdiagonal_correlation(true_ld, unguarded_ld),
        "defended": offdiagonal_correlation(true_ld, defended_ld),
        "true_mean_abs": float(np.mean(np.abs(true_ld[off]))),
        "true_ld": true_ld,
        "unguarded_ld": unguarded_ld,
        "defended_ld": defended_ld,
    }


def plot_heatmaps(scores: dict, positions: np.ndarray, path: Path, title: str) -> None:
    matrices = (
        ("True genotypes", scores["true_ld"], None),
        ("Unguarded reconstruction", scores["unguarded_ld"], scores["unguarded"]),
        ("Defended reconstruction", scores["defended_ld"], scores["defended"]),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), constrained_layout=True)
    image = None
    for axis, (label, matrix, correlation) in zip(axes, matrices):
        image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="RdBu_r", interpolation="nearest", aspect="equal")
        heading = label if correlation is None else f"{label}\nLD correlation {correlation:.3f}"
        axis.set_title(heading)
        axis.set_xlabel("SNP in genomic order")
        axis.set_ylabel("SNP in genomic order")
        axis.set_xticks([])
        axis.set_yticks([])
    span = f"{int(positions[0]):,}–{int(positions[-1]):,} bp" if positions.size else ""
    figure.suptitle(f"{title}  {span}".rstrip())
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04, label="Pearson r")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def score_episode(episode: Path) -> dict:
    truth = np.load(episode / "truth_window.npy")
    queries = np.load(episode / "query_indices.npy")
    positions = np.load(episode / "positions.npy")
    unguarded = np.load(episode / "unguarded_recon.npy")
    defended = np.load(episode / "defended_recon.npy")
    ordered_truth, ordered_unguarded, ordered_positions = queried_genotypes(truth, unguarded, queries, positions)
    _unused, ordered_defended, _positions = queried_genotypes(truth, defended, queries, positions)
    scores = score_genotypes(ordered_truth, ordered_unguarded, ordered_defended)
    scores["positions"] = ordered_positions
    scores["n_snps"] = int(ordered_positions.size)
    if ordered_positions.size > 1:
        gaps = np.diff(ordered_positions.astype(np.int64))
        scores["median_gap_bp"] = int(np.median(gaps))
    else:
        scores["median_gap_bp"] = 0
    return scores


def format_panel_line(prefix: str, scores: dict) -> str:
    return (
        f"{prefix}: {scores['n_snps']} SNPs, "
        f"{int(scores['positions'][0]):,}–{int(scores['positions'][-1]):,} bp, "
        f"median gap {scores['median_gap_bp']:,} bp. "
        f"True mean |r| off diagonal {scores['true_mean_abs']:.3f}. "
        f"Unguarded LD correlation {scores['unguarded']:.3f}. "
        f"Defended LD correlation {scores['defended']:.3f}."
    )


def _replace(command: list[str], flag: str, value: str) -> None:
    command[command.index(flag) + 1] = value


def episode_command(
    population: str,
    prefix: str,
    cap: float,
    weights: Path,
    detector: Path,
    out_dir: Path,
    seed: int,
    n_queries: int,
    cuda_device: int,
) -> list[str]:
    gene_size = max(DEFAULT_GENE_SIZE, n_queries)
    command = imputation_command(
        population,
        prefix,
        cap,
        weights,
        detector,
        out_dir,
        seed,
        BEACON_SIZE,
        n_queries,
        gene_size,
        cuda_device,
        False,
        ROOT / "tools",
        None,
        False,
    )
    _replace(command, "--data", str(ROOT / "data" / population))
    if not detector.is_file() and "--suspicion-detector-path" in command:
        flag = command.index("--suspicion-detector-path")
        del command[flag : flag + 2]
    return command


def run_panel(population: str, prefix: str, out_dir: Path, n_queries: int, seed: int, cuda_device: int) -> dict:
    cap, weights = checkpoint_for(prefix, ROOT / "results" / "mae-knee-v2")
    detector = ROOT / "results" / "panel-detectors-v3" / prefix / "suspicion_detector.json"
    episode = out_dir / "episode"
    command = episode_command(
        population, prefix, cap, weights, detector, episode, seed, n_queries, cuda_device
    )
    if not detector.is_file():
        print(
            f"WARNING: {detector} is not in this checkout. "
            "The defended reconstruction will use the fixed-weight suspicion gate.",
            flush=True,
        )
    print(
        f"\n=== {population} {prefix} | {n_queries} queries | MAE cap {cap:.6f} | weights {weights} ===",
        flush=True,
    )
    completed = subprocess.run(command, cwd=str(ROOT))
    if completed.returncode != 0:
        raise SystemExit(f"{prefix} reconstruction failed with exit code {completed.returncode}.")
    scores = score_episode(episode)
    plot_heatmaps(scores, scores["positions"], out_dir / "ld_heatmaps.png", prefix)
    (out_dir / "summary.txt").write_text(format_panel_line(prefix, scores) + "\n")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=50, help="SNPs the attacker queries. Sequence length stays 50.")
    parser.add_argument("--panels", nargs="*", default=None, help="Population:prefix pairs. Default is all nine panels.")
    parser.add_argument("--from-episode", type=Path, default=None, help="Score one finished episode directory and exit.")
    parser.add_argument("--out", type=Path, default=Path("results/ld-shattering"))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    if args.queries < 2:
        raise SystemExit("--queries must be at least 2.")

    if args.from_episode is not None:
        scores = score_episode(args.from_episode)
        label = args.from_episode.name
        out_dir = args.out / label
        plot_heatmaps(scores, scores["positions"], out_dir / "ld_heatmaps.png", label)
        text = format_panel_line(label, scores) + "\n"
        (out_dir / "summary.txt").write_text(text)
        print(text, end="")
        print(f"Wrote {out_dir / 'ld_heatmaps.png'}")
        return

    names = list(PANELS if not args.panels else [tuple(value.split(":", 1)) for value in args.panels])
    for population, prefix in names:
        if ":" in population or not prefix:
            raise SystemExit(f"Use POPULATION:prefix, not {population}:{prefix}.")
    out_dir = args.out / f"queries-{args.queries}"
    lines = [
        f"Linkage correlation of reconstructed SNPs. {args.queries} queries, seed {args.seed}.",
        "Each number is the Pearson correlation between off-diagonal entries of the true linkage matrix and the reconstruction.",
        "The diagonal is omitted because it is 1 in every matrix. SNPs are sorted by genomic position.",
        "",
    ]
    for population, prefix in names:
        scores = run_panel(population.upper(), prefix, out_dir / prefix, args.queries, args.seed, args.cuda_device)
        lines.append(format_panel_line(prefix, scores))
    summary = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.txt").write_text(summary)
    print(summary, end="")
    print(f"Wrote {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
