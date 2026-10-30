from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Module import ReconstructionTool
from helpers.cohort_data import build_binary_cohort
from helpers.disease_panels import ensure_disease_loci_in_pool
from helpers.snp_manifest import honest_bp_window_indices, load_manifest
from helpers.snp_stratification import select_stratified_indices

QUERY_KINDS = ("mixed", "optimal", "random")
GUESS_KINDS = ("coin", "maf")


def query_indices(mafs: np.ndarray, n_queries: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    
    if kind not in QUERY_KINDS:
        raise ValueError(f"Unknown query set {kind!r}; use one of {QUERY_KINDS}.")
    frequencies = np.asarray(mafs, dtype=np.float64).reshape(-1)
    if n_queries < 1 or n_queries > frequencies.size:
        raise ValueError(f"Need between 1 and {frequencies.size} queries, got {n_queries}.")
    if kind == "random":
        return np.sort(rng.choice(frequencies.size, size=n_queries, replace=False))
    order_high = torch.argsort(torch.as_tensor(frequencies), descending=True).numpy()
    if kind == "optimal":
        return order_high[:n_queries]
    n_high = int(n_queries * 0.8)
    n_low = n_queries - n_high
    order_low = torch.argsort(torch.as_tensor(frequencies), descending=False).numpy()
    return np.concatenate((order_high[:n_high], order_low[:n_low]))


def draw_guess(
    n_snps: int,
    n_people: int,
    kind: str,
    maf: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    
    if kind == "coin":
        return rng.integers(0, 2, size=(n_snps, n_people), dtype=np.int8)
    if kind == "maf":
        if maf is None:
            raise ValueError("A public-MAF guess needs one frequency per queried SNP.")
        rates = np.clip(np.asarray(maf, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if rates.size != n_snps:
            raise ValueError(f"Expected {n_snps} frequencies, got {rates.size}.")
        return (rng.random((n_snps, n_people)) < rates[:, None]).astype(np.int8)
    raise ValueError(f"Unknown guess {kind!r}; use one of {GUESS_KINDS}.")


def matched_f1(truth: np.ndarray, guess: np.ndarray) -> float:
    truth_bits = np.asarray(truth, dtype=np.int8)
    guess_bits = np.asarray(guess, dtype=np.int8)
    if truth_bits.shape != guess_bits.shape or truth_bits.ndim != 2:
        raise ValueError("Truth and guess must be (SNPs, people) arrays of the same shape.")
    if truth_bits.shape[1] == 0:
        raise ValueError("The beacon has no people to match.")
    disagreement = np.mean(truth_bits[:, :, None] != guess_bits[:, None, :], axis=0)
    _rows, columns = linear_sum_assignment(disagreement)
    ordered = guess_bits[:, columns]
    return float(ReconstructionTool.calculate_f1(truth_bits, ordered))


def unmatched_f1(truth: np.ndarray, guess: np.ndarray) -> float:
    
    return float(ReconstructionTool.calculate_f1(np.asarray(truth, dtype=np.int8), np.asarray(guess, dtype=np.int8)))


def load_eval_beacon(
    data_dir: str,
    prefix: str,
    *,
    gene_size: int,
    beacon_size: int,
    corr_size: int,
    seed: int,
    eval_cohort: str,
) -> tuple[np.ndarray, np.ndarray]:
    
    np.random.seed(seed)
    mafs = np.load(os.path.join(data_dir, f"{prefix}_cleaned_mafs.npy"))
    binary_raw = np.load(os.path.join(data_dir, f"{prefix}_cleaned_binary.npy"))
    manifest = load_manifest(data_dir, prefix)
    if manifest is None:
        raise SystemExit(f"No SNP manifest for {prefix} under {data_dir}.")
    window = honest_bp_window_indices(manifest, prefix)
    if window is None or window.size < gene_size:
        have = 0 if window is None else int(window.size)
        raise SystemExit(f"The disease window for {prefix} has {have} SNPs, need {gene_size}.")
    stratified = select_stratified_indices(mafs, window, gene_size, np.random)
    stratified = ensure_disease_loci_in_pool(
        stratified,
        window,
        manifest,
        gene_size,
        rng=np.random.default_rng(seed),
    )
    if stratified.size != gene_size:
        raise SystemExit(f"Panel has {stratified.size} SNPs after disease pinning, expected {gene_size}.")
    binary, _pool, panel_snps = build_binary_cohort(
        binary_raw,
        stratified,
        beacon_size=beacon_size,
        att_count=corr_size,
        eval_cohort=eval_cohort,
        strict_50_50=False,
    )
    if panel_snps != gene_size:
        raise SystemExit(f"Cohort slice has {panel_snps} SNPs, expected {gene_size}.")
    if binary.shape[0] < beacon_size:
        raise SystemExit(f"Cohort has {binary.shape[0]} people, need a beacon of {beacon_size}.")
    beacon = (binary[:beacon_size] > 0.5).astype(np.int8)
    panel_mafs = np.round(np.asarray(mafs[stratified], dtype=np.float64), 3)
    return beacon, panel_mafs


def score_guesses(
    beacon: np.ndarray,
    panel_mafs: np.ndarray,
    queries: np.ndarray,
    *,
    guess: str,
    trials: int,
    rng: np.random.Generator,
) -> dict[str, float | list[float]]:
    truth = beacon[:, queries].T
    queried_maf = panel_mafs[queries]
    matched: list[float] = []
    unmatched: list[float] = []
    for _ in range(trials):
        draw = draw_guess(truth.shape[0], truth.shape[1], guess, queried_maf, rng)
        matched.append(matched_f1(truth, draw))
        unmatched.append(unmatched_f1(truth, draw))
    matched_arr = np.asarray(matched, dtype=np.float64)
    unmatched_arr = np.asarray(unmatched, dtype=np.float64)
    return {
        "positive_rate": float(np.mean(truth)),
        "matched_f1_mean": float(matched_arr.mean()),
        "matched_f1_std": float(matched_arr.std(ddof=1)) if matched_arr.size > 1 else 0.0,
        "matched_f1_min": float(matched_arr.min()),
        "matched_f1_max": float(matched_arr.max()),
        "unmatched_f1_mean": float(unmatched_arr.mean()),
        "matched_f1_per_trial": [float(value) for value in matched_arr],
    }


def run(args: argparse.Namespace) -> dict:
    beacon, panel_mafs = load_eval_beacon(
        args.data,
        args.pop_prefix,
        gene_size=args.gene_size,
        beacon_size=args.beacon_size,
        corr_size=args.corr_size,
        seed=args.seed,
        eval_cohort=args.eval_cohort,
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    for kind in args.queries:
        chosen = query_indices(panel_mafs, args.max_queries, kind, rng)
        for guess in args.guess:
            summary = score_guesses(
                beacon,
                panel_mafs,
                chosen,
                guess=guess,
                trials=args.trials,
                rng=rng,
            )
            rows.append(
                {
                    "guess": guess,
                    "queries": kind,
                    "n_queries": int(args.max_queries),
                    "n_people": int(beacon.shape[0]),
                    **summary,
                }
            )
    payload = {
        "pop_prefix": args.pop_prefix,
        "eval_cohort": args.eval_cohort,
        "seed": int(args.seed),
        "beacon_size": int(args.beacon_size),
        "gene_size": int(args.gene_size),
        "max_queries": int(args.max_queries),
        "trials": int(args.trials),
        "rows": rows,
    }
    return payload


def format_report(payload: dict) -> str:
    header = (
        f"{payload['pop_prefix']} cohort {payload['eval_cohort']}: "
        f"{payload['beacon_size']} people, {payload['max_queries']} queries, "
        f"{payload['gene_size']}-SNP panel, seed {payload['seed']}, "
        f"{payload['trials']} draws"
    )
    lines = [header, ""]
    lines.append(f"{'guess':<8} {'queries':<10} {'carried':>8} {'matched F1':>18} {'unmatched F1':>14}")
    for row in payload["rows"]:
        matched = f"{row['matched_f1_mean']:.4f} ± {row['matched_f1_std']:.4f}"
        lines.append(
            f"{row['guess']:<8} {row['queries']:<10} {row['positive_rate']:>8.3f} "
            f"{matched:>18} {row['unmatched_f1_mean']:>14.4f}"
        )
    lines.append("")
    lines.append(
        "Matched F1 is the metric. Unmatched F1 leaves people in storage order. "
        "A coin guess flips every allele. A MAF guess draws each allele from the cohort "
        "frequency and still ignores which people are in the beacon."
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 of a reconstruction that ignores beacon answers.")
    parser.add_argument("--data", default="data/EUR")
    parser.add_argument("--pop-prefix", default="eur_chr6")
    parser.add_argument("--eval-cohort", default="B", choices=("A", "B"))
    parser.add_argument("--gene-size", type=int, default=1000)
    parser.add_argument("--beacon-size", type=int, default=50)
    parser.add_argument("--corr-size", type=int, default=50)
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--queries", nargs="+", default=list(QUERY_KINDS), choices=QUERY_KINDS)
    parser.add_argument("--guess", nargs="+", default=list(GUESS_KINDS), choices=GUESS_KINDS)
    parser.add_argument("--output", default="results/random-guess-f1/eur_chr6.json")
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.max_queries > args.gene_size:
        parser.error("--max-queries cannot exceed --gene-size")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run(args)
    text = format_report(payload)
    print(text)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote {destination}")


if __name__ == "__main__":
    main()
