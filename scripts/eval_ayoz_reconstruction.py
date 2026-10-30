
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers.disease_panels import ensure_disease_loci_in_pool
from helpers.query_suspicion import AGENT_SUSPICION_DIM
from helpers.snp_manifest import honest_bp_window_indices, load_manifest
from helpers.snp_stratification import select_stratified_indices

POPULATION = "EUR"
PREFIX = "eur_chr6"
DEFAULT_M = (1, 3, 5, 10)
WANDB_PROJECT = "BRA-RL-Defense"
WANDB_GROUP = "eur-chr6-ayoz-defended"

CHECKPOINT = Path("results/mae-knee-v2/eur_chr6/mae-0.125926/run2/weights_best_eval")
DETECTOR = Path("results/panel-detectors-v3/eur_chr6/suspicion_detector.json")
MAE_CAP = 0.125926

PANEL_SNPS = 1000
QUERY_CHUNK = 50
TOKEN_DIM = 12 + AGENT_SUSPICION_DIM + 3 + 2 * 4 + 2 * 4 + 8


def flipped_indices(before_present: np.ndarray, after_present: np.ndarray) -> np.ndarray:
    
    before = np.asarray(before_present, dtype=bool).reshape(-1)
    after = np.asarray(after_present, dtype=bool).reshape(-1)
    if before.shape != after.shape:
        raise ValueError("Beacon snapshots must cover the same SNPs")
    return np.flatnonzero(~before & after)


def presence_row(genotypes: np.ndarray) -> np.ndarray:
    
    cohort = np.asarray(genotypes)
    if cohort.ndim != 2:
        raise ValueError("Genotypes must be people by SNP")
    if cohort.shape[0] == 0:
        return np.zeros(cohort.shape[1], dtype=bool)
    return np.max(cohort, axis=0) > 0


def linkage_similarity(reference: np.ndarray) -> np.ndarray:
    
    genotypes = np.asarray(reference, dtype=np.float64)
    if genotypes.ndim != 2 or genotypes.shape[1] == 0:
        raise ValueError("Reference genotypes must be people by SNP, with at least one SNP")
    if genotypes.shape[0] < 2:
        raise ValueError("Linkage needs at least two people outside the beacon")
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(genotypes, rowvar=False)
    if np.ndim(corr) == 0:
        corr = np.array([[1.0]])
    undefined = ~np.isfinite(corr).all(axis=1)
    similarity = np.abs(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
    np.fill_diagonal(similarity, 1.0)
    if np.any(undefined):
        similarity[undefined, :] = 1e-6
        similarity[:, undefined] = 1e-6
        np.fill_diagonal(similarity, 1.0)
    return similarity


def cluster_flips(similarity: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    
    matrix = np.asarray(similarity, dtype=np.float64)
    n_snps = int(matrix.shape[0])
    if matrix.shape != (n_snps, n_snps):
        raise ValueError("Similarity must be square")
    if n_snps == 0:
        return np.zeros(0, dtype=int)
    clusters = int(max(1, min(int(n_clusters), n_snps)))
    if clusters == 1:
        return np.zeros(n_snps, dtype=int)
    model = SpectralClustering(
        n_clusters=clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=int(seed),
        n_init=10,
    )
    return np.asarray(model.fit_predict(matrix), dtype=int)


def precision_recall(truth: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    
    present = np.asarray(truth, dtype=bool).reshape(-1)
    called = np.asarray(predicted, dtype=bool).reshape(-1)
    if present.shape != called.shape:
        raise ValueError("Truth and prediction must cover the same SNPs")
    true_positive = int(np.sum(called & present))
    false_positive = int(np.sum(called & ~present))
    false_negative = int(np.sum(~called & present))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else float("nan")
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else float("nan")
    return precision, recall


def f1_score(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall) or precision + recall == 0.0:
        return float("nan")
    return float(2.0 * precision * recall / (precision + recall))


def score_clusters(labels: np.ndarray, victim: np.ndarray) -> dict[str, float]:
    
    cluster_ids = np.asarray(labels, dtype=int).reshape(-1)
    truth = np.asarray(victim, dtype=bool).reshape(-1)
    if cluster_ids.shape != truth.shape:
        raise ValueError("Cluster labels and the victim must cover the same flipped SNPs")
    if cluster_ids.size == 0:
        empty = float("nan")
        return {
            "oracle_precision": empty,
            "oracle_recall": empty,
            "oracle_f1": empty,
            "blind_precision": empty,
            "blind_recall": empty,
            "blind_f1": empty,
            "n_clusters": 0,
        }
    per_cluster: list[tuple[float, float, float]] = []
    for cluster_id in np.unique(cluster_ids):
        precision, recall = precision_recall(truth, cluster_ids == cluster_id)
        per_cluster.append((precision, recall, f1_score(precision, recall)))
    finite = [row for row in per_cluster if np.isfinite(row[2])]
    if finite:
        oracle = max(finite, key=lambda row: row[2])
    else:
        oracle = max(per_cluster, key=lambda row: (np.nan_to_num(row[0]), np.nan_to_num(row[1])))
    blind_precision = _mean_finite([row[0] for row in per_cluster])
    blind_recall = _mean_finite([row[1] for row in per_cluster])
    return {
        "oracle_precision": float(oracle[0]),
        "oracle_recall": float(oracle[1]),
        "oracle_f1": float(oracle[2]),
        "blind_precision": blind_precision,
        "blind_recall": blind_recall,
        "blind_f1": f1_score(blind_precision, blind_recall),
        "n_clusters": int(len(per_cluster)),
    }


def agreement_correlation(genotypes: np.ndarray) -> np.ndarray:
    
    bits = np.asarray(genotypes, dtype=np.float64).T
    n_people = int(bits.shape[1])
    overlap = bits @ bits.T
    row = bits.sum(axis=1, keepdims=True)
    return (2.0 * overlap + n_people - row - row.T) / max(n_people, 1)


def published_checkpoint(path: Path | None = None) -> Path:
    
    checkpoint = CHECKPOINT if path is None else Path(path)
    actor = checkpoint / "actor.pth"
    if not actor.is_file():
        raise SystemExit(f"Missing published EUR chr6 actor: {actor}")
    return checkpoint


def load_actor(checkpoint: Path):
    
    from td import TD3

    actor = TD3(
        seq_len=QUERY_CHUNK,
        token_dim=TOKEN_DIM,
        action_dim=1,
        inference_only=True,
    )
    actor.load(str(checkpoint))
    return actor


def _defender_args(n_people: int, n_snps: int, detector: Path) -> argparse.Namespace:
    return argparse.Namespace(
        gene_size=n_snps,
        beacon_size=n_people,
        corr_size=0,
        max_queries=min(QUERY_CHUNK, n_snps),
        seq_len=QUERY_CHUNK,
        mae_window=QUERY_CHUNK,
        mae_limit=MAE_CAP,
        mae_limit_ramp=False,
        eval_cohort="B",
        evaluate=True,
        attacker_type="mixed",
        force_attacker_type=True,
        beacon_type="agent",
        shuffle_queries=False,
        fixed_query_indices=np.arange(min(QUERY_CHUNK, n_snps)),
        query_selector=None,
        skip_f1_eval=True,
        skip_terminal_bra=False,
        suspicion_detector_path=str(detector),
        suspicion_features=True,
        suspicion_gate=True,
        rarity_token_features=True,
        ld_token_features=True,
        ld_history_k=4,
        ld_panel_k=4,
        attacker_knowledge_features=True,
        attacker_knowledge_dim=8,
        no_svd_embeddings=True,
        hard_mae_budget=True,
        enforce_mae_budget=True,
        reward_only_mae_budget=False,
        binary_beacon=False,
        hub_only_noise=False,
        sparse_reward=True,
        privacy_f1_baseline=0.7,
        f1_reward_scale=400.0,
        cer_penalty_weight=500.0,
        unified_suspicion_reward=True,
        pop_prefix=PREFIX,
        token_dim=TOKEN_DIM,
        bra_fast=True,
        device="cpu",
        query_mixture_training=False,
        honest_episode_fraction=0.0,
        mixed_episode_fraction=0.0,
    )


def reported_presence(
    genotypes: np.ndarray,
    mafs: np.ndarray,
    actor,
    detector: Path,
) -> np.ndarray:
    
    from env import Env

    people, n_snps = genotypes.shape
    if mafs.shape != (n_snps,):
        raise ValueError(f"MAFs cover {mafs.shape} SNPs, genotypes have {n_snps}")
    answers = np.zeros(n_snps, dtype=bool)
    for start in range(0, n_snps, PANEL_SNPS):
        stop = min(start + PANEL_SNPS, n_snps)
        block = np.asarray(genotypes[:, start:stop], dtype=np.float32)
        block_mafs = np.asarray(mafs[start:stop], dtype=np.float64)
        linkage = agreement_correlation(block)
        env = Env(_defender_args(people, stop - start, detector), block_mafs, block, linkage, linkage)
        offset = 0
        width = stop - start
        while offset < width:
            chunk = np.arange(offset, min(offset + QUERY_CHUNK, width))

            def _select(_genomes, _maf, chosen=chunk):
                return chosen

            env.args.query_selector = _select
            env.args.max_queries = int(chunk.size)
            env.args.shuffle_queries = False
            env.reset()
            done = False
            while not done:
                _state, _reward, done, _info = env.step(beacon_agent=actor)
            local = env.attacker.queries.detach().cpu().numpy().astype(int)
            reported = np.asarray(env.beacon.responses.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
            if local.shape != reported.shape:
                raise RuntimeError(
                    f"Defender returned {reported.size} answers for {local.size} queries"
                )
            answers[start + local] = reported > 0.0
            offset += int(chunk.size)
    return answers


def reconstruct_update(
    beacon: np.ndarray,
    newcomers: np.ndarray,
    reference: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
    before: np.ndarray | None = None,
    after: np.ndarray | None = None,
) -> dict[str, float]:
    
    if before is None:
        before = presence_row(beacon)
    if after is None:
        after = presence_row(np.vstack([beacon, newcomers]))
    flipped = flipped_indices(before, after)
    victim_scores: list[dict[str, float]] = []
    n_clusters_used = 0
    if flipped.size:
        similarity = linkage_similarity(reference[:, flipped])
        labels = cluster_flips(similarity, n_clusters, seed)
        n_clusters_used = int(len(np.unique(labels)))
        
        
        for person in newcomers:
            bits = person[flipped] > 0
            if not np.any(bits):
                continue
            victim_scores.append(score_clusters(labels, bits))
    summary = _average_scores(victim_scores)
    summary["n_victims_scored"] = len(victim_scores)
    minor_alleles = newcomers > 0
    vulnerable = minor_alleles[:, flipped] if flipped.size else np.zeros((newcomers.shape[0], 0), dtype=bool)
    per_person = minor_alleles.sum(axis=1)
    covered = vulnerable.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        coverage = np.divide(covered, per_person, out=np.zeros(per_person.shape, dtype=float), where=per_person > 0)
    summary.update(
        {
            "n_flipped": int(flipped.size),
            "n_clusters": n_clusters_used,
            "mean_victim_minor_alleles": float(per_person.mean()) if per_person.size else 0.0,
            "mean_vulnerable_fraction": float(coverage.mean()) if coverage.size else 0.0,
        }
    )
    return summary


def _mean_finite(values) -> float:
    numbers = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if np.isfinite(number):
            numbers.append(number)
    if not numbers:
        return float("nan")
    return float(np.mean(numbers))


def _average_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = (
        "oracle_precision",
        "oracle_recall",
        "blind_precision",
        "blind_recall",
    )
    averaged = {key: _mean_finite(row.get(key) for row in rows) for key in keys}
    averaged["oracle_f1"] = f1_score(averaged["oracle_precision"], averaged["oracle_recall"])
    averaged["blind_f1"] = f1_score(averaged["blind_precision"], averaged["blind_recall"])
    return averaged


def build_panel(gene_size: int, panel_seed: int) -> np.ndarray:
    
    data_dir = ROOT / "data" / POPULATION
    mafs = np.load(data_dir / f"{PREFIX}_cleaned_mafs.npy", mmap_mode="r")
    manifest = load_manifest(str(data_dir), PREFIX)
    if manifest is None:
        raise SystemExit(f"Missing SNP manifest for {PREFIX}")
    window = honest_bp_window_indices(manifest, PREFIX)
    if window is None or window.size < 1:
        raise SystemExit("EUR chr6 has no HLA window in the manifest")
    if gene_size <= 0 or gene_size >= window.size:
        return np.asarray(window, dtype=int)
    rng = np.random.default_rng(panel_seed)
    selected = select_stratified_indices(np.asarray(mafs), window, gene_size, rng)
    selected = ensure_disease_loci_in_pool(
        selected,
        window,
        manifest,
        gene_size,
        rng=rng,
    )
    if selected.size != gene_size:
        raise SystemExit(f"Panel has {selected.size} SNPs, expected {gene_size}")
    return np.asarray(selected, dtype=int)


def load_panel_genotypes(indices: np.ndarray) -> np.ndarray:
    binary = np.load(ROOT / "data" / POPULATION / f"{PREFIX}_cleaned_binary.npy", mmap_mode="r")
    return np.asarray(binary[:, indices], dtype=np.int8)


def load_panel_mafs(indices: np.ndarray) -> np.ndarray:
    mafs = np.load(ROOT / "data" / POPULATION / f"{PREFIX}_cleaned_mafs.npy", mmap_mode="r")
    return np.asarray(mafs[indices], dtype=np.float64)


def split_people(n_people: int, beacon_size: int, n_new: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if beacon_size + n_new >= n_people:
        raise ValueError(
            f"Beacon {beacon_size} plus {n_new} newcomers leaves no linkage reference in {n_people} people"
        )
    order = np.random.default_rng(seed).permutation(n_people)
    newcomers = order[:n_new]
    beacon = order[n_new : n_new + beacon_size]
    reference = order[n_new + beacon_size :]
    return beacon, newcomers, reference


def run_repeats(
    genotypes: np.ndarray,
    mafs: np.ndarray,
    actor,
    detector: Path,
    *,
    n_new: int,
    beacon_size: int,
    repeats: int,
    base_seed: int,
) -> list[dict[str, float]]:
    rows = []
    for repeat in range(repeats):
        seed = base_seed + repeat
        beacon_idx, new_idx, ref_idx = split_people(genotypes.shape[0], beacon_size, n_new, seed)
        beacon = genotypes[beacon_idx]
        newcomers = genotypes[new_idx]
        before = reported_presence(beacon, mafs, actor, detector)
        after = reported_presence(np.vstack([beacon, newcomers]), mafs, actor, detector)
        summary = reconstruct_update(
            beacon,
            newcomers,
            genotypes[ref_idx],
            n_clusters=n_new,
            seed=seed,
            before=before,
            after=after,
        )
        summary["repeat"] = repeat
        summary["seed"] = seed
        summary["m"] = n_new
        rows.append(summary)
        print(
            f"m={n_new} repeat {repeat + 1}/{repeats} | flipped {summary['n_flipped']} | "
            f"oracle P {summary['oracle_precision']:.3f} R {summary['oracle_recall']:.3f} | "
            f"blind P {summary['blind_precision']:.3f} R {summary['blind_recall']:.3f}",
            flush=True,
        )
    return rows


def _json_ready(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return _json_ready(value.item())
    return value


def log_wandb(n_new: int, rows: list[dict[str, float]], args: argparse.Namespace) -> None:
    if not args.wandb:
        return
    import wandb

    wandb.init(
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=f"eur-chr6-ayoz-m{n_new}",
        tags=["ayoz", PREFIX, "reconstruction", f"m{n_new}"],
        config={
            "m": n_new,
            "beacon_size": args.beacon_size,
            "gene_size": args.gene_size,
            "repeats": args.repeats,
            "panel_seed": args.panel_seed,
            "attack": "ayoz-spectral",
            "defense": str(args.checkpoint),
            "mae_cap": MAE_CAP,
        },
    )
    for row in rows:
        wandb.log({f"repeat/{key}": value for key, value in row.items() if isinstance(value, float)}, step=int(row["repeat"]))
    means = _average_scores(rows)
    wandb.summary.update({f"mean/{key}": value for key, value in means.items()})
    wandb.summary["mean/n_flipped"] = float(np.mean([row["n_flipped"] for row in rows]))
    wandb.finish()


def run_m(n_new: int, args: argparse.Namespace) -> None:
    panel_path = Path(args.results_root) / "panel_indices.npy"
    if not panel_path.is_file():
        raise SystemExit(f"Missing panel indices: {panel_path}")
    genotypes = load_panel_genotypes(np.load(panel_path))
    mafs = load_panel_mafs(np.load(panel_path))
    args.gene_size = int(genotypes.shape[1])
    checkpoint = published_checkpoint(Path(args.checkpoint))
    detector = Path(args.detector)
    if not detector.is_file():
        raise SystemExit(f"Missing detector: {detector}")
    actor = load_actor(checkpoint)
    output_dir = Path(args.results_root) / f"m{n_new}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_repeats(
        genotypes,
        mafs,
        actor,
        detector,
        n_new=n_new,
        beacon_size=args.beacon_size,
        repeats=args.repeats,
        base_seed=args.seed,
    )
    payload = {"m": n_new, "beacon_size": args.beacon_size, "gene_size": args.gene_size, "repeats": rows}
    (output_dir / "metrics.json").write_text(json.dumps(_json_ready(payload), indent=2) + "\n")
    log_wandb(n_new, rows, args)


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


def selected_m(args: argparse.Namespace) -> list[int]:
    if args.only:
        return [int(value) for value in args.only]
    if args.m is not None:
        return [int(args.m)]
    return list(DEFAULT_M)


def prepare_panel(args: argparse.Namespace) -> int:
    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    indices = build_panel(args.gene_size, args.panel_seed)
    np.save(root / "panel_indices.npy", indices)
    (root / "panel_meta.json").write_text(
        json.dumps(
            {
                "population": POPULATION,
                "prefix": PREFIX,
                "gene_size": int(indices.size),
                "panel_seed": args.panel_seed,
                "window": "HLA chr6 31,800,027–33,049,983 bp",
                "beacon_answer": "yes when the frozen actor reports a frequency above zero",
                "attack": "Ayoz et al. spectral clustering on no-to-yes SNPs",
                "defense": str(args.checkpoint),
                "mae_cap": MAE_CAP,
            },
            indent=2,
        )
        + "\n"
    )
    return int(indices.size)


def child_command(n_new: int, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--m",
        str(n_new),
        "--results-root",
        args.results_root,
        "--beacon-size",
        str(args.beacon_size),
        "--gene-size",
        str(args.gene_size),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--panel-seed",
        str(args.panel_seed),
        "--checkpoint",
        args.checkpoint,
        "--detector",
        args.detector,
    ]
    if not args.wandb:
        command.append("--no-wandb")
    return command


def schedule(args: argparse.Namespace, gpus: list[int], n_snps: int) -> list[str]:
    pending = selected_m(args)
    print(
        f"Scheduling Ayoz reconstruction for m={pending} on CUDA "
        f"{', '.join(str(gpu) for gpu in gpus)}. "
        f"Beacon {args.beacon_size}, {n_snps} HLA SNPs, {args.repeats} repeats. "
        f"WandB {'on' if args.wandb else 'off'}.",
        flush=True,
    )
    active: list[tuple[subprocess.Popen, int, int, Path]] = []
    failed: list[str] = []
    spawn_delay = 2.0
    last_spawn = 0.0
    while pending or active:
        still_running: list[tuple[subprocess.Popen, int, int, Path]] = []
        for proc, n_new, gpu, log_path in active:
            code = proc.poll()
            if code is None:
                still_running.append((proc, n_new, gpu, log_path))
                continue
            if code == 0:
                print(f"Finished m={n_new} on CUDA {gpu}", flush=True)
            else:
                print(f"Failed m={n_new} on CUDA {gpu} (exit {code}). See {log_path}", flush=True)
                failed.append(str(n_new))
        active = still_running
        busy = {gpu for _, _, gpu, _ in active}
        while pending and len(active) < len(gpus):
            free = [gpu for gpu in gpus if gpu not in busy]
            if not free:
                break
            if active and time.time() - last_spawn < spawn_delay:
                break
            n_new = pending.pop(0)
            gpu = free[0]
            busy.add(gpu)
            log_path = Path(args.results_root) / f"m{n_new}" / "worker.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"Starting m={n_new} on CUDA {gpu} -> {log_path}", flush=True)
            with log_path.open("w") as log_file:
                proc = subprocess.Popen(
                    child_command(n_new, args),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(ROOT),
                )
            active.append((proc, n_new, gpu, log_path))
            last_spawn = time.time()
        if pending or active:
            time.sleep(1)
    return failed


def summarize(root: Path) -> None:
    rows = []
    for path in sorted(root.glob("m*/metrics.json")):
        payload = json.loads(path.read_text())
        repeats = payload["repeats"]
        means = _average_scores(repeats)
        flipped = [repeat["n_flipped"] for repeat in repeats]
        covered = [repeat["mean_vulnerable_fraction"] for repeat in repeats]
        rows.append(
            {
                "m": payload["m"],
                "repeats": len(repeats),
                "beacon_size": payload["beacon_size"],
                "gene_size": payload["gene_size"],
                "mean_n_flipped": float(np.mean(flipped)) if flipped else 0.0,
                "mean_vulnerable_fraction": float(np.mean(covered)) if covered else 0.0,
                **means,
            }
        )
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(_json_ready(rows), indent=2) + "\n")
    if rows:
        with (root / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(_json_ready(row) for row in rows)
    print("\nAyoz reconstruction, frozen EUR chr6 defender, HLA panel")
    print(f"{'m':>4} {'flipped':>8} {'oracle P':>9} {'oracle R':>9} {'blind P':>8} {'blind R':>8}")
    for row in rows:
        print(
            f"{row['m']:4d} {row['mean_n_flipped']:8.1f} "
            f"{row['oracle_precision']:9.3f} {row['oracle_recall']:9.3f} "
            f"{row['blind_precision']:8.3f} {row['blind_recall']:8.3f}"
        )
    print(f"Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default=None, help="Comma-separated CUDA indices. One update size m per device.")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--results-root", default="results/eur-chr6-ayoz-defended")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--detector", default=str(DETECTOR))
    parser.add_argument("--beacon-size", type=int, default=50)
    parser.add_argument(
        "--gene-size",
        type=int,
        default=0,
        help="SNPs drawn from the HLA window. 0 keeps the whole window.",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3, help="Base seed for people splits. Repeat r uses seed+r.")
    parser.add_argument("--panel-seed", type=int, default=3)
    parser.add_argument("--m", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--only",
        nargs="+",
        type=int,
        help="Update sizes to run. Default: 1 3 5 10",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.beacon_size < 1 or args.repeats < 1:
        raise SystemExit("--beacon-size and --repeats must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.m is None:
            raise SystemExit("A worker needs --m")
        run_m(int(args.m), args)
        return
    if not args.gpus:
        raise SystemExit("Pass --gpus with the CUDA indices to use, for example --gpus 0,1,2,3")
    published_checkpoint(Path(args.checkpoint))
    if not Path(args.detector).is_file():
        raise SystemExit(f"Missing detector: {args.detector}")
    for n_new in selected_m(args):
        if n_new < 1:
            raise SystemExit(f"Update size m must be positive, got {n_new}")
    n_snps = prepare_panel(args)
    failed = schedule(args, parse_gpus(args.gpus), n_snps)
    summarize(Path(args.results_root))
    if failed:
        raise SystemExit(f"Failed update sizes: {', '.join(failed)}")


if __name__ == "__main__":
    main()
