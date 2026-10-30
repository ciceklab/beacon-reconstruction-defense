from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager

import numpy as np

CACHE_VERSION = 2
DEFAULT_CACHE_DIR = "./cache/privacy_baselines"


def build_baseline_cache_key(args, panel_indices: np.ndarray) -> str:
    panel = np.asarray(panel_indices, dtype=np.int64)
    payload = {
        "version": CACHE_VERSION,
        "pop_prefix": getattr(args, "pop_prefix", None),
        "gene_size": int(args.gene_size),
        "max_queries": int(args.max_queries),
        "beacon_size": int(args.beacon_size),
        "corr_size": int(args.corr_size) if args.corr_size is not None else None,
        "seed": int(args.seed),
        "eval_cohort": str(args.eval_cohort),
        "strict_50_50": bool(args.strict_50_50),
        "attacker_type": str(args.attacker_type),
        "force_attacker_type": bool(args.force_attacker_type),
        "shuffle_queries": bool(args.shuffle_queries),
        "bra_fast": bool(args.bra_fast),
        "binary_beacon": bool(getattr(args, "binary_beacon", False)),
        "panel_sha256": hashlib.sha256(panel.tobytes()).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"{key}.json")


@contextmanager
def baseline_cache_lock(cache_dir: str, key: str):
    
    os.makedirs(cache_dir, exist_ok=True)
    lock_file = open(_cache_path(cache_dir, key) + ".lock", "w")
    try:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        yield
    finally:
        lock_file.close()


def load_baseline_cache(
    cache_dir: str,
    key: str,
    expected_meta: dict,
) -> dict[str, float] | None:
    path = _cache_path(cache_dir, key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("version") != CACHE_VERSION:
        return None
    if payload.get("meta") != expected_meta:
        return None
    values = payload.get("baselines", {})
    if "train" not in values:
        return None
    return {name: float(value) for name, value in values.items()}


def save_baseline_cache(
    cache_dir: str,
    key: str,
    meta: dict,
    baselines: dict[str, float],
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, key)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(
            {
                "version": CACHE_VERSION,
                "meta": meta,
                "baselines": {
                    name: float(value) for name, value in baselines.items()
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )
    os.replace(tmp, path)
    return path
