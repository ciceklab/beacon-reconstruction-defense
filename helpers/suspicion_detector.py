from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

DETECTOR_VERSION = 3

DEFAULT_PERSISTENCE = 3


@dataclass
class SuspicionDetector:
    

    coef: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    threshold: float
    fpr_budget: float = 0.10
    auc: float = float("nan")
    episode_auc: float = float("nan")
    honest_fpr: float = float("nan")
    mixed_tpr: float = float("nan")
    calibration_honest_fpr: float = float("nan")
    calibration_mixed_tpr: float = float("nan")
    n_features: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coef = np.asarray(self.coef, dtype=np.float64).reshape(-1)
        self.feature_mean = np.asarray(self.feature_mean, dtype=np.float64).reshape(-1)
        self.feature_std = np.asarray(self.feature_std, dtype=np.float64).reshape(-1)
        self.n_features = int(self.coef.size)

    def score_scale(self) -> str:
        return str(self.meta.get("score_scale", "probability"))

    def persistence(self) -> int:
        try:
            return max(1, int(self.meta.get("persistence", 1)))
        except (TypeError, ValueError):
            return 1

    def _logits(self, features: np.ndarray) -> tuple[bool, np.ndarray]:
        x = np.asarray(features, dtype=np.float64)
        single = x.ndim == 1
        x = x.reshape(-1, self.n_features)
        std = np.where(self.feature_std > 1e-12, self.feature_std, 1.0)
        z = (x - self.feature_mean) / std
        return single, z @ self.coef + self.intercept

    def decision_function(self, features: np.ndarray) -> np.ndarray | float:
        
        single, logits = self._logits(features)
        return float(logits[0]) if single else logits

    def predict_proba(self, features: np.ndarray) -> np.ndarray | float:
        
        single, logits = self._logits(features)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        return float(probs[0]) if single else probs

    def operating_scores(self, features: np.ndarray) -> np.ndarray | float:
        
        if self.score_scale() == "logit":
            return self.decision_function(features)
        return self.predict_proba(features)

    def is_adversarial(self, features: np.ndarray) -> bool:
        score = float(np.asarray(self.operating_scores(features)).reshape(-1)[0])
        return bool(score >= self.threshold)

    def to_dict(self) -> dict:
        return {
            "version": DETECTOR_VERSION,
            "coef": self.coef.tolist(),
            "intercept": float(self.intercept),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "threshold": float(self.threshold),
            "fpr_budget": float(self.fpr_budget),
            "auc": float(self.auc),
            "episode_auc": float(self.episode_auc),
            "honest_fpr": float(self.honest_fpr),
            "mixed_tpr": float(self.mixed_tpr),
            "calibration_honest_fpr": float(self.calibration_honest_fpr),
            "calibration_mixed_tpr": float(self.calibration_mixed_tpr),
            "n_features": int(self.n_features),
            "meta": self.meta,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "SuspicionDetector":
        return cls(
            coef=d["coef"],
            intercept=d["intercept"],
            feature_mean=d["feature_mean"],
            feature_std=d["feature_std"],
            threshold=d["threshold"],
            fpr_budget=d.get("fpr_budget", 0.10),
            auc=d.get("auc", float("nan")),
            episode_auc=d.get("episode_auc", float("nan")),
            honest_fpr=d.get("honest_fpr", float("nan")),
            mixed_tpr=d.get("mixed_tpr", float("nan")),
            calibration_honest_fpr=d.get("calibration_honest_fpr", float("nan")),
            calibration_mixed_tpr=d.get("calibration_mixed_tpr", float("nan")),
            n_features=d.get("n_features", len(d["coef"])),
            meta=d.get("meta", {}),
        )

    @classmethod
    def load(cls, path: str) -> "SuspicionDetector":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def with_threshold(self, threshold: float, **metric_updates) -> "SuspicionDetector":
        payload = self.to_dict()
        payload["threshold"] = float(threshold)
        for key, value in metric_updates.items():
            if key in payload:
                payload[key] = value
            else:
                payload.setdefault("meta", {})[key] = value
        return self.from_dict(payload)


def _persistence_level(step_scores: np.ndarray, persistence: int) -> float:
    
    values = np.asarray(step_scores, dtype=np.float64)
    run = max(1, int(persistence))
    if values.size == 0 or values.size < run:
        return float("-inf")
    if run == 1:
        return float(np.max(values))
    windows = np.lib.stride_tricks.sliding_window_view(values, run)
    return float(np.max(np.min(windows, axis=1)))


def _episode_operating_dataset(
    scores: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
    persistence: int = 1,
    label: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    episode_ids = np.asarray(episode_ids)
    grouped: dict[object, list[float]] = {}
    grouped_label: dict[object, int] = {}
    order: list[object] = []
    for score, y, eid in zip(scores, labels, episode_ids):
        key = eid.item() if isinstance(eid, np.generic) else eid
        if key not in grouped:
            grouped[key] = []
            grouped_label[key] = int(y)
            order.append(key)
        elif grouped_label[key] != int(y):
            raise ValueError(f"Episode {key!r} contains multiple labels")
        grouped[key].append(float(score))
    order = sorted(grouped.keys())
    levels = [
        _persistence_level(grouped[key], persistence)
        for key in order
    ]
    episode_labels = [grouped_label[key] for key in order]
    if label is None:
        return np.asarray(levels, dtype=np.float64), np.asarray(episode_labels, dtype=int)
    chosen = [level for level, y in zip(levels, episode_labels) if y == label]
    return np.asarray(chosen, dtype=np.float64)


def _threshold_for_episode_fpr(
    probs: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
    fpr_budget: float,
    persistence: int = 1,
) -> tuple[float, float, float]:
    
    labels = np.asarray(labels, dtype=int)
    episode_ids = np.asarray(episode_ids)
    if not 0.0 <= fpr_budget < 1.0:
        raise ValueError("fpr_budget must be in [0, 1)")

    honest_max = _episode_operating_dataset(probs, labels, episode_ids, persistence, label=0)
    mixed_max = _episode_operating_dataset(probs, labels, episode_ids, persistence, label=1)

    if honest_max.size == 0:
        threshold = 0.5
    else:
        
        
        
        
        
        ordered = np.sort(honest_max)
        allowed_false_positives = int(np.floor(fpr_budget * honest_max.size))
        pivot_index = honest_max.size - allowed_false_positives - 1
        pivot_index = int(np.clip(pivot_index, 0, honest_max.size - 1))
        threshold = float(np.nextafter(ordered[pivot_index], np.inf))

    honest_fpr = float(np.mean(honest_max >= threshold)) if honest_max.size else float("nan")
    mixed_tpr = float(np.mean(mixed_max >= threshold)) if mixed_max.size else float("nan")
    return threshold, honest_fpr, mixed_tpr


def _episode_max_dataset(
    probs: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    episode_ids = np.asarray(episode_ids)
    episode_scores: list[float] = []
    episode_labels: list[int] = []
    for eid in np.unique(episode_ids):
        mask = episode_ids == eid
        unique_labels = np.unique(labels[mask])
        if unique_labels.size != 1:
            raise ValueError(f"Episode {eid!r} contains multiple labels: {unique_labels.tolist()}")
        episode_scores.append(float(np.max(probs[mask])))
        episode_labels.append(int(unique_labels[0]))
    return np.asarray(episode_scores), np.asarray(episode_labels)


def evaluate_detector(
    detector: SuspicionDetector,
    features: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
) -> dict[str, float | int]:
    
    from sklearn.metrics import roc_auc_score

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    eids = np.asarray(episode_ids)
    scores = np.asarray(detector.operating_scores(X), dtype=np.float64)
    episode_scores, episode_labels = _episode_operating_dataset(
        scores, y, eids, detector.persistence()
    )

    def safe_auc(targets: np.ndarray, scores: np.ndarray) -> float:
        try:
            return float(roc_auc_score(targets, scores))
        except ValueError:
            return float("nan")

    honest = episode_labels == 0
    mixed = episode_labels == 1
    return {
        "step_auc": safe_auc(y, scores),
        "episode_auc": safe_auc(episode_labels, episode_scores),
        "honest_fpr": float(np.mean(episode_scores[honest] >= detector.threshold)),
        "mixed_tpr": float(np.mean(episode_scores[mixed] >= detector.threshold)),
        "n_honest_episodes": int(np.sum(honest)),
        "n_mixed_episodes": int(np.sum(mixed)),
        "honest_peak_mean": float(np.mean(episode_scores[honest])),
        "mixed_peak_mean": float(np.mean(episode_scores[mixed])),
    }


def operating_point_metrics(
    episode_scores: np.ndarray,
    episode_labels: np.ndarray,
    threshold: float,
    *,
    fpr_budget: float = 0.10,
    min_tpr: float = 0.90,
) -> dict[str, float | int | bool]:
    
    scores = np.asarray(episode_scores, dtype=np.float64)
    labels = np.asarray(episode_labels, dtype=int)
    honest = labels == 0
    mixed = labels == 1
    fpr = float(np.mean(scores[honest] >= threshold)) if np.any(honest) else float("nan")
    tpr = float(np.mean(scores[mixed] >= threshold)) if np.any(mixed) else float("nan")
    return {
        "threshold": float(threshold),
        "honest_fpr": fpr,
        "mixed_tpr": tpr,
        "youden": float(tpr - fpr) if np.isfinite(tpr) and np.isfinite(fpr) else float("nan"),
        "n_honest_episodes": int(np.sum(honest)),
        "n_mixed_episodes": int(np.sum(mixed)),
        "passes_fpr_budget": bool(np.isfinite(fpr) and fpr <= fpr_budget + 1e-12),
        "passes_min_tpr": bool(np.isfinite(tpr) and tpr >= min_tpr - 1e-12),
        "passes_recovery_detector": bool(
            np.isfinite(fpr)
            and np.isfinite(tpr)
            and fpr <= fpr_budget + 1e-12
            and tpr >= min_tpr - 1e-12
        ),
    }


def sweep_episode_thresholds(
    episode_scores: np.ndarray,
    episode_labels: np.ndarray,
    *,
    fpr_budget: float = 0.10,
    min_tpr: float = 0.90,
    extra_thresholds: list[float] | None = None,
) -> dict:
    
    scores = np.asarray(episode_scores, dtype=np.float64)
    labels = np.asarray(episode_labels, dtype=int)
    honest = scores[labels == 0]
    extras = [] if extra_thresholds is None else [float(t) for t in extra_thresholds]
    parts = [np.array([0.0, 1.0], dtype=np.float64), scores]
    if extras:
        parts.append(np.asarray(extras, dtype=np.float64))
    if honest.size:
        parts.append(np.nextafter(honest, np.inf))
    candidates = np.unique(np.concatenate(parts))
    candidates = candidates[np.isfinite(candidates)]
    curve = [
        operating_point_metrics(
            scores, labels, float(t), fpr_budget=fpr_budget, min_tpr=min_tpr
        )
        for t in candidates
    ]

    def _pick(rows: list[dict], key):
        return max(rows, key=key) if rows else None

    budget_rows = [row for row in curve if row["passes_fpr_budget"]]
    tpr_rows = [row for row in curve if row["passes_min_tpr"]]
    return {
        "fpr_budget": float(fpr_budget),
        "min_tpr": float(min_tpr),
        "curve": curve,
        "budget_max_tpr": _pick(
            budget_rows,
            lambda row: (float(row["mixed_tpr"]), -float(row["honest_fpr"]), float(row["threshold"])),
        ),
        "min_tpr_min_fpr": _pick(
            tpr_rows,
            lambda row: (-float(row["honest_fpr"]), float(row["threshold"]), float(row["mixed_tpr"])),
        ),
        "youden": _pick(
            curve,
            lambda row: (float(row["youden"]), float(row["mixed_tpr"]), -float(row["honest_fpr"])),
        ),
    }


def split_episode_two_way(
    labels: np.ndarray,
    episode_ids: np.ndarray,
    *,
    select_fraction: float = 0.5,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    
    if not 0.0 < select_fraction < 1.0:
        raise ValueError("select_fraction must be in (0, 1)")
    y = np.asarray(labels, dtype=int)
    eids = np.asarray(episode_ids)
    rng = np.random.default_rng(seed)
    select_ids: list[object] = []
    confirm_ids: list[object] = []
    for label in (0, 1):
        class_ids = np.unique(eids[y == label])
        if class_ids.size < 2:
            raise ValueError(
                f"Need at least 2 episodes for class {label}; got {class_ids.size}"
            )
        rng.shuffle(class_ids)
        n_select = max(1, int(np.floor(class_ids.size * select_fraction)))
        n_select = min(n_select, class_ids.size - 1)
        select_ids.extend(class_ids[:n_select].tolist())
        confirm_ids.extend(class_ids[n_select:].tolist())
    return np.isin(eids, np.asarray(select_ids)), np.isin(eids, np.asarray(confirm_ids))


def compact_operating_curve(curve: list[dict]) -> list[dict]:
    
    compact: list[dict] = []
    for row in curve:
        key = (round(float(row["honest_fpr"]), 6), round(float(row["mixed_tpr"]), 6))
        if compact:
            prev = compact[-1]
            prev_key = (round(float(prev["honest_fpr"]), 6), round(float(prev["mixed_tpr"]), 6))
            if key == prev_key:
                compact[-1] = row
                continue
        compact.append(row)
    return compact


def threshold_sweep_report(
    detector: SuspicionDetector,
    features: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
    *,
    fpr_budget: float = 0.10,
    min_tpr: float = 0.90,
    select_fraction: float = 0.5,
    split_seed: int = 17,
) -> dict:
    
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    eids = np.asarray(episode_ids)
    probs = np.asarray(detector.operating_scores(X), dtype=np.float64)
    select_mask, confirm_mask = split_episode_two_way(
        y, eids, select_fraction=select_fraction, seed=split_seed
    )
    select_scores, select_labels = _episode_operating_dataset(
        probs[select_mask], y[select_mask], eids[select_mask], detector.persistence()
    )
    confirm_scores, confirm_labels = _episode_operating_dataset(
        probs[confirm_mask], y[confirm_mask], eids[confirm_mask], detector.persistence()
    )
    all_scores, all_labels = _episode_operating_dataset(
        probs, y, eids, detector.persistence()
    )
    extra = [float(detector.threshold)]
    kw = dict(fpr_budget=fpr_budget, min_tpr=min_tpr, extra_thresholds=extra)
    select_sweep = sweep_episode_thresholds(select_scores, select_labels, **kw)
    all_sweep = sweep_episode_thresholds(all_scores, all_labels, **kw)
    confirm_at = {
        "original": operating_point_metrics(
            confirm_scores,
            confirm_labels,
            float(detector.threshold),
            fpr_budget=fpr_budget,
            min_tpr=min_tpr,
        )
    }
    for name in ("budget_max_tpr", "min_tpr_min_fpr", "youden"):
        chosen = select_sweep[name]
        if chosen is not None:
            confirm_at[name] = operating_point_metrics(
                confirm_scores,
                confirm_labels,
                float(chosen["threshold"]),
                fpr_budget=fpr_budget,
                min_tpr=min_tpr,
            )
    select_sweep = dict(select_sweep)
    all_sweep = dict(all_sweep)
    select_sweep["curve"] = compact_operating_curve(select_sweep["curve"])
    all_sweep["curve"] = compact_operating_curve(all_sweep["curve"])
    return {
        "original_threshold": float(detector.threshold),
        "split_seed": int(split_seed),
        "select_fraction": float(select_fraction),
        "n_select_episodes": int(select_scores.size),
        "n_confirm_episodes": int(confirm_scores.size),
        "select": select_sweep,
        "confirm": confirm_at,
        "all_data": all_sweep,
        "recommended": "budget_max_tpr",
    }


def global_episode_ids(episode_ids: np.ndarray, panel_ids: np.ndarray) -> np.ndarray:
    
    panels = np.asarray(panel_ids)
    local_ids = np.asarray(episode_ids)
    if len(panels) != len(local_ids):
        raise ValueError("panel_ids and episode_ids must have equal lengths")
    keys: dict[tuple[str, str], int] = {}
    out = np.empty(len(local_ids), dtype=np.int64)
    for i, (panel, eid) in enumerate(zip(panels.tolist(), local_ids.tolist())):
        key = (str(panel), str(eid))
        if key not in keys:
            keys[key] = len(keys)
        out[i] = keys[key]
    return out


def split_within_panels(
    labels: np.ndarray,
    episode_ids: np.ndarray,
    panel_ids: np.ndarray,
    *,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
    seed: int = 17,
) -> dict[str, np.ndarray]:
    
    y = np.asarray(labels, dtype=int)
    eids = np.asarray(episode_ids)
    panels = np.asarray(panel_ids).astype(str)
    if not (len(y) == len(eids) == len(panels)):
        raise ValueError("labels, episode_ids, and panel_ids must have equal lengths")
    masks = {name: np.zeros(len(y), dtype=bool) for name in ("train", "calibration", "test")}
    for panel in np.unique(panels):
        step_mask = panels == panel
        local = split_episode_groups(
            y[step_mask],
            eids[step_mask],
            train_fraction=train_fraction,
            calibration_fraction=calibration_fraction,
            seed=seed,
        )
        local_index = np.flatnonzero(step_mask)
        for name, local_mask in local.items():
            masks[name][local_index[np.asarray(local_mask, dtype=bool)]] = True
    return masks


def per_panel_operating_points(
    episode_scores: np.ndarray,
    episode_labels: np.ndarray,
    episode_panels: np.ndarray,
    threshold: float,
    *,
    fpr_budget: float = 0.10,
    min_tpr: float = 0.90,
) -> dict[str, dict]:
    scores = np.asarray(episode_scores, dtype=np.float64)
    labels = np.asarray(episode_labels, dtype=int)
    panels = np.asarray(episode_panels).astype(str)
    return {
        str(panel): operating_point_metrics(
            scores[panels == panel],
            labels[panels == panel],
            threshold,
            fpr_budget=fpr_budget,
            min_tpr=min_tpr,
        )
        for panel in np.unique(panels)
    }


def choose_universal_threshold(
    episode_scores: np.ndarray,
    episode_labels: np.ndarray,
    episode_panels: np.ndarray,
    *,
    fpr_budget: float = 0.10,
    min_tpr: float = 0.90,
) -> dict:
    
    scores = np.asarray(episode_scores, dtype=np.float64)
    labels = np.asarray(episode_labels, dtype=int)
    panels = np.asarray(episode_panels).astype(str)
    honest_scores = scores[labels == 0]
    parts = [np.array([0.0, 1.0], dtype=np.float64)]
    if honest_scores.size:
        parts.append(np.nextafter(honest_scores, np.inf))
    candidates = np.unique(np.concatenate(parts))
    candidates = candidates[np.isfinite(candidates)]
    best: dict | None = None
    best_key: tuple | None = None
    for threshold in candidates:
        by_panel = per_panel_operating_points(
            scores,
            labels,
            panels,
            float(threshold),
            fpr_budget=fpr_budget,
            min_tpr=min_tpr,
        )
        fprs = [float(row["honest_fpr"]) for row in by_panel.values()]
        tprs = [float(row["mixed_tpr"]) for row in by_panel.values()]
        worst_fpr = max(fprs)
        min_tpr_obs = min(tprs)
        mean_tpr = float(np.mean(tprs))
        within_budget = bool(worst_fpr <= fpr_budget + 1e-12)
        key = (within_budget, min_tpr_obs, mean_tpr, -worst_fpr, float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "threshold": float(threshold),
                "worst_honest_fpr": float(worst_fpr),
                "min_mixed_tpr": float(min_tpr_obs),
                "mean_mixed_tpr": mean_tpr,
                "all_panels_within_fpr": within_budget,
                "all_panels_pass": bool(within_budget and min_tpr_obs >= min_tpr - 1e-12),
                "per_panel": by_panel,
            }
    if best is None:
        raise ValueError("No threshold candidates")
    return best


def fit_universal_detector(
    features: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
    panel_ids: np.ndarray,
    *,
    fpr_budget: float = 0.10,
    min_tpr: float = 0.90,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
    split_seed: int = 17,
    C: float = 1.0,
    meta: dict | None = None,
) -> tuple[SuspicionDetector, dict]:
    
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    raw_eids = np.asarray(episode_ids)
    panels = np.asarray(panel_ids).astype(str)
    if X.ndim != 2:
        raise ValueError(f"features must be 2-D; got shape {X.shape}")
    if not (len(X) == len(y) == len(raw_eids) == len(panels)):
        raise ValueError("features, labels, episode_ids, and panel_ids must have equal lengths")
    eids = global_episode_ids(raw_eids, panels)
    for eid in np.unique(eids):
        episode_labels = np.unique(y[eids == eid])
        if episode_labels.size != 1:
            raise ValueError(
                f"Episode {int(eid)} contains multiple labels: {episode_labels.tolist()}"
            )
    masks = split_within_panels(
        y,
        eids,
        panels,
        train_fraction=train_fraction,
        calibration_fraction=calibration_fraction,
        seed=split_seed,
    )
    if np.any(masks["train"] & masks["calibration"]) or np.any(masks["train"] & masks["test"]) or np.any(masks["calibration"] & masks["test"]):
        raise RuntimeError("Universal detector splits overlap")

    mean = X[masks["train"]].mean(axis=0)
    std = X[masks["train"]].std(axis=0)
    std_safe = np.where(std > 1e-12, std, 1.0)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit((X[masks["train"]] - mean) / std_safe, y[masks["train"]])
    detector = SuspicionDetector(
        coef=clf.coef_.reshape(-1),
        intercept=float(clf.intercept_[0]),
        feature_mean=mean,
        feature_std=std_safe,
        threshold=0.5,
        fpr_budget=fpr_budget,
        meta={
            "score_scale": "logit",
            "persistence": DEFAULT_PERSISTENCE,
        },
    )

    def _peaks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        step_scores = np.asarray(detector.operating_scores(X[mask]), dtype=np.float64)
        scores, episode_labels = _episode_operating_dataset(
            step_scores, y[mask], eids[mask], detector.persistence()
        )
        episode_panels: list[str] = []
        for eid in np.unique(eids[mask]):
            episode_panels.append(str(panels[mask][eids[mask] == eid][0]))
        return scores, episode_labels, np.asarray(episode_panels)

    calibration_choice = choose_universal_threshold(
        *_peaks(masks["calibration"]),
        fpr_budget=fpr_budget,
        min_tpr=min_tpr,
    )
    detector.threshold = float(calibration_choice["threshold"])
    test_scores, test_labels, test_panels = _peaks(masks["test"])
    test_per_panel = per_panel_operating_points(
        test_scores,
        test_labels,
        test_panels,
        detector.threshold,
        fpr_budget=fpr_budget,
        min_tpr=min_tpr,
    )
    test_pooled = operating_point_metrics(
        test_scores,
        test_labels,
        detector.threshold,
        fpr_budget=fpr_budget,
        min_tpr=min_tpr,
    )
    test_worst_fpr = max(float(row["honest_fpr"]) for row in test_per_panel.values())
    test_min_tpr = min(float(row["mixed_tpr"]) for row in test_per_panel.values())
    detector.honest_fpr = float(test_worst_fpr)
    detector.mixed_tpr = float(test_min_tpr)
    detector.calibration_honest_fpr = float(calibration_choice["worst_honest_fpr"])
    detector.calibration_mixed_tpr = float(calibration_choice["min_mixed_tpr"])
    pooled_eval = evaluate_detector(detector, X[masks["test"]], y[masks["test"]], eids[masks["test"]])
    detector.auc = float(pooled_eval["step_auc"])
    detector.episode_auc = float(pooled_eval["episode_auc"])
    report = {
        "panels": sorted(set(panels.tolist())),
        "fpr_budget": float(fpr_budget),
        "min_tpr": float(min_tpr),
        "threshold": float(detector.threshold),
        "threshold_rule": "maximize worst-panel TPR subject to every panel's calibration FPR staying within budget",
        "calibration": calibration_choice,
        "test_per_panel": test_per_panel,
        "test_pooled": test_pooled,
        "test_worst_honest_fpr": float(test_worst_fpr),
        "test_min_mixed_tpr": float(test_min_tpr),
        "test_all_panels_pass": bool(
            test_worst_fpr <= fpr_budget + 1e-12 and test_min_tpr >= min_tpr - 1e-12
        ),
    }
    detector.meta = _stamp_checklist_meta(meta)
    detector.meta["universal"] = report
    return detector, report


def split_episode_groups(
    labels: np.ndarray,
    episode_ids: np.ndarray,
    *,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
    seed: int = 17,
) -> dict[str, np.ndarray]:
    
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError("train_fraction + calibration_fraction must be < 1")

    y = np.asarray(labels, dtype=int)
    eids = np.asarray(episode_ids)
    rng = np.random.default_rng(seed)
    split_ids: dict[str, list[object]] = {"train": [], "calibration": [], "test": []}

    for label in (0, 1):
        class_ids = np.unique(eids[y == label])
        if class_ids.size < 3:
            raise ValueError(
                f"Need at least 3 episodes for class {label}; got {class_ids.size}"
            )
        rng.shuffle(class_ids)
        n_train = max(1, int(np.floor(class_ids.size * train_fraction)))
        n_cal = max(1, int(np.floor(class_ids.size * calibration_fraction)))
        if n_train + n_cal >= class_ids.size:
            n_cal = max(1, class_ids.size - n_train - 1)
        split_ids["train"].extend(class_ids[:n_train].tolist())
        split_ids["calibration"].extend(class_ids[n_train:n_train + n_cal].tolist())
        split_ids["test"].extend(class_ids[n_train + n_cal:].tolist())

    return {
        name: np.isin(eids, np.asarray(ids))
        for name, ids in split_ids.items()
    }


def fit_detector(
    features: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
    *,
    fpr_budget: float = 0.10,
    C: float = 1.0,
    meta: dict | None = None,
) -> SuspicionDetector:
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if X.ndim != 2:
        raise ValueError(f"features must be 2-D (n_steps, n_features); got shape {X.shape}")

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std > 1e-12, std, 1.0)
    Z = (X - mean) / std_safe

    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit(Z, y)

    logits = clf.decision_function(Z)
    try:
        auc = float(roc_auc_score(y, logits))
    except ValueError:
        auc = float("nan")

    threshold, honest_fpr, mixed_tpr = _threshold_for_episode_fpr(
        logits, y, episode_ids, fpr_budget, persistence=DEFAULT_PERSISTENCE
    )

    return SuspicionDetector(
        coef=clf.coef_.reshape(-1),
        intercept=float(clf.intercept_[0]),
        feature_mean=mean,
        feature_std=std_safe,
        threshold=threshold,
        fpr_budget=fpr_budget,
        auc=auc,
        honest_fpr=honest_fpr,
        mixed_tpr=mixed_tpr,
        n_features=int(X.shape[1]),
        meta=_stamp_checklist_meta(meta),
    )


def fit_detector_split(
    features: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
    *,
    fpr_budget: float = 0.10,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
    split_seed: int = 17,
    C: float = 1.0,
    meta: dict | None = None,
) -> SuspicionDetector:
    
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    eids = np.asarray(episode_ids)
    if X.ndim != 2:
        raise ValueError(f"features must be 2-D (n_steps, n_features); got shape {X.shape}")
    if not (len(X) == len(y) == len(eids)):
        raise ValueError("features, labels, and episode_ids must have equal lengths")

    masks = split_episode_groups(
        y,
        eids,
        train_fraction=train_fraction,
        calibration_fraction=calibration_fraction,
        seed=split_seed,
    )
    train_mask = masks["train"]
    calibration_mask = masks["calibration"]
    test_mask = masks["test"]

    mean = X[train_mask].mean(axis=0)
    std = X[train_mask].std(axis=0)
    std_safe = np.where(std > 1e-12, std, 1.0)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit((X[train_mask] - mean) / std_safe, y[train_mask])

    provisional = SuspicionDetector(
        coef=clf.coef_.reshape(-1),
        intercept=float(clf.intercept_[0]),
        feature_mean=mean,
        feature_std=std_safe,
        threshold=0.5,
        fpr_budget=fpr_budget,
        meta=_stamp_checklist_meta(None),
    )
    calibration_scores = np.asarray(
        provisional.operating_scores(X[calibration_mask]), dtype=np.float64
    )
    threshold, calibration_fpr, calibration_tpr = _threshold_for_episode_fpr(
        calibration_scores,
        y[calibration_mask],
        eids[calibration_mask],
        fpr_budget,
        persistence=provisional.persistence(),
    )
    provisional.threshold = threshold
    test_metrics = evaluate_detector(
        provisional,
        X[test_mask],
        y[test_mask],
        eids[test_mask],
    )

    split_counts = {
        name: {
            "honest": int(np.unique(eids[mask & (y == 0)]).size),
            "mixed": int(np.unique(eids[mask & (y == 1)]).size),
        }
        for name, mask in masks.items()
    }
    detector_meta = _stamp_checklist_meta(meta)
    detector_meta.update(
        {
            "split_seed": int(split_seed),
            "train_fraction": float(train_fraction),
            "calibration_fraction": float(calibration_fraction),
            "test_fraction": float(1.0 - train_fraction - calibration_fraction),
            "split_counts": split_counts,
            "test_metrics": test_metrics,
        }
    )
    provisional.auc = float(test_metrics["step_auc"])
    provisional.episode_auc = float(test_metrics["episode_auc"])
    provisional.honest_fpr = float(test_metrics["honest_fpr"])
    provisional.mixed_tpr = float(test_metrics["mixed_tpr"])
    provisional.calibration_honest_fpr = calibration_fpr
    provisional.calibration_mixed_tpr = calibration_tpr
    provisional.meta = detector_meta
    return provisional


def collect_calibration_data(env, n_honest: int, n_mixed: int):
    
    saved = {
        "beacon_type": env.args.beacon_type,
        "attacker_type": getattr(env.args, "attacker_type", "mixed"),
        "force_attacker_type": getattr(env.args, "force_attacker_type", False),
        "evaluate": getattr(env.args, "evaluate", False),
        "skip_terminal_bra": getattr(env.args, "skip_terminal_bra", False),
    }
    feats: list[list[float]] = []
    labels: list[int] = []
    ep_ids: list[int] = []
    eid = 0
    n_total = int(n_honest) + int(n_mixed)
    try:
        env.args.beacon_type = "none"
        env.args.evaluate = False
        env.args.force_attacker_type = True
        env.args.skip_terminal_bra = True
        for atk, label, n in (("honest", 0, n_honest), ("mixed", 1, n_mixed)):
            for _ in range(n):
                env.args.attacker_type = atk
                env.reset()
                done = False
                info = {}
                while not done:
                    _, _, done, info = env.step(beacon_agent=None)
                for fv in info.get("episode_suspicion_features", []):
                    feats.append(list(fv))
                    labels.append(label)
                    ep_ids.append(eid)
                eid += 1
                if n_total >= 20 and eid % 50 == 0:
                    print(
                        f"  collected {eid}/{n_total} query-stream episodes "
                        "(BRA/SVD/TD3 skipped)",
                        flush=True,
                    )
    finally:
        env.args.beacon_type = saved["beacon_type"]
        env.args.attacker_type = saved["attacker_type"]
        env.args.force_attacker_type = saved["force_attacker_type"]
        env.args.evaluate = saved["evaluate"]
        env.args.skip_terminal_bra = saved["skip_terminal_bra"]
    return (
        np.asarray(feats, dtype=np.float64),
        np.asarray(labels, dtype=int),
        np.asarray(ep_ids, dtype=int),
    )


def _stamp_checklist_meta(meta: dict | None) -> dict:
    from helpers.query_suspicion import DETECTOR_FEATURE_NAMES, LD_THRESHOLD_PERCENTILE

    stamped = dict(meta or {})
    stamped["score_scale"] = "logit"
    stamped["persistence"] = DEFAULT_PERSISTENCE
    stamped["ld_threshold"] = f"panel_percentile_{int(LD_THRESHOLD_PERCENTILE)}"
    stamped["feature_names"] = list(DETECTOR_FEATURE_NAMES)
    return stamped


def update_persistent_engagement(
    score: float,
    threshold: float,
    streak: int,
    latched: bool,
    persistence: int,
) -> tuple[int, bool]:
    
    if latched:
        return int(streak), True
    if float(score) >= float(threshold):
        streak = int(streak) + 1
    else:
        streak = 0
    return streak, streak >= max(1, int(persistence))


def logit_gate(logit: float, threshold: float, *, floor: float = 0.6) -> float:
    
    if float(logit) < float(threshold):
        return 0.0
    excess = 1.0 / (1.0 + np.exp(-(float(logit) - float(threshold))))
    return float(np.clip(floor + (1.0 - floor) * excess, 0.0, 1.0))


def calibrated_gate(prob: float, threshold: float, *, floor: float = 0.6, power: float = 1.0) -> float:
    
    p = float(np.clip(prob, 0.0, 1.0))
    if p < threshold:
        return 0.0
    span = max(1.0 - threshold, 1e-6)
    t = ((p - threshold) / span) ** power
    return float(np.clip(floor + (1.0 - floor) * t, 0.0, 1.0))


def calibrated_latched_gate(
    prob: float,
    previous_peak: float,
    threshold: float,
    *,
    floor: float = 0.6,
    power: float = 1.0,
) -> tuple[float, float]:
    
    peak = max(
        float(np.clip(previous_peak, 0.0, 1.0)),
        float(np.clip(prob, 0.0, 1.0)),
    )
    return peak, calibrated_gate(peak, threshold, floor=floor, power=power)
