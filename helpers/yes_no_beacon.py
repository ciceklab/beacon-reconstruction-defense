from __future__ import annotations

import numpy as np


def presence_counts(frequencies, beacon_size: int) -> np.ndarray:
    
    if int(beacon_size) < 1:
        raise ValueError("beacon_size must be positive")
    values = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    return np.round(values * int(beacon_size))


def yes_no_attack_targets(
    released_frequencies,
    public_mafs,
    beacon_size: int,
) -> np.ndarray:
    
    released = np.asarray(released_frequencies, dtype=np.float64).reshape(-1)
    mafs = np.asarray(public_mafs, dtype=np.float64).reshape(-1)
    if released.shape != mafs.shape:
        raise ValueError("released frequencies and public MAFs must have the same length")
    one_person = 1.0 / float(beacon_size)
    present = presence_counts(released, beacon_size) >= 1.0
    guessed = np.maximum(np.clip(mafs, 0.0, 1.0), one_person)
    return np.where(present, guessed, 0.0).astype(np.float64)


def yes_no_release(
    gated_action: float,
    true_frequency: float,
    public_maf: float,
    beacon_size: int,
    lie_threshold: float = 0.5,
) -> dict[str, float | bool]:
    
    truth = 1.0 if presence_counts([true_frequency], beacon_size)[0] >= 1.0 else 0.0
    lie = abs(float(gated_action)) > float(lie_threshold)
    bit = (1.0 - truth) if lie else truth
    truthful_target = float(yes_no_attack_targets([truth], [public_maf], beacon_size)[0])
    target = float(yes_no_attack_targets([bit], [public_maf], beacon_size)[0])
    return {
        "bit": float(bit),
        "lie": lie,
        "distortion": float(abs(target - truthful_target)),
    }


def format_yes_no_stats(stats: dict) -> str:
    def rate(key: str) -> str:
        value = stats.get(key)
        if value is None:
            return "n/a"
        return f"{float(value):.3f}"

    return (
        f"flips {rate('binary_flip_rate')} | "
        f"suppressed yes {rate('binary_false_no_rate')} | "
        f"invented yes {rate('binary_false_yes_rate')} | "
        f"honest yes {rate('binary_honest_yes_rate')}"
    )


def yes_no_bit_stats(
    released_frequencies,
    true_frequencies,
    beacon_size: int,
) -> dict[str, float | None]:
    
    released_yes = presence_counts(released_frequencies, beacon_size) >= 1.0
    honest_yes = presence_counts(true_frequencies, beacon_size) >= 1.0
    if released_yes.shape != honest_yes.shape:
        raise ValueError("released and true frequencies must have the same length")
    n = int(released_yes.size)
    if n == 0:
        raise ValueError("bit stats need at least one query")
    flips = released_yes != honest_yes
    n_yes = int(honest_yes.sum())
    n_no = n - n_yes

    def rate(count: int, denom: int) -> float | None:
        if denom == 0:
            return None
        return float(count / denom)

    return {
        "binary_flip_rate": float(flips.mean()),
        "binary_released_yes_rate": float(released_yes.mean()),
        "binary_honest_yes_rate": float(honest_yes.mean()),
        "binary_false_no_rate": rate(int((honest_yes & ~released_yes).sum()), n_yes),
        "binary_false_yes_rate": rate(int((~honest_yes & released_yes).sum()), n_no),
    }
