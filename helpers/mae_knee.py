from __future__ import annotations


def select_mae_knee(caps: list[float], relative_drops: list[float]) -> dict:
    if len(caps) != len(relative_drops):
        raise ValueError("caps and relative_drops must have the same length")
    if len(caps) < 3:
        raise ValueError("a knee needs at least three caps")
    order = sorted(range(len(caps)), key=lambda index: float(caps[index]))
    ordered_caps = [float(caps[index]) for index in order]
    ordered_drops = [float(relative_drops[index]) for index in order]
    if any(ordered_caps[index] <= ordered_caps[index - 1] for index in range(1, len(ordered_caps))):
        raise ValueError("caps must be unique")

    slopes = [
        (ordered_drops[index] - ordered_drops[index - 1])
        / (ordered_caps[index] - ordered_caps[index - 1])
        for index in range(1, len(ordered_caps))
    ]
    interior = []
    for index in range(1, len(ordered_caps) - 1):
        decrease = slopes[index - 1] - slopes[index]
        interior.append(
            {
                "cap": ordered_caps[index],
                "relative_drop": ordered_drops[index],
                "slope_in": slopes[index - 1],
                "slope_out": slopes[index],
                "marginal_decrease": decrease,
            }
        )
    largest_decrease = max(point["marginal_decrease"] for point in interior)
    if largest_decrease <= 0.0:
        selected = ordered_caps[-1]
        inside = False
    else:
        selected = min(
            point["cap"]
            for point in interior
            if point["marginal_decrease"] == largest_decrease
        )
        inside = True
    return {
        "selected_cap": selected,
        "knee_inside_grid": inside,
        "points": [
            {"cap": cap, "relative_drop": drop}
            for cap, drop in zip(ordered_caps, ordered_drops)
        ],
        "interior": interior,
    }


def _curve_caps(chosen: dict) -> list[float]:
    
    if "points" in chosen:
        caps = [float(point["cap"]) for point in chosen["points"]]
    elif "selection_curve" in chosen:
        caps = [float(point["mae_limit"]) for point in chosen["selection_curve"]]
    else:
        raise KeyError("knee result has neither points nor selection_curve")
    caps.sort()
    return caps


def unresolved_interval(chosen: dict, ceiling: float = 0.5) -> tuple[float, float] | None:
    
    caps = _curve_caps(chosen)
    if len(caps) < 2:
        raise ValueError("an interval needs at least two caps")
    selected = float(chosen["selected_cap"])
    index = min(range(len(caps)), key=lambda item: abs(caps[item] - selected))
    if not chosen["knee_inside_grid"]:
        last = caps[-1]
        step = last - caps[-2]
        right = min(float(ceiling), last + step)
        if right <= last:
            return None
        return (last, right)
    return (caps[index - 1], caps[index + 1])


def replay_search(
    wide: list[float],
    measured: list[float],
    drops_by_cap: dict[float, float],
    *,
    refine_count: int,
    max_waves: int,
    resolution: float,
    ceiling: float,
) -> dict:
    
    if refine_count < 1:
        raise ValueError("refine_count must be positive")
    if max_waves < 1:
        raise ValueError("max_waves must be positive")
    measured_set = {round(float(cap), 6) for cap in measured}
    drops = {round(float(cap), 6): float(drop) for cap, drop in drops_by_cap.items()}
    required: list[float] = []
    for cap in wide:
        key = round(float(cap), 6)
        if key not in required:
            required.append(key)
    wave = 1
    while True:
        missing = [cap for cap in required if cap not in measured_set]
        if missing:
            basis = [cap for cap in required if cap in measured_set]
            return {"missing": missing, "closed": False, "wave": wave, "basis": basis}
        if wave >= max_waves:
            return {"missing": [], "closed": True, "wave": wave, "basis": list(required)}
        ordered = sorted(required)
        try:
            drop_row = [drops[cap] for cap in ordered]
        except KeyError as exc:
            raise KeyError(f"missing relative drop for cap {exc}") from exc
        chosen = select_mae_knee(ordered, drop_row)
        interval = unresolved_interval(chosen, ceiling=ceiling)
        width = None if interval is None else interval[1] - interval[0]
        if interval is None or width <= resolution:
            return {"missing": [], "closed": True, "wave": wave, "basis": list(required)}
        added = [
            round(float(cap), 6)
            for cap in caps_inside(interval[0], interval[1], refine_count, ordered)
        ]
        if not added:
            return {"missing": [], "closed": True, "wave": wave, "basis": list(required)}
        required = sorted(set(required) | set(added))
        wave += 1


def caps_inside(
    left: float,
    right: float,
    count: int,
    existing: list[float],
    tol: float = 1e-4,
) -> list[float]:
    
    if count < 1:
        raise ValueError("count must be positive")
    if right <= left:
        raise ValueError("interval must be non-empty")
    proposed = []
    span = float(right) - float(left)
    for step in range(1, count + 1):
        value = round(float(left) + span * step / (count + 1), 6)
        if any(abs(value - old) <= tol for old in existing):
            continue
        if value <= float(left) or value >= float(right):
            continue
        proposed.append(value)
    return proposed
