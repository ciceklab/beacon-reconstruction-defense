import json
import os

def run_no_defense_baseline(env, args, n_episodes: int=10):
    saved_beacon_type = args.beacon_type
    args.beacon_type = 'none'
    env.args.beacon_type = 'none'
    f1_scores = []
    mae_scores = []
    try:
        for _ in range(n_episodes):
            env.reset()
            done = False
            info = {}
            while not done:
                _, _, done, info = env.step(beacon_agent=None)
            f1_scores.append(float(info.get('f1', 1.0)))
            mae_scores.append(float(info.get('mae', 0.0)))
    finally:
        args.beacon_type = saved_beacon_type
        env.args.beacon_type = saved_beacon_type
    return (f1_scores, mae_scores)

def save_baseline_json(path: str, f1_scores, mae_scores, args):
    import statistics as st
    payload = {'episodes': len(f1_scores), 'beacon_type': 'none', 'binary_beacon': bool(getattr(args, 'binary_beacon', False)), 'f1_mean': float(st.mean(f1_scores)), 'f1_std': float(st.pstdev(f1_scores)) if len(f1_scores) > 1 else 0.0, 'f1_min': float(min(f1_scores)), 'f1_max': float(max(f1_scores)), 'f1_per_episode': f1_scores, 'mae_mean': float(st.mean(mae_scores)), 'mae_per_episode': mae_scores, 'seed': int(args.seed), 'fixed_window_start': getattr(args, 'fixed_window_start', None), 'shuffle_queries': bool(getattr(args, 'shuffle_queries', False)), 'privacy_f1_baseline': float(getattr(args, 'privacy_f1_baseline', 0.0))}
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return payload

def load_baseline_json(path: str):
    with open(path) as f:
        return json.load(f)


def run_random_flip(env, n_episodes: int):
    
    if getattr(env.args, "beacon_type", None) != "random":
        raise ValueError("run_random_flip expects beacon_type='random'")
    f1_scores = []
    mae_scores = []
    cer_scores = []
    for episode in range(n_episodes):
        env.reset()
        done = False
        info = {}
        while not done:
            _, _, done, info = env.step(beacon_agent=None)
        f1_scores.append(float(info.get("f1", 0.0)))
        mae_scores.append(float(info.get("mae", 0.0)))
        cer_scores.append(float(info.get("Clinical_Error_Ratio", 0.0)))
        print(
            f"random-flip {episode + 1}/{n_episodes} | "
            f"F1={f1_scores[-1]:.4f} | MAE={mae_scores[-1]:.4f} | CER={cer_scores[-1]:.3f}",
            flush=True,
        )
    return (f1_scores, mae_scores, cer_scores)


def save_random_flip_json(path: str, f1_scores, mae_scores, cer_scores, args):
    import statistics as st

    def _mean(values):
        return float(st.mean(values)) if values else 0.0

    def _std(values):
        return float(st.pstdev(values)) if len(values) > 1 else 0.0

    payload = {
        "method": "random-flip",
        "flip_prob": float(getattr(args, "random_flip_prob", 0.5)),
        "attacker_type": str(getattr(args, "attacker_type", "mixed")),
        "episodes": len(f1_scores),
        "f1_mean": _mean(f1_scores),
        "f1_std": _std(f1_scores),
        "f1_min": float(min(f1_scores)) if f1_scores else 0.0,
        "f1_max": float(max(f1_scores)) if f1_scores else 0.0,
        "f1_per_episode": f1_scores,
        "mae_mean": _mean(mae_scores),
        "mae_std": _std(mae_scores),
        "mae_per_episode": mae_scores,
        "clinical_error_ratio_mean": _mean(cer_scores),
        "clinical_error_ratio_per_episode": cer_scores,
        "seed": int(args.seed),
        "eval_cohort": str(getattr(args, "eval_cohort", "")),
        "bra_fast": bool(getattr(args, "bra_fast", False)),
        "shuffle_queries": bool(getattr(args, "shuffle_queries", False)),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload
