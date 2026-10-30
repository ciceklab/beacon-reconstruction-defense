from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from helpers.snp_stratification import STRATA_COMMON_TO_RARE, STRATUM_LABELS, assign_strata

ROLES = ("honest", "mixed")
ROLE_LABELS = {"honest": "Honest researcher", "mixed": "Mixed attacker"}
ROLE_COLORS = {"honest": "#4C78A8", "mixed": "#E45756"}


def collect_role_traces(env, beacon_agent, role: str, n_episodes: int) -> list[dict]:
    
    import copy

    rows: list[dict] = []
    for episode in range(n_episodes):
        role_args = copy.copy(env.args)
        role_args.attacker_type = role
        role_args.force_attacker_type = True
        role_args.evaluate = True
        role_args.skip_f1_eval = True
        env.reset(role_args)
        done = False
        info: dict = {}
        while not done:
            _binfo, _rewards, done, info = env.step(beacon_agent=beacon_agent)
        mafs = list(info.get("episode_mafs") or [])
        noises = list(info.get("episode_noises") or [])
        panel = getattr(env, "episode_honest_panel", None) or {}
        panel_id = str(panel.get("id", ""))
        for step, (maf, noise) in enumerate(zip(mafs, noises)):
            rows.append(
                {
                    "role": role,
                    "episode": episode,
                    "step": step,
                    "maf": float(maf),
                    "abs_noise": float(abs(noise)),
                    "stratum": str(assign_strata(np.array([maf]))[0]),
                    "panel_id": panel_id,
                }
            )
        print(
            f"  {role} episode {episode + 1}/{n_episodes}"
            f"{(' ' + panel_id) if panel_id else ''}"
            f" | queries={len(mafs)}"
            f" | mean|noise|={float(np.mean(noises)) if noises else 0.0:.4f}",
            flush=True,
        )
    return rows


def summarize_traces(rows: list[dict]) -> list[dict]:
    
    summary: list[dict] = []
    for role in ROLES:
        role_rows = [row for row in rows if row["role"] == role]
        n_role_episodes = len({int(row["episode"]) for row in role_rows})
        for stratum in STRATA_COMMON_TO_RARE:
            picked = [row for row in role_rows if row["stratum"] == stratum]
            episode_means = []
            for episode in sorted({int(row["episode"]) for row in picked}):
                values = [row["abs_noise"] for row in picked if int(row["episode"]) == episode]
                if values:
                    episode_means.append(float(np.mean(values)))
            per_episode = (len(picked) / n_role_episodes) if n_role_episodes else 0.0
            summary.append(
                {
                    "role": role,
                    "stratum": stratum,
                    "n_queries": len(picked),
                    "n_episodes": n_role_episodes,
                    "mean_queries_per_episode": per_episode if n_role_episodes else "",
                    "mean_abs_noise": float(np.mean(episode_means)) if episode_means else "",
                    "std_episode_mean": float(np.std(episode_means)) if len(episode_means) > 1 else "",
                }
            )
    return summary


def write_trace_tables(rows: list[dict], summary: list[dict], out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{stem}_queries.csv"
    summary_path = out_dir / f"{stem}_summary.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["role", "episode", "step", "maf", "abs_noise", "stratum", "panel_id"],
        )
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "role",
                "stratum",
                "n_queries",
                "n_episodes",
                "mean_queries_per_episode",
                "mean_abs_noise",
                "std_episode_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)
    return raw_path, summary_path


def load_summary_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _queries_per_episode(row: dict) -> float:
    
    raw = row.get("mean_queries_per_episode", "")
    if raw not in ("", None):
        return float(raw)
    n_queries = float(row.get("n_queries") or 0)
    n_episodes = float(row.get("n_episodes") or 0)
    if n_episodes > 0:
        return n_queries / n_episodes
    return n_queries


def _format_count(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}"


def plot_strata_noise(summary: list[dict], out_path: Path, *, title: str) -> None:
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lookup = {(row["role"], row["stratum"]): row for row in summary}
    x = np.arange(len(STRATA_COMMON_TO_RARE))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ymax = 0.0
    for offset, role in ((-width / 2, "honest"), (width / 2, "mixed")):
        heights = []
        errors = []
        counts = []
        for stratum in STRATA_COMMON_TO_RARE:
            row = lookup.get((role, stratum), {})
            raw = row.get("mean_abs_noise", "")
            height = float(raw) if raw not in ("", None) else 0.0
            per_episode = _queries_per_episode(row)
            if per_episode == 0:
                height = 0.0
            err_raw = row.get("std_episode_mean", "")
            error = float(err_raw) if err_raw not in ("", None) else 0.0
            heights.append(height)
            errors.append(error if per_episode else 0.0)
            counts.append(per_episode)
            ymax = max(ymax, height + error)
        bars = ax.bar(
            x + offset,
            heights,
            width=width,
            color=ROLE_COLORS[role],
            edgecolor="white",
            linewidth=0.6,
            label=ROLE_LABELS[role],
            zorder=3,
            yerr=errors,
            capsize=3,
            error_kw={"elinewidth": 0.8, "ecolor": "#333333", "capthick": 0.8},
        )
        for bar, count, height, error in zip(bars, counts, heights, errors):
            label = f"n={_format_count(count)}"
            if count == 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.0,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#666666",
                )
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + error,
                label,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([STRATUM_LABELS[name] for name in STRATA_COMMON_TO_RARE], fontsize=8.5)
    ax.set_ylabel("Mean absolute frequency noise")
    ax.set_title(title, fontsize=12, fontweight="semibold")
    ax.set_ylim(0, max(ymax * 1.22, 0.02))
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "Error bars: standard deviation across episodes. n is the mean query count in that stratum per episode.",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def collect_and_plot(args, env) -> None:
    
    from td import TD3

    checkpoint = getattr(args, "resume_beacon", None)
    if not checkpoint:
        raise SystemExit("--trace-strata-noise requires --resume-beacon")
    n_episodes = int(getattr(args, "strata_episodes", None) or args.episodes)
    out_dir = Path(getattr(args, "strata_out_dir", None) or "figures/checklist-v3")
    stem = str(getattr(args, "strata_stem", None) or f"{args.pop_prefix}_maf_strata_noise")
    k_epochs = args.td3_updates_per_episode or args.max_queries
    run_device = str(getattr(args, "device", "cuda:4"))
    print(f"Strata trace actor device: {run_device}")
    beacon_agent = TD3(
        seq_len=args.seq_len,
        token_dim=args.token_dim,
        action_dim=1,
        max_action=1.0,
        gamma=0.99,
        K_epochs=k_epochs,
        reward_clip=args.reward_clip,
        policy_delay=2,
        critic_loss_type=args.critic_loss,
        huber_beta=args.huber_beta,
        explore_noise_min=args.explore_noise_min,
        inference_only=True,
        device_override=run_device,
    )
    print(f"Loading frozen defender from {checkpoint}")
    beacon_agent.load(checkpoint)
    rows: list[dict] = []
    for role in ROLES:
        print(f"\nTracing {role} queries ({n_episodes} episodes, terminal BRA skipped)")
        rows.extend(collect_role_traces(env, beacon_agent, role, n_episodes))
    summary = summarize_traces(rows)
    raw_path, summary_path = write_trace_tables(rows, summary, out_dir, stem)
    figure_path = out_dir / f"{stem}.png"
    plot_strata_noise(summary, figure_path, title=figure_title(str(args.pop_prefix)))
    print(f"\nSaved {summary_path}")
    print(f"Saved {raw_path}")
    print(f"Saved {figure_path}")
    print(f"Saved {figure_path.with_suffix('.pdf')}")
    _print_summary(summary)


def plot_from_summary(summary_csv: Path, out_path: Path, title: str) -> None:
    summary = load_summary_csv(summary_csv)
    plot_strata_noise(summary, out_path, title=title)


def figure_title(prefix: str) -> str:
    population, _, chrom = prefix.partition("_")
    chrom_label = chrom.replace("chr", "chromosome ") if chrom else "panel"
    return (
        f"{population.upper()} {chrom_label} — injected noise by MAF stratum\n"
        "checklist v3 defender, held-out cohort B"
    )


def _print_summary(summary: list[dict]) -> None:
    print(f"\n{'stratum':16s} {'role':8s} {'n/ep':>6s} {'mean|noise|':>12s} {'sd':>8s}")
    for row in summary:
        mean = row["mean_abs_noise"]
        std = row["std_episode_mean"]
        mean_txt = f"{float(mean):.4f}" if mean != "" else "—"
        std_txt = f"{float(std):.4f}" if std != "" else "—"
        print(
            f"{row['stratum']:16s} {row['role']:8s} "
            f"{_format_count(_queries_per_episode(row)):>6s} {mean_txt:>12s} {std_txt:>8s}"
        )
