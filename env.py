import copy
import random
from collections import deque

import numpy as np
import torch

from attacker import ATTACKER_TYPES, Attacker
from beacon import Beacon
from embeddings import fit_beacon_embeddings, mia_vulnerability_score
from helpers.disease_panels import get_disease_panel
from helpers.snp_manifest import infer_chromosome
from helpers.yes_no_beacon import format_yes_no_stats, yes_no_attack_targets, yes_no_bit_stats, yes_no_release
from helpers.query_suspicion import (
    AGENT_SUSPICION_DIM,
    aggregate_suspicion,
    compute_suspicion_features,
    compute_suspicion_gate,
    panel_ld_threshold,
)
from helpers.suspicion_detector import (
    SuspicionDetector,
    calibrated_latched_gate,
    logit_gate,
    update_persistent_engagement,
)
from ld_knowledge_features import (
    compute_attacker_knowledge,
    ld_history_features,
    ld_panel_anchor_features,
    precompute_panel_ld_anchors,
)
from utility_metrics import evaluate_episode_utility, evaluate_utility_preservation
from helpers.maf_tools import MAFHelper
from helpers.heuristic_actor import gated_heuristic_action, heuristic_action
from Module import evaluate_defense_f1

COHORT_A_DRAW = 64


def cohort_a_draw_count(beacon_size: int, training_pool: int) -> int:
    
    if int(beacon_size) <= COHORT_A_DRAW:
        return COHORT_A_DRAW
    if int(training_pool) < int(beacon_size):
        raise ValueError(
            f"Cohort A has {int(training_pool)} training people, which cannot fill "
            f"a {int(beacon_size)}-person beacon."
        )
    return int(training_pool)


def attack_safe_weights(attack_probability: float) -> tuple[float, float]:
    attack = float(np.clip(attack_probability, 0.0, 1.0))
    return attack, 1.0 - attack


def randomized_allele_frequency(genotypes: np.ndarray, flip_prob: float, uniforms: np.ndarray) -> float:
    bits = np.asarray(genotypes, dtype=np.float64).reshape(-1)
    draws = np.asarray(uniforms, dtype=np.float64).reshape(-1)
    if bits.shape != draws.shape:
        raise ValueError("uniforms must have one draw per allele")
    prob = float(flip_prob)
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"flip_prob must be in [0, 1], got {flip_prob}")
    flipped = np.where(draws < prob, 1.0 - bits, bits)
    if flipped.size == 0:
        return 0.0
    return float(flipped.mean())


class Env:

    def __init__(self, args, maf, binary, corr_matrix_agent, corr_matrix_attacker):
        self.maf = maf[:args.gene_size]
        self.binary = binary[:, :args.gene_size]
        self.args = copy.copy(args)
        self.C_agent = corr_matrix_agent
        self.C_attacker = corr_matrix_attacker
        self.episode = 0
        self.mae_window = getattr(args, "mae_window", 50)
        self.noise_history = deque(maxlen=self.mae_window)
        self.token_history = deque(maxlen=getattr(args, "seq_len", 20))
        self.episode_attacker_type = str(getattr(args, "attacker_type", "optimal"))
        self.episode_honest_panel = None
        beacon_case, mafs = self.get_populations()
        self.maf_helper = MAFHelper(mafs)
        self.attacker = Attacker(args=self.args, maf_helper=self.maf_helper, full_corr_matrix=self.C_attacker)
        self.beacon = Beacon(self.args, beacon_case, mafs, self.C_agent)
        self.query_index = 0
        self.max_steps = min(self.args.max_queries, len(self.attacker.queries))
        self.attacker_agent_actions = []
        self.episode_mafs = []
        self.episode_noises = []
        self.episode_hub_scores = []
        self.episode_suspicion_scores: list[float] = []
        self.episode_suspicion_probs: list[float] = []
        self.episode_suspicion_features: list[np.ndarray] = []
        self.episode_suspicion_peak = 0.0
        self.episode_suspicion_streak = 0
        self.episode_suspicion_latched = False
        self._cached_ld_threshold: float | None = None
        self.episode_budget_penalties = 0
        self.high_util_streak = 0
        self.hub_scores_history: list[float] = []
        self.suspicion_detector = self._load_suspicion_detector()
        self._init_embeddings_and_caps(beacon_case)

    @property
    def rolling_mae_utilization(self) -> float:
        if len(self.noise_history) == 0:
            return 0.0
        return float(np.mean(self.noise_history)) / max(self._effective_mae_limit(), 1e-09)

    def _effective_mae_limit(self) -> float:
        base = float(self.args.mae_limit)
        if not getattr(self.args, "mae_limit_ramp", False):
            return base
        ramp_eps = max(1, int(getattr(self.args, "mae_ramp_episodes", 200)))
        start_frac = float(getattr(self.args, "mae_ramp_start_frac", 0.6))
        progress = min(1.0, self.episode / ramp_eps)
        frac = start_frac + (1.0 - start_frac) * progress
        return base * frac

    def _init_embeddings_and_caps(self, beacon_case):
        beacon_matrix = beacon_case.numpy()
        self.beacon_matrix = beacon_matrix
        n_snps = len(self.maf)
        if getattr(self.args, "skip_terminal_bra", False):
            
            self.snp_embeddings = None
            self.human_embeddings = None
            self.panel_ld_idx = np.zeros((n_snps, 0), dtype=np.int32)
            self.panel_ld_vals = np.zeros((n_snps, 0), dtype=np.float32)
            self.log_maf = np.zeros(n_snps, dtype=np.float32)
            self.maf_percentile_rank = np.zeros(n_snps, dtype=np.float32)
            self.mia_scores = np.zeros(n_snps, dtype=np.float32)
            self.cap_multiplier = 1.0
            return
        use_svd = not getattr(self.args, "no_svd_embeddings", False)
        if use_svd:
            aux_dim = getattr(self.args, "latent_aux_dim", getattr(self.args, "latent_dim", 8))
            self.snp_embeddings, self.human_embeddings, _ = fit_beacon_embeddings(
                beacon_matrix, n_components=aux_dim
            )
        else:
            self.snp_embeddings = None
            self.human_embeddings = None
        c_agent = np.asarray(self.C_agent, dtype=np.float64)
        k_panel = int(getattr(self.args, "ld_panel_k", 4))
        self.panel_ld_idx, self.panel_ld_vals = precompute_panel_ld_anchors(
            c_agent, k=k_panel, noise_threshold=self.beacon.noise_threshold
        )
        maf_array = np.array(self.maf, dtype=np.float64)
        self.log_maf = np.log10(maf_array + 1e-06).astype(np.float32)
        order = np.argsort(maf_array)
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, len(maf_array), dtype=np.float32)
        self.maf_percentile_rank = ranks
        if self.human_embeddings is not None:
            mia = np.array(
                [
                    mia_vulnerability_score(i, beacon_matrix, self.human_embeddings)
                    for i in range(len(maf_array))
                ],
                dtype=np.float32,
            )
            self.mia_scores = mia / max(float(mia.max()), 1e-06)
        else:
            self.mia_scores = np.zeros(len(maf_array), dtype=np.float32)
        top_20_percentile_maf = float(np.percentile(maf_array, 80))
        max_step_burst = self.args.mae_limit * 4.0
        self.cap_multiplier = max_step_burst / max(top_20_percentile_maf, 0.001)

    def _sample_episode_attacker_type(self) -> str:
        if getattr(self.args, "evaluate", False) or getattr(self.args, "baseline_only", False):
            return str(getattr(self.args, "attacker_type", "optimal"))

        forced = getattr(self.args, "attacker_type", None)
        if forced in ATTACKER_TYPES and getattr(self.args, "force_attacker_type", False):
            return forced

        if not getattr(self.args, "query_mixture_training", True):
            return str(getattr(self.args, "attacker_type", "optimal"))

        honest_frac = float(getattr(self.args, "honest_episode_fraction", 0.20))
        mixed_frac = float(getattr(self.args, "mixed_episode_fraction", 0.80))
        draw = random.random()
        if draw < honest_frac:
            return "honest"
        if draw < honest_frac + mixed_frac:
            return "mixed"
        return "optimal"

    def _configure_episode_attacker(self) -> None:
        self.episode_attacker_type = self._sample_episode_attacker_type()
        ep_args = copy.copy(self.args)
        ep_args.attacker_type = self.episode_attacker_type
        if self.episode_attacker_type == "honest":
            chrom = infer_chromosome(getattr(self.args, "pop_prefix", "") or "")
            self.episode_honest_panel = get_disease_panel(
                num_queries=int(self.args.max_queries),
                chromosome=chrom if chrom > 0 else None,
            )
            ep_args.honest_panel = self.episode_honest_panel
        else:
            self.episode_honest_panel = None
            ep_args.honest_panel = None
        self.args = ep_args

    def reset(self, args=None):
        if args is not None:
            self.args = copy.copy(args)
        self.episode += 1
        self._configure_episode_attacker()
        beacon_case, mafs = self.get_populations()
        selector = getattr(self.args, "query_selector", None)
        if selector is not None:
            self.episode_attacker_type = "identifying"
            genomes = beacon_case.detach().cpu().numpy() if torch.is_tensor(beacon_case) else np.asarray(beacon_case)
            chosen = np.asarray(selector(genomes, np.asarray(self.maf, dtype=np.float64)), dtype=int)
            self.args.fixed_query_indices = chosen
        else:
            self.args.fixed_query_indices = None
        self.query_index = 0
        self.noise_history.clear()
        self.token_history.clear()
        self.attacker = Attacker(args=self.args, maf_helper=self.maf_helper, full_corr_matrix=self.C_attacker)
        self.beacon = Beacon(self.args, beacon_case, mafs, self.C_agent)
        self.max_steps = min(self.args.max_queries, len(self.attacker.queries))
        self.attacker_agent_actions = []
        self.episode_mafs = []
        self.episode_noises = []
        self.episode_hub_scores = []
        self.episode_suspicion_scores = []
        self.episode_suspicion_probs = []
        self.episode_suspicion_features = []
        self.episode_suspicion_streak = 0
        self.episode_suspicion_latched = False
        self._cached_ld_threshold = None
        self.episode_suspicion_peak = (
            float("-inf")
            if self.suspicion_detector is not None and self.suspicion_detector.score_scale() == "logit"
            else 0.0
        )
        self.episode_budget_penalties = 0
        self.high_util_streak = 0
        self.hub_scores_history = []
        seq_len = max(getattr(self.args, "seq_len", self.args.max_queries), self.args.max_queries)
        self.token_history = deque(maxlen=seq_len)
        self._init_embeddings_and_caps(beacon_case)

    def build_state_sequence(self):
        seq_len = getattr(self.args, "seq_len", 20)
        token_dim = getattr(self.args, "token_dim", 19)
        tokens = list(self.token_history)
        if len(tokens) >= seq_len:
            seq = np.stack(tokens[-seq_len:], axis=0)
        elif len(tokens) == 0:
            seq = np.zeros((seq_len, token_dim), dtype=np.float32)
        else:
            pad = np.zeros((seq_len - len(tokens), token_dim), dtype=np.float32)
            seq = np.vstack([pad, np.stack(tokens, axis=0)])
        return seq.astype(np.float32)

    def _projected_rolling_mean(self, abs_noise: float) -> float:
        history = list(self.noise_history)
        window = self.noise_history.maxlen
        if len(history) < window:
            new_sum = sum(history) + abs_noise
            return new_sum / (len(history) + 1)
        oldest = history[0]
        new_sum = sum(history) - oldest + abs_noise
        return new_sum / window

    def _max_abs_noise_for_budget(self) -> float:
        history = list(self.noise_history)
        window = self.noise_history.maxlen
        limit = self._effective_mae_limit()
        if len(history) < window:
            allowed_sum = limit * (len(history) + 1) - sum(history)
        else:
            allowed_sum = limit * window - sum(history) + history[0]
        return max(0.0, allowed_sum)

    def _cap_noise_to_rolling_budget(self, abs_epsilon: float) -> float:
        max_allowed = self._max_abs_noise_for_budget()
        return min(abs_epsilon, max_allowed)

    def _load_suspicion_detector(self):
        path = getattr(self.args, "suspicion_detector_path", None)
        if not path:
            return None
        try:
            det = SuspicionDetector.load(path)
            print(
                f" Loaded suspicion detector: {path} | step_AUC={det.auc:.4f} "
                f"episode_AUC={det.episode_auc:.4f} "
                f"threshold={det.threshold:.4f} honest_FPR={det.honest_fpr:.3f} mixed_TPR={det.mixed_tpr:.3f}"
            )
            return det
        except Exception as e:  
            print(f" WARNING: could not load suspicion detector {path!r} ({e}); using fixed-weight fallback.")
            return None

    def _dynamic_ld_threshold(self) -> float:
        if self._cached_ld_threshold is None:
            self._cached_ld_threshold = panel_ld_threshold(np.asarray(self.C_agent, dtype=np.float64))
        return float(self._cached_ld_threshold)

    def _suspicion_feature_pair(
        self,
        queried: np.ndarray,
        hub_score_pre: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        
        shared = dict(
            queried_indices=queried,
            maf_values=np.asarray(self.maf, dtype=np.float64),
            corr_matrix=np.asarray(self.C_agent, dtype=np.float64),
            hub_scores_history=self.hub_scores_history,
            current_hub=hub_score_pre,
        )
        detector_features = compute_suspicion_features(
            **shared,
            ld_noise_threshold=self._dynamic_ld_threshold(),
        )
        agent_features = compute_suspicion_features(
            **shared,
            ld_noise_threshold=self.beacon.noise_threshold,
        )[:AGENT_SUSPICION_DIM]
        return agent_features, detector_features

    def _compute_step_suspicion(self, attacker_action: int, hub_score_pre: float) -> tuple[np.ndarray, np.ndarray, float, float]:
        prior = self.beacon.attacker_actions.numpy().astype(int)
        if prior.size == 0:
            queried = np.array([int(attacker_action)], dtype=int)
        else:
            queried = np.append(prior, int(attacker_action))

        agent_features, detector_features = self._suspicion_feature_pair(queried, hub_score_pre)
        if self.suspicion_detector is not None:
            detector = self.suspicion_detector
            scored_features = detector_features
            if detector.n_features != int(detector_features.shape[-1]):
                if detector.n_features == int(agent_features.shape[-1]):
                    scored_features = agent_features
                else:
                    raise ValueError(
                        f"Suspicion detector expects {detector.n_features} clues, "
                        f"but this checklist computed {detector_features.shape[-1]}."
                    )
            floor = float(getattr(self.args, "suspicion_attack_floor", 0.6))
            if detector.score_scale() == "logit":
                score = float(detector.decision_function(scored_features))
                self.episode_suspicion_streak, self.episode_suspicion_latched = update_persistent_engagement(
                    score,
                    detector.threshold,
                    self.episode_suspicion_streak,
                    self.episode_suspicion_latched,
                    detector.persistence(),
                )
                self.episode_suspicion_peak = max(self.episode_suspicion_peak, score)
                gate = 1.0
                if getattr(self.args, "suspicion_gate", True):
                    gate = (
                        logit_gate(self.episode_suspicion_peak, detector.threshold, floor=floor)
                        if self.episode_suspicion_latched
                        else 0.0
                    )
                self._remember_suspicion_probability(scored_features, score)
                return agent_features, detector_features, score, gate

            score = float(detector.predict_proba(scored_features))
            gate = 1.0
            if getattr(self.args, "suspicion_gate", True):
                self.episode_suspicion_peak, gate = calibrated_latched_gate(
                    score,
                    self.episode_suspicion_peak,
                    detector.threshold,
                    floor=floor,
                )
            else:
                self.episode_suspicion_peak = max(self.episode_suspicion_peak, score)
            self._remember_suspicion_probability(scored_features, score)
            return agent_features, detector_features, score, gate

        score = aggregate_suspicion(agent_features)
        self.episode_suspicion_peak = max(self.episode_suspicion_peak, score)
        gate = 1.0
        if getattr(self.args, "suspicion_gate", True):
            gate = compute_suspicion_gate(
                self.episode_suspicion_peak,
                mode=str(getattr(self.args, "suspicion_gate_mode", "authorization")),
                threshold=float(getattr(self.args, "suspicion_gate_threshold", 0.45)),
                power=float(getattr(self.args, "suspicion_gate_power", 1.25)),
                honest_ceiling=float(getattr(self.args, "suspicion_honest_ceiling", 0.42)),
                attack_floor=float(getattr(self.args, "suspicion_attack_floor", 0.65)),
            )
        self._remember_suspicion_probability(agent_features, score)
        return agent_features, detector_features, score, gate

    def _episode_suspicion_mean(self) -> float:
        if self.episode_suspicion_scores:
            return float(np.mean(self.episode_suspicion_scores))
        return 0.0

    def _remember_suspicion_probability(self, features: np.ndarray, score: float) -> None:
        
        detector = self.suspicion_detector
        if detector is None:
            probability = float(np.clip(score, 0.0, 1.0))
        else:
            probability = float(detector.predict_proba(features))
        self.episode_suspicion_probs.append(probability)

    def _episode_suspicion_probability(self) -> float:
        if self.episode_suspicion_probs:
            return float(np.mean(self.episode_suspicion_probs))
        return 0.0

    def _step_query_stream_only(self, attacker_action: int, f_true: float):
        
        hub_score_pre = self.beacon.hub_score(int(attacker_action))
        _agent_features, detector_features, suspicion_score, suspicion_gate = self._compute_step_suspicion(
            attacker_action, hub_score_pre
        )
        self.episode_suspicion_scores.append(suspicion_score)
        self.episode_suspicion_features.append(np.asarray(detector_features, dtype=np.float32))
        self.noise_history.append(0.0)
        self.episode_mafs.append(float(self.maf[attacker_action]))
        self.episode_noises.append(0.0)
        self.episode_hub_scores.append(hub_score_pre)
        self.hub_scores_history.append(hub_score_pre)
        self.beacon.update(0.0, attacker_action, f_true)
        self.attacker.update(0.0, attacker_action)
        self.query_index += 1
        done = self.query_index >= self.max_steps
        seq_len = getattr(self.args, "seq_len", 20)
        token_dim = getattr(self.args, "token_dim", 19)
        next_state = np.zeros((seq_len, token_dim), dtype=np.float32)
        info = {
            "f1": 0.0,
            "mae": 0.0,
            "privacy_reward": 0.0,
            "utility_reward": 0.0,
            "final_step_reward": 0.0,
            "rolling_mae_utilization": 0.0,
            "episode_budget_penalties": 0,
            "suspicion_score": suspicion_score,
            "suspicion_gate": suspicion_gate,
            "episode_suspicion_mean": self._episode_suspicion_mean() if done else suspicion_score,
            "episode_suspicion_probability": self._episode_suspicion_probability() if done else suspicion_score,
            "episode_suspicion_peak": self.episode_suspicion_peak if done else suspicion_score,
            "episode_attacker_type": self.episode_attacker_type,
            "f1_delta": 0.0,
            "relative_f1_drop": 0.0,
            "episode_mafs": self.episode_mafs if done else [],
            "episode_noises": self.episode_noises if done else [],
            "episode_hub_scores": self.episode_hub_scores if done else [],
            "episode_suspicion_scores": self.episode_suspicion_scores if done else [],
            "episode_suspicion_features": (
                [f.tolist() for f in self.episode_suspicion_features] if done else []
            ),
        }
        return (
            [next_state, np.array([0.0], dtype=np.float32), 0.0, done],
            [0.0, 0],
            done,
            info,
        )

    def step(self, beacon_agent=None):
        f1_score = 0.0
        mae = 0.0
        f1_delta = 0.0
        privacy_reward = 0.0
        utility_reward = 0.0
        step_reward = 0.0
        suspicion_score = 0.0
        suspicion_gate = 1.0
        rel_drop = 0.0

        attacker_action = self.attacker.act(self.query_index)
        self.attacker_agent_actions.append(attacker_action)
        f_true = self.beacon.beacon_case[:, attacker_action].float().mean().item()
        if getattr(self.args, "skip_terminal_bra", False):
            return self._step_query_stream_only(int(attacker_action), float(f_true))
        state_seq = self.build_state_sequence()
        pre_feats = self.beacon.get_bra_features(attacker_action, f_true, self.rolling_mae_utilization)
        hub_score_pre = float(pre_feats["hub_score"])

        agent_features, detector_features, suspicion_score, suspicion_gate = self._compute_step_suspicion(
            attacker_action, hub_score_pre
        )
        self.episode_suspicion_scores.append(suspicion_score)
        self.episode_suspicion_features.append(np.asarray(detector_features, dtype=np.float32))

        current_maf = float(self.maf[attacker_action])
        heuristic_policy = getattr(self.args, "heuristic_policy", None)
        if heuristic_policy:
            
            if not hasattr(self, "_heuristic_rng"):
                self._heuristic_rng = np.random.default_rng(int(getattr(self.args, "seed", 0)))
            raw_action = heuristic_action(
                str(heuristic_policy),
                maf=current_maf,
                maf_percentile=float(self.maf_percentile_rank[int(attacker_action)]),
                rng=self._heuristic_rng,
            )
        elif getattr(self.args, "baseline_gated_heuristic", False):
            
            raw_action = gated_heuristic_action(current_maf, float(np.random.random()))
        elif self.args.beacon_type == "agent":
            evaluate_mode = getattr(self.args, "evaluate", False)
            beacon_action = beacon_agent.select_action(state_seq, evaluate_mode)
            raw_action = float(np.asarray(beacon_action).flatten()[0])
        else:
            raw_action = 0.0

        if getattr(self.args, "hub_only_noise", False):
            hub_min = float(getattr(self.args, "hub_noise_min", 0.25))
            attack_threshold = (
                self.suspicion_detector.threshold
                if self.suspicion_detector is not None
                else float(getattr(self.args, "suspicion_gate_threshold", 0.45))
            )
            if self.episode_suspicion_peak >= attack_threshold:
                raw_action *= max(hub_score_pre, 0.5)
            elif hub_score_pre < hub_min:
                raw_action = 0.0
            else:
                raw_action *= hub_score_pre

        if getattr(self.args, "suspicion_gate", True):
            raw_action *= suspicion_gate

        num_individuals = self.beacon.beacon_case.shape[0]
        if getattr(self.args, "binary_beacon", False):
            released = yes_no_release(raw_action, f_true, current_maf, num_individuals)
            y_noisy = float(released["bit"])
            abs_eps = float(released["distortion"])
            actual_epsilon = y_noisy - float(f_true)
            inefficiency_tax = 0.0
            if (
                released["lie"]
                and getattr(self.args, "enforce_mae_budget", True)
                and getattr(self.args, "hard_mae_budget", True)
                and self._projected_rolling_mean(abs_eps) > self._effective_mae_limit()
            ):
                y_noisy = float(yes_no_release(0.0, f_true, current_maf, num_individuals)["bit"])
                abs_eps = 0.0
                actual_epsilon = y_noisy - float(f_true)
                self.episode_budget_penalties += 1
                inefficiency_tax = getattr(self.args, "budget_inefficiency_tax", 0.15)
        elif getattr(self.args, "reward_only_mae_budget", False):
            effective_mae = self._effective_mae_limit()
            burst = effective_mae * 4.0
            intended_epsilon = float(raw_action) * burst
            min_human_shift = 1.0 / num_individuals
            proportional_cap = current_maf * self.cap_multiplier
            biological_noise_cap = max(proportional_cap, min_human_shift)
            actual_epsilon = float(np.clip(intended_epsilon, -biological_noise_cap, biological_noise_cap))
            raw_y = np.clip(f_true + actual_epsilon, 0.0, 1.0)
            people_count = np.round(raw_y * num_individuals)
            y_noisy = people_count / num_individuals
            actual_epsilon = float(y_noisy - f_true)
            abs_eps = abs(actual_epsilon)
            inefficiency_tax = 0.0
            if self._projected_rolling_mean(abs_eps) > effective_mae:
                self.episode_budget_penalties += 1
                inefficiency_tax = getattr(self.args, "budget_inefficiency_tax", 0.15)
        elif getattr(self.args, "enforce_mae_budget", True):
            effective_mae = self._effective_mae_limit()
            max_step_burst = effective_mae * 4.0
            rolling_cap = self._max_abs_noise_for_budget()
            max_allowable_noise = min(max_step_burst, rolling_cap) if rolling_cap > 0 else 0.0
            intended_epsilon = raw_action * max_allowable_noise
            min_human_shift = 1.0 / num_individuals
            proportional_cap = current_maf * self.cap_multiplier
            biological_noise_cap = max(proportional_cap, min_human_shift)
            if max_allowable_noise <= 0:
                hard_limit = 0.0
            else:
                hard_limit = min(max_allowable_noise, biological_noise_cap)
            actual_epsilon = intended_epsilon
            if abs(actual_epsilon) > hard_limit:
                actual_epsilon = hard_limit * np.sign(intended_epsilon)
            abs_eps = abs(float(actual_epsilon))
            projected_mean = self._projected_rolling_mean(abs_eps)
            inefficiency_tax = 0.0
            if (
                getattr(self.args, "hard_mae_budget", True)
                and (projected_mean > effective_mae)
            ):
                actual_epsilon = 0.0
                y_noisy = f_true
                abs_eps = 0.0
                self.episode_budget_penalties += 1
                inefficiency_tax = getattr(self.args, "budget_inefficiency_tax", 0.15)
            elif hard_limit == 0.0:
                actual_epsilon = 0.0
                y_noisy = f_true
            else:
                raw_y = np.clip(f_true + actual_epsilon, 0.0, 1.0)
                people_count = np.round(raw_y * num_individuals)
                y_noisy = people_count / num_individuals
                actual_epsilon = y_noisy - f_true
                abs_eps = abs(float(actual_epsilon))
                projected_mean = self._projected_rolling_mean(abs_eps)
                if projected_mean > effective_mae:
                    self.episode_budget_penalties += 1
                    inefficiency_tax = getattr(self.args, "budget_inefficiency_tax", 0.15)
                    capped_abs = self._cap_noise_to_rolling_budget(abs_eps)
                    if capped_abs < abs_eps:
                        actual_epsilon = capped_abs * np.sign(actual_epsilon)
                        y_noisy = np.clip(f_true + actual_epsilon, 0.0, 1.0)
                        abs_eps = abs(float(actual_epsilon))
        else:
            
            actual_epsilon = float(np.clip(raw_action, -1.0, 1.0))
            inefficiency_tax = 0.0
            raw_y = np.clip(f_true + actual_epsilon, 0.0, 1.0)
            people_count = np.round(raw_y * num_individuals)
            y_noisy = people_count / num_individuals
            actual_epsilon = float(y_noisy - f_true)
            abs_eps = abs(actual_epsilon)

        if self.args.beacon_type == "random" and not getattr(self.args, "baseline_gated_heuristic", False):
            genotypes = self.beacon.beacon_case[:, int(attacker_action)].detach().float().cpu().numpy()
            y_noisy = randomized_allele_frequency(
                genotypes,
                float(getattr(self.args, "random_flip_prob", 0.5)),
                np.random.random(genotypes.shape[0]),
            )
            actual_epsilon = float(y_noisy - f_true)
            abs_eps = abs(actual_epsilon)
            inefficiency_tax = 0.0

        self.noise_history.append(abs_eps)
        self.episode_mafs.append(float(self.maf[attacker_action]))
        self.episode_noises.append(abs_eps)
        self.episode_hub_scores.append(hub_score_pre)

        total_q = max(int(self.max_steps), 1)
        queries_remaining_frac = float(max(self.max_steps - self.query_index, 0)) / total_q
        budget_util = self.rolling_mae_utilization

        rarity_extra = None
        if getattr(self.args, "rarity_token_features", False):
            rarity_extra = np.array(
                [
                    self.log_maf[attacker_action],
                    self.maf_percentile_rank[attacker_action],
                    self.mia_scores[attacker_action],
                ],
                dtype=np.float32,
            )
        ld_extra = None
        if getattr(self.args, "ld_token_features", False):
            history = self.beacon.attacker_actions.numpy().astype(int)
            k_hist = int(getattr(self.args, "ld_history_k", 4))
            k_panel = int(getattr(self.args, "ld_panel_k", 4))
            hist_block = ld_history_features(self.C_agent, attacker_action, history, k=k_hist)
            panel_block = ld_panel_anchor_features(
                self.panel_ld_idx, self.panel_ld_vals, attacker_action, k=k_panel
            )
            ld_extra = np.concatenate([hist_block, panel_block])
        knowledge_extra = None
        if getattr(self.args, "attacker_knowledge_features", False):
            past = self.beacon.attacker_actions.numpy().astype(int)
            history = np.append(past, attacker_action)
            true_hist = [
                float(self.beacon.beacon_case[:, int(i)].float().mean().item()) for i in history
            ]
            noisy_hist = list(self.beacon.responses.numpy().flatten()) + [float(y_noisy)]
            hub_arr = np.array(self.hub_scores_history + [hub_score_pre], dtype=np.float32)
            knowledge_extra = compute_attacker_knowledge(
                self.C_agent,
                history,
                true_hist,
                noisy_hist,
                queries_remaining_frac,
                self.rolling_mae_utilization,
                hub_scores_history=hub_arr,
                noise_threshold=self.beacon.noise_threshold,
            )
        snp_aux = None
        if self.snp_embeddings is not None:
            aux_dim = getattr(self.args, "latent_aux_dim", 8)
            snp_aux = self.snp_embeddings[:, :aux_dim]

        suspicion_extra = agent_features if getattr(self.args, "suspicion_features", True) else None
        token = self.beacon.build_token(
            attacker_action,
            float(y_noisy),
            self.rolling_mae_utilization,
            snp_embeddings_aux=snp_aux,
            queries_remaining_frac=queries_remaining_frac,
            budget_utilization=budget_util,
            suspicion_extra=suspicion_extra,
            rarity_extra=rarity_extra,
            ld_extra=ld_extra,
            knowledge_extra=knowledge_extra,
            minimal=bool(getattr(self.args, "maf_detector_state", False)),
        )
        self.hub_scores_history.append(hub_score_pre)
        self.token_history.append(token)
        self.beacon.update(actual_epsilon, attacker_action, y_noisy)
        self.attacker.update(actual_epsilon, attacker_action)
        self.query_index += 1
        done = self.query_index >= self.max_steps
        next_state_seq = self.build_state_sequence()

        step_reward = self._shaped_step_reward(
            abs_eps=abs_eps,
            hub_score_pre=hub_score_pre,
            current_maf=current_maf,
            inefficiency_tax=inefficiency_tax,
            suspicion=suspicion_score,
        )

        binary_stats = None
        if done:
            mode_label = self.episode_attacker_type
            if self.episode_honest_panel:
                mode_label = f"{mode_label} ({self.episode_honest_panel['id']})"
            mae = float(np.mean(self.noise_history)) if len(self.noise_history) > 0 else 0.0
            if getattr(self.args, "skip_f1_eval", False):
                
                f1_score = 0.0
            else:
                print(
                    f"--- Running BRA Optimizer for {self.max_steps} queries "
                    f"[{mode_label} | mean suspicion={self._episode_suspicion_mean():.3f}] ---"
                )
                queried_indices = self.attacker.attacker_actions.clone().detach().long()
                noisy_freqs = self.beacon.responses.clone().detach().float()
                if getattr(self.args, "binary_beacon", False):
                    
                    actions = queried_indices.detach().cpu().numpy().astype(int)
                    released = noisy_freqs.detach().cpu().numpy()
                    true_freq = (
                        self.beacon.beacon_case[:, actions].float().mean(dim=0).detach().cpu().numpy()
                    )
                    public = np.asarray(self.maf, dtype=np.float64)[actions]
                    beacon_size = int(self.beacon.beacon_case.shape[0])
                    targets = yes_no_attack_targets(released, public, beacon_size)
                    binary_stats = yes_no_bit_stats(released, true_freq, beacon_size)
                    noisy_freqs = torch.tensor(targets, dtype=torch.float32)
                true_db = self.beacon.beacon_case.T
                is_fast_mode = getattr(self.args, "bra_fast", False)
                f1_score, reconstructed = evaluate_defense_f1(
                    queried_indices=queried_indices,
                    noisy_freqs=noisy_freqs,
                    true_db=true_db,
                    full_corr_matrix=self.C_attacker,
                    device=self.args.device,
                    fast_mode=is_fast_mode,
                    return_reconstruction=True,
                )
                self.last_reconstruction = np.asarray(reconstructed, dtype=np.uint8).T
                self.last_query_indices = queried_indices.detach().cpu().numpy().astype(int)
            target_f1 = self.args.privacy_f1_baseline
            f1_delta = target_f1 - f1_score
            f1_scale = getattr(self.args, "f1_reward_scale", 2000.0)
            utility = evaluate_utility_preservation(self.episode_mafs, self.episode_noises)
            cer_weight = getattr(self.args, "cer_penalty_weight", 500.0)
            cer_penalty = utility["Clinical_Error_Ratio"] * cer_weight

            score_mean = self._episode_suspicion_mean()
            attack_weight, safe_weight = attack_safe_weights(self._episode_suspicion_probability())
            use_unified = getattr(self.args, "unified_suspicion_reward", True)
            if use_unified:
                privacy_term = f1_delta * f1_scale
                utility_term = cer_penalty
                privacy_reward = attack_weight * privacy_term
                utility_reward = -safe_weight * utility_term
                step_reward += privacy_reward + utility_reward
            else:
                privacy_reward = f1_delta * f1_scale
                step_reward += privacy_reward - cer_penalty

            sparse = getattr(self.args, "sparse_reward", False)
            if not sparse and f1_score > target_f1:
                ob_pen = getattr(self.args, "f1_overbaseline_penalty", 0.0)
                excess = f1_score - target_f1
                if use_unified:
                    step_reward -= ob_pen * excess * f1_scale * attack_weight
                else:
                    step_reward -= ob_pen * excess * f1_scale
            if not sparse and getattr(self.args, "enforce_mae_budget", True):
                util_target = getattr(self.args, "util_target", 0.7)
                util_under_w = getattr(self.args, "util_underuse_penalty", 0.0)
                if util_under_w > 0:
                    shortfall = max(0.0, util_target - self.rolling_mae_utilization)
                    step_reward -= util_under_w * shortfall * 50.0 * attack_weight
            if not sparse:
                rarity_bonus_w = getattr(self.args, "rarity_preservation_bonus", 0.0)
                if rarity_bonus_w > 0:
                    step_reward += rarity_bonus_w * utility["Rarity_Preservation_Score"] * safe_weight
            proxy_scale = 0.0 if sparse else getattr(self.args, "proxy_reward_scale", 0.0)
            if proxy_scale > 0:
                from bra_proxy import subset_corr_mismatch

                queried = self.attacker.attacker_actions.clone().detach().cpu().numpy().astype(int)
                noisy = self.beacon.responses.clone().detach().cpu().numpy().astype(np.float64)
                proxy_loss = subset_corr_mismatch(
                    self.beacon.beacon_case,
                    queried,
                    noisy,
                    self.C_attacker,
                    device=str(getattr(self.args, "device", "cpu")),
                )
                hub_scale = hub_score_pre if getattr(self.args, "proxy_hub_only", True) else 1.0
                step_reward += proxy_scale * hub_scale * attack_weight / (1.0 + proxy_loss)
                info_proxy = proxy_loss
            else:
                info_proxy = 0.0

            rel_drop = f1_delta / target_f1 if target_f1 > 0 else 0.0
            if not getattr(self.args, "skip_f1_eval", False):
                print(
                    f"Episode Done [{mode_label}]. F1: {f1_score:.4f} | Target: {target_f1:.4f} | "
                    f"rel_drop: {rel_drop:.2%} | Rolling MAE: {mae:.4f} | "
                    f"Suspicion: {score_mean:.3f} | P(adv): {attack_weight:.3f} | Privacy R: {privacy_reward:.4f} | "
                    f"Utility R: {utility_reward:.4f} | CER: {cer_penalty:.4f} | "
                    f"Rarity preserve: {utility['Rarity_Preservation_Score']:.3f} | "
                    f"MAF-Noise ρ: {utility['Noise_MAF_Spearman_Corr']:.3f} | "
                    f"mean|noise|: {float(np.mean(self.episode_noises)):.4f} | Proxy: {info_proxy:.2f}"
                )
                if binary_stats is not None:
                    print(f"Yes/no beacon | {format_yes_no_stats(binary_stats)}")

        info = {
            "f1": f1_score if done else 0.0,
            "mae": mae if done else 0.0,
            "privacy_reward": privacy_reward if done else 0.0,
            "utility_reward": utility_reward if done else 0.0,
            "final_step_reward": step_reward if done else 0.0,
            "rolling_mae_utilization": self.rolling_mae_utilization,
            "episode_budget_penalties": self.episode_budget_penalties,
            "suspicion_score": suspicion_score,
            "suspicion_gate": suspicion_gate,
            "episode_suspicion_mean": self._episode_suspicion_mean() if done else suspicion_score,
            "episode_suspicion_probability": self._episode_suspicion_probability() if done else suspicion_score,
            "episode_suspicion_peak": self.episode_suspicion_peak if done else suspicion_score,
            "episode_attacker_type": self.episode_attacker_type,
            "f1_delta": float(f1_delta) if done else 0.0,
            "relative_f1_drop": float(rel_drop) if done else 0.0,
            "episode_mafs": self.episode_mafs if done else [],
            "episode_noises": self.episode_noises if done else [],
            "episode_hub_scores": self.episode_hub_scores if done else [],
            "episode_suspicion_scores": self.episode_suspicion_scores if done else [],
            "episode_suspicion_features": (
                [f.tolist() for f in self.episode_suspicion_features] if done else []
            ),
        }
        if binary_stats is not None:
            info.update(binary_stats)
        if done and getattr(self.args, "eval_utility", False):
            utility = evaluate_episode_utility(
                self.episode_mafs,
                self.episode_noises,
                self.episode_hub_scores,
                rare_threshold=float(getattr(self.args, "rare_maf_threshold", 0.05)),
                hub_threshold=float(getattr(self.args, "hub_noise_min", 0.25)),
            )
            info.update(utility)
        return (
            [next_state_seq, np.array([actual_epsilon], dtype=np.float32), step_reward, done],
            [step_reward, 0],
            done,
            info,
        )

    def _shaped_step_reward(
        self,
        abs_eps: float,
        hub_score_pre: float,
        current_maf: float,
        inefficiency_tax: float,
        suspicion: float,
    ) -> float:
        reward = -inefficiency_tax
        scale = max(getattr(self.args, "reward_normalize_scale", 1.0), 1e-06)
        if getattr(self.args, "sparse_reward", False):
            return reward / scale

        suspicion = float(np.clip(suspicion, 0.0, 1.0))
        use_unified = getattr(self.args, "unified_suspicion_reward", True)
        safe_weight = (1.0 - suspicion) if use_unified else 0.0
        attack_weight = suspicion if use_unified else 1.0

        hub_min = float(getattr(self.args, "hub_noise_min", 0.25))
        hub_waste_w = float(getattr(self.args, "hub_waste_weight", 0.0))
        if hub_waste_w > 0 and abs_eps > 0:
            waste = abs_eps * max(0.0, hub_min - hub_score_pre)
            reward -= hub_waste_w * waste * attack_weight

        rarity_w = float(getattr(self.args, "rarity_noise_penalty", 0.0))
        rare_thresh = float(getattr(self.args, "rare_maf_threshold", 0.05))
        if rarity_w > 0 and abs_eps > 0 and (current_maf <= rare_thresh):
            reward -= rarity_w * abs_eps / max(current_maf, 0.01)

        low_susp_w = float(getattr(self.args, "low_suspicion_noise_penalty", 15.0))
        if use_unified and low_susp_w > 0 and abs_eps > 0:
            reward -= low_susp_w * abs_eps * safe_weight / scale

        budget_on = bool(getattr(self.args, "enforce_mae_budget", True))
        util_target = float(getattr(self.args, "util_target", 0.7))
        util_pen = float(getattr(self.args, "util_penalty_scale", 0.0))
        if budget_on and util_pen > 0 and self.rolling_mae_utilization > util_target:
            reward -= util_pen * (self.rolling_mae_utilization - util_target)

        util_under_bonus = float(getattr(self.args, "util_underuse_bonus", 0.0))
        if budget_on and util_under_bonus > 0 and abs_eps > 0 and (hub_score_pre >= hub_min):
            shortfall = max(0.0, util_target * 0.55 - self.rolling_mae_utilization)
            reward += util_under_bonus * abs_eps * shortfall * hub_score_pre * attack_weight

        sustained_thresh = float(getattr(self.args, "util_sustained_thresh", 0.9))
        sustained_pen = float(getattr(self.args, "util_sustained_penalty", 0.0))
        if budget_on and self.rolling_mae_utilization >= sustained_thresh:
            self.high_util_streak += 1
        else:
            self.high_util_streak = 0
        if budget_on:
            reward -= sustained_pen * self.high_util_streak
        return reward / scale

    def get_populations(self):
        beacon_size = self.args.beacon_size
        if self.args.eval_cohort == "A":
            n_rows = int(self.binary.shape[0])
            attacker_rows = int(getattr(self.args, "corr_size", 0) or 0)
            training_pool = n_rows - attacker_rows if 0 < attacker_rows < n_rows else n_rows
            draw_from = cohort_a_draw_count(int(beacon_size), training_pool)
            pool_indices = np.random.choice(draw_from, beacon_size, replace=False)
            self.beacon_row_indices = np.asarray(pool_indices, dtype=int)
            beacon_case = self.binary[pool_indices]
        else:
            self.beacon_row_indices = np.arange(int(beacon_size), dtype=int)
            beacon_case = self.binary[:beacon_size]
        mafs = torch.Tensor(self.maf)
        return (torch.Tensor(beacon_case), mafs)
