import numpy as np
import torch

BRA_CORE_DIM = 12
IDX_MAF, IDX_ANOMALY, IDX_Y_NOISY = (0, 1, 2)
IDX_SIGNED_MAX_CORR, IDX_COMPROMISE, IDX_FORWARD = (3, 4, 5)
IDX_SHIELD, IDX_PRIOR_ERR, IDX_ROLLING_UTIL, IDX_HUB = (6, 7, 8, 9)
IDX_QUERIES_REMAINING, IDX_BUDGET_UTIL = (10, 11)


class Beacon:

    def __init__(self, args, case, mafs, corr_matrix=None):
        self.args = args
        self.mafs = mafs
        self.beacon_case = case
        self.corr_matrix = corr_matrix if corr_matrix is not None else np.zeros((1, 1))
        self.attacker_actions = torch.tensor([], dtype=torch.long)
        self.responses = torch.tensor([], dtype=torch.float32)
        self.sum_noise_magnitude = 0.0
        self.noise_threshold = getattr(args, "ld_noise_threshold", 0.3486)

    def hub_score(self, snp_idx: int) -> float:
        
        if len(self.attacker_actions) == 0:
            return 0.0
        history_indices = self.attacker_actions.numpy().astype(int)
        return float(np.clip(np.max(np.abs(self.corr_matrix[int(snp_idx), history_indices])), 0.0, 1.0))

    def get_bra_features(self, snp_idx: int, y_noisy: float, rolling_mae_utilization: float):
        snp_idx = int(snp_idx)
        current_maf = float(self.mafs[snp_idx].item())
        f_true = float(self.beacon_case[:, snp_idx].float().mean().item())
        y_noisy = float(y_noisy)
        anomaly_delta = f_true - current_maf
        if len(self.attacker_actions) == 0:
            signed_max_corr = 0.0
            haplotype_compromise_density = 0.0
            forward_leakage_risk = 0.0
            propagated_noise_shield = 0.0
            attacker_prior_error = anomaly_delta
            hub_score = 0.0
        else:
            history_indices = self.attacker_actions.numpy().astype(int)
            ld_correlations = self.corr_matrix[snp_idx, history_indices]
            abs_correlations = np.abs(ld_correlations)
            max_idx = int(np.argmax(abs_correlations))
            signed_max_corr = float(ld_correlations[max_idx])
            hub_score = float(np.clip(abs_correlations[max_idx], 0.0, 1.0))
            thresh_mask = abs_correlations > self.noise_threshold
            compromised_linkage = float(np.sum(abs_correlations[thresh_mask]))
            all_abs_corr = np.abs(self.corr_matrix[snp_idx])
            meaningful_total = all_abs_corr > self.noise_threshold
            total_linkage = float(np.sum(all_abs_corr[meaningful_total]))
            haplotype_compromise_density = compromised_linkage / max(total_linkage, 1e-09)
            history_set = set(history_indices.tolist())
            history_set.add(snp_idx)
            unqueried = [i for i in range(self.corr_matrix.shape[0]) if i not in history_set]
            if unqueried:
                forward_leakage_risk = float(np.max(np.abs(self.corr_matrix[snp_idx, unqueried])))
            else:
                forward_leakage_risk = 0.0
            past_noise = self.responses.numpy() - self.beacon_case[:, history_indices].float().mean(dim=0).numpy()
            if len(history_indices) > 0:
                propagated_noise_shield = float(np.mean(ld_correlations * past_noise))
            else:
                propagated_noise_shield = 0.0
            Sigma_tH = self.corr_matrix[snp_idx, history_indices]
            Sigma_HH = self.corr_matrix[np.ix_(history_indices, history_indices)]
            lambda_reg = 0.05
            Sigma_HH_reg = Sigma_HH + np.eye(len(history_indices)) * lambda_reg
            try:
                optimal_weights = np.linalg.solve(Sigma_HH_reg, Sigma_tH)
            except np.linalg.LinAlgError:
                optimal_weights = np.zeros(len(history_indices))
                optimal_weights[max_idx] = signed_max_corr
            history_mafs = self.mafs[history_indices].numpy().flatten()
            history_responses = self.responses.numpy().flatten()
            history_deltas = history_responses - history_mafs
            multivariate_guess = float(np.clip(current_maf + np.dot(optimal_weights, history_deltas), 0.0, 1.0))
            attacker_prior_error = float(f_true - multivariate_guess)
        return {
            "current_maf": current_maf,
            "anomaly_delta": float(anomaly_delta),
            "y_noisy": y_noisy,
            "signed_max_corr": signed_max_corr,
            "haplotype_compromise_density": float(haplotype_compromise_density),
            "forward_leakage_risk": float(forward_leakage_risk),
            "propagated_noise_shield": float(propagated_noise_shield),
            "attacker_prior_error": float(attacker_prior_error),
            "rolling_mae_utilization": float(rolling_mae_utilization),
            "hub_score": hub_score,
        }

    def build_token(
        self,
        snp_idx: int,
        y_noisy: float,
        rolling_mae_utilization: float,
        snp_embeddings_aux=None,
        *,
        queries_remaining_frac: float = 0.0,
        budget_utilization: float | None = None,
        suspicion_extra: np.ndarray | None = None,
        rarity_extra=None,
        ld_extra=None,
        knowledge_extra=None,
        minimal: bool = False,
    ):
        feats = self.get_bra_features(snp_idx, y_noisy, rolling_mae_utilization)
        if minimal:
            parts = [np.array([feats["current_maf"]], dtype=np.float32)]
            if suspicion_extra is not None:
                parts.append(np.asarray(suspicion_extra, dtype=np.float32).flatten())
            return np.concatenate(parts).astype(np.float32)
        budget_util = float(
            budget_utilization if budget_utilization is not None else feats["rolling_mae_utilization"]
        )
        core = np.array(
            [
                feats["current_maf"],
                feats["anomaly_delta"],
                feats["y_noisy"],
                feats["signed_max_corr"],
                feats["haplotype_compromise_density"],
                feats["forward_leakage_risk"],
                feats["propagated_noise_shield"],
                feats["attacker_prior_error"],
                feats["rolling_mae_utilization"],
                feats["hub_score"],
                float(np.clip(queries_remaining_frac, 0.0, 1.0)),
                float(np.clip(budget_util, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        parts = [core]
        if suspicion_extra is not None:
            parts.append(np.asarray(suspicion_extra, dtype=np.float32).flatten())
        if rarity_extra is not None:
            parts.append(np.asarray(rarity_extra, dtype=np.float32).flatten())
        if ld_extra is not None:
            parts.append(np.asarray(ld_extra, dtype=np.float32).flatten())
        if knowledge_extra is not None:
            parts.append(np.asarray(knowledge_extra, dtype=np.float32).flatten())
        if snp_embeddings_aux is not None and len(snp_embeddings_aux) > 0:
            aux = np.asarray(snp_embeddings_aux[snp_idx], dtype=np.float32).flatten()
            parts.append(aux)
        return np.concatenate(parts).astype(np.float32)

    def update(self, epsilon, attacker_action, y_noisy):
        y_noisy_tensor = torch.as_tensor([y_noisy], dtype=torch.float32)
        attacker_action_tensor = torch.as_tensor([attacker_action], dtype=torch.long)
        self.attacker_actions = torch.cat((self.attacker_actions, attacker_action_tensor))
        self.responses = torch.cat((self.responses, y_noisy_tensor))
        self.sum_noise_magnitude += abs(epsilon)
