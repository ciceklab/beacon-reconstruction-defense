from __future__ import annotations

import numpy as np


def build_binary_cohort(
    full_binary_raw: np.ndarray,
    stratified_indices: np.ndarray,
    *,
    beacon_size: int,
    att_count: int,
    eval_cohort: str,
    strict_50_50: bool,
) -> tuple[np.ndarray, int, int]:
    
    full_binary = full_binary_raw[:, stratified_indices]
    panel_gene_size = full_binary.shape[1]

    if strict_50_50:
        total_individuals = full_binary.shape[0]
        if total_individuals < beacon_size + att_count:
            raise ValueError(
                f"Dataset only has {total_individuals} people; need "
                f"{beacon_size} beacon + {att_count} attacker."
            )
        beacon_pool = full_binary[:beacon_size]
        attacker_panel = full_binary[-att_count:]
        binary = np.vstack((beacon_pool, attacker_panel))
        pool_size_for_c_agent = beacon_size
    else:
        attacker_panel = full_binary[-att_count:]
        cohort_b_holdout = full_binary[-(att_count + beacon_size):-att_count]
        training_pool = full_binary[:-(att_count + beacon_size)]
        if eval_cohort == "B":
            binary = np.vstack((cohort_b_holdout, attacker_panel))
            pool_size_for_c_agent = beacon_size
        elif eval_cohort in ("BL", "BXL"):
            binary = np.vstack((cohort_b_holdout, training_pool, attacker_panel))
            pool_size_for_c_agent = beacon_size
        else:
            binary = np.vstack((training_pool, attacker_panel))
            pool_size_for_c_agent = len(training_pool)

    return binary, pool_size_for_c_agent, panel_gene_size
