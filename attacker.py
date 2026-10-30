import numpy as np
import torch

from helpers.camouflage import histogram_matched_queries, maf_profile
from helpers.disease_panels import get_disease_panel, list_diseases_for_chromosome, resolve_honest_queries
from helpers.maf_tools import MAFHelper
from helpers.snp_manifest import infer_chromosome

ATTACKER_TYPES = ("optimal", "mixed", "honest", "camouflaged", "random")


class Attacker:

    def __init__(self, args, maf_helper: MAFHelper, full_corr_matrix=None):
        self.args = args
        self.mafs = maf_helper.get_mafs()
        self.full_corr_matrix = full_corr_matrix
        self.attacker_actions = torch.tensor([], dtype=torch.long)
        self.attacker_type = str(getattr(args, "attacker_type", "optimal"))
        if self.attacker_type not in ATTACKER_TYPES:
            raise ValueError(
                f"Unknown attacker_type {self.attacker_type!r}; use one of {ATTACKER_TYPES}."
            )
        self.honest_panel: dict | None = None
        self.query_gene_core_mask: np.ndarray | None = None
        self.queries = self._build_queries()

    def _build_queries(self) -> torch.Tensor:
        fixed = getattr(self.args, "fixed_query_indices", None)
        if fixed is not None:
            selected = torch.as_tensor(fixed, dtype=torch.long).reshape(-1)
            print(
                f"--- Identifying SNPs: {int(selected.numel())} rarest carried alleles "
                "of the beacon members ---"
            )
            if getattr(self.args, "shuffle_queries", False):
                print("--- SHUFFLE TEST ACTIVE: Randomizing identifying query timeline ---")
                selected = selected[torch.randperm(selected.numel())]
            return selected
        if self.attacker_type == "optimal":
            return self._init_optimal_queries()
        if self.attacker_type == "mixed":
            return self._init_mixed_queries()
        if self.attacker_type == "camouflaged":
            return self._init_camouflaged_queries()
        if self.attacker_type == "random":
            return self._init_random_queries()
        return self._init_honest_queries()

    def act(self, current_step):
        return self.queries[current_step].item()

    def update(self, epsilon, attacker_action):
        attacker_action_tensor = torch.as_tensor([attacker_action], dtype=torch.long)
        self.attacker_actions = torch.cat((self.attacker_actions, attacker_action_tensor))

    def _init_random_queries(self):
        
        n_panel = int(self.mafs.numel())
        n_draw = int(self.args.max_queries)
        if n_draw > n_panel:
            raise ValueError(
                f"Random attacker needs {n_draw} queries but the panel has {n_panel} SNPs."
            )
        seed = int(getattr(self.args, "seed", 0)) + int(getattr(self.args, "episode", 0))
        chosen = np.random.default_rng(seed).choice(n_panel, size=n_draw, replace=False)
        selected = torch.tensor(chosen, dtype=torch.long)
        print(f"--- Random attacker: {n_draw} SNPs drawn uniformly from the {n_panel}-SNP panel ---")
        if getattr(self.args, "shuffle_queries", False):
            print("--- SHUFFLE TEST ACTIVE: Randomizing random query timeline ---")
            selected = selected[torch.randperm(selected.size(0))]
        return selected

    def _init_optimal_queries(self):
        sorted_indices = torch.argsort(self.mafs, descending=True)
        selected_queries = sorted_indices[: self.args.max_queries]
        if getattr(self.args, "shuffle_queries", False):
            print("--- SHUFFLE TEST ACTIVE: Randomizing optimal query timeline ---")
            shuffle_idx = torch.randperm(selected_queries.size(0))
            selected_queries = selected_queries[shuffle_idx]
        return selected_queries

    def _init_mixed_queries(self):
        sorted_indices_desc = torch.argsort(self.mafs, descending=True)
        sorted_indices_asc = torch.argsort(self.mafs, descending=False)
        num_high_maf = int(self.args.max_queries * 0.8)
        num_clinical = self.args.max_queries - num_high_maf
        high_maf_queries = sorted_indices_desc[:num_high_maf]
        clinical_queries = sorted_indices_asc[:num_clinical]
        selected_queries = torch.cat((high_maf_queries, clinical_queries))
        if getattr(self.args, "shuffle_queries", False):
            print("--- SHUFFLE TEST ACTIVE: Randomizing mixed query timeline ---")
            shuffle_idx = torch.randperm(selected_queries.size(0))
            selected_queries = selected_queries[shuffle_idx]
        return selected_queries

    def _init_camouflaged_queries(self):
        
        panel_mafs = np.asarray(self.mafs.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        targets = honest_reference_mafs(self.args, self.mafs)
        n_bins = int(getattr(self.args, "camouflage_bins", 10))
        ld_greedy = bool(getattr(self.args, "camouflage_ld_greedy", True))
        n_panel = panel_mafs.size
        corr = self.full_corr_matrix
        if corr is None:
            corr = np.eye(n_panel, dtype=np.float64)
        seed = int(getattr(self.args, "seed", 0)) + int(getattr(self.args, "episode", 0)) + 7919
        rng = np.random.default_rng(seed)
        chosen = histogram_matched_queries(
            targets,
            panel_mafs,
            np.asarray(corr, dtype=np.float64),
            n_draw=int(self.args.max_queries),
            rng=rng,
            n_bins=n_bins,
            ld_greedy=ld_greedy,
        )
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        target_hist = maf_profile(targets, edges=edges)
        chosen_hist = maf_profile(panel_mafs[chosen], edges=edges)
        mode = "LD-greedy" if ld_greedy else "MAF-only"
        print(
            f"--- Camouflaged attacker ({mode}, {n_bins} MAF bins) | "
            f"honest histogram {np.round(target_hist, 2).tolist()} | "
            f"query histogram {np.round(chosen_hist, 2).tolist()} ---"
        )
        selected = torch.tensor(chosen, dtype=torch.long)
        if getattr(self.args, "shuffle_queries", False):
            print("--- SHUFFLE TEST ACTIVE: Randomizing camouflaged query timeline ---")
            shuffle_idx = torch.randperm(selected.size(0))
            selected = selected[shuffle_idx]
        return selected

    def _init_honest_queries(self):
        preset = getattr(self.args, "honest_panel", None)
        disease_id = preset["id"] if preset else getattr(self.args, "episode_disease_id", None)
        seed = int(getattr(self.args, "seed", 0)) + int(getattr(self.args, "episode", 0))
        rng = np.random.default_rng(seed)

        selected_queries, panel = resolve_honest_queries(
            self.mafs,
            data_dir=getattr(self.args, "data", None),
            pop_prefix=getattr(self.args, "pop_prefix", None),
            disease_id=disease_id,
            max_queries=int(self.args.max_queries),
            pool_global_indices=getattr(self.args, "pool_global_indices", None),
            rng=rng,
            gene_window_bp=int(getattr(self.args, "disease_gene_window_bp", 200_000)),
        )
        self.honest_panel = panel
        mask = panel.get("gene_core_mask")
        self.query_gene_core_mask = np.asarray(mask, dtype=bool) if mask else None

        if selected_queries.size(0) != self.args.max_queries:
            raise ValueError(
                f"Honest panel {panel['id']} resolved {selected_queries.size(0)} queries, "
                f"expected {self.args.max_queries}."
            )

        core_hits = int(self.query_gene_core_mask.sum()) if self.query_gene_core_mask is not None else 0
        resolved = panel.get("resolved_loci") or []
        loci_summary = ", ".join(
            f"{x['rsid']}@{x.get('matched_position', x['position'])}" for x in resolved[:4]
        )
        print(
            f"--- Honest researcher panel: {panel['name']} ({panel['id']}) | "
            f"{core_hits} disease-core SNPs | loci: {loci_summary or 'filler only'} ---"
        )

        if getattr(self.args, "shuffle_queries", False):
            print(f"--- SHUFFLE TEST ACTIVE: Randomizing honest panel ({panel['id']}) timeline ---")
            shuffle_idx = torch.randperm(selected_queries.size(0))
            selected_queries = selected_queries[shuffle_idx]
            if self.query_gene_core_mask is not None:
                self.query_gene_core_mask = self.query_gene_core_mask[shuffle_idx.numpy()]
        return selected_queries


def honest_reference_mafs(args, mafs: torch.Tensor) -> np.ndarray:
    
    values = np.asarray(mafs.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    chrom = infer_chromosome(getattr(args, "pop_prefix", "") or "")
    diseases = list_diseases_for_chromosome(chrom) if chrom > 0 else []
    if not diseases:
        queries, _panel = resolve_honest_queries(
            mafs,
            data_dir=getattr(args, "data", None),
            pop_prefix=getattr(args, "pop_prefix", None),
            disease_id=None,
            max_queries=int(args.max_queries),
            pool_global_indices=getattr(args, "pool_global_indices", None),
            rng=np.random.default_rng(int(getattr(args, "seed", 0))),
            gene_window_bp=int(getattr(args, "disease_gene_window_bp", 200_000)),
        )
        return values[queries.numpy()]
    pooled: list[np.ndarray] = []
    for offset, disease in enumerate(diseases):
        rng = np.random.default_rng(int(getattr(args, "seed", 0)) + offset)
        panel = get_disease_panel(disease.disease_id, num_queries=int(args.max_queries))
        queries, _panel = resolve_honest_queries(
            mafs,
            data_dir=getattr(args, "data", None),
            pop_prefix=getattr(args, "pop_prefix", None),
            disease_id=panel["id"],
            max_queries=int(args.max_queries),
            pool_global_indices=getattr(args, "pool_global_indices", None),
            rng=rng,
            gene_window_bp=int(getattr(args, "disease_gene_window_bp", 200_000)),
        )
        pooled.append(values[queries.numpy()])
    return np.concatenate(pooled)
