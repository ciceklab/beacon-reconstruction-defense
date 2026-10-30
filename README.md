# Defending Against Genome Reconstruction Attacks in Genomic Data Sharing Beacons

TD3 defense for GA4GH-style beacons that release allele frequencies. A 7-feature logistic detector gates a continuous-action policy, which adds frequency noise under a hard rolling MAE budget. The attacker is the alternating-optimization genome reconstruction attack of Saleem, Çiçek, and Sav (Bioinformatics, 2025).

Panel genotypes are not in this repository. Place each cohort under `data/{EUR,AFR,AMR}` as `{prefix}_cleaned_mafs.npy` and `{prefix}_cleaned_binary.npy`.

## Installation

```bash
pip install -r requirements.txt
```


## Data Preparation

- **Used Dataset:** [Download Dataset (Google Drive Link)](<https://drive.google.com/drive/folders/1a15Gp_MkJrfH4rXyzfU1S8dwubyx1M_O?usp=sharing>)
- **Full Dataset:** [International Genome Sample Resource (IGSR)](<https://www.internationalgenome.org/data/>)

Note: The provided Google Drive link contains the dataset used in our experiments. To proceed with the full dataset, you can use our data preprocessing pipeline, provided in the following scripts:

- `build_snp_manifests.py`
- `build_all_manifests.py`

These scripts automate the dataset preprocessing and manifest generation steps required to prepare the full IGSR dataset for subsequent analysis and experimentation.

## What reproduces the paper

| Paper result | Script |
|---|---|
| SNP manifests and disease-locus checks | `scripts/build_all_manifests.py`, `scripts/validate_disease_panels.py` |
| Supplementary Table 3 sample IDs | `scripts/list_cohort_samples.py` (tables in `reports/cohort_samples/`) |
| Suspicion detector (200 calibration episodes) | `scripts/calibrate_panel_detectors.py` |
| Supplementary Figure 1, cohort MAE caps | `scripts/probe_mae_knee.py` |
| Figure 2, nine panels | `scripts/train_panel_defenders.py` |
| Figure 3 and Supplementary Table 2, EUR chr6 stress tests | `scripts/run_eur_chr6_robustness.py` |
| Figure 3 A/D, query budgets | `scripts/run_eur_chr6_query_budget.py` |
| Figure 3 B/E, gated heuristics and camoflauged attacker | `scripts/run_eur_chr6_gatedh_camo_tests.py` |
| Figure 3 C/F, beacon size including EUR N=75 with a 40-person reference | `scripts/run_eur_chr6_beacon_size.py` |
| Figure 4 A noise by MAF stratum | `scripts/plot_strata_noise.py` |
| Figure 4 B ablations | `scripts/run_eur_chr6_ablations.py` |
| Supplementary Figure 4, yes/no beacon | `scripts/eval_binary_beacon.py` |
| Supplementary Table 4, random guess attacker | `scripts/eval_random_guess_f1.py` |
| Supplementary Figures 2 and 5, Ayöz et al. temporal attack | `scripts/eval_ayoz_reconstruction.py`, `scripts/eval_ayoz_afr_chr6.py` |
| African supplement (Supplementary Figures 5–10) | `scripts/run_afr_chr6_robustness.py`, `scripts/run_afr_chr6_gatedh_camo_tests.py`, `scripts/run_afr_chr6_ablations.py`, `scripts/run_afr_chr6_beacon_size.py`, `scripts/run_afr_chr6_query_budget.py` |
| LD Heatmaps (Supplementary Figures 11–13) | `scripts/eval_ld_shattering.py` |



EUR and AFR launchers are separate because the published checkpoints, MAE caps (0.125926 on EUR chr6, 0.25 on AFR chr6), and cohort sizes differ. The random-answer baseline (`p = 0.5`) is `main.py --random-flip`.

## Toy Example run

[`notebooks/eur_chr6_toy.ipynb`](notebooks/eur_chr6_toy.ipynb) builds a synthetic HLA panel in the `eur_chr6` file layout and runs the chromosome 6 recipe at laptop scale: a short suspicion-detector calibration, then TD3 with MAE cap 0.125926 on cohort A. It checks the training loop. The paper figures come from the real cohort and the full episode budget.


## Single training run

```bash
python main.py \
  --data data/EUR \
  --pop_prefix eur_chr6 \
  --mae-limit 0.125926 \
  --max_queries 50 \
  --hard-mae-budget \
  --eval_cohort B \
  --attacker_type mixed \
  --eval_utility
```

## Layout

```
main.py                         Training and evaluation entry point
notebooks/eur_chr6_toy.ipynb    Synthetic EUR chr6 training for reviewers
env.py                          Beacon MDP, MAE cap, suspicion gate
engine.py                       TD3 training loop
td.py                           Actor-critic with a transformer encoder
attacker.py                     Mixed, optimal, honest, and camouflaged query policies
Module.py                       Reconstruction attack and allele-level F1
beacon.py                       Per-query token features
ld_knowledge_features.py        Linkage and attacker-knowledge features
utility_metrics.py              MAE, clinical error on rare variants, and utility logs
baseline_probe.py               Unguarded and random-answer baselines
bra_proxy.py                    Training-time reconstruction proxy
helpers/                        Detector, disease panels, heuristic actor, yes/no beacon
scripts/                        Paper experiment launchers
tests/                          Unit tests for those launchers and helpers
reports/cohort_samples/         1000 Genomes sample IDs used in Supplementary Table 3
```