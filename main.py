from attacker import Attacker
from helpers.cohort_data import build_binary_cohort
from helpers.query_suspicion import AGENT_SUSPICION_DIM
from helpers.maf_tools import MAFHelper
from helpers.snp_stratification import select_stratified_indices
import numpy as np
import pickle
import argparse
import json
import os
import random
import torch
import wandb
from device_util import announce_device, configure_cpu_threads, get_torch_device
from Module import ReconstructionTool, evaluate_defense_f1
from env import Env
from td import TD3
from engine import train_TD_beacon

def reproducibility(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
configure_cpu_threads()

def args_create():
    parser = argparse.ArgumentParser(description='BRA Defense TD3 Agent')
    parser.add_argument('--cuda-device', default=None, type=int, help='CUDA device index (default: 0)')
    parser.add_argument('--data', default='./data/EUR', type=str, help='Dataset Path')
    parser.add_argument('--gene_size', default=50, type=int, help='States gene size')
    parser.add_argument('--beacon_size', default=50, type=int, help='Beacon population size')
    parser.add_argument('--pop_prefix', default=None, type=str, help='Population prefix (e.g., pur, mxl, yri)')
    parser.add_argument('--strict_50_50', action='store_true', help='Use strict 50/50 split for diverse OOD populations')
    parser.add_argument('--episodes', default=1000, type=int, help='Number of episodes for training agent.')
    parser.add_argument('--max_queries', default=50, type=int, help='Maximum queries per episode')
    parser.add_argument('--seq-len', default=None, type=int, help='Token history length (default: max_queries)')
    parser.add_argument('--token-dim', default=18, type=int, help='Per-token dim (10 BRA core + latent_aux_dim)')
    parser.add_argument('--latent-dim', default=16, type=int, help='Deprecated alias; use --latent-aux-dim')
    parser.add_argument('--latent-aux-dim', default=8, type=int, help='Supplemental SVD dimensions in each token')
    parser.add_argument('--rarity-token-features', action='store_true', help='Append log MAF, MAF percentile rank, MIA score to each token')
    parser.add_argument('--rare-maf-threshold', default=0.05, type=float, help='MAF below this is treated as rare for penalties/metrics')
    parser.add_argument('--rarity-noise-penalty', default=0.0, type=float, help='Per-step penalty scale for noise on rare SNPs')
    parser.add_argument('--rarity-preservation-bonus', default=0.0, type=float, help='Terminal bonus scale for Rarity_Preservation_Score')
    parser.add_argument('--util-underuse-penalty', default=2.0, type=float, help='Terminal penalty when rolling MAE util below util-target on attack-like episodes')
    parser.add_argument('--util-underuse-bonus', default=0.0, type=float, help='Per-step bonus encouraging budget use when util is very low')
    parser.add_argument('--mae-window', default=50, type=int, help='Rolling window for MAE budget deque')
    parser.add_argument('--cer-penalty-weight', default=500.0, type=float, help='Terminal reward penalty scale for clinical error ratio')
    parser.add_argument('--f1-reward-scale', default=100.0, type=float, help='Terminal privacy reward multiplier on (baseline_f1 - attacker_f1)')
    parser.add_argument('--budget-inefficiency-tax', default=0.05, type=float, help='Per-step penalty when rolling MAE cap clips noise (budget clip penalty)')
    parser.add_argument('--hub-waste-weight', default=2.0, type=float, help='Per-step penalty scale for noise on low-LD SNPs')
    parser.add_argument('--proxy-reward-scale', default=5.0, type=float, help='Per-step reward for reducing BRA correlation proxy')
    parser.add_argument('--td3-updates-per-episode', default=None, type=int, help='TD3 gradient steps per episode (default: max_queries)')
    parser.add_argument('--reward-clip', default=50.0, type=float, help='Clamp TD3 Q targets to [-clip, clip]')
    parser.add_argument('--buffer-warmup', default=500, type=int, help='Min replay transitions before TD3 update')
    parser.add_argument('--state_dim', default=9, type=int, help='Deprecated; use --seq-len and --token-dim')
    parser.add_argument('--mae-limit', default=0.05, type=float, help='Max rolling mean |noise| per query')
    parser.add_argument('--privacy-f1-baseline', default=0.7, type=float, help='Reconstruction F1 above which privacy cliff applies')
    parser.add_argument('--no-enforce-mae-budget', action='store_true', help='No rolling MAE cap: actor noise is only limited by the suspicion gate and the [0, 1] beacon answer')
    parser.add_argument('--update_freq', default=5, type=int, help='Train Agent model frequency')
    parser.add_argument(
        '--attacker_type',
        default='optimal',
        choices=['optimal', 'mixed', 'honest', 'camouflaged', 'random'],
        type=str,
        help='Attacker for eval/baseline; training samples a mixture unless --force-attacker-type',
    )
    parser.add_argument(
        '--camouflage-bins',
        default=10,
        type=int,
        help='Equal-width MAF bins the camouflaged attacker copies from honest disease panels',
    )
    parser.add_argument(
        '--no-camouflage-ld',
        action='store_false',
        dest='camouflage_ld_greedy',
        help='Match the honest MAF histogram without preferring high-LD variants inside each bin',
    )
    parser.add_argument(
        '--force-attacker-type',
        action='store_true',
        help='Use --attacker-type for every training episode (disable query mixture)',
    )
    parser.add_argument(
        '--query-mixture-training',
        action='store_true',
        default=True,
        help='Sample optimal/mixed/honest episodes during training',
    )
    parser.add_argument(
        '--no-query-mixture-training',
        action='store_false',
        dest='query_mixture_training',
    )
    parser.add_argument(
        '--honest-episode-fraction',
        default=0.20,
        type=float,
        help='Fraction of training episodes with honest disease-panel queries',
    )
    parser.add_argument(
        '--mixed-episode-fraction',
        default=0.80,
        type=float,
        help='Fraction of training episodes with mixed adversarial queries',
    )
    parser.add_argument(
        '--suspicion-features',
        action='store_true',
        default=True,
        help='Append online suspicion features to each state token',
    )
    parser.add_argument(
        '--no-suspicion-features',
        action='store_false',
        dest='suspicion_features',
    )
    parser.add_argument(
        '--suspicion-gate',
        action='store_true',
        default=True,
        help='Multiply actor noise by online suspicion gate (default-on)',
    )
    parser.add_argument(
        '--no-suspicion-gate',
        action='store_false',
        dest='suspicion_gate',
    )
    parser.add_argument(
        '--suspicion-gate-mode',
        default='authorization',
        choices=['authorization', 'soft', 'hard'],
        help='authorization: strong noise once attack-like; soft: weak ramp; hard: threshold on/off',
    )
    parser.add_argument(
        '--suspicion-gate-threshold',
        default=0.45,
        type=float,
        help='Gate threshold (authorization/soft ramp center / hard cutoff)',
    )
    parser.add_argument(
        '--suspicion-gate-power',
        default=1.25,
        type=float,
        help='Exponent for soft suspicion gate ramp (ignored in authorization mode)',
    )
    parser.add_argument(
        '--suspicion-honest-ceiling',
        default=0.42,
        type=float,
        help='Suspicion scores at or below this get zero noise authorization',
    )
    parser.add_argument(
        '--suspicion-attack-floor',
        default=0.65,
        type=float,
        help='Minimum noise gate once suspicion exceeds gate threshold (authorization mode)',
    )
    parser.add_argument(
        '--unified-suspicion-reward',
        action='store_true',
        default=True,
        help='Blend privacy vs utility terminal reward by episode suspicion mean',
    )
    parser.add_argument(
        '--no-unified-suspicion-reward',
        action='store_false',
        dest='unified_suspicion_reward',
    )
    parser.add_argument(
        '--low-suspicion-noise-penalty',
        default=15.0,
        type=float,
        help='Per-step penalty for noise when suspicion is low',
    )
    parser.add_argument(
        '--disease-gene-window-bp',
        default=200_000,
        type=int,
        help='Max distance (bp) to map catalog SNP to nearest panel variant',
    )
    parser.add_argument('--shuffle-queries', action='store_true', default=False, help='Randomizes the order of the optimal attacker queries')
    parser.add_argument('--eval_utility', action='store_true', help='Run the utility preservation analysis to see which SNPs are modified')
    parser.add_argument('--beacon_type', default='agent', type=str)
    parser.add_argument(
        '--random-flip',
        action='store_true',
        help='Replace every beacon answer with randomized allele flips and measure attacker F1, then exit',
    )
    parser.add_argument(
        '--random-flip-prob',
        default=0.5,
        type=float,
        help='Per-allele flip probability for --random-flip. 0.5 makes the answer independent of the cohort',
    )
    parser.add_argument(
        '--random-flip-episodes',
        default=30,
        type=int,
        help='BRA episodes for --random-flip',
    )
    parser.add_argument(
        '--binary-beacon',
        action='store_true',
        help=(
            'Yes/no beacon. A gated action with magnitude above 0.5 flips the presence bit; '
            'a quiet action reports the truth. The attacker sees 0 on a no and the public MAF '
            'on a yes. The rolling MAE cap limits the size of those lies.'
        ),
    )
    parser.add_argument('--eval_cohort', default='A', choices=['A', 'B', 'BL', 'BXL'], type=str, help='Evaluation cohort selection')
    parser.add_argument('--seed', default=3, type=int, help='Seed for reproducibility')
    parser.add_argument('--user_risk', default=0.2, type=float, help='Risk Level for End User')
    parser.add_argument('--evaluate', action='store_true', help='Evaluation mode (freezes weights)')
    parser.add_argument(
        '--trace-strata-noise',
        action='store_true',
        help='Replay a frozen defender and plot mean |noise| in each of the five MAF strata',
    )
    parser.add_argument('--strata-episodes', default=None, type=int, help='Episodes per role for --trace-strata-noise')
    parser.add_argument('--strata-out-dir', default='figures/checklist-v3', type=str)
    parser.add_argument('--resume-beacon', default=None, type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--results-dir', default='./results/train', type=str, help='path to cache')
    parser.add_argument('--corr_size', default=None, type=int, help='Size of the correlated SNP set')
    parser.add_argument('--use_wandb', action='store_true', help='Toggle W&B logging')
    parser.add_argument('--run_name', default=None, type=str, help='Explicit WandB run name')
    parser.add_argument('--tags', default=[], nargs='+', type=str, help='List of WandB tags')
    parser.add_argument('--bra-fast', action='store_true', help='Slightly faster BRA (~10,15 less inner work); F1 should track full mode closely')
    parser.add_argument('--bra-turbo', action='store_true', help='Aggressive BRA shortcut (large F1 drop vs full); overrides --bra-fast')
    parser.add_argument('--fixed-window-start', default=None, type=int, help='Fixed genomic window start index (reproducible SNP panel)')
    parser.add_argument('--baseline-only', action='store_true', help='Run no-defense baseline episodes and exit (no TD3 training)')
    parser.add_argument('--baseline-episodes', default=10, type=int, help='Episodes for --baseline-only (zero beacon noise)')
    parser.add_argument('--baseline-cache-dir', default='./cache/privacy_baselines', help='Directory for configuration-keyed BRA anchor cache')
    parser.add_argument('--no-baseline-cache', action='store_true', help='Disable BRA anchor cache')
    parser.add_argument('--refresh-baseline-cache', action='store_true', help='Recompute and replace cached BRA anchors')
    parser.add_argument('--calibrate-detector', action='store_true', help='Fit the online adversary detector on honest vs mixed episodes and exit (no TD3 training)')
    parser.add_argument('--collect-detector-features', action='store_true', help='Save honest vs mixed query-stream features and exit (no fit, no BRA)')
    parser.add_argument('--calibration-episodes', default=200, type=int, help='Episodes per class (honest, mixed) for detector calibration')
    parser.add_argument('--detector-fpr-budget', default=0.10, type=float, help='Max fraction of honest episodes allowed to trip the detector (operating point)')
    parser.add_argument('--detector-train-fraction', default=0.60, type=float, help='Episode fraction used to fit detector coefficients')
    parser.add_argument('--detector-calibration-fraction', default=0.20, type=float, help='Episode fraction used only to select the FPR threshold')
    parser.add_argument('--detector-split-seed', default=17, type=int, help='Seed for grouped detector train/calibration/test splits')
    parser.add_argument('--evaluate-detector', action='store_true', help='Evaluate an existing detector on honest vs mixed no-defense episodes and exit')
    parser.add_argument('--sweep-detector-thresholds', action='store_true', help='Grid-search engage-once threshold for a frozen detector and exit (query-stream only)')
    parser.add_argument('--detector-eval-episodes', default=100, type=int, help='Episodes per class for --evaluate-detector / --sweep-detector-thresholds')
    parser.add_argument('--min-detector-tpr', default=0.90, type=float, help='Minimum mixed TPR used when sweeping detector thresholds')
    parser.add_argument('--write-swept-detector', action='store_true', default=True, help='Write suspicion_detector_swept.json with the selected threshold')
    parser.add_argument('--no-write-swept-detector', action='store_false', dest='write_swept_detector')
    parser.add_argument('--suspicion-detector-path', default=None, type=str, help='Path to a calibrated detector JSON; enables calibrated online gating')
    parser.add_argument('--util-target', default=0.85, type=float, help='Target rolling MAE utilization (penalty above this)')
    parser.add_argument('--util-penalty-scale', default=3.0, type=float, help='Per-step penalty scale for util above util-target')
    parser.add_argument('--f1-overbaseline-penalty', default=2.0, type=float, help='Extra multiplier on terminal privacy when F1 exceeds baseline')
    parser.add_argument('--critic-loss', default='huber', choices=['huber', 'mse'], help='TD3 critic loss (huber recommended)')
    parser.add_argument('--huber-beta', default=1.0, type=float, help='Smooth L1 beta for Huber critic loss')
    parser.add_argument(
        '--baseline-gated-heuristic',
        action='store_true',
        help='Replace the TD3 actor with a rare-allele-silent full-budget rule; the suspicion gate and rolling MAE cap stay on',
    )
    parser.add_argument(
        '--heuristic-policy',
        default=None,
        choices=['uniform', 'full', 'common', 'proportional'],
        help='Replace the TD3 actor with this gated rule. The suspicion gate and rolling MAE cap stay on',
    )
    parser.add_argument('--hard-mae-budget', action='store_true', default=True, help='Zero noise when projected MAE would exceed budget')
    parser.add_argument('--no-hard-mae-budget', action='store_false', dest='hard_mae_budget')
    parser.add_argument(
        '--reward-only-mae-budget',
        action='store_true',
        help='Keep over-cap noise and charge the budget tax. No rolling clip and no zeroing',
    )
    parser.add_argument(
        '--maf-detector-state',
        action='store_true',
        help='Token is the query MAF plus the detector features. Linkage fields are omitted',
    )
    parser.add_argument('--hub-only-noise', action='store_true', default=False, help='Scale/zero noise by LD hub score (off by default; mixed BRA targets high-MAF not hubs)')
    parser.add_argument('--no-hub-only-noise', action='store_false', dest='hub_only_noise')
    parser.add_argument('--hub-noise-min', default=0.25, type=float, help='Min hub score to allow noise injection')
    parser.add_argument('--proxy-hub-only', action='store_true', default=True, help='Scale proxy reward by hub score')
    parser.add_argument('--no-proxy-hub-only', action='store_false', dest='proxy_hub_only')
    parser.add_argument('--reward-normalize-scale', default=5.0, type=float, help='Divide per-step rewards by this scale')
    parser.add_argument('--util-sustained-thresh', default=0.9, type=float, help='Rolling util above this increments sustained penalty')
    parser.add_argument('--util-sustained-penalty', default=0.25, type=float, help='Per-step penalty per consecutive high-util step')
    parser.add_argument('--mae-limit-ramp', action='store_true', default=True, help='Ramp MAE limit from start_frac to full over episodes')
    parser.add_argument('--no-mae-limit-ramp', action='store_false', dest='mae_limit_ramp')
    parser.add_argument('--mae-ramp-episodes', default=200, type=int, help='Episodes to reach full MAE limit')
    parser.add_argument('--mae-ramp-start-frac', default=0.6, type=float, help='Initial MAE limit fraction')
    parser.add_argument('--explore-noise-min', default=0.04, type=float, help='Floor for action exploration noise')
    parser.add_argument('--explore-decay', default=0.998, type=float, help='Exploration decay after explore-decay-start')
    parser.add_argument('--explore-decay-start', default=100, type=int, help='Episode to switch to slower explore decay')
    parser.add_argument('--eval-freq', default=10, type=int, help='Eval checkpoint frequency (full training)')
    parser.add_argument('--eval-episodes', default=3, type=int, help='Frozen-policy MIXED eval episodes per checkpoint (privacy; used for best-checkpoint selection)')
    parser.add_argument('--honest-eval-episodes', default=None, type=int, help='Frozen-policy HONEST eval episodes per checkpoint (utility; default: max(2, eval-episodes//2))')
    parser.add_argument('--early-stop-patience', default=0, type=int, help='Stop after N eval checks without F1 improvement (0=off)')
    parser.add_argument('--sparse-reward', action='store_true', help='Terminal F1+CER + inefficiency tax only (no heuristic shaping)')
    parser.add_argument('--wandb-minimal', action='store_true', help='Log only essential metrics to W&B')
    parser.add_argument('--wandb-group', default=None, type=str, help='W&B run group (e.g. group eval runs per train config)')
    parser.add_argument('--ld-token-features', action='store_true', help='Top-K LD (sign*r, |r|) to history + panel anchors')
    parser.add_argument('--ld-history-k', default=4, type=int, help='Top-K LD links to attacker query history')
    parser.add_argument('--ld-panel-k', default=4, type=int, help='Top-K static panel LD anchors per SNP')
    parser.add_argument('--attacker-knowledge-features', action='store_true', help='Attacker knowledge / danger scalars in each token')
    parser.add_argument('--attacker-knowledge-dim', default=8, type=int, help='Size of knowledge vector (must match ld_knowledge_features)')
    parser.add_argument('--no-svd-embeddings', action='store_true', help='Disable TruncatedSVD aux (use with --ld-token-features)')
    args = parser.parse_args()
    if args.gene_size < args.max_queries:
        print(f'Warning: gene_size ({args.gene_size}) is smaller than max_queries ({args.max_queries}).')
        print(f'Adjusting gene_size to {args.max_queries} to ensure the attacker has enough targets.')
        args.gene_size = args.max_queries
    passed_seq_len = args.seq_len
    if args.seq_len is None:
        args.seq_len = args.max_queries
    if args.evaluate and passed_seq_len is not None:
        if args.max_queries > args.seq_len:
            print(
                f'Eval keeps sequence length {args.seq_len} '
                f'for a {args.max_queries}-query episode.'
            )
    else:
        args.seq_len = max(args.seq_len, args.max_queries)
    bra_core = 12
    suspicion_extra = AGENT_SUSPICION_DIM if getattr(args, 'suspicion_features', True) else 0
    if getattr(args, 'maf_detector_state', False):
        args.token_dim = 1 + suspicion_extra
    else:
        rarity_extra = 3 if getattr(args, 'rarity_token_features', False) else 0
        ld_extra = 0
        if getattr(args, 'ld_token_features', False):
            k_hist = int(getattr(args, 'ld_history_k', 4))
            k_panel = int(getattr(args, 'ld_panel_k', 4))
            ld_extra = 2 * k_hist + 2 * k_panel
        knowledge_extra = int(getattr(args, 'attacker_knowledge_dim', 8)) if getattr(args, 'attacker_knowledge_features', False) else 0
        svd_extra = 0 if getattr(args, 'no_svd_embeddings', False) else args.latent_aux_dim
        args.token_dim = bra_core + suspicion_extra + rarity_extra + ld_extra + knowledge_extra + svd_extra
    mix_total = float(getattr(args, 'honest_episode_fraction', 0.0)) + float(
        getattr(args, 'mixed_episode_fraction', 0.0)
    )
    if mix_total > 1.0:
        raise ValueError(
            f"honest_episode_fraction + mixed_episode_fraction must be <= 1.0 (got {mix_total})."
        )
    args.enforce_mae_budget = not args.no_enforce_mae_budget
    os.makedirs(args.results_dir, exist_ok=True)
    if args.random_flip:
        if not 0.0 <= float(args.random_flip_prob) <= 1.0:
            raise ValueError('--random-flip-prob must be between 0 and 1')
        if int(args.random_flip_episodes) < 1:
            raise ValueError('--random-flip-episodes must be at least 1')
        args.beacon_type = 'random'
        args.evaluate = True
        args.force_attacker_type = True
        args.eval_utility = True
    standalone_mode = any(
        bool(getattr(args, name, False))
        for name in ('evaluate', 'baseline_only', 'calibrate_detector', 'collect_detector_features', 'evaluate_detector', 'sweep_detector_thresholds', 'trace_strata_noise', 'random_flip')
    )
    if not standalone_mode:
        args.results_dir = os.path.join(args.results_dir, 'run' + str(len(os.listdir(args.results_dir))))
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.results_dir + '/weights', exist_ok=True)
    device = get_torch_device(cuda_index=args.cuda_device)
    announce_device(device)
    args.device = str(device)
    print(args)
    return args
args = args_create()
reproducibility(args.seed)
print(f'\n Loading Compiled Data Arrays...')
mafs_file = f'{args.pop_prefix}_cleaned_mafs.npy' if args.pop_prefix else 'cleaned_mafs.npy'
binary_file = f'{args.pop_prefix}_cleaned_binary.npy' if args.pop_prefix else 'cleaned_binary.npy'
mafs = np.load(os.path.join(args.data, mafs_file))
full_binary_raw = np.load(os.path.join(args.data, binary_file))
total_snps = len(mafs)
window_size = 5000
use_honest_window = (
    getattr(args, 'query_mixture_training', True)
    and float(getattr(args, 'honest_episode_fraction', 0.0)) > 0.0
    and args.pop_prefix
)
if args.fixed_window_start is not None:
    start_idx = int(args.fixed_window_start)
    if start_idx + window_size > total_snps:
        raise ValueError(f'fixed-window-start {start_idx} + window {window_size} exceeds {total_snps} SNPs')
    window_indices = np.arange(start_idx, start_idx + window_size)
    print(f' Fixed genomic window: start_idx={start_idx} ({window_indices.size} SNPs)')
elif use_honest_window:
    from helpers.snp_manifest import honest_bp_window_indices, load_manifest

    manifest_for_window = load_manifest(args.data, args.pop_prefix)
    honest_window = (
        honest_bp_window_indices(manifest_for_window, args.pop_prefix)
        if manifest_for_window is not None
        else None
    )
    if honest_window is not None and honest_window.size >= args.gene_size:
        window_indices = honest_window
        lo_bp = int(manifest_for_window.positions[window_indices[0]])
        hi_bp = int(manifest_for_window.positions[window_indices[-1]])
        print(
            f' Honest disease window: {window_indices.size:,} SNPs '
            f'(chr{manifest_for_window.chromosome} {lo_bp:,}–{hi_bp:,} bp)'
        )
    else:
        start_idx = np.random.randint(0, total_snps - window_size)
        window_indices = np.arange(start_idx, start_idx + window_size)
else:
    start_idx = np.random.randint(0, total_snps - window_size)
    window_indices = np.arange(start_idx, start_idx + window_size)
stratified_indices = select_stratified_indices(
    mafs,
    window_indices,
    args.gene_size,
    np.random,
)
if (
    getattr(args, 'query_mixture_training', True)
    and float(getattr(args, 'honest_episode_fraction', 0.0)) > 0.0
    and args.pop_prefix
):
    from helpers.disease_panels import catalog_locus_global_indices, ensure_disease_loci_in_pool
    from helpers.snp_manifest import load_manifest

    manifest = load_manifest(args.data, args.pop_prefix)
    if manifest is not None:
        in_window = catalog_locus_global_indices(
            manifest, gene_window_bp=int(getattr(args, 'disease_gene_window_bp', 200_000))
        ) & {int(x) for x in window_indices}
        stratified_indices = ensure_disease_loci_in_pool(
            stratified_indices,
            window_indices,
            manifest,
            args.gene_size,
            gene_window_bp=int(getattr(args, 'disease_gene_window_bp', 200_000)),
        )
        pinned_in_pool = len(in_window & {int(x) for x in stratified_indices})
        print(
            f' Pinned disease catalog SNPs in pool: {pinned_in_pool}/{len(in_window)} '
            f'in-window loci (pool size={stratified_indices.size})'
        )
maf_values = np.round(mafs[stratified_indices], 3)
args.pool_global_indices = stratified_indices.copy()
print(f' Sliced {args.gene_size} functional SNPs from {total_snps:,} viable candidates.')
if args.beacon_size >= 150:
    attacker_db_count = 150
elif args.beacon_size >= 100:
    attacker_db_count = 100
else:
    attacker_db_count = args.beacon_size
if args.corr_size is not None:
    attacker_db_count = args.corr_size
att_count = args.corr_size if args.corr_size is not None else attacker_db_count
if args.strict_50_50:
    pop_name = args.pop_prefix.upper() if args.pop_prefix else 'OOD POPULATION'
    print(f'\n--- EXECUTING STRICT 50/50 SPLIT FOR {pop_name} ---')
else:
    print('\n--- EXECUTING WGS SUPER-COHORT SPLIT (EUR Default) ---')
    if args.eval_cohort == 'B':
        print('Loading Strict OOD Holdout (Cohort B)...')
    elif args.eval_cohort == 'BL':
        print('--- OUT-OF-DISTRIBUTION (OOD) EVALUATION ACTIVE (COHORT BL) ---')
    elif args.eval_cohort == 'BXL':
        print('--- OUT-OF-DISTRIBUTION (OOD) EVALUATION ACTIVE (COHORT BXL) ---')
binary, pool_size_for_c_agent, panel_gene_size = build_binary_cohort(
    full_binary_raw,
    stratified_indices,
    beacon_size=args.beacon_size,
    att_count=att_count,
    eval_cohort=args.eval_cohort,
    strict_50_50=args.strict_50_50,
)
if not args.strict_50_50 and args.eval_cohort == 'A':
    print(f' Loading Training Pool (Cohort A) with {pool_size_for_c_agent} individuals...')
if panel_gene_size != args.gene_size:
    raise ValueError(
        f'Panel helper returned {panel_gene_size} SNPs, expected {args.gene_size}'
    )
print(f'Loaded {binary.shape[0]} individuals and {binary.shape[1]} SNPs.')

def main():
    print('Calculating Asymmetric Correlation Matrices...')
    temp_tool = ReconstructionTool((args.gene_size, args.beacon_size), np.zeros((1, 1)), device=args.device)
    beacon_db = binary[:pool_size_for_c_agent, :args.gene_size]
    C_agent = temp_tool.calculate_current_correlations(torch.tensor(beacon_db.T, dtype=torch.float32, device=args.device)).cpu().numpy()
    detector_only = bool(
        getattr(args, 'calibrate_detector', False)
        or getattr(args, 'collect_detector_features', False)
        or getattr(args, 'evaluate_detector', False)
        or getattr(args, 'sweep_detector_thresholds', False)
    )
    trace_noise = bool(getattr(args, 'trace_strata_noise', False))
    if trace_noise:
        args.skip_f1_eval = True
        args.evaluate = True
        args.force_attacker_type = True
        args.privacy_f1_baseline = 0.0
        args.privacy_f1_baseline_full = 0.0
        C_attacker = np.zeros_like(C_agent)
        print(' Strata-noise trace: policy forward only. Terminal BRA is skipped.')
    elif detector_only:
        args.skip_terminal_bra = True
        args.no_svd_embeddings = True
        args.privacy_f1_baseline = 0.0
        args.privacy_f1_baseline_full = 0.0
        C_attacker = np.zeros_like(C_agent)
        print(' Detector-only mode: skipping 0% BRA baseline, SVD/MIA, TD3, and terminal reconstruction.')
    else:
        public_db = binary[-att_count:, :args.gene_size]
        C_attacker = temp_tool.calculate_current_correlations(torch.tensor(public_db.T, dtype=torch.float32, device=args.device)).cpu().numpy()
        if getattr(args, 'random_flip', False):
            args.privacy_f1_baseline = 0.0
            args.privacy_f1_baseline_full = 0.0
            print(
                f' Random-flip eval: each allele is flipped with p={args.random_flip_prob:.2f} '
                f'({args.random_flip_episodes} episodes, attacker={args.attacker_type}). '
                'Skipping the 0% noise anchor.'
            )
        else:
            print('\n======================================================')
            print(' CALCULATING DYNAMIC 0% NOISE BASELINE F1')
            print('======================================================')
        if (
            not getattr(args, 'random_flip', False)
        ):
            print(f" Attacker: {args.attacker_type} | shuffle_queries={bool(getattr(args, 'shuffle_queries', False))} | bra_fast={bool(getattr(args, 'bra_fast', False))}")
            if getattr(args, 'binary_beacon', False):
                print(' Yes/no beacon: |gated action| > 0.5 flips the presence bit. Attacker sees 0 on a no and the public MAF on a yes.')
            if getattr(args, 'query_mixture_training', True) and not getattr(args, 'force_attacker_type', False):
                opt_frac = 1.0 - float(args.honest_episode_fraction) - float(args.mixed_episode_fraction)
                print(
                    f" Query mixture: {opt_frac:.0%} optimal / {args.mixed_episode_fraction:.0%} mixed / "
                    f"{args.honest_episode_fraction:.0%} honest"
                )
            print(
                f" Suspicion gate: {getattr(args, 'suspicion_gate', True)} ({getattr(args, 'suspicion_gate_mode', 'soft')}, "
                f"threshold={getattr(args, 'suspicion_gate_threshold', 0.45)}) | "
                f"unified reward: {getattr(args, 'unified_suspicion_reward', True)} | token_dim={args.token_dim}"
            )
            from helpers.baseline_cache import (
                baseline_cache_lock,
                build_baseline_cache_key,
                load_baseline_cache,
                save_baseline_cache,
            )

            cache_key, cache_meta = build_baseline_cache_key(args, stratified_indices)
            cache_dir = str(args.baseline_cache_dir)
            use_cache = not bool(args.no_baseline_cache)

            def compute_baselines() -> dict[str, float]:
                dummy_maf_helper = MAFHelper(torch.Tensor(maf_values))
                dummy_attacker = Attacker(
                    args=args,
                    maf_helper=dummy_maf_helper,
                    full_corr_matrix=C_attacker,
                )
                queried_indices = []
                max_q = min(args.max_queries, len(dummy_attacker.queries))
                for step in range(max_q):
                    action = dummy_attacker.act(step)
                    queried_indices.append(action)
                    dummy_attacker.update(0.0, action)
                queried_indices_tensor = torch.tensor(queried_indices, dtype=torch.long)
                beacon_db_tensor = torch.tensor(
                    binary[:args.beacon_size, :args.gene_size],
                    dtype=torch.float32,
                )
                true_freqs = beacon_db_tensor[:, queried_indices_tensor].mean(dim=0)
                if getattr(args, 'binary_beacon', False):
                    from helpers.yes_no_beacon import yes_no_attack_targets
                    public = np.asarray(maf_values, dtype=np.float64)[queried_indices]
                    true_freqs = torch.tensor(
                        yes_no_attack_targets(
                            true_freqs.detach().cpu().numpy(),
                            public,
                            int(args.beacon_size),
                        ),
                        dtype=torch.float32,
                    )
                trials = 5 if args.gene_size <= 200 else 3
                baseline_modes: list[tuple[str, bool]] = [
                    ('train', bool(getattr(args, 'bra_fast', False)))
                ]
                if getattr(args, 'bra_fast', False):
                    baseline_modes.append(('full', False))
                computed: dict[str, float] = {}
                for label, fast_mode in baseline_modes:
                    mode_name = 'BRA-fast' if fast_mode else 'BRA-full'
                    f1_scores = []
                    for i in range(trials):
                        print(f'{mode_name} baseline trial {i + 1}/{trials}...')
                        baseline_f1 = evaluate_defense_f1(
                            queried_indices=queried_indices_tensor,
                            noisy_freqs=true_freqs.clone(),
                            true_db=beacon_db_tensor.T,
                            full_corr_matrix=C_attacker,
                            device=args.device,
                            fast_mode=fast_mode,
                        )
                        f1_scores.append(float(baseline_f1))
                    computed[label] = sum(f1_scores) / len(f1_scores)
                return computed

            baselines = None
            if use_cache and not args.refresh_baseline_cache:
                baselines = load_baseline_cache(cache_dir, cache_key, cache_meta)
            if baselines is None and use_cache:
                with baseline_cache_lock(cache_dir, cache_key):
                    if not args.refresh_baseline_cache:
                        baselines = load_baseline_cache(cache_dir, cache_key, cache_meta)
                    if baselines is None:
                        baselines = compute_baselines()
                        cache_path = save_baseline_cache(
                            cache_dir,
                            cache_key,
                            cache_meta,
                            baselines,
                        )
                        print(f'  [baseline cache] saved {cache_path}')
            elif baselines is None:
                baselines = compute_baselines()
            else:
                print(f'  [baseline cache] loaded key {cache_key[:12]} from {cache_dir}')
            args.privacy_f1_baseline = float(baselines['train'])
            args.privacy_f1_baseline_full = float(baselines.get('full', baselines['train']))
            if getattr(args, 'bra_fast', False):
                print(f'--> BRA-fast anchor (train rewards): {args.privacy_f1_baseline:.4f}')
                print(f'--> BRA-full anchor (checkpoints/eval): {args.privacy_f1_baseline_full:.4f}')
            else:
                print(f'--> Calculated 0% Noise Baseline F1 : {args.privacy_f1_baseline:.4f}')
            print("--> The agent's reward cliff has been safely anchored to this true baseline.")
            print('======================================================\n')
    env = Env(args, maf_values, binary, corr_matrix_agent=C_agent, corr_matrix_attacker=C_attacker)
    env.args.privacy_f1_baseline = float(args.privacy_f1_baseline)
    env.args.privacy_f1_baseline_full = float(args.privacy_f1_baseline_full)
    print(f'Calculated Agent Matrix: {C_agent.shape} | Attacker Matrix: {C_attacker.shape}')
    if getattr(args, 'random_flip', False):
        from baseline_probe import run_random_flip, save_random_flip_json

        n_flip = int(args.random_flip_episodes)
        print(
            f'\n--- Random-flip beacon | p={args.random_flip_prob:.2f} | '
            f'{n_flip} episodes | attacker={args.attacker_type} | '
            f'BRA={"fast" if args.bra_fast else "full"} ---'
        )
        f1s, maes, cers = run_random_flip(env, n_episodes=n_flip)
        flip_path = os.path.join(args.results_dir, 'random_flip.json')
        payload = save_random_flip_json(flip_path, f1s, maes, cers, args)
        print(
            f"Random-flip F1 mean={payload['f1_mean']:.4f} std={payload['f1_std']:.4f} | "
            f"MAE mean={payload['mae_mean']:.4f} | CER mean={payload['clinical_error_ratio_mean']:.3f} "
            f"(saved {flip_path})"
        )
        return
    if trace_noise:
        from helpers.strata_noise import collect_and_plot
        collect_and_plot(args, env)
        return
    if detector_only:
        if getattr(args, 'collect_detector_features', False):
            from helpers.suspicion_detector import collect_calibration_data
            n_cal = int(args.calibration_episodes)
            print(
                f'\n--- Collecting detector features '
                f'({n_cal} honest + {n_cal} mixed query-stream episodes) ---'
            )
            X, y, eids = collect_calibration_data(env, n_honest=n_cal, n_mixed=n_cal)
            features_path = os.path.join(args.results_dir, 'detector_features.npz')
            np.savez_compressed(
                features_path,
                features=np.asarray(X, dtype=np.float32),
                labels=np.asarray(y, dtype=np.int32),
                episode_ids=np.asarray(eids, dtype=np.int32),
            )
            meta_path = os.path.join(args.results_dir, 'detector_features_meta.json')
            with open(meta_path, 'w') as f:
                json.dump(
                    {
                        'pop_prefix': args.pop_prefix,
                        'eval_cohort': args.eval_cohort,
                        'gene_size': int(args.gene_size),
                        'max_queries': int(args.max_queries),
                        'seed': int(args.seed),
                        'n_rows': int(X.shape[0]),
                        'n_episodes': int(len(set(eids.tolist()))),
                        'n_features': int(X.shape[1]),
                    },
                    f,
                    indent=2,
                )
            print(f'Saved {X.shape[0]} feature rows from {len(set(eids.tolist()))} episodes to {features_path}')
            return
        if getattr(args, 'calibrate_detector', False):
            from helpers.suspicion_detector import collect_calibration_data, fit_detector_split
            n_cal = int(args.calibration_episodes)
            fpr = float(args.detector_fpr_budget)
            train_frac = float(args.detector_train_fraction)
            calibration_frac = float(args.detector_calibration_fraction)
            split_seed = int(args.detector_split_seed)
            print(
                f'\n--- Calibrating online adversary detector '
                f'({n_cal} honest + {n_cal} mixed query-stream episodes, FPR budget {fpr:.0%}) ---'
            )
            print(
                f'Episode-grouped split: train={train_frac:.0%}, '
                f'calibration={calibration_frac:.0%}, '
                f'test={1.0 - train_frac - calibration_frac:.0%}, seed={split_seed}'
            )
            X, y, eids = collect_calibration_data(env, n_honest=n_cal, n_mixed=n_cal)
            print(f'Collected {X.shape[0]} per-step feature vectors from {len(set(eids.tolist()))} episodes (dim={X.shape[1]}).')
            detector = fit_detector_split(
                X,
                y,
                eids,
                fpr_budget=fpr,
                train_fraction=train_frac,
                calibration_fraction=calibration_frac,
                split_seed=split_seed,
                meta={'pop_prefix': args.pop_prefix, 'gene_size': int(args.gene_size),
                      'max_queries': int(args.max_queries), 'calibration_episodes': n_cal},
            )
            det_path = os.path.join(args.results_dir, 'suspicion_detector.json')
            detector.save(det_path)
            print(
                f'Calibration threshold={detector.threshold:.4f} | '
                f'calibration honest FPR={detector.calibration_honest_fpr:.3f} | '
                f'calibration mixed TPR={detector.calibration_mixed_tpr:.3f}'
            )
            print(
                f'HELD-OUT TEST: step AUC={detector.auc:.4f} | '
                f'episode AUC={detector.episode_auc:.4f} | '
                f'honest FPR={detector.honest_fpr:.3f} | mixed TPR={detector.mixed_tpr:.3f}'
            )
            print(f'Feature weights (standardized): {np.round(detector.coef, 3).tolist()}')
            print(f'Saved detector to {det_path}')
            return
        from helpers.suspicion_detector import (
            collect_calibration_data,
            evaluate_detector,
            threshold_sweep_report,
        )
        if env.suspicion_detector is None:
            raise ValueError('--evaluate-detector / --sweep-detector-thresholds requires --suspicion-detector-path')
        n_eval = int(args.detector_eval_episodes)
        print(
            f'\n--- Collecting detector query streams '
            f'({n_eval} honest + {n_eval} mixed, no BRA) ---'
        )
        X, y, eids = collect_calibration_data(
            env,
            n_honest=n_eval,
            n_mixed=n_eval,
        )
        if getattr(args, 'evaluate_detector', False):
            metrics = evaluate_detector(env.suspicion_detector, X, y, eids)
            payload = {
                'detector_path': args.suspicion_detector_path,
                'pop_prefix': args.pop_prefix,
                'gene_size': int(args.gene_size),
                'max_queries': int(args.max_queries),
                'threshold': float(env.suspicion_detector.threshold),
                **metrics,
            }
            output_path = os.path.join(args.results_dir, 'detector_transfer_metrics.json')
            with open(output_path, 'w') as f:
                json.dump(payload, f, indent=2)
            print(
                f"Detector transfer: step AUC={metrics['step_auc']:.4f} | "
                f"episode AUC={metrics['episode_auc']:.4f} | "
                f"honest FPR={metrics['honest_fpr']:.3f} | "
                f"mixed TPR={metrics['mixed_tpr']:.3f}"
            )
            print(f'Saved detector transfer metrics to {output_path}')
            if not getattr(args, 'sweep_detector_thresholds', False):
                return
        fpr = float(args.detector_fpr_budget)
        min_tpr = float(args.min_detector_tpr)
        report = threshold_sweep_report(
            env.suspicion_detector,
            X,
            y,
            eids,
            fpr_budget=fpr,
            min_tpr=min_tpr,
            split_seed=int(args.detector_split_seed),
        )
        sweep_path = os.path.join(args.results_dir, 'threshold_sweep.json')
        with open(sweep_path, 'w') as f:
            json.dump(
                {
                    'detector_path': args.suspicion_detector_path,
                    'pop_prefix': args.pop_prefix,
                    'eval_cohort': args.eval_cohort,
                    'gene_size': int(args.gene_size),
                    'max_queries': int(args.max_queries),
                    'seed': int(args.seed),
                    **report,
                },
                f,
                indent=2,
            )
        print(
            f'\nThreshold sweep | original={report["original_threshold"]:.4f} | '
            f'select n={report["n_select_episodes"]} confirm n={report["n_confirm_episodes"]}'
        )
        confirm = report['confirm']
        for name in ('original', 'budget_max_tpr', 'min_tpr_min_fpr', 'youden'):
            row = confirm.get(name)
            if not row:
                print(f'  {name}: unavailable on select split')
                continue
            flag = 'PASS' if row['passes_recovery_detector'] else 'fail'
            print(
                f'  {name:16s}  t={row["threshold"]:.4f}  '
                f'FPR={row["honest_fpr"]:.3f}  TPR={row["mixed_tpr"]:.3f}  '
                f'Youden={row["youden"]:.3f}  [{flag}]'
            )
        recommended = report['select'].get('budget_max_tpr')
        if recommended is None:
            print(' No select-split threshold stays within the FPR budget; not writing a swept detector.')
        elif getattr(args, 'write_swept_detector', True):
            confirm_best = confirm.get('budget_max_tpr') or {}
            swept = env.suspicion_detector.with_threshold(
                float(recommended['threshold']),
                honest_fpr=float(confirm_best.get('honest_fpr', recommended['honest_fpr'])),
                mixed_tpr=float(confirm_best.get('mixed_tpr', recommended['mixed_tpr'])),
                fpr_budget=fpr,
            )
            swept.meta = dict(swept.meta or {})
            swept.meta['threshold_sweep'] = {
                'source_detector': args.suspicion_detector_path,
                'rule': 'budget_max_tpr',
                'eval_cohort': args.eval_cohort,
                'select': recommended,
                'confirm': confirm_best,
            }
            swept_path = os.path.join(args.results_dir, 'suspicion_detector_swept.json')
            swept.save(swept_path)
            print(f'Saved swept detector to {swept_path}')
        print(f'Saved sweep curve to {sweep_path}')
        return
    if args.use_wandb:
        final_run_name = args.run_name if args.run_name else os.path.basename(args.results_dir)
        init_kw = dict(project='BRA-RL-Defense', name=final_run_name, tags=args.tags, config=vars(args))
        if getattr(args, 'wandb_group', None):
            init_kw['group'] = args.wandb_group
        wandb.init(**init_kw)
        wandb.define_metric('Episode')
        for prefix in ('Attack/*', 'Honest/*', 'Eval/*', 'Utility/*', 'Suspicion/*'):
            wandb.define_metric(prefix, step_metric='Episode')
    k_epochs = args.td3_updates_per_episode
    if k_epochs is None:
        k_epochs = args.max_queries
    beacon_agent = TD3(seq_len=args.seq_len, token_dim=args.token_dim, action_dim=1, max_action=1.0, gamma=0.99, K_epochs=k_epochs, reward_clip=args.reward_clip, policy_delay=2, critic_loss_type=args.critic_loss, huber_beta=args.huber_beta, explore_noise_min=args.explore_noise_min)
    if args.resume_beacon:
        beacon_agent.load(args.resume_beacon)
    from baseline_probe import run_no_defense_baseline, save_baseline_json
    baseline_path = os.path.join(args.results_dir, 'baseline_no_defense.json')
    if getattr(args, 'baseline_only', False):
        n_base = int(args.baseline_episodes)
        print(f'\n--- No-defense baseline ({n_base} episodes, zero noise) ---')
        f1s, maes = run_no_defense_baseline(env, args, n_episodes=n_base)
        payload = save_baseline_json(baseline_path, f1s, maes, args)
        print(f"Baseline F1 mean={payload['f1_mean']:.4f} std={payload['f1_std']:.4f} (saved {baseline_path})")
        return
    if getattr(args, 'evaluate', False):
        n_base = min(10, int(args.episodes))
        print(f'\n--- No-defense baseline ({n_base} episodes, same cohort/attacker/BRA as eval) ---')
        f1s, maes = run_no_defense_baseline(env, args, n_episodes=n_base)
        payload = save_baseline_json(baseline_path, f1s, maes, args)
        print(f"No-defense F1 mean={payload['f1_mean']:.4f} | BRA-full anchor={getattr(args, 'privacy_f1_baseline_full', args.privacy_f1_baseline):.4f} | (saved {baseline_path})")
    train_TD_beacon(args, env, beacon_agent)
if __name__ == '__main__':
    main()
