"""PCVRHyFormer training entry point (baseline_fast: accelerated training).

Usage:
    python train.py [--num_epochs 10] [--batch_size 1024] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from utils import (
    set_seed,
    EarlyStopping,
    create_logger,
    add_training_accel_arguments,
    apply_training_backend,
    reconcile_cudnn_determinism,
    dataloader_extras,
    effective_learning_rates,
    maybe_compile_model,
    training_accel_options_from_args,
    assert_baseline_model_input,
    assert_trainer_builds_baseline_model_input,
    resolve_rankmixer_ns_token_counts,
    validate_rank_mixer_t_config,
    BCE_POS_WEIGHT_MAX,
)
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import ModelInput, PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def validate_loss_args(args: argparse.Namespace) -> None:
    weighted_flags = (
        args.bce_pos_weight > 0
        or args.train_positive_rate > 0
        or args.estimate_train_positive_rate
    )

    if args.loss_type == 'bce':
        if weighted_flags:
            raise ValueError(
                'loss_type=bce (E0) does not use --bce_pos_weight, '
                '--train_positive_rate, or --estimate_train_positive_rate; '
                'use loss_type=bce_weighted for E_pos'
            )
        return

    if args.loss_type == 'focal':
        if weighted_flags:
            raise ValueError(
                'loss_type=focal does not use bce_weighted flags; '
                'remove --bce_pos_weight / --train_positive_rate / '
                '--estimate_train_positive_rate'
            )
        return

    sources = sum([
        args.bce_pos_weight > 0,
        args.train_positive_rate > 0,
        args.estimate_train_positive_rate,
    ])
    if args.bce_pos_weight != 0.0:
        if args.bce_pos_weight <= 0.0:
            raise ValueError(
                f'--bce_pos_weight must be in (0, {BCE_POS_WEIGHT_MAX}], '
                f'got {args.bce_pos_weight}'
            )
        if args.bce_pos_weight > BCE_POS_WEIGHT_MAX:
            raise ValueError(
                f'--bce_pos_weight must be <= {BCE_POS_WEIGHT_MAX}, '
                f'got {args.bce_pos_weight}'
            )
    if sources != 1:
        raise ValueError(
            'loss_type=bce_weighted requires exactly one weight source: '
            '--bce_pos_weight > 0, OR --train_positive_rate in (0,1), OR '
            '--estimate_train_positive_rate (got %d sources)' % sources
        )
    if args.train_positive_rate > 0 and not (0.0 < args.train_positive_rate < 1.0):
        raise ValueError(
            f'--train_positive_rate must be in (0, 1), got {args.train_positive_rate}'
        )


def validate_request_social_calendar_args(
    use_request_time_ns: bool,
    use_request_social_calendar: bool,
    social_calendar_table_path: str,
) -> None:
    if not use_request_social_calendar:
        return
    if not use_request_time_ns:
        raise ValueError(
            "--use_request_social_calendar requires --use_request_time_ns"
        )
    if not os.path.isfile(social_calendar_table_path):
        raise ValueError(
            f"--use_request_social_calendar: calendar table not found: "
            f"{social_calendar_table_path}"
        )


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument(
        '--loss_type', type=str, default='bce',
        choices=['bce', 'bce_weighted', 'focal'],
        help='bce=E0; bce_weighted=E_pos (pos_weight); focal=Focal Loss',
    )
    parser.add_argument(
        '--bce_pos_weight', type=float, default=0.0,
        help='Positive-class weight for bce_weighted (PyTorch pos_weight). '
             'If > 0, used directly and ignores train_positive_rate / estimate.',
    )
    parser.add_argument(
        '--train_positive_rate', type=float, default=0.0,
        help='Training set positive rate in (0,1). Used when loss_type=bce_weighted '
             'and bce_pos_weight<=0 to compute pos_weight=(1-p)/p.',
    )
    parser.add_argument(
        '--estimate_train_positive_rate', action='store_true', default=False,
        help='Scan train Parquet label_type to estimate p+ before training '
             '(only when loss_type=bce_weighted and bce_pos_weight<=0 and '
             'train_positive_rate<=0).',
    )
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    parser.add_argument('--ns_groups_json', type=str, default='',
                        help='Path to the NS-groups JSON file. Empty string (default, '
                             'same as run.sh) places each int feature in its own singleton '
                             'group. If the path does not exist, singleton groups are used.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')
    parser.set_defaults(use_request_time_ns=True)
    parser.add_argument(
        '--use_request_time_ns',
        action='store_true',
        dest='use_request_time_ns',
        help='Append one NS token from request-time calendar features '
             '(hour/dow/dom/weekend). Default: enabled. Pair with '
             '--user_ns_tokens 4 (see run.sh) to keep T=16 for d_model=64.',
    )
    parser.add_argument(
        '--no-use_request_time_ns',
        action='store_false',
        dest='use_request_time_ns',
        help='Disable request calendar NS token (baseline ablation: use '
             '--user_ns_tokens 5 with this flag).',
    )

    parser.set_defaults(use_int_dense_alignment=False)
    parser.add_argument(
        '--use_int_dense_alignment',
        action='store_true',
        dest='use_int_dense_alignment',
        help='Fuse shared int/dense fids before NS concat; project dense_only fids '
             'to one residual NS token.',
    )
    parser.add_argument(
        '--no-use_int_dense_alignment',
        action='store_false',
        dest='use_int_dense_alignment',
        help='Disable int-dense alignment (same as context-tokens dense mega-token).',
    )
    _default_cal = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cn_request_calendar.csv"
    )
    parser.set_defaults(use_request_social_calendar=False)
    parser.add_argument(
        '--use_request_social_calendar',
        action='store_true',
        dest='use_request_social_calendar',
    )
    parser.add_argument(
        '--no-use_request_social_calendar',
        action='store_false',
        dest='use_request_social_calendar',
    )
    parser.add_argument(
        '--social_calendar_table_path',
        type=str,
        default=_default_cal,
    )
    _default_pairs = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'alignment_pairs.json',
    )
    parser.add_argument(
        '--alignment_pairs_json',
        type=str,
        default=_default_pairs,
        help='JSON mapping int fids to dense fids for alignment.',
    )

    add_training_accel_arguments(parser)

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH', os.path.join(args.log_dir or './tf_events', 'events'))

    # Validate required paths
    if not args.data_dir:
        raise ValueError("TRAIN_DATA_PATH environment variable or --data_dir argument must be set")
    if not args.ckpt_dir:
        args.ckpt_dir = './checkpoints'
    if not args.log_dir:
        args.log_dir = './logs'

    return args


def main() -> None:
    args = parse_args()

    accel_options = training_accel_options_from_args(args)
    apply_training_backend(accel_options)

    assert_baseline_model_input(ModelInput)
    assert_trainer_builds_baseline_model_input(PCVRHyFormerRankingTrainer)

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    if args.tf_events_dir:
        Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    reconcile_cudnn_determinism(accel_options)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    tf_events_dir = args.tf_events_dir or os.path.join(args.log_dir, 'tf_events')
    writer = SummaryWriter(tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    validate_request_social_calendar_args(
        args.use_request_time_ns,
        args.use_request_social_calendar,
        args.social_calendar_table_path,
    )

    validate_loss_args(args)

    estimated_rate = None
    estimated_n_pos = None
    estimated_n_total = None
    if (
        args.loss_type == 'bce_weighted'
        and args.estimate_train_positive_rate
    ):
        from dataset import estimate_train_positive_rate
        estimated_rate, estimated_n_pos, estimated_n_total = (
            estimate_train_positive_rate(
                data_dir=args.data_dir,
                valid_ratio=args.valid_ratio,
                train_ratio=args.train_ratio,
            )
        )

    from utils import resolve_bce_pos_weight
    effective_bce_pos_weight = resolve_bce_pos_weight(
        loss_type=args.loss_type,
        bce_pos_weight=args.bce_pos_weight,
        train_positive_rate=args.train_positive_rate,
        estimated_positive_rate=estimated_rate,
    )
    if args.loss_type == 'bce_weighted':
        rate_for_log = (
            args.train_positive_rate
            if args.train_positive_rate > 0
            else estimated_rate
        )
        logging.info(
            f'Resolved bce_weighted: train_positive_rate={rate_for_log} '
            f'bce_pos_weight={effective_bce_pos_weight:.6f}'
        )

    train_config = {
        **vars(args),
        'effective_bce_pos_weight': effective_bce_pos_weight,
    }
    if estimated_rate is not None:
        train_config['estimated_train_positive_rate'] = estimated_rate
        train_config['estimated_train_n_pos'] = estimated_n_pos
        train_config['estimated_train_n_total'] = estimated_n_total

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    _pf = dataloader_extras(accel_options, args.num_workers).get("prefetch_factor")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        prefetch_factor=_pf,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        use_request_social_calendar=args.use_request_social_calendar,
        social_calendar_table_path=args.social_calendar_table_path,
    )

    if args.use_request_social_calendar:
        lut = getattr(pcvr_dataset, '_social_calendar', None)
        if lut is not None and lut.sorted_date_keys.size == 0:
            logging.warning(
                "Social calendar table is empty; request_holiday_type/promo_id "
                "will be all zeros."
            )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    fid_alignments = None
    dense_residual_slices = None
    residual_dense_dim = 0
    if args.use_int_dense_alignment:
        if pcvr_dataset.user_dense_schema.total_dim <= 0:
            raise ValueError("use_int_dense_alignment requires user_dense_feats")
        if not os.path.isfile(args.alignment_pairs_json):
            raise ValueError(
                f"use_int_dense_alignment: alignment_pairs_json not found: "
                f"{args.alignment_pairs_json}"
            )
        from alignment import (
            load_alignment_pairs,
            validate_fid_alignments_covered_by_ns_groups,
        )
        fid_alignments, dense_residual_slices, residual_dense_dim = load_alignment_pairs(
            args.alignment_pairs_json,
            pcvr_dataset.user_int_schema,
            pcvr_dataset.user_dense_schema,
        )
        validate_fid_alignments_covered_by_ns_groups(fid_alignments, user_ns_groups)
        logging.info(
            f"Int-dense alignment: {len(fid_alignments)} fid pairs, "
            f"residual_dense_dim={residual_dense_dim}"
        )

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    num_user_ns, num_item_ns = resolve_rankmixer_ns_token_counts(
        args.ns_tokenizer_type,
        args.user_ns_tokens,
        args.item_ns_tokens,
        len(user_ns_groups),
        len(item_ns_groups),
    )
    validate_rank_mixer_t_config(
        d_model=args.d_model,
        num_queries=args.num_queries,
        num_sequences=len(pcvr_dataset.seq_domains),
        num_user_ns=num_user_ns,
        num_item_ns=num_item_ns,
        user_dense_dim=pcvr_dataset.user_dense_schema.total_dim,
        item_dense_dim=pcvr_dataset.item_dense_schema.total_dim,
        use_request_time_ns=args.use_request_time_ns,
        use_int_dense_alignment=args.use_int_dense_alignment,
        residual_dense_dim=residual_dense_dim,
        rank_mixer_mode=args.rank_mixer_mode,
    )

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "use_request_time_ns": args.use_request_time_ns,
        "use_request_social_calendar": args.use_request_social_calendar,
        "use_int_dense_alignment": args.use_int_dense_alignment,
        "fid_alignments": fid_alignments,
        "dense_residual_slices": dense_residual_slices,
        "residual_dense_dim": residual_dense_dim,
    }

    model = PCVRHyFormer(**model_args).to(args.device)
    model = maybe_compile_model(model, accel_options, batch_size=args.batch_size)

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    if args.use_request_time_ns:
        logging.info("Request calendar NS token enabled (expect user_ns_tokens=4 for T=16)")
    if args.use_request_social_calendar:
        logging.info("Request social calendar enabled (holiday_type + promo_id)")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    dense_lr, sparse_lr = effective_learning_rates(
        args.lr, args.sparse_lr, args.batch_size, accel_options,
    )

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=dense_lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        bce_pos_weight=effective_bce_pos_weight,
        sparse_lr=sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        alignment_pairs_path=(
            args.alignment_pairs_json
            if args.use_int_dense_alignment and os.path.exists(args.alignment_pairs_json)
            else None
        ),
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=train_config,
        accel_options=accel_options,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
