"""PCVRHyFormer inference for request-calendar + int-dense alignment solution.

Upload this file to the evaluation container together with ``dataset.py``,
``model.py``, ``trainer.py``, and ``alignment.py`` from the same directory.
Include ``alignment_pairs.json`` in the checkpoint when trained with alignment.
Do **not** use the repo-root ``evaluation/infer.py``.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory (``model.pt`` + ``train_config.json``).
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Directory for ``predictions.json``.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    FeatureSchema,
    PCVRParquetDataset,
    NUM_TIME_BUCKETS,
    SOCIAL_CALENDAR_CKPT_BASENAME,
)
from model import PCVRHyFormer, ModelInput


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# Must match ``train.py`` / ``run.sh`` defaults for this solution.
_FALLBACK_MODEL_CFG = {
    'd_model': 64,
    'emb_dim': 64,
    'num_queries': 2,
    'num_hyformer_blocks': 2,
    'num_heads': 4,
    'seq_encoder_type': 'transformer',
    'hidden_mult': 4,
    'dropout_rate': 0.01,
    'seq_top_k': 50,
    'seq_causal': False,
    'action_num': 1,
    'num_time_buckets': NUM_TIME_BUCKETS,
    'rank_mixer_mode': 'full',
    'use_rope': False,
    'rope_base': 10000.0,
    'emb_skip_threshold': 0,
    'seq_id_threshold': 10000,
    'ns_tokenizer_type': 'rankmixer',
    'user_ns_tokens': 4,
    'item_ns_tokens': 2,
    'use_request_time_ns': True,
    'use_request_social_calendar': False,
    'use_int_dense_alignment': False,
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 16

_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}, "
        f"falling back to hardcoded defaults. "
        f"Shape mismatch may occur if training used non-default hyperparameters.")
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == 'num_time_buckets':
            if 'num_time_buckets' in train_config:
                cfg[key] = train_config['num_time_buckets']
            elif 'use_time_buckets' in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config['use_time_buckets'] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    f"train_config missing both 'num_time_buckets' and 'use_time_buckets', "
                    f"using fallback = {cfg[key]}")
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(
                f"train_config missing '{key}', using fallback = {cfg[key]}")
    return cfg


def resolve_social_calendar_table_path(
    train_config: Dict[str, Any],
    model_dir: str,
    script_dir: str,
) -> str:
    ckpt_sidecar = os.path.join(model_dir, SOCIAL_CALENDAR_CKPT_BASENAME)
    if os.path.isfile(ckpt_sidecar):
        return ckpt_sidecar
    raw = train_config.get("social_calendar_table_path") or ""
    if raw and os.path.isabs(raw) and os.path.isfile(raw):
        return raw
    if raw:
        ckpt_rel = os.path.join(model_dir, os.path.basename(raw))
        if os.path.isfile(ckpt_rel):
            return ckpt_rel
        script_rel = os.path.join(script_dir, os.path.basename(raw))
        if os.path.isfile(script_rel):
            return script_rel
    return os.path.join(script_dir, SOCIAL_CALENDAR_CKPT_BASENAME)


def resolve_alignment_pairs_path(model_dir: str, train_config: Dict[str, Any]) -> str:
    raw = train_config.get('alignment_pairs_json', 'alignment_pairs.json')
    if os.path.isabs(raw) and os.path.isfile(raw):
        return raw
    candidate = os.path.join(model_dir, os.path.basename(raw))
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"alignment_pairs.json not found: tried {candidate!r} "
        f"(checkpoint must include alignment_pairs sidecar)"
    )


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    train_config: Dict[str, Any],
    model_dir: str,
    ns_groups_json: Optional[str] = None,
    device: str = 'cpu',
) -> PCVRHyFormer:
    user_ns_groups: List[List[int]]
    item_ns_groups: List[List[int]]
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }
        try:
            user_ns_groups = [
                [user_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['user_ns_groups'].values()
            ]
            item_ns_groups = [
                [item_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['item_ns_groups'].values()
            ]
        except KeyError as exc:
            raise KeyError(
                f"NS-groups JSON references fid {exc.args[0]} which is not "
                f"present in the checkpoint's schema.json. The ns_groups.json "
                f"and schema.json must come from the same training run."
            ) from exc
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]

    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)

    fid_alignments = None
    dense_residual_slices = None
    residual_dense_dim = 0
    use_align = bool(model_cfg.get('use_int_dense_alignment'))
    if use_align:
        from alignment import load_alignment_pairs
        pairs_path = resolve_alignment_pairs_path(model_dir, train_config)
        fid_alignments, dense_residual_slices, residual_dense_dim = load_alignment_pairs(
            pairs_path,
            dataset.user_int_schema,
            dataset.user_dense_schema,
        )

    hyformer_cfg = {k: v for k, v in model_cfg.items() if k != 'use_int_dense_alignment'}
    logging.info(f"Building PCVRHyFormer with cfg: {hyformer_cfg}, use_align={use_align}")
    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        use_int_dense_alignment=use_align,
        fid_alignments=fid_alignments,
        dense_residual_slices=dense_residual_slices,
        residual_dense_dim=residual_dense_dim,
        **hyformer_cfg,
    ).to(device)

    return model


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    state_dict = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means the "
            "model constructed by build_model does NOT match the checkpoint. "
            "Check that train_config.json in the ckpt dir is present and matches "
            "the training hyperparameters (use_request_time_ns, use_int_dense_alignment, "
            "user_ns_tokens, alignment_pairs.json sidecar, ...).")
        raise e


def get_ckpt_path() -> Optional[str]:
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path:
        return None
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    return None


def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
    """Convert a batch dict to ``ModelInput`` (mirrors ``trainer._make_model_input``)."""
    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch['_seq_domains']
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f'{domain}_len']
        B, _, L = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f'{domain}_time_bucket',
            torch.zeros(B, L, dtype=torch.long, device=device))

    return ModelInput(
        user_int_feats=device_batch['user_int_feats'],
        item_int_feats=device_batch['item_int_feats'],
        user_dense_feats=device_batch['user_dense_feats'],
        item_dense_feats=device_batch['item_dense_feats'],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
        request_hour=device_batch['request_hour'],
        request_dow=device_batch['request_dow'],
        request_dom=device_batch['request_dom'],
        request_weekend=device_batch['request_weekend'],
        request_holiday_type=device_batch['request_holiday_type'],
        request_promo_id=device_batch['request_promo_id'],
    )


def main() -> None:
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info(f"Using schema: {schema_path}")

    train_config = load_train_config(model_dir)
    model_cfg = resolve_model_cfg(train_config)

    use_social = bool(model_cfg.get('use_request_social_calendar', False))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    table_path = resolve_social_calendar_table_path(train_config, model_dir, script_dir)
    if use_social and not os.path.isfile(table_path):
        raise FileNotFoundError(
            f"use_request_social_calendar=True but calendar table not found: {table_path}"
        )

    sml_str = train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    batch_size = int(train_config.get('batch_size', _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get('num_workers', _FALLBACK_NUM_WORKERS))

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
        use_request_social_calendar=use_social,
        social_calendar_table_path=table_path if use_social else None,
    )
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")

    ns_groups_json = train_config.get('ns_groups_json', None)
    if ns_groups_json:
        local_candidate = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        train_config=train_config,
        model_dir=model_dir,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"The directory contains: {os.listdir(model_dir) if model_dir and os.path.isdir(model_dir) else 'N/A'}. "
            "This typically means the training job wrote only the sidecar "
            "files (schema.json / train_config.json) for this step but did "
            "not persist model.pt — a symptom of a race between "
            "_remove_old_best_dirs and EarlyStopping.save_checkpoint."
        )
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=torch.cuda.is_available(),
    )

    all_probs = []
    all_user_ids = []
    logging.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get('user_id', [])

            logits, _ = model.predict(model_input)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)

            if (batch_idx + 1) % 100 == 0:
                logging.info(f"  Processed {(batch_idx + 1) * batch_size} samples")

    logging.info(f"Inference complete: {len(all_probs)} predictions")

    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }

    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()
