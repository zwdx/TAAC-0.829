import os
import random
import copy
import logging
import time
import argparse
import inspect
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Dict, Any, FrozenSet, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogFormatter:
    """Custom ``logging.Formatter`` that prefixes every record with the
    wall-clock timestamp and the elapsed wall-clock time since this
    formatter instance was constructed.

    The prefix format is ``"<locale-date> <locale-time> - H:MM:SS"``, which
    is convenient for tracking long-running training runs where both the
    absolute time and the time-since-start are useful.

    Multi-line messages are re-indented so that continuation lines align
    with the beginning of the message (not the prefix).
    """

    def __init__(self) -> None:
        # Anchor used to compute the elapsed-time part of the log prefix.
        # Can be reset at runtime via ``create_logger(...).reset_time()``.
        self.start_time: float = time.time()

    def format(self, record: logging.LogRecord) -> str:
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s" % (
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        # Indent continuation lines so they line up with the message body,
        # not with the timestamp prefix.
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(filepath: str) -> logging.Logger:
    """Create and configure the root logger for a training/inference run.

    The returned logger has two handlers attached:

    * A ``FileHandler`` bound to ``filepath`` (opened in write mode,
      truncating any previous content) that records ``DEBUG``-level and
      above messages for post-mortem inspection.
    * A ``StreamHandler`` to stderr that only echoes ``INFO``-level and
      above messages, keeping the console output concise.

    Both handlers share a ``LogFormatter`` so the console and the log file
    stay in sync. Any pre-existing handlers on the root logger are removed
    to avoid duplicate lines when this function is called multiple times.

    Args:
        filepath: Destination path of the log file. Opened in ``"w"`` mode,
            so previous contents are overwritten.

    Returns:
        The root ``logging.Logger`` instance. The returned object is
        augmented with a ``reset_time()`` attribute that resets the
        elapsed-time clock used by the log prefix. This is useful when the
        "interesting" phase of a run starts well after process launch
        (e.g. after schema building and data loading).
    """
    log_formatter = LogFormatter()

    file_handler = logging.FileHandler(filepath, "w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Allow callers to reset the elapsed-time clock shown in the log prefix.
    def reset_time() -> None:
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time  # type: ignore[attr-defined]

    return logger


class EarlyStopping:
    """Early-stop training when the validation metric plateaus.

    The tracker assumes a *higher-is-better* metric (typical for AUC or
    accuracy). A candidate ``score`` is considered an improvement iff
    ``score > best_score + delta``; otherwise the internal ``counter`` is
    incremented and training is requested to stop once
    ``counter >= patience``.

    On every improvement the current ``model.state_dict()`` is both
    deep-copied in memory (``self.best_model``) and persisted to disk at
    ``checkpoint_path``. The most recent *improving* score is cached in
    ``self.best_saved_score`` so callers can skip redundant IO.

    Attributes:
        checkpoint_path: Destination path for the best ``state_dict``.
        patience: Number of non-improving calls tolerated before
            ``early_stop`` is flipped to ``True``.
        verbose: If ``True``, emit an ``INFO`` line whenever a checkpoint
            is written.
        counter: Number of consecutive non-improving calls seen so far.
        best_score: Best score observed; ``None`` until the first call.
        early_stop: Set to ``True`` once ``counter >= patience``.
        delta: Minimum absolute improvement required to reset ``counter``.
        best_model: In-memory deep copy of the best ``state_dict``.
        best_saved_score: Score associated with the last checkpoint
            actually written to disk.
        best_extra_metrics: Optional auxiliary metrics captured at the
            best-score step (e.g. logloss, other AUCs).
        label: Short prefix (e.g. ``"val"``) prepended to log lines to
            disambiguate multiple trackers running in parallel.
    """

    def __init__(
        self,
        checkpoint_path: str,
        label: str = "",
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0,
    ) -> None:
        self.checkpoint_path: str = checkpoint_path
        self.patience: int = patience
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.delta: float = delta
        self.best_model: Optional[Dict[str, torch.Tensor]] = None
        self.best_saved_score: float = 0.0
        self.best_extra_metrics: Optional[Dict[str, Any]] = None
        self.label: str = label
        if self.label != "":
            self.label += " "

    def _is_not_improved(self, score: float) -> bool:
        """Return ``True`` iff ``score`` fails to beat ``best_score + delta``.

        Used as the gating condition for incrementing the patience counter.
        ``best_score`` must have been seeded by a prior ``__call__``.
        """
        assert self.best_score is not None, "call __call__ first to seed best_score"
        if score > self.best_score + self.delta:
            return False
        return True

    def __call__(
        self,
        score: float,
        model: nn.Module,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Feed a new validation score into the tracker.

        Three branches, in order:

        1. First call (``best_score is None``): seed the tracker, persist a
           checkpoint, and cache the model weights.
        2. Not improved: increment ``counter`` and log the progress; flip
           ``early_stop`` once ``counter >= patience``.
        3. Improved: reset ``counter`` to ``0``, update ``best_score`` and
           ``best_extra_metrics``, refresh the in-memory ``best_model``,
           and write a new checkpoint to disk.

        Args:
            score: Scalar validation metric (higher is better, e.g. AUC).
            model: Model whose ``state_dict`` is snapshotted on
                improvement. Only the parameters are saved, not the
                optimizer state.
            extra_metrics: Optional dict of auxiliary metrics recorded at
                the same step, e.g.
                ``{"best_val_AUC": ..., "best_val_logloss": ...}``. Stored
                verbatim as ``self.best_extra_metrics``; not interpreted
                by ``EarlyStopping`` itself.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_extra_metrics = extra_metrics
            self.best_saved_score = 0.0
            self.save_checkpoint(score, model)
            self.best_model = copy.deepcopy(get_model_state_dict(model))
        elif self._is_not_improved(score):
            self.counter += 1
            logging.info(f'{self.label}earlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            logging.info(f'{self.label}earlyStopping counter reset!')
            self.best_score = score
            self.best_model = copy.deepcopy(get_model_state_dict(model))
            self.best_extra_metrics = extra_metrics
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score: float, model: nn.Module) -> None:
        """Persist ``model.state_dict()`` to ``self.checkpoint_path``.

        Creates any missing parent directories, writes atomically via
        ``torch.save``, and records ``score`` as ``self.best_saved_score``
        so subsequent callers can detect "no new improvement since last
        save" without re-reading the checkpoint file.

        Args:
            score: Validation score associated with the weights being
                saved. Exposed to callers via ``best_saved_score`` after
                the write completes.
            model: Model whose parameters are being snapshotted. Only
                ``state_dict()`` is written; optimizer and scheduler state
                are explicitly *not* included.
        """
        if self.verbose:
            logging.info('Validation score increased. Saving model ...')
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(get_model_state_dict(model), self.checkpoint_path)
        self.best_saved_score = score


def set_seed(seed: int) -> None:
    """Seed every RNG that can influence training reproducibility.

    Seeds ``random``, the ``PYTHONHASHSEED`` env var, NumPy, the CPU
    PyTorch generator and all CUDA generators, then forces cuDNN into
    deterministic mode.

    Note that full bitwise determinism on GPU also requires disabling
    cuDNN auto-tuning (``torch.backends.cudnn.benchmark = False``) and may
    come with a non-trivial throughput cost; this helper intentionally
    only toggles ``deterministic`` to preserve speed for common use cases.

    Args:
        seed: Non-negative integer seed shared by all RNGs listed above.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Route 0 (E_pos): shared bounds for computed and CLI --bce_pos_weight.
BCE_POS_WEIGHT_MIN: float = 1.0
BCE_POS_WEIGHT_MAX: float = 100.0


def compute_bce_pos_weight_from_positive_rate(
    positive_rate: float,
    eps: float = 1e-6,
    min_pos_weight: float = BCE_POS_WEIGHT_MIN,
    max_pos_weight: float = BCE_POS_WEIGHT_MAX,
) -> float:
    """Map train positive rate p+ to PyTorch BCE pos_weight (1-p+)/p+."""
    p = float(positive_rate)
    if not (0.0 < p < 1.0):
        raise ValueError(
            f"positive_rate must be in (0, 1), got {positive_rate}"
        )
    w = (1.0 - p) / max(p, eps)
    return float(min(max(w, min_pos_weight), max_pos_weight))


def weighted_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float,
    reduction: str = 'mean',
) -> torch.Tensor:
    """BCEWithLogits with positive-class weight (Route 0 / E_pos)."""
    if pos_weight <= 0.0:
        raise ValueError(f"pos_weight must be > 0, got {pos_weight}")
    pw = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pw, reduction=reduction,
    )


def resolve_bce_pos_weight(
    loss_type: str,
    bce_pos_weight: float,
    train_positive_rate: float,
    estimated_positive_rate: Optional[float] = None,
) -> float:
    """Return effective pos_weight; 0.0 when loss is not bce_weighted."""
    if loss_type != 'bce_weighted':
        return 0.0
    if bce_pos_weight > 0.0:
        w = float(bce_pos_weight)
        if w > BCE_POS_WEIGHT_MAX:
            raise ValueError(
                f"bce_pos_weight must be <= {BCE_POS_WEIGHT_MAX}, got {w}"
            )
        return w
    if train_positive_rate > 0.0:
        return compute_bce_pos_weight_from_positive_rate(train_positive_rate)
    if estimated_positive_rate is not None:
        return compute_bce_pos_weight_from_positive_rate(estimated_positive_rate)
    raise ValueError(
        "loss_type=bce_weighted requires one of: "
        "--bce_pos_weight > 0, --train_positive_rate in (0,1), "
        "or --estimate_train_positive_rate"
    )


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: (N,) raw logits (before sigmoid).
        targets: (N,) binary labels {0, 1}.
        alpha: positive-class weight in (0, 1). When positives dominate,
            use alpha < 0.5 to downweight the positive class.
        gamma: focusing parameter. gamma=0 degenerates to standard BCE;
            gamma=2 is the standard value.
        reduction: 'mean' | 'sum' | 'none'.
    """
    p = torch.sigmoid(logits)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * bce_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


# ---------------------------------------------------------------------------
# Training acceleration (flat upload: lives in utils.py, no extra modules)
# ---------------------------------------------------------------------------

REF_BATCH_SIZE = 256


@dataclass(frozen=True)
class TrainingAccelOptions:
    amp: str
    tf32: bool
    torch_compile: bool
    cudnn_benchmark: bool
    scale_lr_with_batch: bool
    scale_sparse_lr_with_batch: bool
    dataloader_tune: bool
    prefetch_factor: int
    amp_eval: bool

    @property
    def amp_enabled(self) -> bool:
        return self.amp == "bf16"


def add_training_accel_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training acceleration")
    group.add_argument("--amp", choices=("off", "bf16"), default="bf16")
    group.add_argument("--tf32", dest="tf32", action="store_true", default=True)
    group.add_argument("--no-tf32", dest="tf32", action="store_false")
    group.add_argument("--torch_compile", dest="torch_compile", action="store_true", default=True)
    group.add_argument("--no_torch_compile", dest="torch_compile", action="store_false")
    group.add_argument("--cudnn_benchmark", dest="cudnn_benchmark", action="store_true", default=True)
    group.add_argument("--no_cudnn_benchmark", dest="cudnn_benchmark", action="store_false")
    group.add_argument("--scale_lr_with_batch", dest="scale_lr_with_batch", action="store_true", default=True)
    group.add_argument("--no_scale_lr_with_batch", dest="scale_lr_with_batch", action="store_false")
    group.add_argument("--scale_sparse_lr_with_batch", dest="scale_sparse_lr_with_batch", action="store_true", default=True)
    group.add_argument("--no_scale_sparse_lr_with_batch", dest="scale_sparse_lr_with_batch", action="store_false")
    group.add_argument("--dataloader_tune", dest="dataloader_tune", action="store_true", default=True)
    group.add_argument("--no_dataloader_tune", dest="dataloader_tune", action="store_false")
    group.add_argument("--prefetch_factor", type=int, default=3)
    group.add_argument("--amp_eval", dest="amp_eval", action="store_true", default=True)
    group.add_argument("--no_amp_eval", dest="amp_eval", action="store_false")


def training_accel_options_from_args(args: argparse.Namespace) -> TrainingAccelOptions:
    return TrainingAccelOptions(
        amp=args.amp,
        tf32=args.tf32,
        torch_compile=args.torch_compile,
        cudnn_benchmark=args.cudnn_benchmark,
        scale_lr_with_batch=args.scale_lr_with_batch,
        scale_sparse_lr_with_batch=args.scale_sparse_lr_with_batch,
        dataloader_tune=args.dataloader_tune,
        prefetch_factor=args.prefetch_factor,
        amp_eval=args.amp_eval,
    )


def get_model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """State dict safe for save/load when ``model`` is ``torch.compile``-wrapped."""
    inner = getattr(model, "_orig_mod", model)
    return inner.state_dict()


def apply_training_backend(options: TrainingAccelOptions) -> None:
    if options.tf32 and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        logging.info("training_accel: TF32 matmul precision=high")
    if options.cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("training_accel: cudnn.benchmark=True")


def reconcile_cudnn_determinism(options: TrainingAccelOptions) -> None:
    """``set_seed`` enables cudnn deterministic; benchmark mode needs it off."""
    if options.cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False


def batch_lr_scale(batch_size: int) -> float:
    return float(batch_size) / float(REF_BATCH_SIZE)


def effective_learning_rates(
    base_dense_lr: float,
    base_sparse_lr: float,
    batch_size: int,
    options: TrainingAccelOptions,
) -> Tuple[float, float]:
    if batch_size == REF_BATCH_SIZE:
        return base_dense_lr, base_sparse_lr
    ratio = batch_lr_scale(batch_size)
    dense = base_dense_lr * ratio if options.scale_lr_with_batch else base_dense_lr
    sparse = base_sparse_lr * ratio if options.scale_sparse_lr_with_batch else base_sparse_lr
    if ratio != 1.0 and (options.scale_lr_with_batch or options.scale_sparse_lr_with_batch):
        logging.info(
            "training_accel: batch_size=%s ref=%s ratio=%.4f -> dense_lr=%s sparse_lr=%s",
            batch_size, REF_BATCH_SIZE, ratio, dense, sparse,
        )
    return dense, sparse


_COMPILE_MAX_BATCH_SIZE = 1024


def maybe_compile_model(
    model: nn.Module,
    options: TrainingAccelOptions,
    batch_size: int = 256,
) -> nn.Module:
    if not options.torch_compile:
        return model
    if batch_size >= _COMPILE_MAX_BATCH_SIZE:
        logging.warning(
            "training_accel: torch.compile skipped (batch_size=%s >= %s)",
            batch_size, _COMPILE_MAX_BATCH_SIZE,
        )
        return model
    if not torch.cuda.is_available():
        logging.warning("training_accel: torch.compile skipped (CUDA not available)")
        return model
    if not hasattr(torch, "compile"):
        logging.warning("training_accel: torch.compile unavailable, skipping")
        return model
    try:
        compiled = torch.compile(model)
        logging.info("training_accel: torch.compile enabled")
        return compiled
    except Exception as exc:
        logging.warning("training_accel: torch.compile failed (%s), using eager", exc)
        return model


def dataloader_extras(options: TrainingAccelOptions, num_workers: int) -> Dict[str, Any]:
    if num_workers <= 0 or not options.dataloader_tune:
        return {}
    return {"prefetch_factor": max(1, int(options.prefetch_factor))}


class AmpStepHelper:
    def __init__(self, options: TrainingAccelOptions, device: str) -> None:
        self.options = options
        self.device = device

    def _device_type(self) -> str:
        return "cuda" if str(self.device).startswith("cuda") else "cpu"

    def train_autocast(self) -> Any:
        if not self.options.amp_enabled or self._device_type() != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def eval_autocast(self) -> Any:
        if not (self.options.amp_enabled and self.options.amp_eval):
            return nullcontext()
        if self._device_type() != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def backward_and_step(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        dense_optimizer: torch.optim.Optimizer,
        sparse_optimizer: Optional[torch.optim.Optimizer],
        max_norm: float = 1.0,
    ) -> None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, foreach=False)
        dense_optimizer.step()
        if sparse_optimizer is not None:
            sparse_optimizer.step()


def resolve_rankmixer_ns_token_counts(
    ns_tokenizer_type: str,
    user_ns_tokens: int,
    item_ns_tokens: int,
    num_user_groups: int,
    num_item_groups: int,
) -> Tuple[int, int]:
    """Resolve effective user/item NS token counts (matches ``PCVRHyFormer.__init__``)."""
    if ns_tokenizer_type == 'rankmixer':
        num_user = user_ns_tokens if user_ns_tokens > 0 else num_user_groups
        num_item = item_ns_tokens if item_ns_tokens > 0 else num_item_groups
        return num_user, num_item
    if ns_tokenizer_type == 'group':
        return num_user_groups, num_item_groups
    raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")


def compute_pcvr_num_ns(
    num_user_ns: int,
    num_item_ns: int,
    user_dense_dim: int,
    item_dense_dim: int,
    use_request_time_ns: bool,
    use_int_dense_alignment: bool = False,
    residual_dense_dim: int = 0,
) -> int:
    """Total NS token count (matches ``PCVRHyFormer.num_ns``)."""
    if user_dense_dim > 0:
        if use_int_dense_alignment:
            user_dense_tokens = 1 if residual_dense_dim > 0 else 0
        else:
            user_dense_tokens = 1
    else:
        user_dense_tokens = 0
    return (
        num_user_ns
        + user_dense_tokens
        + num_item_ns
        + (1 if item_dense_dim > 0 else 0)
        + (1 if use_request_time_ns else 0)
    )


def validate_rank_mixer_t_config(
    d_model: int,
    num_queries: int,
    num_sequences: int,
    num_user_ns: int,
    num_item_ns: int,
    user_dense_dim: int,
    item_dense_dim: int,
    use_request_time_ns: bool,
    rank_mixer_mode: str,
    use_int_dense_alignment: bool = False,
    residual_dense_dim: int = 0,
) -> None:
    """Fail fast when ``rank_mixer_mode=full`` and ``d_model % T != 0``."""
    if rank_mixer_mode != 'full':
        return

    num_ns = compute_pcvr_num_ns(
        num_user_ns,
        num_item_ns,
        user_dense_dim,
        item_dense_dim,
        use_request_time_ns,
        use_int_dense_alignment=use_int_dense_alignment,
        residual_dense_dim=residual_dense_dim,
    )
    T = num_queries * num_sequences + num_ns
    if d_model % T == 0:
        return

    valid_T = [t for t in range(1, d_model + 1) if d_model % t == 0]
    hints: List[str] = []
    if use_request_time_ns and num_user_ns != 4:
        hints.append(
            "with --use_request_time_ns, use --user_ns_tokens 4 (and --item_ns_tokens 2, "
            "--num_queries 2) so num_ns=8 and T=16 for d_model=64"
        )
    if not use_request_time_ns and num_user_ns == 4 and user_dense_dim > 0:
        hints.append(
            "with --no-use_request_time_ns, use --user_ns_tokens 5 for the baseline ablation "
            "(num_ns=8, T=16 for d_model=64)"
        )
    if use_request_time_ns and num_user_ns == 5:
        hints.append(
            "do not combine --use_request_time_ns with --user_ns_tokens 5 "
            "(adds a 9th NS token; T=17 is invalid for d_model=64)"
        )

    hint_msg = " ".join(hints) if hints else (
        f"adjust --user_ns_tokens / --item_ns_tokens / --num_queries / "
        f"--use_request_time_ns so T={T} divides d_model={d_model}"
    )
    raise ValueError(
        f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
        f"{num_queries}*{num_sequences}+{num_ns}={T}. "
        f"Valid T for d_model={d_model}: {valid_T}. Hint: {hint_msg}"
    )


_BASELINE_MODEL_INPUT_FIELDS: FrozenSet[str] = frozenset({
    "user_int_feats", "item_int_feats", "user_dense_feats", "item_dense_feats",
    "seq_data", "seq_lens", "seq_time_buckets",
    "request_hour", "request_dow", "request_dom", "request_weekend",
    "request_holiday_type", "request_promo_id",
})


def assert_baseline_model_input(model_input_cls: Type) -> None:
    fields = frozenset(getattr(model_input_cls, "_fields", ()))
    if fields != _BASELINE_MODEL_INPUT_FIELDS:
        extra = sorted(fields - _BASELINE_MODEL_INPUT_FIELDS)
        missing = sorted(_BASELINE_MODEL_INPUT_FIELDS - fields)
        parts = []
        if extra:
            parts.append(f"unexpected fields {extra}")
        if missing:
            parts.append(f"missing fields {missing}")
        raise RuntimeError(
            f"model.ModelInput does not match expected contract ({'; '.join(parts)})."
        )


def assert_trainer_builds_baseline_model_input(trainer_cls: Type) -> None:
    try:
        source = inspect.getsource(trainer_cls._make_model_input)
    except (TypeError, OSError):
        return
    if "seq_inter_buckets" in source:
        raise RuntimeError(
            "trainer._make_model_input passes seq_inter_buckets; "
            "upload baseline_fast trainer.py only (flat files in script dir)."
        )
