import argparse

import pytest
import torch

from utils import (
    BCE_POS_WEIGHT_MAX,
    compute_bce_pos_weight_from_positive_rate,
    resolve_bce_pos_weight,
    weighted_binary_cross_entropy_with_logits,
)


def test_compute_pos_weight_typical_ctr_rate():
    w = compute_bce_pos_weight_from_positive_rate(0.01)
    assert 90.0 <= w <= 100.0


def test_compute_pos_weight_rejects_invalid_rate():
    with pytest.raises(ValueError, match="positive_rate must be in"):
        compute_bce_pos_weight_from_positive_rate(0.0)


def test_resolve_manual_pos_weight():
    w = resolve_bce_pos_weight('bce_weighted', 7.5, 0.0, None)
    assert w == 7.5


def test_resolve_from_train_positive_rate():
    w = resolve_bce_pos_weight('bce_weighted', 0.0, 0.1, None)
    assert abs(w - 9.0) < 1e-5


def test_resolve_bce_returns_zero():
    assert resolve_bce_pos_weight('bce', 99.0, 0.5, 0.01) == 0.0


def test_weighted_bce_higher_pos_weight_increases_loss_on_positive():
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0])
    loss_hi = weighted_binary_cross_entropy_with_logits(
        logits, targets, pos_weight=10.0,
    )
    loss_lo = weighted_binary_cross_entropy_with_logits(
        logits, targets, pos_weight=1.0,
    )
    assert loss_hi > loss_lo


def test_resolve_requires_source():
    with pytest.raises(ValueError, match='bce_weighted requires'):
        resolve_bce_pos_weight('bce_weighted', 0.0, 0.0, None)


def _ns(**kwargs):
    base = dict(
        loss_type='bce',
        bce_pos_weight=0.0,
        train_positive_rate=0.0,
        estimate_train_positive_rate=False,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_validate_rejects_estimate_on_bce_e0():
    from train import validate_loss_args

    with pytest.raises(ValueError, match='bce_weighted'):
        validate_loss_args(_ns(estimate_train_positive_rate=True))


def test_validate_rejects_pos_weight_on_bce_e0():
    from train import validate_loss_args

    with pytest.raises(ValueError, match='bce_weighted'):
        validate_loss_args(_ns(bce_pos_weight=9.0))


def test_validate_rejects_weighted_flags_on_focal():
    from train import validate_loss_args

    with pytest.raises(ValueError, match='focal does not use'):
        validate_loss_args(_ns(loss_type='focal', bce_pos_weight=1.0))


def test_validate_bce_weighted_requires_exactly_one_source():
    from train import validate_loss_args

    with pytest.raises(ValueError, match='exactly one'):
        validate_loss_args(_ns(loss_type='bce_weighted'))


def test_validate_bce_weighted_rejects_two_sources():
    from train import validate_loss_args

    with pytest.raises(ValueError, match='exactly one'):
        validate_loss_args(_ns(
            loss_type='bce_weighted',
            train_positive_rate=0.01,
            estimate_train_positive_rate=True,
        ))


def test_validate_accepts_bce_weighted_estimate_only():
    from train import validate_loss_args

    validate_loss_args(_ns(
        loss_type='bce_weighted',
        estimate_train_positive_rate=True,
    ))


def test_validate_rejects_negative_bce_pos_weight():
    from train import validate_loss_args

    with pytest.raises(ValueError, match='bce_pos_weight must be in'):
        validate_loss_args(_ns(loss_type='bce_weighted', bce_pos_weight=-1.0))


def test_validate_rejects_excessive_bce_pos_weight():
    from train import validate_loss_args

    with pytest.raises(ValueError, match=r'must be <='):
        validate_loss_args(_ns(
            loss_type='bce_weighted',
            bce_pos_weight=BCE_POS_WEIGHT_MAX + 1.0,
        ))


def test_resolve_rejects_excessive_manual_pos_weight():
    with pytest.raises(ValueError, match='bce_pos_weight must be <='):
        resolve_bce_pos_weight('bce_weighted', BCE_POS_WEIGHT_MAX + 1.0, 0.0, None)


def test_estimate_rejects_zero_positives_in_train(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataset import estimate_train_positive_rate

    labels = pa.array([0, 0, 0, 0, 2, 0, 0, 0], type=pa.int64())
    pq.write_table(
        pa.table({'label_type': labels}),
        tmp_path / "part.parquet",
        row_group_size=4,
    )
    with pytest.raises(ValueError, match='no positive labels'):
        estimate_train_positive_rate(str(tmp_path), valid_ratio=0.5)


def test_estimate_train_positive_rate_on_synthetic_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataset import estimate_train_positive_rate

    labels = pa.array([2, 0, 0, 0, 0, 0, 0, 2], type=pa.int64())
    table = pa.table({'label_type': labels})
    path = tmp_path / "part.parquet"
    pq.write_table(table, path, row_group_size=4)

    rate, n_pos, n_total = estimate_train_positive_rate(
        str(tmp_path), valid_ratio=0.5, train_ratio=1.0,
    )
    assert n_total == 4
    assert n_pos == 1
    assert abs(rate - 0.25) < 1e-6
