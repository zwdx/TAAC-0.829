import numpy as np
import torch

from dataset import compute_request_calendar_features


def test_compute_request_calendar_features_treats_zero_timestamp_as_epoch():
    ts = np.array([0, 1714924800], dtype=np.int64)
    hour, dow, dom, weekend = compute_request_calendar_features(ts)
    assert hour.shape == (2,)
    assert hour[1] == 0
    assert dom[1] == 5


def test_compute_request_calendar_features_shanghai_midnight():
    # 2024-05-06 00:00:00 CST = 2024-05-05 16:00:00 UTC
    ts = np.array([1714924800], dtype=np.int64)
    hour, dow, dom, weekend = compute_request_calendar_features(ts)
    assert hour[0] == 0
    assert dom[0] == 5  # May 6 -> dom index 5
    assert weekend[0] == 0


def test_modelinput_request_fields_exist():
    from model import ModelInput

    fields = ModelInput._fields
    for name in (
        "request_hour",
        "request_dow",
        "request_dom",
        "request_weekend",
        "request_holiday_type",
        "request_promo_id",
    ):
        assert name in fields


def test_assert_baseline_model_input_accepts_extended_tuple():
    from model import ModelInput
    from utils import assert_baseline_model_input

    assert_baseline_model_input(ModelInput)


def test_request_calendar_ns_forward_shape():
    from model import ModelInput, PCVRHyFormer

    B, L = 2, 4
    domains = ["seq_a", "seq_b", "seq_c", "seq_d"]
    mi = ModelInput(
        user_int_feats=torch.zeros(B, 2, dtype=torch.long),
        item_int_feats=torch.zeros(B, 1, dtype=torch.long),
        user_dense_feats=torch.zeros(B, 4),
        item_dense_feats=torch.zeros(B, 0),
        seq_data={d: torch.zeros(B, 1, L, dtype=torch.long) for d in domains},
        seq_lens={d: torch.full((B,), 2, dtype=torch.long) for d in domains},
        seq_time_buckets={d: torch.zeros(B, L, dtype=torch.long) for d in domains},
        request_hour=torch.tensor([9, 21], dtype=torch.long),
        request_dow=torch.tensor([0, 5], dtype=torch.long),
        request_dom=torch.tensor([0, 14], dtype=torch.long),
        request_weekend=torch.tensor([0, 1], dtype=torch.long),
        request_holiday_type=torch.zeros(B, dtype=torch.long),
        request_promo_id=torch.zeros(B, dtype=torch.long),
    )
    model = PCVRHyFormer(
        user_int_feature_specs=[(10, 0, 1), (10, 1, 1)],
        item_int_feature_specs=[(10, 0, 1)],
        user_dense_dim=4,
        item_dense_dim=0,
        seq_vocab_sizes={d: [10] for d in domains},
        user_ns_groups=[[0], [1]],
        item_ns_groups=[[0]],
        d_model=64,
        emb_dim=8,
        num_queries=2,
        num_hyformer_blocks=1,
        num_heads=4,
        num_time_buckets=0,
        rank_mixer_mode="full",
        ns_tokenizer_type="rankmixer",
        user_ns_tokens=4,
        item_ns_tokens=2,
        use_request_time_ns=True,
    )
    assert model.num_ns == 8
    T = 2 * len(domains) + model.num_ns
    assert T == 16
    assert 64 % T == 0
    logits = model(mi)
    assert logits.shape == (B, 1)


def test_trainer_make_model_input_passes_request_fields():
    from trainer import PCVRHyFormerRankingTrainer
    from model import ModelInput

    trainer = PCVRHyFormerRankingTrainer.__new__(PCVRHyFormerRankingTrainer)
    trainer.device = torch.device("cpu")

    domains = ["seq_a", "seq_b", "seq_c", "seq_d"]
    B, L = 2, 3
    batch = {
        "user_int_feats": torch.zeros(B, 1, dtype=torch.long),
        "item_int_feats": torch.zeros(B, 1, dtype=torch.long),
        "user_dense_feats": torch.zeros(B, 2),
        "item_dense_feats": torch.zeros(B, 0),
        "_seq_domains": domains,
        "request_hour": torch.tensor([8, 20], dtype=torch.long),
        "request_dow": torch.tensor([1, 6], dtype=torch.long),
        "request_dom": torch.tensor([2, 29], dtype=torch.long),
        "request_weekend": torch.tensor([0, 1], dtype=torch.long),
        "request_holiday_type": torch.zeros(B, dtype=torch.long),
        "request_promo_id": torch.zeros(B, dtype=torch.long),
    }
    for d in domains:
        batch[d] = torch.zeros(B, 1, L, dtype=torch.long)
        batch[f"{d}_len"] = torch.full((B,), 2, dtype=torch.long)
        batch[f"{d}_time_bucket"] = torch.zeros(B, L, dtype=torch.long)

    mi = trainer._make_model_input(batch)
    assert isinstance(mi, ModelInput)
    assert mi.request_hour.shape == (B,)
