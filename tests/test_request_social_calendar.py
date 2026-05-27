import inspect
import json

import numpy as np
import pytest
import torch

from dataset import (
    SOCIAL_CALENDAR_CKPT_BASENAME,
    SocialCalendarLookup,
    _DEFAULT_SOCIAL_CALENDAR_CSV,
    load_social_calendar_table,
    lookup_request_social_calendar,
)


def test_default_social_calendar_csv_path_is_flat():
    assert _DEFAULT_SOCIAL_CALENDAR_CSV.endswith(SOCIAL_CALENDAR_CKPT_BASENAME)
    assert "data/calendar" not in _DEFAULT_SOCIAL_CALENDAR_CSV.replace("\\", "/")


def test_load_social_calendar_table_sorted_keys(tmp_path):
    p = tmp_path / "cal.csv"
    p.write_text(
        "date,holiday_type,promo_id\n"
        "2024-06-18,0,1\n"
        "2024-05-01,1,0\n",
        encoding="utf-8",
    )
    lut = load_social_calendar_table(str(p))
    assert isinstance(lut, SocialCalendarLookup)
    assert lut.sorted_date_keys.tolist() == [20240501, 20240618]
    assert lut.holiday_type.tolist() == [1, 0]
    assert lut.promo_id.tolist() == [0, 1]


def test_lookup_request_social_calendar_hit_and_miss():
    ts_hit = np.array([1718640000], dtype=np.int64)
    ts_miss = np.array([1704067200], dtype=np.int64)
    lut = SocialCalendarLookup(
        np.array([20240618], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
    )
    h1, p1 = lookup_request_social_calendar(ts_hit, lut)
    assert h1[0] == 0 and p1[0] == 1
    h2, p2 = lookup_request_social_calendar(ts_miss, lut)
    assert h2[0] == 0 and p2[0] == 0


def test_load_empty_csv_returns_empty_lookup(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("date,holiday_type,promo_id\n", encoding="utf-8")
    lut = load_social_calendar_table(str(p))
    assert lut.sorted_date_keys.size == 0
    h, pr = lookup_request_social_calendar(np.array([1718640000], dtype=np.int64), lut)
    assert h[0] == 0 and pr[0] == 0


def test_lookup_extreme_timestamp_returns_zeros():
    lut = SocialCalendarLookup(
        np.array([20240618], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
    )
    ts = np.array([9223372036854775807], dtype=np.int64)
    h, pr = lookup_request_social_calendar(ts, lut)
    assert h[0] == 0 and pr[0] == 0


def test_load_rejects_duplicate_dates(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text(
        "date,holiday_type,promo_id\n2024-01-01,0,0\n2024-01-01,1,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_social_calendar_table(str(p))


def test_get_pcvr_data_signature_includes_social_calendar_kwargs():
    from dataset import get_pcvr_data

    params = inspect.signature(get_pcvr_data).parameters
    assert "use_request_social_calendar" in params
    assert "social_calendar_table_path" in params


def test_load_schema_does_not_reference_social_calendar_params():
    from dataset import PCVRParquetDataset

    source = inspect.getsource(PCVRParquetDataset._load_schema)
    assert "use_request_social_calendar" not in source


def test_lookup_enabled_returns_nonzero_promo(tmp_path):
    p = tmp_path / "cal.csv"
    p.write_text("date,holiday_type,promo_id\n2024-06-18,0,1\n", encoding="utf-8")
    lut = load_social_calendar_table(str(p))
    ts = np.array([1718640000], dtype=np.int64)
    h, pr = lookup_request_social_calendar(ts, lut)
    assert int(pr[0]) == 1


def test_get_pcvr_data_impl_shares_lookup():
    from dataset import get_pcvr_data

    src = inspect.getsource(get_pcvr_data)
    assert "social_lookup = load_social_calendar_table" in src
    assert src.count("social_calendar_lookup=social_lookup") >= 2


def test_modelinput_includes_social_calendar_fields():
    from model import ModelInput

    assert "request_holiday_type" in ModelInput._fields
    assert "request_promo_id" in ModelInput._fields


def test_request_calendar_ns_social_forward_shape():
    from model import RequestCalendarNSTokenizer

    B, d = 3, 64
    tok = RequestCalendarNSTokenizer(d, use_social_calendar=True)
    out = tok(
        torch.zeros(B, dtype=torch.long),
        torch.zeros(B, dtype=torch.long),
        torch.zeros(B, dtype=torch.long),
        torch.zeros(B, dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1, 0], dtype=torch.long),
    )
    assert out.shape == (B, 1, d)


def test_request_calendar_ns_without_social_ignores_extra_ids():
    from model import RequestCalendarNSTokenizer

    B, d = 2, 64
    tok = RequestCalendarNSTokenizer(d, use_social_calendar=False)
    assert not hasattr(tok, "emb_holiday_type")
    out = tok(
        torch.zeros(B, dtype=torch.long),
        torch.zeros(B, dtype=torch.long),
        torch.zeros(B, dtype=torch.long),
        torch.zeros(B, dtype=torch.long),
        torch.ones(B, dtype=torch.long),
        torch.ones(B, dtype=torch.long),
    )
    assert out.shape == (B, 1, d)


def test_pcvr_hyformer_social_calendar_num_ns_unchanged_with_alignment():
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
        request_holiday_type=torch.tensor([0, 1], dtype=torch.long),
        request_promo_id=torch.tensor([0, 2], dtype=torch.long),
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
        use_request_social_calendar=True,
        use_int_dense_alignment=False,
    )
    assert model.num_ns == 8
    logits = model(mi)
    assert logits.shape == (B, 1)


def test_trainer_sidecar_copies_social_calendar_csv(tmp_path):
    from trainer import PCVRHyFormerRankingTrainer

    cal_src = tmp_path / "src_cal.csv"
    cal_src.write_text("date,holiday_type,promo_id\n2024-06-18,0,1\n", encoding="utf-8")
    ckpt_dir = tmp_path / "ckpt"
    trainer = PCVRHyFormerRankingTrainer.__new__(PCVRHyFormerRankingTrainer)
    trainer.train_config = {
        "use_request_social_calendar": True,
        "social_calendar_table_path": str(cal_src),
    }
    trainer.schema_path = None
    trainer.ns_groups_path = None
    trainer.alignment_pairs_path = None
    trainer._write_sidecar_files(str(ckpt_dir))
    sidecar = ckpt_dir / SOCIAL_CALENDAR_CKPT_BASENAME
    assert sidecar.is_file()
    cfg = json.loads((ckpt_dir / "train_config.json").read_text(encoding="utf-8"))
    assert cfg["social_calendar_table_path"] == SOCIAL_CALENDAR_CKPT_BASENAME
