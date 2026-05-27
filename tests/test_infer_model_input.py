import torch

from model import ModelInput


def test_infer_fallback_cfg_matches_run_sh():
    from infer import _FALLBACK_MODEL_CFG, _MODEL_CFG_KEYS

    assert "use_request_time_ns" in _MODEL_CFG_KEYS
    assert _FALLBACK_MODEL_CFG["use_request_time_ns"] is True
    assert _FALLBACK_MODEL_CFG["user_ns_tokens"] == 4
    assert _FALLBACK_MODEL_CFG["item_ns_tokens"] == 2
    assert _FALLBACK_MODEL_CFG["num_queries"] == 2
    assert "use_int_dense_alignment" in _MODEL_CFG_KEYS
    assert _FALLBACK_MODEL_CFG["use_int_dense_alignment"] is False
    assert "use_request_social_calendar" in _MODEL_CFG_KEYS
    assert _FALLBACK_MODEL_CFG["use_request_social_calendar"] is False


def test_resolve_model_cfg_passes_use_request_time_ns():
    from infer import resolve_model_cfg

    cfg = resolve_model_cfg({"use_request_time_ns": False, "user_ns_tokens": 5})
    assert cfg["use_request_time_ns"] is False
    assert cfg["user_ns_tokens"] == 5


def test_batch_to_model_input_includes_request_calendar_fields():
    from infer import _batch_to_model_input

    device = "cpu"
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

    mi = _batch_to_model_input(batch, device)
    assert isinstance(mi, ModelInput)
    assert mi.request_hour.shape == (B,)
