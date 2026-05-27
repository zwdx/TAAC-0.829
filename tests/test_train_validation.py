import re
from pathlib import Path

import pytest

from utils import (
    compute_pcvr_num_ns,
    resolve_rankmixer_ns_token_counts,
    validate_rank_mixer_t_config,
)


def test_resolve_rankmixer_ns_token_counts_auto_falls_back_to_group_count():
    num_user, num_item = resolve_rankmixer_ns_token_counts(
        ns_tokenizer_type="rankmixer",
        user_ns_tokens=0,
        item_ns_tokens=0,
        num_user_groups=7,
        num_item_groups=4,
    )
    assert num_user == 7
    assert num_item == 4


def test_compute_pcvr_num_ns_with_request_calendar_token():
    assert compute_pcvr_num_ns(
        num_user_ns=4,
        num_item_ns=2,
        user_dense_dim=10,
        item_dense_dim=0,
        use_request_time_ns=True,
    ) == 8


def test_validate_rejects_enabled_request_ns_with_user_tokens_5():
    with pytest.raises(ValueError, match=r"user_ns_tokens 4"):
        validate_rank_mixer_t_config(
            d_model=64,
            num_queries=2,
            num_sequences=4,
            num_user_ns=5,
            num_item_ns=2,
            user_dense_dim=10,
            item_dense_dim=0,
            use_request_time_ns=True,
            rank_mixer_mode="full",
        )


def test_validate_rejects_disabled_request_ns_with_user_tokens_4():
    with pytest.raises(ValueError, match=r"user_ns_tokens 5"):
        validate_rank_mixer_t_config(
            d_model=64,
            num_queries=2,
            num_sequences=4,
            num_user_ns=4,
            num_item_ns=2,
            user_dense_dim=10,
            item_dense_dim=0,
            use_request_time_ns=False,
            rank_mixer_mode="full",
        )


def test_validate_accepts_enabled_recipe():
    validate_rank_mixer_t_config(
        d_model=64,
        num_queries=2,
        num_sequences=4,
        num_user_ns=4,
        num_item_ns=2,
        user_dense_dim=10,
        item_dense_dim=0,
        use_request_time_ns=True,
        rank_mixer_mode="full",
    )


def test_validate_accepts_baseline_ablation_recipe():
    validate_rank_mixer_t_config(
        d_model=64,
        num_queries=2,
        num_sequences=4,
        num_user_ns=5,
        num_item_ns=2,
        user_dense_dim=10,
        item_dense_dim=0,
        use_request_time_ns=False,
        rank_mixer_mode="full",
    )


def test_validate_rank_mixer_with_int_dense_alignment():
    assert compute_pcvr_num_ns(
        4,
        2,
        918,
        0,
        True,
        use_int_dense_alignment=True,
        residual_dense_dim=100,
    ) == 8
    validate_rank_mixer_t_config(
        d_model=64,
        num_queries=2,
        num_sequences=4,
        num_user_ns=4,
        num_item_ns=2,
        user_dense_dim=918,
        item_dense_dim=0,
        use_request_time_ns=True,
        use_int_dense_alignment=True,
        residual_dense_dim=100,
        rank_mixer_mode="full",
    )


def test_validate_request_social_calendar_requires_request_time_ns():
    from train import validate_request_social_calendar_args

    with pytest.raises(ValueError, match="use_request_time_ns"):
        validate_request_social_calendar_args(
            use_request_time_ns=False,
            use_request_social_calendar=True,
            social_calendar_table_path="dummy.csv",
        )


def test_validate_request_social_calendar_requires_existing_table(tmp_path):
    from train import validate_request_social_calendar_args

    missing = str(tmp_path / "no.csv")
    with pytest.raises(ValueError, match="calendar table not found"):
        validate_request_social_calendar_args(
            use_request_time_ns=True,
            use_request_social_calendar=True,
            social_calendar_table_path=missing,
        )


def test_validate_skips_when_rank_mixer_mode_not_full():
    validate_rank_mixer_t_config(
        d_model=64,
        num_queries=2,
        num_sequences=4,
        num_user_ns=99,
        num_item_ns=2,
        user_dense_dim=10,
        item_dense_dim=0,
        use_request_time_ns=True,
        rank_mixer_mode="separate",
    )


def test_train_validate_rank_mixer_does_not_pass_social_calendar_kw():
    train_py = Path(__file__).resolve().parents[1] / "train.py"
    src = train_py.read_text(encoding="utf-8")
    m = re.search(
        r"validate_rank_mixer_t_config\((.*?)\n    \)\n\n    model_args",
        src,
        re.DOTALL,
    )
    assert m is not None
    assert "use_request_social_calendar" not in m.group(1)


def test_train_model_args_includes_use_request_social_calendar():
    train_py = Path(__file__).resolve().parents[1] / "train.py"
    src = train_py.read_text(encoding="utf-8")
    model_block = src.split("model_args = {")[1].split("}")[0]
    assert '"use_request_social_calendar"' in model_block
    assert "args.use_request_social_calendar" in model_block
