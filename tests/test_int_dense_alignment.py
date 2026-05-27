"""Tests for int-dense fid-level alignment."""

import pytest
import torch

from dataset import FeatureSchema
from alignment import (
    CrossModalPairAlign,
    IntDensePairFusionBank,
    ResolvedFidAlignment,
    gather_dense_slice,
    load_alignment_pairs,
    validate_fid_alignments_covered_by_ns_groups,
)


def _tiny_schemas():
    us = FeatureSchema()
    us.add(89, 1)
    us.add(90, 1)
    ds = FeatureSchema()
    ds.add(89, 1)
    ds.add(90, 1)
    ds.add(61, 3)
    return us, ds


def test_load_alignment_pairs_residual_dim(tmp_path):
    cfg = tmp_path / "pairs.json"
    cfg.write_text(
        '{"user_pairs":[{"name":"P1","int_fids":[89,90],"dense_fids":[89,90]}],'
        '"dense_only_fids":[61]}',
        encoding="utf-8",
    )
    us, ds = _tiny_schemas()
    fid_alignments, residual_slices, residual_dim = load_alignment_pairs(str(cfg), us, ds)
    assert len(fid_alignments) == 2
    assert fid_alignments[0].dense_dim == 1
    assert residual_dim == 3
    assert residual_slices == [(2, 3)]


def test_gather_dense_slice_single_and_multi():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.allclose(gather_dense_slice(x, [(1, 1)]), torch.tensor([[2.0]]))
    assert torch.allclose(gather_dense_slice(x, [(0, 1), (2, 2)]), torch.tensor([[1.0, 3.0, 4.0]]))


def test_cross_modal_pair_align_shape():
    m = CrossModalPairAlign(dense_dim=5, embed_dim=8)
    int_emb = torch.randn(4, 8)
    dense = torch.randn(4, 5)
    out = m(int_emb, dense)
    assert out.shape == (4, 8)


def test_rankmixer_forward_with_pair_fusion_changes_output():
    from model import RankMixerNSTokenizer

    specs = [(10, 0, 1), (10, 1, 1)]
    fid0 = ResolvedFidAlignment("P_f0", 0, (0, 1), 1)
    bank = IntDensePairFusionBank([fid0], embed_dim=4)
    tok = RankMixerNSTokenizer(
        specs, [[0], [1]], emb_dim=4, d_model=8, num_ns_tokens=2, pair_fusion_bank=bank,
    )
    int_feats = torch.tensor([[1, 2], [3, 4]])
    dense = torch.tensor([[0.5, 0.0], [0.1, 0.0]])
    out_with = tok(int_feats, dense)
    tok_baseline = RankMixerNSTokenizer(
        specs, [[0], [1]], emb_dim=4, d_model=8, num_ns_tokens=2,
    )
    out_without = tok_baseline(int_feats)
    assert out_with.shape == (2, 2, 8)
    assert not torch.allclose(out_with, out_without)


def test_pcvr_forward_int_dense_alignment_num_ns_unchanged():
    from model import ModelInput, PCVRHyFormer

    domains = ["seq_a", "seq_b", "seq_c", "seq_d"]
    B, L = 2, 4
    fid_align = [
        ResolvedFidAlignment("P0", 0, (0, 1), 1),
        ResolvedFidAlignment("P1", 1, (1, 1), 1),
    ]
    residual_slices = [(2, 2)]
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
        use_int_dense_alignment=True,
        fid_alignments=fid_align,
        dense_residual_slices=residual_slices,
        residual_dense_dim=2,
    )
    assert model.num_ns == 8
    assert 64 % (2 * len(domains) + model.num_ns) == 0
    mi = ModelInput(
        user_int_feats=torch.zeros(B, 2, dtype=torch.long),
        item_int_feats=torch.zeros(B, 1, dtype=torch.long),
        user_dense_feats=torch.randn(B, 4),
        item_dense_feats=torch.zeros(B, 0),
        seq_data={d: torch.zeros(B, 1, L, dtype=torch.long) for d in domains},
        seq_lens={d: torch.full((B,), 2, dtype=torch.long) for d in domains},
        seq_time_buckets={d: torch.zeros(B, L, dtype=torch.long) for d in domains},
        request_hour=torch.zeros(B, dtype=torch.long),
        request_dow=torch.zeros(B, dtype=torch.long),
        request_dom=torch.zeros(B, dtype=torch.long),
        request_weekend=torch.zeros(B, dtype=torch.long),
        request_holiday_type=torch.zeros(B, dtype=torch.long),
        request_promo_id=torch.zeros(B, dtype=torch.long),
    )
    logits = model(mi)
    assert logits.shape == (B, 1)


def test_load_alignment_pairs_rejects_duplicate_int_spec_idx(tmp_path):
    cfg = tmp_path / "pairs.json"
    cfg.write_text(
        '{"user_pairs":[{"name":"P1","int_fids":[89,89],"dense_fids":[89,90]}],'
        '"dense_only_fids":[61]}',
        encoding="utf-8",
    )
    us = FeatureSchema()
    us.add(89, 1)
    us.add(90, 1)
    ds = FeatureSchema()
    ds.add(89, 1)
    ds.add(90, 1)
    ds.add(61, 3)
    with pytest.raises(ValueError, match="duplicate int_spec_idx"):
        load_alignment_pairs(str(cfg), us, ds)


def test_validate_fid_alignments_rejects_missing_ns_group_coverage():
    fid_alignments = [
        ResolvedFidAlignment("P0", 0, (0, 1), 1),
        ResolvedFidAlignment("P1", 1, (1, 1), 1),
    ]
    user_ns_groups = [[0]]
    with pytest.raises(ValueError, match="not in any user_ns_groups"):
        validate_fid_alignments_covered_by_ns_groups(fid_alignments, user_ns_groups)


def test_validate_fid_alignments_accepts_full_coverage():
    fid_alignments = [
        ResolvedFidAlignment("P0", 0, (0, 1), 1),
        ResolvedFidAlignment("P1", 1, (1, 1), 1),
    ]
    validate_fid_alignments_covered_by_ns_groups(fid_alignments, [[0], [1]])


def test_pcvr_dense_proj_matches_baseline_when_alignment_disabled():
    from model import PCVRHyFormer

    domains = ["seq_a", "seq_b", "seq_c", "seq_d"]
    kw = dict(
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
    baseline = PCVRHyFormer(**kw)
    off = PCVRHyFormer(**kw, use_int_dense_alignment=False)
    assert off.use_int_dense_alignment is False
    assert off.has_user_dense_ns is True
    assert baseline.has_user_dense_ns is True
    assert off.num_ns == baseline.num_ns == 8
    assert off.user_dense_proj[0].in_features == 4
    assert baseline.user_dense_proj[0].in_features == 4
    assert off.pair_fusion_bank is None
