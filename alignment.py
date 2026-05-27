"""Int–Dense fid-level alignment config and fusion modules."""

from __future__ import annotations

import json
from typing import Dict, List, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from dataset import FeatureSchema


class ResolvedFidAlignment(
    tuple,
):
    """Resolved per-fid alignment metadata.

    Fields: name, int_spec_idx, dense_slice, dense_dim.
    """

    __slots__ = ()

    def __new__(
        cls,
        name: str,
        int_spec_idx: int,
        dense_slice: Tuple[int, int],
        dense_dim: int,
    ) -> "ResolvedFidAlignment":
        return tuple.__new__(cls, (name, int_spec_idx, dense_slice, dense_dim))

    @property
    def name(self) -> str:
        return self[0]

    @property
    def int_spec_idx(self) -> int:
        return self[1]

    @property
    def dense_slice(self) -> Tuple[int, int]:
        return self[2]

    @property
    def dense_dim(self) -> int:
        return self[3]


def _fid_to_spec_idx(schema: "FeatureSchema", fid: int) -> int:
    for i, (entry_fid, _, _) in enumerate(schema.entries):
        if entry_fid == fid:
            return i
    raise ValueError(f"fid {fid} not found in schema (total_dim={schema.total_dim})")


def gather_dense_slice(
    dense_feats: torch.Tensor,
    slices: List[Tuple[int, int]],
) -> torch.Tensor:
    """Slice user_dense_feats by (offset, length) list."""
    if not slices:
        raise ValueError("gather_dense_slice requires at least one slice")
    if len(slices) == 1:
        off, ln = slices[0]
        return dense_feats[:, off : off + ln]
    parts = [dense_feats[:, off : off + ln] for off, ln in slices]
    return torch.cat(parts, dim=-1)


def load_alignment_pairs(
    path: str,
    user_int_schema: "FeatureSchema",
    user_dense_schema: "FeatureSchema",
) -> Tuple[List[ResolvedFidAlignment], List[Tuple[int, int]], int]:
    """Load alignment JSON and resolve schema offsets."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    fid_alignments: List[ResolvedFidAlignment] = []
    used_dense_fids: set = set()
    used_int_spec_idx: set = set()

    for pair in cfg.get("user_pairs", []):
        int_fids = pair["int_fids"]
        dense_fids = pair["dense_fids"]
        if len(int_fids) != len(dense_fids):
            raise ValueError(
                f"pair {pair.get('name')}: len(int_fids)={len(int_fids)} "
                f"!= len(dense_fids)={len(dense_fids)}"
            )
        pair_name = pair["name"]
        for int_fid, dense_fid in zip(int_fids, dense_fids):
            if dense_fid in used_dense_fids:
                raise ValueError(f"dense fid {dense_fid} appears in multiple pairs")
            used_dense_fids.add(dense_fid)
            int_spec_idx = _fid_to_spec_idx(user_int_schema, int_fid)
            if int_spec_idx in used_int_spec_idx:
                raise ValueError(
                    f"duplicate int_spec_idx={int_spec_idx} (int_fid={int_fid}) in alignment config"
                )
            used_int_spec_idx.add(int_spec_idx)
            off, ln = user_dense_schema.get_offset_length(dense_fid)
            fid_alignments.append(
                ResolvedFidAlignment(
                    name=f"{pair_name}_f{int_fid}",
                    int_spec_idx=int_spec_idx,
                    dense_slice=(off, ln),
                    dense_dim=ln,
                )
            )

    dense_only_fids = cfg.get("dense_only_fids", [])
    residual_slices: List[Tuple[int, int]] = []
    for dense_fid in dense_only_fids:
        if dense_fid in used_dense_fids:
            raise ValueError(f"dense fid {dense_fid} in both pair and dense_only_fids")
        used_dense_fids.add(dense_fid)
        residual_slices.append(user_dense_schema.get_offset_length(dense_fid))

    residual_slices.sort(key=lambda x: x[0])
    residual_dim = sum(ln for _, ln in residual_slices)
    aligned_dim = sum(f.dense_dim for f in fid_alignments)

    if aligned_dim + residual_dim != user_dense_schema.total_dim:
        raise ValueError(
            f"aligned_dim({aligned_dim}) + residual_dim({residual_dim}) "
            f"!= user_dense total_dim({user_dense_schema.total_dim})"
        )

    schema_dense_fids = set(user_dense_schema.feature_ids)
    if used_dense_fids != schema_dense_fids:
        missing = schema_dense_fids - used_dense_fids
        extra = used_dense_fids - schema_dense_fids
        raise ValueError(
            f"dense fid coverage mismatch: missing={missing}, extra={extra}"
        )

    return fid_alignments, residual_slices, residual_dim


def validate_fid_alignments_covered_by_ns_groups(
    fid_alignments: List[ResolvedFidAlignment],
    user_ns_groups: List[List[int]],
) -> None:
    """Ensure every aligned int fid is embedded via user_ns_groups.

    When alignment is on, paired dense dims are not in the residual mega-token.
    If an int_spec_idx never appears in any NS group, its dense signal is dropped.
    """
    covered = {idx for group in user_ns_groups for idx in group}
    missing = [fa for fa in fid_alignments if fa.int_spec_idx not in covered]
    if not missing:
        return
    names = [fa.name for fa in missing]
    spec_indices = [fa.int_spec_idx for fa in missing]
    raise ValueError(
        f"int-dense alignment entries not in any user_ns_groups (paired dense would "
        f"be dropped from NS): names={names}, int_spec_idx={spec_indices}. "
        f"Covered int_spec_idx={sorted(covered)}"
    )


class CrossModalPairAlign(nn.Module):
    """Fuse one int embedding with its dense slice in embed_dim space."""

    def __init__(self, dense_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.dense_proj = nn.Sequential(
            nn.Linear(dense_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.int_refine = nn.LayerNorm(embed_dim)
        self.fuse_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, int_emb: torch.Tensor, dense_slice: torch.Tensor) -> torch.Tensor:
        d = self.dense_proj(dense_slice)
        i = self.int_refine(int_emb)
        g = self.fuse_gate(torch.cat([i, d], dim=-1))
        return self.out_norm(g * i + (1.0 - g) * d)


class IntDensePairFusionBank(nn.Module):
    """Per-fid aligners keyed by int feature spec index."""

    def __init__(
        self,
        fid_alignments: List[ResolvedFidAlignment],
        embed_dim: int,
    ) -> None:
        super().__init__()
        self._by_int_spec_idx: Dict[int, ResolvedFidAlignment] = {
            f.int_spec_idx: f for f in fid_alignments
        }
        self.aligners = nn.ModuleDict({
            f.name: CrossModalPairAlign(f.dense_dim, embed_dim) for f in fid_alignments
        })

    def fuse_fid_embedding(
        self,
        fid_spec_idx: int,
        fid_emb: torch.Tensor,
        dense_feats: torch.Tensor,
    ) -> torch.Tensor:
        meta = self._by_int_spec_idx.get(fid_spec_idx)
        if meta is None:
            return fid_emb
        dense_slice = gather_dense_slice(dense_feats, [meta.dense_slice])
        return self.aligners[meta.name](fid_emb, dense_slice)
