# opt_semantic_feature_alignment

在 `opt_time_request_context_tokens` 基础上，于 NS concat **之前**对 shared int/dense fid 做逐 fid 门控融合（`alignment.py`），未配对 dense 维（如 fid 61、87）投影为 1 个 residual NS token。

## 默认 CLI（`run.sh`）

- `--use_request_time_ns` + `--use_int_dense_alignment`
- `--user_ns_tokens 4` `--item_ns_tokens 2` `--num_queries 2` → `num_ns=8`, `T=16`（`d_model=64`）

## Item / 序列（G5 uplift）

针对 I3（MeanPool）的候选条件化序列摘要见 **`solution/opt_item_seq_cross_modal_alignment/`**（须上传 `target_attention.py`）。本包保持 semantic 基线用于对照与 `--no-use_target_attn_pool` 类 ablation 的 ckpt 来源。

## Ablation

```bash
bash run.sh --no-use_int_dense_alignment
```

## Checkpoint

- 需侧车 `alignment_pairs.json`（训练时由 trainer 复制到 ckpt 目录）
- **不兼容** context-tokens 或旧版全局 MLP 对齐 ckpt（`strict=True` 加载失败）

## `alignment_pairs.json` 与 schema

- 每个 `user_dense` fid 必须出现在某 pair 的 `dense_fids` 或 `dense_only_fids` 中，且维度之和等于 `user_dense_schema.total_dim`（启动时校验）。
- 每个 pair 的 `int_fids[k]` 必须出现在 `user_ns_groups` 的某个组里（否则 paired dense 不会进入 NS）。
- 按竞赛 `schema.json` 维护 JSON；与示例 schema 不一致时训练会在 `load_alignment_pairs` 阶段失败。
