#!/bin/bash
# All script/*.py live in one flat directory on the platform (no subfolders).
cd "$(dirname "$0")" 2>/dev/null || true
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python3 -u train.py \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --use_request_time_ns \
    --use_int_dense_alignment \
    --alignment_pairs_json "${PWD}/alignment_pairs.json" \
    --use_request_social_calendar \
    --social_calendar_table_path "${PWD}/cn_request_calendar.csv" \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --batch_size 1024 \
    "$@"
