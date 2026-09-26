#!/usr/bin/env bash
# Zero-shot accuracy with LM Evaluation Harness (v0.4.3, pinned in pyproject).
#   bash scripts/run_lm_eval.sh <model-or-checkpoint-dir> <out.json> [extra lm_eval args]
#   TASKS=arc_easy,boolq overrides the task list.
set -euo pipefail
uv run lm_eval --model hf \
  --model_args "pretrained=$1,dtype=${DTYPE:-float16}" \
  --tasks "${TASKS:-arc_easy,arc_challenge,hellaswag,winogrande,boolq}" \
  --num_fewshot 0 --batch_size auto --output_path "$2" "${@:3}"
