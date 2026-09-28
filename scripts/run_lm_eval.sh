#!/usr/bin/env bash
# Zero-shot accuracy with LM Evaluation Harness (v0.4.3, pinned in pyproject).
#   bash scripts/run_lm_eval.sh <model-or-checkpoint-dir> <out.json> [extra lm_eval args]
#   TASKS=arc_easy,boolq_aps overrides the task list.
set -euo pipefail
# Device: LM_DEVICE overrides; otherwise cuda if available, else mps, else cpu.
DEV=${LM_DEVICE:-$(uv run python -c "import torch;print('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')")}
uv run lm_eval --model hf \
  --model_args "pretrained=$1,dtype=${DTYPE:-float16}" --device "$DEV" \
  --tasks "${TASKS:-arc_easy,arc_challenge,hellaswag,winogrande,boolq_aps}" \
  --include_path lm_eval_tasks --num_fewshot 0 --batch_size "${BATCH:-auto}" --output_path "$2" "${@:3}"
