#!/usr/bin/env bash
# Quick end-to-end check on Qwen-2.5-0.5B (ungated, ~1 GB): download, compress
# ONE layer, evaluate perplexity on a few windows and two zero-shot tasks on a
# small subset. The numbers are not comparable to the paper; the point is that
# every stage runs and produces a result file.
#
#   bash scripts/smoke_test.sh [model]              # default qwen2.5-0.5b; bf16 + senseed
#   SMOKE_SCHEMES="bf16 seedlm squant senseed" bash scripts/smoke_test.sh
#
# Compression is the slow step. SenSeed wants a GPU (DEVICES=auto); SeedLM and
# S-Quant run on CPU and are much slower even for one layer, so they are opt-in.
set -euo pipefail
cd "$(dirname "$0")/.."
M=${1:-qwen2.5-0.5b}   # any name from download_models.py --list
SCHEMES=${SMOKE_SCHEMES:-"bf16 senseed"}
# SenSeed compression: all CUDA GPUs if present, otherwise the (slow) numpy backend.
if [ -z "${DEVICES+x}" ]; then
  DEVICES=$(uv run python -c "import torch;print('auto' if torch.cuda.is_available() else '')")
fi
LAYERS=${LAYERS:-0}
mkdir -p results/smoke ckpt data
source scripts/_progress.sh
n=0; for s in $SCHEMES; do [ "$s" = bf16 ] && n=$((n+2)) || n=$((n+3)); done
progress_init "smoke-$M" "$n"

[ -d "models/$M" ] || uv run python scripts/download_models.py "$M"
[ -f data/wikitext2_test.txt ] || uv run python scripts/download_data.py

for s in $SCHEMES; do
  ck="ckpt/smoke-$M-$s"; dt=float16
  case $s in
    bf16)    ck="models/$M"; dt=bfloat16 ;;
    seedlm|squant)
             stage "$s: compress layers $LAYERS"
             uv run python experiments/compress_checkpoint.py --model "models/$M" \
               --out "$ck" --method "$s" --layers "$LAYERS" --resume ;;
    senseed) stage "senseed: compress layers $LAYERS"
             uv run python experiments/compress_checkpoint.py --model "models/$M" \
               --out "$ck" --method senseed --layers "$LAYERS" --devices "$DEVICES" --resume ;;
    *) echo "unknown scheme $s"; exit 1 ;;
  esac
  stage "$s: WikiText-2 perplexity"
  uv run python experiments/eval_hf.py --model "$ck" --text-file data/wikitext2_test.txt \
    --seqlen 2048 --windows 8 --dtype "$dt" --out "results/smoke/$s.ppl.json"
  stage "$s: zero-shot (arc_easy, boolq_aps)"
  BATCH=${BATCH:-8} DTYPE=$dt TASKS=arc_easy,boolq_aps bash scripts/run_lm_eval.sh "$ck" \
    "results/smoke/$s.lmeval.json" --limit 50
done
progress_done
echo "smoke results:"; ls results/smoke
