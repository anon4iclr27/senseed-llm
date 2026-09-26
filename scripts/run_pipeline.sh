#!/usr/bin/env bash
# One model, all four schemes: compress -> WikiText-2 PPL -> zero-shot tasks.
#   bash scripts/run_pipeline.sh llama-2-7b [schemes...]
# schemes default to: bf16 seedlm squant senseed.  Restartable: an arm whose
# results/<model>-<scheme>.ppl.json exists is skipped.
set -euo pipefail
M=${1:?model name under models/}; shift || true
SCHEMES=${*:-"bf16 seedlm squant senseed"}
DEVICES=${DEVICES:-auto}; TEXT=data/wikitext2_test.txt
mkdir -p results ckpt
for s in $SCHEMES; do
  tag="$M-$s"; ck="ckpt/$tag"; res="results/$tag.ppl.json"
  [ -f "$res" ] && { echo "skip $tag"; continue; }
  case $s in
    bf16)   ck="models/$M"; dt=bfloat16 ;;
    seedlm|squant)
            uv run python experiments/compress_checkpoint.py --model "models/$M" \
              --out "$ck" --method "$s" --resume; dt=float16 ;;
    senseed) uv run python experiments/compress_checkpoint.py --model "models/$M" \
              --out "$ck" --method senseed --devices "$DEVICES" --resume; dt=float16 ;;
    *) echo "unknown scheme $s"; exit 1 ;;
  esac
  uv run python experiments/eval_hf.py --model "$ck" --text-file "$TEXT" \
    --seqlen 2048 --dtype "$dt" --out "$res"
  DTYPE=$dt bash scripts/run_lm_eval.sh "$ck" "results/$tag.lmeval.json"
done
