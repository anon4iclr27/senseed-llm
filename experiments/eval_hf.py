#!/usr/bin/env python3
"""WikiText-2 perplexity with stock transformers, on a checkpoint from
``compress_checkpoint.py``.

The numpy harness in ``eval_ppl.py`` is a from-scratch forward pass, and a
reviewer is entitled to distrust it: its fp16 number cannot be compared with a
published table unless the tokenisation, the window convention and the
label-shift all happen to match.  This runs the loop everybody else runs --
``AutoModelForCausalLM``, non-overlapping 2048-token windows over the
concatenated test split, mean cross-entropy -- so the fp16 row lands where the
literature's fp16 row lands, and the compressed rows are then comparable too.

    python experiments/eval_hf.py --model ckpt/llama2-7b-senseed \\
        --text-file wiki.txt --device cuda:0 --out results/hf_qwen7b_senseed.json

The convention is the one GPTQ introduced and everything since has copied: the
model is called with ``labels=input_ids``, so it scores ``seqlen - 1`` positions
per window and the per-window losses are averaged unweighted.  ``exact_nll``
below is the token-weighted version, which is what the numpy harness reports;
the two differ by well under a thousandth and both are printed so neither can
be quietly swapped for the other.

**This script needs the checkpoint to already be compressed.**  It applies
nothing itself -- that is the point of splitting the two steps, and it means
this file could be replaced by lm-evaluation-harness without changing any
result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import _bootstrap  # noqa: F401  (sys.path)


def load_text(args):
    if args.text_file:
        return open(os.path.expanduser(args.text_file), encoding="utf-8").read()
    from datasets import load_dataset
    ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split)
    return "\n\n".join(ds["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="",
                    help="default: --model (a compressed checkpoint carries "
                         "the original tokenizer)")
    ap.add_argument("--text-file", default="",
                    help="plain text, e.g. from scripts/make_wikitext.py.  "
                         "Preferred: it needs no network and pins the exact "
                         "bytes every arm saw.")
    ap.add_argument("--hf-dataset", default="wikitext")
    ap.add_argument("--hf-config", default="wikitext-2-raw-v1")
    ap.add_argument("--hf-split", default="test")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=0,
                    help="0 = every non-overlapping window in the text")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.path.expanduser(args.model)
    tok_path = os.path.expanduser(args.tokenizer or args.model)
    dtype = getattr(torch, args.dtype)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.eval().to(args.device)
    print(f"loaded {model_path} in {time.time() - t0:.0f}s, "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, "
          f"{args.dtype} on {args.device}", flush=True)

    ids = tok(load_text(args), return_tensors="pt").input_ids
    n = ids.numel() // args.seqlen
    if args.windows:
        n = min(n, args.windows)
    if n == 0:
        raise SystemExit(f"text gives {ids.numel()} tokens, fewer than one "
                         f"window of {args.seqlen}")
    print(f"{ids.numel()} tokens -> {n} windows of {args.seqlen}", flush=True)

    losses, t1 = [], time.time()
    with torch.no_grad():
        for i in range(n):
            batch = ids[:, i * args.seqlen:(i + 1) * args.seqlen].to(args.device)
            loss = model(batch, labels=batch).loss.float().item()
            losses.append(loss)
            print(f"  window {i}: ppl {math.exp(loss):.4f}", flush=True)

    # The convention: unweighted mean of per-window mean cross-entropy.
    nll = sum(losses) / n
    # Token-weighted, which is what eval_ppl.py reports.  Each window scores
    # seqlen - 1 positions, and every window has the same length, so these two
    # coincide here; they are both printed because they stop coinciding the
    # moment a partial window is allowed in.
    exact = sum(l * (args.seqlen - 1) for l in losses) / (n * (args.seqlen - 1))

    ppl = math.exp(nll)
    print(f"\n{os.path.basename(model_path)}  {n * args.seqlen} tokens  "
          f"cross-entropy {nll:.5f}  perplexity {ppl:.4f}  "
          f"({time.time() - t1:.0f}s)")
    print("compare the fp16 arm against the fp16 row of the table you are "
          "citing;\nif it does not land there, the tokenisation or the window "
          "convention differs\nand no compressed row below it is comparable "
          "either.")

    if args.out:
        json.dump(dict(config=vars(args), tokens=n * args.seqlen, nll=nll,
                       exact_nll=exact, ppl=ppl, windows_nll=losses,
                       gate_tally={}),
                  open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
