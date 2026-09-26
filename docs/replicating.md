# Replicating the paper's runs

Everything below is run from the repository root. The commands have not been
executed end to end on the full-size models; see "Status" at the bottom before
relying on a step.

## 0. Hardware and time
* Evaluation (perplexity, zero-shot) needs one GPU with enough memory for the
  model in fp16 (about 14 GB for a 7B model, 26 GB for Llama-2-13B).
* Compression dominates the cost. The seed search is roughly 90% of it. SeedLM
  and S-Quant run on CPU pools (tens of CPU-hours per billion weights); SenSeed
  runs on GPUs (`--devices auto`) and takes several GPU-hours per 7B model.
  Measure first (section 5) and extrapolate before committing a machine.
* Disk: each compressed checkpoint is about the size of the fp16 model
  (13 GB for 7B). `ckpt/` is git-ignored.

## 1. Environment
```bash
bash setup.sh                           # installs uv if needed, creates .venv
uv run huggingface-cli login            # Llama models are gated: accept licences first
uv run pytest -q                        # about one minute, CPU only
```

## 1b. Smoke test (Qwen-2.5-0.5B)
```bash
bash scripts/smoke_test.sh        # downloads Qwen-2.5-0.5B (ungated) and WikiText-2
```
Compresses layer 0 only, then runs perplexity on 8 windows and ARC-Easy/BoolQ
on 50 examples each, for the BF16 and SenSeed arms (`SMOKE_SCHEMES="bf16 seedlm
squant senseed"` adds the CPU arms, which are much slower). Outputs go to
`results/smoke/`. The numbers are not comparable to the paper. Not yet executed.

## 2. Models and data
```bash
uv run python scripts/download_models.py          # -> models/{llama-2-7b,llama-2-13b,llama-3-8b,mistral-7b}
uv run python scripts/download_data.py            # -> data/wikitext2_test.txt + cached zero-shot datasets
```
Models: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`,
`meta-llama/Meta-Llama-3-8B`, `mistralai/Mistral-7B-v0.1`. Pass names to
download a subset, e.g. `download_models.py llama-2-7b`.

## 3. Metrics (the paper's protocol)
| Metric | Command | Settings |
|---|---|---|
| WikiText-2 perplexity | `experiments/eval_hf.py` | test split, raw text joined by blank lines, 2048-token non-overlapping windows |
| Zero-shot accuracy | `scripts/run_lm_eval.sh` | LM Evaluation Harness 0.4.3, tasks `arc_easy, arc_challenge, hellaswag, winogrande, boolq`, 0-shot |

Single-checkpoint examples (any Hugging Face-format directory works):
```bash
uv run python experiments/eval_hf.py --model models/llama-2-7b \
    --text-file data/wikitext2_test.txt --seqlen 2048 --dtype bfloat16 \
    --out results/llama-2-7b-bf16.ppl.json
DTYPE=bfloat16 bash scripts/run_lm_eval.sh models/llama-2-7b results/llama-2-7b-bf16.lmeval.json
```
The perplexity JSON holds the token count, mean NLL and perplexity. Use the
same `--text-file` for every arm: the token count must match across arms.
Compressed checkpoints are stored as fp16 weights (reconstructed values), so
evaluate them with `--dtype float16` / `DTYPE=float16`.

## 4. Table 1 / Table 2 arms
### 4.1 Unquantized reference
Evaluate `models/<name>` as in section 3 (paper reports the FP16/BF16 baseline).

### 4.2 SeedLM (our implementation of the published method)
4-bit point `(S,k)=(16,3)`, block size 8, 4-bit mantissas and exponent.
```bash
uv run python experiments/compress_checkpoint.py --model models/llama-2-7b \
    --out ckpt/llama-2-7b-seedlm --method seedlm --resume
```
then evaluate `ckpt/llama-2-7b-seedlm` (section 3). `--resume` skips shards that
are already written, so an interrupted run can be restarted.

### 4.3 S-Quant (our implementation)
The paper's S-Quant rows are the published numbers; the implementation here is
used for the weight-space comparison (reconstruction error) and can also be
written out as a checkpoint:
```bash
uv run python experiments/compress_checkpoint.py --model models/llama-2-7b \
    --out ckpt/llama-2-7b-squant --method squant --squant-rth 0.90 --resume
```
Adjust `--squant-rth` to sweep the explained-energy threshold.

### 4.4 SenSeed, 4.0 bits/weight ("Mode 2")
```bash
uv run python experiments/compress_checkpoint.py --model models/llama-2-7b \
    --out ckpt/llama-2-7b-senseed --method senseed --slope 1.0 \
    --devices auto --resume
```
This computes the data-free importance from the checkpoint (RMSNorm gains,
one propagation hop for `o_proj`/`down_proj`), allocates `k` per block at a
mean rate of 4.000 bits/weight, and writes a standard checkpoint.

### 4.5 SenSeed at a lower rate ("Mode 1", e.g. 3.77 bits)
This uses the rate-distortion allocation over the `(S,k)` grid and is a
multi-step procedure:
1. Build a per-`k` cache of compressed weights for the grid points with
   `scripts/seed_schedule.py --model models/<name> --out ckpt/... --cache ...`
   (see `--help`; `--seed-bits`, `--k`, `--grid`, `--devices`).
2. Assemble uniform arms with `scripts/make_uniform.py --points S14k4,S16k4,...`
   and evaluate each with `eval_hf.py`. Their excess loss over the fp16 arm is
   the damage curve `e(g)`, stored as `results/<prefix><k>_*-S<S>.json`.
3. Allocate and write the checkpoint at a target rate:
   ```bash
   uv run python scripts/joint_allocate.py --model models/llama-2-7b \
       --fp16 results/fp16.json --results results --surface-prefix ev7b_k \
       --target 3.77 --out ckpt/llama-2-7b-senseed-3p77
   uv run python scripts/joint_allocate.py ... --dry-run   # report allocation only
   ```
4. Evaluate the written checkpoint as in section 3.

### 4.6 All arms for one model
```bash
bash scripts/run_pipeline.sh llama-2-7b                # bf16 seedlm squant senseed
bash scripts/run_pipeline.sh llama-2-7b seedlm         # a subset
DEVICES=cuda:0,cuda:1 bash scripts/run_pipeline.sh llama-3-8b senseed
```
Results: `results/<model>-<scheme>.ppl.json` and `.lmeval.json`. An arm whose
perplexity JSON exists is skipped, so the script is restartable. Repeat for
`llama-2-13b`, `llama-3-8b` and `mistral-7b`.

## 5. Cost check before a full run
Compress a single layer and time it:
```bash
uv run python experiments/compress_checkpoint.py --model models/llama-2-7b \
    --out ckpt/_calib --method senseed --layers 0 --devices auto --stats-only
```
`ckpt/_calib/compression_stats.json` gives device-seconds per tensor; multiply
by the number of layers for the full model.

## 6. Not covered by this release
* AWQ and S-Quant numbers in the paper's tables are the published values; no AWQ
  code is included.
* The bit-flip robustness study and the ASIC hardware evaluation are not part of
  this release.
* Zero-shot accuracy for SeedLM in the paper comes from the implementation here;
  the published SeedLM numbers used in comparisons are quoted, not recomputed.

## Status (read before relying on this)
* Verified: the unit tests, and encode/decode of all three codecs on small random
  matrices.
* Not verified: any run on real checkpoints, `setup.sh`, the download scripts,
  `run_lm_eval.sh`, `run_pipeline.sh`, and the checkpoint writer including the
  `squant` method.
* This code snapshot predates parts of the paper's SenSeed method: the FP16
  outlier columns (paper section 3.5), the output-side factor and importance
  floor (section 3.3) are not implemented, and `--method senseed` uses the
  k-schedule rather than the hull allocation. Numbers from these commands may
  therefore differ from the paper's tables.
