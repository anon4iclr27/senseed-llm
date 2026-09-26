# SenSeed: sensitivity-aware seed-based weight compression for LLMs

Anonymous code release for double-blind review. Licensed under Apache-2.0 (see LICENSE).

SenSeed compresses LLM weights into LFSR seeds (as SeedLM and S-Quant do) but
allocates bits by weight *sensitivity*, estimated from the checkpoint alone
(no calibration data, no per-block side information). See
[docs/schemes.md](docs/schemes.md) for how each compared scheme is implemented
(Unquantized BF16, SeedLM, S-Quant, SenSeed).

## Setup
```bash
bash setup.sh                       # installs uv, creates .venv 
uv run huggingface-cli login        # Llama models are gated; accept their licences first
uv run python scripts/download_models.py   # -> ./models/{llama-2-7b,llama-2-13b,llama-3-8b,mistral-7b}
uv run python scripts/download_data.py     # WikiText-2 -> ./data, caches ARC-E/C, HellaSwag, WinoGrande, BoolQ
```

## Run
Choose the model by name (`uv run python scripts/download_models.py --list` shows them):
```bash
bash scripts/run_pipeline.sh llama-2-7b            # all four schemes: PPL + zero-shot
bash scripts/run_pipeline.sh llama-2-7b seedlm     # one scheme
```
Results land in `results/`. Compressing a 7B model with the seed search needs
GPUs (`DEVICES=auto`) and many hours; SeedLM/S-Quant run on CPU pools.

## Smoke tests
```bash
uv run pytest -q                 # unit tests, CPU, ~1 min
bash scripts/smoke_test.sh       # end-to-end on Qwen-2.5-0.5B (one layer, small eval)
```

## Layout
See [docs/replicating.md](docs/replicating.md) for how to reproduce the paper runs step by step.

`senseed/` codecs and data-free sensitivity · `experiments/` checkpoint
compression and perplexity evaluation · `scripts/` setup, lm-eval,
allocation tools · `docs/` implementation notes · `tests/`.
