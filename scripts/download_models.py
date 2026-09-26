#!/usr/bin/env python3
"""Download the evaluated models into ./models/<name>/ (needs HF login for Llama).

    uv run python scripts/download_models.py                 # all four
    uv run python scripts/download_models.py llama-2-7b      # a subset
    uv run python scripts/download_models.py qwen2.5-0.5b   # smoke-test model (not in the default set)
"""
import argparse
import os

MODELS = {
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}
SMOKE = {"qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B"}   # ungated; used by scripts/smoke_test.sh
ALL = {**MODELS, **SMOKE}
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(MODELS))
    ap.add_argument("--out", default=ROOT)
    ap.add_argument("--list", action="store_true", help="print available model names and exit")
    args = ap.parse_args()
    if args.list:
        for n, r in ALL.items():
            print(f"{n:16s} {r}")
        return
    from huggingface_hub import snapshot_download
    for n in args.names:
        if n not in ALL:
            raise SystemExit(f"unknown model {n}; choose from {list(ALL)}")
        dst = os.path.join(args.out, n)
        print(f"{ALL[n]} -> {dst}")
        snapshot_download(ALL[n], local_dir=dst,
                          allow_patterns=["*.json", "*.safetensors", "tokenizer.model",
                                          "tokenizer*", "*.txt"],
                          ignore_patterns=["consolidated*", "original/*"])


if __name__ == "__main__":
    main()
