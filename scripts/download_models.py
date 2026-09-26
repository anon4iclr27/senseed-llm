#!/usr/bin/env python3
"""Download the evaluated models into ./models/<name>/ (needs HF login for Llama).

    uv run python scripts/download_models.py                 # all four
    uv run python scripts/download_models.py llama-2-7b      # a subset
"""
import argparse
import os

from huggingface_hub import snapshot_download

MODELS = {
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(MODELS))
    ap.add_argument("--out", default=ROOT)
    args = ap.parse_args()
    for n in args.names:
        if n not in MODELS:
            raise SystemExit(f"unknown model {n}; choose from {list(MODELS)}")
        dst = os.path.join(args.out, n)
        print(f"{MODELS[n]} -> {dst}")
        snapshot_download(MODELS[n], local_dir=dst,
                          allow_patterns=["*.json", "*.safetensors", "tokenizer.model",
                                          "tokenizer*", "*.txt"],
                          ignore_patterns=["consolidated*", "original/*"])


if __name__ == "__main__":
    main()
