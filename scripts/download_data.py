#!/usr/bin/env python3
"""Fetch WikiText-2 (perplexity) and cache the lm-eval-harness zero-shot tasks.

Writes data/wikitext2_test.txt (test split, raw, joined with blank lines --
the exact text every arm is scored on) and warms the Hugging Face datasets cache
for ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande and BoolQ so evaluation can
run offline afterwards.
"""
import os

from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data")

# (dataset, config) as used by lm-eval-harness v0.4.3
TASKS = {
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge"),
    "hellaswag": ("Rowan/hellaswag", None),
    "winogrande": ("allenai/winogrande", "winogrande_xl"),
    "boolq": ("aps/super_glue", "boolq"),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    path = os.path.join(OUT, "wikitext2_test.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(ds["text"]))
    print(f"wikitext-2 test -> {path}")
    for task, (name, cfg) in TASKS.items():
        print(f"caching {task} ({name}{'/' + cfg if cfg else ''})")
        load_dataset(name, cfg) if cfg else load_dataset(name)


if __name__ == "__main__":
    main()
