#!/usr/bin/env python3
"""Assemble evaluable uniform (S,k) arms from the caches.

The band sweep runs on the damage surface e(S,k), and at 7B that surface barely
exists: a cache holds compressed weights, not a checkpoint you can hand to the
evaluator.  This is the one step in between -- un-permute the columns and write
a full model -- done for a list of points and nothing else.  No schedule, no
allocation, no Lagrangian.

**Columns matter.**  Every cache in this project was built on sensitivity-sorted
columns.  An arm assembled without the inverse permutation still loads, still
runs, and still prints a perplexity; it is simply the wrong model.  That failure
is silent, which is why the gather phase is paid here rather than skipped.

    python make_uniform.py --model ~/data/models/llama-2-7b \
        --cache-fmt '~/ckpt/ss7b_k{k}_cache' \
        --points S14k4,S16k4,S12k5,S14k5,S16k5 \
        --out-fmt '~/ckpt/u7b_{tag}'
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _p in (os.path.join(HERE, "scripts"), os.path.join(HERE, "experiments")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from safetensors import safe_open                                  # noqa: E402
from seed_schedule import sensitivity, PROJ, ATTN                  # noqa: E402
from measure_actscale import Shards                                # noqa: E402
from senseed.sensitivity.actscale import ArchSpec                     # noqa: E402

rate = lambda S, k, B=8: (S + 4 * k + 4) / B                       # noqa: E731
parse = lambda t: (int(t.split("k")[0][1:]), int(t.split("k")[1]))  # noqa: E731


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-fmt", required=True,
                    help="quote it: '~/ckpt/ss7b_k{k}_cache'")
    ap.add_argument("--out-fmt", required=True,
                    help="quote it: '~/ckpt/u7b_{tag}'")
    ap.add_argument("--points", required=True, help="S14k4,S16k4,...")
    ap.add_argument("--block", type=int, default=8)
    a_ = ap.parse_args()

    model = os.path.expanduser(a_.model)
    B = a_.block
    pts = [parse(t.strip()) for t in a_.points.split(",") if t.strip()]

    # Refuse before the 12-minute gather, not after it.
    missing = []
    for S, k in pts:
        f = os.path.join(os.path.expanduser(a_.cache_fmt.format(k=k)),
                         f"S{S}.safetensors")
        if not os.path.exists(f):
            missing.append(f)
    if missing:
        raise SystemExit("no cache for:\n  " + "\n  ".join(missing))

    sh = Shards(model)
    arch = ArchSpec.from_config(json.load(open(os.path.join(model, "config.json"))))
    order_of, shape_of, size_of = {}, {}, {}
    L = 0
    while True:
        av = sensitivity(sh, arch, L)
        if av is None:
            break
        p = f"model.layers.{L}."
        for proj in PROJ:
            n = p + ("self_attn." if proj in ATTN else "mlp.") + proj + ".weight"
            W = sh.get(n)
            if W is None:
                continue
            a = np.asarray(av[proj], np.float64)
            if len(a) != W.shape[1]:
                continue
            order_of[n] = np.argsort(-a, kind="stable")
            shape_of[n], size_of[n] = W.shape, int(W.size)
        L += 1
    names = sorted(order_of)
    print(f"{len(names)} linear tensors, "
          f"{sum(size_of.values())/1e6:.0f}M weights\n", flush=True)

    import stbf as stn                                   # noqa: F401
    import safetensors.torch as stt
    import torch

    for S, k in pts:
        tag = f"S{S}k{k}"
        out = os.path.expanduser(a_.out_fmt.format(tag=tag))
        if os.path.exists(os.path.join(out, "model.safetensors")):
            print(f"{tag} exists, skipping ({out})", flush=True)
            continue
        os.makedirs(out, exist_ok=True)
        h = safe_open(os.path.join(
            os.path.expanduser(a_.cache_fmt.format(k=k)),
            f"S{S}.safetensors"), framework="numpy")
        src = {}
        for f in sorted(glob.glob(os.path.join(model, "*.safetensors"))):
            src.update(stt.load_file(f))
        for i_n, n in enumerate(names, 1):
            if i_n % 40 == 0 or i_n == len(names):
                print(f"  {tag} [{i_n}/{len(names)}]", flush=True)
            nb = -(-size_of[n] // B)
            t = h.get_tensor(n).astype(np.float32).reshape(-1)
            if len(t) < nb * B:
                t = np.concatenate([t, np.zeros(nb * B - len(t), np.float32)])
            inv = np.argsort(order_of[n])
            v = np.ascontiguousarray(
                t[:size_of[n]].reshape(shape_of[n])[:, inv])
            src[n] = torch.from_numpy(v).to(src[n].dtype)
        stt.save_file(src, os.path.join(out, "model.safetensors"),
                      metadata={"format": "pt"})
        for fn in os.listdir(model):
            if fn.endswith(".safetensors") or fn.endswith(".index.json"):
                continue
            s = os.path.join(model, fn)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(out, fn))
        json.dump(dict(point=tag, S=S, k=k, uniform=True,
                       mean_rate=rate(S, k)),
                  open(os.path.join(out, "arm.json"), "w"), indent=1)
        print(f"  {tag}  {rate(S,k):.3f} b/w  -> {out}\n", flush=True)


if __name__ == "__main__":
    main()
