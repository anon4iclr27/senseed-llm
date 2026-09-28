#!/usr/bin/env python3
"""Compress a checkpoint once, on GPUs, and write it out as a checkpoint.

Until now compression and evaluation lived in the same process, so every
perplexity run paid for the seed search again.  That is backwards: the search
is ~90% of the cost and its output does not depend on the evaluation.  Splitting
them means the expensive step runs once per (model, method) and everything
afterwards -- more tokens, a different dataset, downstream tasks, a second look
in six months -- is a forward pass.

    python experiments/compress_checkpoint.py --model models/Llama-2-7b-hf \\
        --method senseed --devices cuda:0,cuda:1,cuda:2,cuda:3 \\
        --out ckpt/llama2-7b-senseed

The output directory is a drop-in checkpoint: the same shard layout, the same
index, the same dtypes, config and tokenizer copied through.
``AutoModelForCausalLM.from_pretrained`` loads it with no special casing, which
is the point -- the evaluation code then has nothing to do with this repository
and cannot be accused of flattering it.

**What gets compressed.**  The seven projections of every decoder layer.
Embeddings, the LM head and the norms stay in full precision, as in both prior
papers; the comparison is over the projection matrices only.

**Cost.**  The search is ~90% of it and scales with parameter count: on 12 CPU
cores Qwen-2.5-0.5B took 16.5 h and a 7B model would take about twelve days.
The torch backend moves exactly that loop to the device.  Run
``--method senseed --layers 0`` first and multiply: one layer tells you the whole
model to within a few percent, and it costs minutes.
"""
from __future__ import annotations

import os as _os

# One BLAS thread per worker.  These pools run 64-192 processes; if each one
# also opens a thread per core the machine spends its time in the scheduler.
# Must be set before numpy is imported, which is why it is above the imports.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import _bootstrap  # noqa: F401  (sys.path)

import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from measure_actscale import Shards                       # noqa: E402

from senseed import SEEDLM_4BIT, decompress                  # noqa: E402
from senseed.codec.allocate import (SenSeedConfig, compress_senseed,  # noqa: E402
                                 decompress_senseed)
from senseed.codec.squant import (SQuantConfig, compress_squant,  # noqa: E402
                                  decompress_squant)
from senseed.codec.prune import compress_pruned              # noqa: E402
from senseed.io import save_safetensors                      # noqa: E402
from senseed.sensitivity.actscale import (ArchSpec, attn_out_scale,  # noqa: E402
                                       mlp_act_scale, normed_scale)
from senseed.sensitivity.blockmap import blocks_from_scale   # noqa: E402
from senseed.sensitivity.metrics import informativeness      # noqa: E402

METHODS = ("fp16", "seedlm", "squant", "senseed")
PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
COPY = ("config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
        "merges.txt", "tokenizer.model", "model.safetensors.index.json",
        "added_tokens.json", "chat_template.jinja")

# Default S-Quant point: B=16, S=16, G=8, R_th=0.90.  Override with --squant-rth.
SQUANT_CFG = SQuantConfig()

_DEV = {"value": "numpy"}


def _init(devices):
    """Pin this worker to one device, round robin by process identity."""
    import multiprocessing as mp
    n = mp.current_process()._identity
    i = (n[0] - 1) if n else 0
    _DEV["value"] = devices[i % len(devices)]


def _compress(spec):
    """Compress one tensor, or one horizontal slice of one.

    Blocks live inside a row, so splitting on rows is exact -- no block
    straddles two slices and the result is identical to compressing the whole
    tensor.  It matters because the CPU arms get one core per job: an MLP
    tensor of a 7B model is 68M weights, which is **8.4 hours on one core**, so
    a whole-tensor job makes a shard take as long as its largest tensor however
    many workers are free.  ``eval_ppl.py`` has always split rows; this did not,
    and that was the difference between 3 hours and 34.
    """
    name, part, W, method, a, slope, gate, backend_kw = spec
    W = np.ascontiguousarray(W, np.float32)
    dev = _DEV["value"]
    t0 = time.time()

    if method == "fp16":
        hat = W.astype(np.float16).astype(np.float32)
    elif method == "squant":
        # S-Quant (Wang et al., 2026), reimplemented in senseed/codec/squant.py.
        hat = np.asarray(decompress_squant(compress_squant(W, SQUANT_CFG)),
                         np.float32)
    elif method == "seedlm":
        hat = np.asarray(decompress(compress_pruned(W, SEEDLM_4BIT)), np.float32)
    elif method == "senseed":
        if a is None or (gate and informativeness(a)["top1pct_share"] < gate):
            hat = np.asarray(decompress(compress_pruned(W, SEEDLM_4BIT)),
                             np.float32)
        else:
            order = np.argsort(-a, kind="stable")
            inv = np.argsort(order)
            cfg = SenSeedConfig(B=8, S=16, k_mode="schedule", k_min=2, k_max=6,
                             target_bits=4.0, exp_group=1,
                             schedule_slope=slope)
            res = compress_senseed(np.ascontiguousarray(W[:, order], np.float32),
                                cfg,
                                importance=blocks_from_scale(a[order],
                                                             W.shape[0]),
                                backend=dev, **backend_kw)
            hat = np.asarray(decompress_senseed(res), np.float32)[:, inv]
    else:
        raise ValueError(method)
    return name, part, hat, time.time() - t0, dev


def _hms(sec):
    sec = int(sec)
    return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


class _Tee:
    """Mirror stdout to a log file so a run can be followed with `tail -f`."""

    def __init__(self, stream, path):
        self.stream, self.f = stream, open(path, "a", buffering=1)

    def write(self, x):
        self.stream.write(x)
        self.f.write(x)

    def flush(self):
        self.stream.flush()
        self.f.flush()


def _heartbeat(state, every):
    """Print a line every `every` s so a long job never looks hung."""
    import threading
    stop = threading.Event()

    def run():
        while not stop.wait(every):
            print(f"  ... {_hms(time.time() - state['t0'])} elapsed, "
                  f"{state['done']}/{state['total']} jobs done", flush=True)
    threading.Thread(target=run, daemon=True).start()
    return stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="senseed", choices=METHODS)
    ap.add_argument("--slope", type=float, default=1.0)
    ap.add_argument("--squant-rth", type=float, default=0.90,
                    help="S-Quant explained-energy threshold R_th")
    ap.add_argument("--gate", type=float, default=0.0,
                    help="apply SenSeed only where informativeness(a) reaches "
                         "this; 0 = off.  See experiments/gate_census.py.")
    ap.add_argument("--layers", default="", help="default: all")
    ap.add_argument("--devices", default="",
                    help="comma-separated torch devices, e.g. cuda:0,cuda:1, "
                         "or 'auto' for every visible CUDA device.  Empty = "
                         "the numpy backend on CPU.")
    ap.add_argument("--workers", type=int, default=0,
                    help="GPU pool size; default one per device")
    ap.add_argument("--cpu-workers", type=int, default=0,
                    help="pool size for the tensors that never touch a device "
                         "-- every arm except SenSeed's, and SenSeed's own gated-out "
                         "tensors.  Default min(nproc, 64).")
    ap.add_argument("--block-chunk", type=int, default=0,
                    help="0 = size it from the smallest device's memory")
    ap.add_argument("--seed-chunk", type=int, default=8192)
    ap.add_argument("--row-target", type=float, default=2e6,
                    help="weights per CPU job.  Blocks live inside a row so a "
                         "row split is exact; this only decides how finely the "
                         "work is spread over the pool.")
    ap.add_argument("--stats-only", action="store_true",
                    help="compress and time, write no checkpoint.  For "
                         "calibration: writing a 7B shard costs minutes of "
                         "disk for a measurement that does not need it.")
    ap.add_argument("--dtype", default="source",
                    choices=["source", "F16", "BF16", "F32"])
    ap.add_argument("--resume", action="store_true",
                    help="skip shards already written to --out")
    ap.add_argument("--log", default="",
                    help="log file (default: <out>/compress.log)")
    ap.add_argument("--heartbeat", type=float, default=30,
                    help="seconds between 'still running' lines (0 = off)")
    args = ap.parse_args()
    os.makedirs(os.path.expanduser(args.out), exist_ok=True)
    sys.stdout = _Tee(sys.stdout, args.log or
                      os.path.join(os.path.expanduser(args.out), "compress.log"))
    print(f"\n=== {time.strftime('%F %T')} {' '.join(sys.argv[1:])}", flush=True)
    global SQUANT_CFG
    SQUANT_CFG = SQuantConfig(R_th=args.squant_rth)

    model = os.path.expanduser(args.model)
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    devices = [d.strip() for d in args.devices.split(",") if d.strip()] \
        or ["numpy"]
    block_chunk = args.block_chunk or 4096
    if devices != ["numpy"]:
        from senseed.codec.gpu import device_report, resolve_devices
        print(device_report(), flush=True)
        devices, block_chunk = resolve_devices(devices, args.seed_chunk,
                                               args.block_chunk)
    workers = args.workers or (len(devices) if devices != ["numpy"]
                               else max(1, os.cpu_count() or 1))
    cpu_workers = args.cpu_workers or min(max(1, os.cpu_count() or 1), 64)
    backend_kw = dict(block_chunk=block_chunk, seed_chunk=args.seed_chunk) \
        if devices != ["numpy"] else {}
    if devices != ["numpy"]:
        print(f"{len(devices)} device(s), {workers} gpu worker(s), "
              f"{cpu_workers} cpu worker(s), block_chunk={block_chunk}, "
              f"seed_chunk={args.seed_chunk}", flush=True)

    cfg = json.load(open(os.path.join(model, "config.json")))
    arch = ArchSpec.from_config(cfg)
    sh = Shards(model)
    todo = set(int(x) for x in args.layers.split(",")) if args.layers \
        else set(range(cfg["num_hidden_layers"]))

    # sensitivity vectors, one pass, before anything is compressed
    a_of = {}
    if args.method == "senseed":
        for L in todo:
            p = f"model.layers.{L}."
            g_in = sh.get(p + "input_layernorm.weight")
            g_po = sh.get(p + "post_attention_layernorm.weight")
            if g_in is None or g_po is None:
                continue
            a_of[p + "self_attn.q_proj.weight"] = normed_scale(g_in)
            a_of[p + "self_attn.k_proj.weight"] = normed_scale(g_in)
            a_of[p + "self_attn.v_proj.weight"] = normed_scale(g_in)
            a_of[p + "self_attn.o_proj.weight"] = attn_out_scale(
                sh.get(p + "self_attn.v_proj.weight"), g_in, arch)
            a_of[p + "mlp.gate_proj.weight"] = normed_scale(g_po)
            a_of[p + "mlp.up_proj.weight"] = normed_scale(g_po)
            a_of[p + "mlp.down_proj.weight"] = mlp_act_scale(
                sh.get(p + "mlp.gate_proj.weight"),
                sh.get(p + "mlp.up_proj.weight"), g_po)
        print(f"sensitivity for {len(a_of)} tensors", flush=True)

    def is_target(name):
        if not name.endswith(".weight") or not name.startswith("model.layers."):
            return False
        try:
            L = int(name.split(".")[2])
        except (IndexError, ValueError):
            return False
        return L in todo and any(name.endswith(p + ".weight") for p in PROJ)

    t_all = time.time()
    stats = []
    for fn in sorted(sh._hdr):
        dst = os.path.join(out, fn)
        if args.resume and os.path.exists(dst):
            print(f"skip {fn} (exists)", flush=True)
            continue
        hdr = sh._hdr[fn]
        names = [k for k in hdr if k != "__metadata__"]
        # Split by who actually does the work.  Only SenSeed's schedule runs on a
        # device; SeedLM's pruned search, block floating point and the float16
        # cast are numpy, and giving them one worker per GPU would run a
        # 256-core machine four cores wide.  Measured on Qwen-2.5-7B that is the
        # difference between 3 hours and 8 days for the seedlm arm.
        gpu_jobs, cpu_jobs, nparts = [], [], {}
        for k in names:
            if not is_target(k):
                continue
            a = a_of.get(k)
            W = sh.get(k)
            on_gpu = (args.method == "senseed" and devices != ["numpy"]
                      and a is not None
                      and (not args.gate
                           or informativeness(a)["top1pct_share"] >= args.gate))
            if on_gpu:
                # A device chews a whole tensor efficiently -- 555 s for a 7B
                # down_proj -- and there are 7 per layer, so the four workers
                # stay fed without slicing.
                nparts[k] = 1
                gpu_jobs.append((k, 0, W, args.method, a, args.slope,
                                 args.gate, backend_kw))
                continue
            # A CPU job gets one core.  Slice it so no single job is longer
            # than the pool can hide, and so the pool has enough pieces to
            # balance: whole 7B MLP tensors are 8.4 h each on one core.
            rows = np.array_split(np.arange(W.shape[0]),
                                  max(1, min(W.shape[0],
                                             round(W.size / args.row_target))))
            nparts[k] = len(rows)
            for i, r in enumerate(rows):
                if len(r):
                    cpu_jobs.append((k, i, W[r], args.method, a, args.slope,
                                     args.gate, {}))
        jobs = gpu_jobs + cpu_jobs
        # Under --stats-only nothing is written, so the tensors that would only
        # have been copied are never read: on a 7B shard that is 8 GB of host
        # memory and a few minutes of disk, spent to learn nothing.
        tensors = {} if args.stats_only else \
            {k: sh.get(k) for k in names if not is_target(k)}
        dtypes = {k: (hdr[k]["dtype"] if args.dtype == "source" else args.dtype)
                  for k in names}
        print(f"\n{fn}: {len(gpu_jobs)} on device + {len(cpu_jobs)} on cpu"
              f"{'' if args.stats_only else f', {len(names) - len(jobs)} copied'}",
              flush=True)
        if args.stats_only and not jobs:
            continue

        import multiprocessing as _mp
        done, parts = 0, {}
        state = dict(t0=time.time(), done=0, total=len(jobs))
        hb = _heartbeat(state, args.heartbeat) if args.heartbeat > 0 else None
        for pool_jobs, pool_devices, n, kind in (
                (gpu_jobs, devices, workers, "spawn"),
                (cpu_jobs, ["numpy"], cpu_workers, "fork")):
            if not pool_jobs:
                continue
            # CUDA cannot be re-initialised in a forked child, and the parent
            # has already touched it via device_report().  "spawn" starts each
            # worker from a clean interpreter, which costs a second or two per
            # shard and is the difference between working and not.  The numpy
            # pool has no such constraint and forking is cheaper.
            ctx = _mp.get_context(kind)
            with ProcessPoolExecutor(n, mp_context=ctx, initializer=_init,
                                     initargs=(pool_devices,)) as ex:
                for k, part, hat, dt, dev in ex.map(_compress, pool_jobs):
                    parts.setdefault(k, {})[part] = hat
                    done += 1
                    state["done"] = done
                    stats.append(dict(name=k, part=part, seconds=dt,
                                      device=dev))
                    if nparts[k] == 1 or len(parts[k]) == nparts[k]:
                        tensors[k] = parts.pop(k)[0] if nparts[k] == 1 else \
                            np.concatenate([parts[k][i]
                                            for i in range(nparts[k])], 0)
                        parts.pop(k, None)
                    print(f"  [{done}/{len(jobs)}] {k}"
                          f"{'' if nparts[k] == 1 else f' [{part + 1}/{nparts[k]}]'}"
                          f"  {dt:7.1f}s on {dev}"
                          f"  [{_hms(time.time() - state['t0'])} elapsed, ETA "
                          f"{_hms((time.time() - state['t0']) / done * (len(jobs) - done))}]",
                          flush=True)
        if hb:
            hb.set()

        if args.stats_only:
            del tensors
            continue
        save_safetensors(dst, {k: tensors[k] for k in names}, dtypes)
        print(f"wrote {dst}", flush=True)

    if not args.stats_only:
        for f in COPY:
            src = os.path.join(model, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out, f))

    total = sum(s["seconds"] for s in stats)
    json.dump(dict(config=vars(args), tensors=stats, wall=time.time() - t_all,
                   core_seconds=total),
              open(os.path.join(out, "compression_stats.json"), "w"), indent=1)
    print(f"\n{len(stats)} tensors compressed, {total / 3600:.2f} device-hours, "
          f"{(time.time() - t_all) / 3600:.2f} h wall")
    print(f"checkpoint at {out} -- evaluate it with experiments/eval_hf.py")


if __name__ == "__main__":
    main()
