#!/usr/bin/env python3
"""Seed-bit scheduling: how much storage comes off, and what it costs.

Standalone on purpose -- it imports from ``senseed`` and modifies nothing, so it
can be dropped into a checkout another machine is mid-run against.

**The claim it is built to measure.**  The reference is a uniform 16-bit seed,
which is what SeedLM specifies and what every number in this project so far
used.  At ``k=3, B=8`` that is 4.000 bits/weight, and the seed alone is 2.000 of
them -- half the budget.  The proposal is to keep 16 bits only where the loss is
sensitive and spend fewer elsewhere, so the *mean* seed width drops and with it
the rate.  What gets reported is therefore **storage saved against S=16, and the
perplexity paid for it** -- not accuracy at a matched rate.

    S=16 uniform   4.000 bits/weight     reference
    S_mean = 14    3.750                 6.25% smaller
    S_mean = 12    3.500                12.5%
    S_mean = 10    3.250                18.75%

Each rate also gets a **uniform** arm at the same mean width, because the null
hypothesis is "just shorten every seed" and it has to be excluded.

**Why it needs no codec change.**  Blocks are independent, so compressing the
tensor once per seed width and then taking each block from the width its
schedule assigned is exactly per-block S.  Compression happens once per width
into a cache; every arm is assembled from that cache, so adding arms is free and
the run is restartable.

    python seed_schedule.py --model ~/data/models/qwen2.5-0.5b \\
        --out ~/ckpt/seedsched --seed-bits 14,12,10 --cpu-workers 96

    # same thing on four GPUs, ~100x faster per unit -- see --devices
    python seed_schedule.py --model ~/data/models/llama-2-7b \\
        --out ~/ckpt/ss7b --seed-bits 14,12,10 \\
        --devices cuda:0,cuda:1,cuda:2,cuda:3

Signalling stays at zero: the decoder holds gamma, recomputes the same schedule,
and knows how many bits to read for each block before it reads them.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "scripts"), os.path.join(HERE, "experiments")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from measure_actscale import Shards                              # noqa: E402
from senseed.sensitivity.actscale import (ArchSpec, attn_out_scale,  # noqa: E402
                                       mlp_act_scale, normed_scale)
from senseed.codec.seedlm import SeedLMConfig, compress, decompress  # noqa: E402

PROJ = ["q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"]
ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}

# None = the default start method (fork on Linux).  main() switches this to
# "spawn" when --devices is given; see the pool in phase 1 for why.
_POOL_CTX = None


def block_importance(a, rows, B):
    """Mean sensitivity per block, in the codec's flatten order.

    Computed here rather than imported so the script does not depend on which
    revision of ``blockmap.py`` the checkout happens to sit at.
    """
    a = np.asarray(a, np.float64).ravel()
    full = np.tile(a, rows)
    pad = (-len(full)) % B
    if pad:
        full = np.concatenate([full, np.zeros(pad)])
    m = full.reshape(-1, B).mean(1)
    return m / m.mean() if m.mean() > 0 else np.ones_like(m)


def schedule_S(imp, Sbar, slope, grid):
    """Seed width per block: affine in log importance, mean pinned to ``Sbar``.

    Snapped to ``grid`` because each distinct width costs one compression pass.
    The offset is bisected so the mean survives the snapping and the rate is hit
    exactly -- the same device the basis schedule uses, for the same reason.
    """
    g = np.asarray(sorted(grid), float)
    z = np.log(np.maximum(imp, 1e-300))
    base = Sbar + slope * (z - z.mean())

    # Nearest grid point, by binary search against the midpoints between
    # adjacent grid values rather than a full distance matrix.  The obvious
    # spelling, ``g[np.abs(x[:, None] - g[None, :]).argmin(1)]``, materialises
    # an (nb, len(g)) float64 array -- 225 MB for one 4096x11008 tensor -- and
    # the bisection below calls it 60 times per tensor, which is hours on a 7B
    # model and was the whole cost of assembling an arm.  ``searchsorted``
    # touches nb elements and allocates nothing.  Tie-breaking matches: at a
    # midpoint, side="left" takes the lower grid point, and so did argmin.
    mids = 0.5 * (g[:-1] + g[1:])
    snap = lambda x: g[np.searchsorted(mids, x)]  # noqa: E731

    # The mean is a non-decreasing step function of the offset, so bisect it --
    # but evaluating it does not need a pass over the blocks.  A block sits at
    # or above grid point j+1 exactly when base > mids[j] - t, so sorting base
    # once turns each trial offset into len(g)-1 binary searches for the bin
    # populations, and the mean is those counts against the grid.  The obvious
    # version re-snaps all nb blocks every iteration; at 5.6M blocks that is
    # 5.6 s per tensor against 0.50 s here, all of it repeated work.
    bs = np.sort(base)
    nb = bs.size

    def mean_at(t):
        idx = np.searchsorted(bs, mids - t, side="right")
        return float(np.diff(np.concatenate(([0], idx, [nb]))) @ g) / nb

    # 60 halvings of [-64, 64] is past float64's resolution; 48 pins the offset
    # to ~5e-13, and the snapped result stops moving long before that -- it can
    # only change when a block crosses a midpoint.
    lo, hi = -64.0, 64.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if mean_at(mid) > Sbar:
            hi = mid
        else:
            lo = mid
    return snap(base + lo)


def sensitivity(sh, arch, L):
    p = f"model.layers.{L}."
    g_in = sh.get(p + "input_layernorm.weight")
    g_po = sh.get(p + "post_attention_layernorm.weight")
    if g_in is None or g_po is None:
        return None
    return {"q_proj": normed_scale(g_in), "k_proj": normed_scale(g_in),
            "v_proj": normed_scale(g_in),
            "o_proj": attn_out_scale(sh.get(p + "self_attn.v_proj.weight"),
                                     g_in, arch),
            "gate_proj": normed_scale(g_po), "up_proj": normed_scale(g_po),
            "down_proj": mlp_act_scale(sh.get(p + "mlp.gate_proj.weight"),
                                       sh.get(p + "mlp.up_proj.weight"), g_po)}


def _seedlm_equivalent_cfg(S, k, B):
    """A ``SenSeedConfig`` that is SeedLM, exactly, so the GPU backend can run it.

    SeedLM has no torch implementation in this repository -- ``compress_pruned``
    and friends are numpy and never touch a device, which is why a ``--devices``
    flag on a gated run still crawls "on numpy".  SenSeed's codec *does* have one,
    and SenSeed with a flat allocation **is** SeedLM, so the search can be borrowed
    by configuring the difference away:

    ``k_mode="schedule"``   **not** ``"fixed"``, which is the obvious choice and
    ``schedule_slope=0``    the wrong one.  ``compress_senseed`` only reaches the
                            torch path for its quantisation-aware pass; the
                            *scan* before it is numpy either way, and it is a
                            full sweep of all ``2**S`` seeds -- 43% of the
                            function by its own comment.  That scan is skipped
                            only under ``k_mode == "schedule" and quant_aware
                            and mask is None``.  With ``k_min == k_max`` the
                            allocator's clip pins every block to ``k`` whatever
                            the slope, so "schedule" at slope 0 is a flat
                            allocation -- bit-identical output to "fixed", and
                            the difference between a GPU that is busy and four
                            idle cards next to four pegged CPU cores.
    ``exp_group=1``         one 4-bit exponent per block, as SeedLM stores it.
                            The pipeline already hardcodes 1 everywhere.
    ``basis_layout="row"``  SeedLM's fill order (``"column"`` is S-Quant's).
    ``k_min = k_max = k``   **not cosmetic.**  ``k_max`` is what the basis is
                            built with (``SQuantConfig(..., k_max=cfg.k_max)``),
                            i.e. the number of columns the LFSR stream is laid
                            across before the first k are taken -- the *stride*
                            each basis column walks through the generator.
                            SeedLM lays it across ``P``.  Leaving the default
                            ``k_max=6`` while asking for ``k=3`` gives a
                            different basis for every block and silently
                            produces a different codec: same rate, different
                            numbers.  That is the fill-order confound in
                            ``rate_sweep_and_fill_confound.md``, which took days
                            to find the first time and was worth 2.02x at P=1.

    The two backends agree on the reconstruction error they achieve but can pick
    different seeds where two candidates tie, because the reduction order in a
    batched matmul differs.  So a cache must not mix them -- ``main`` refuses to.
    """
    from senseed.codec.allocate import SenSeedConfig
    return SenSeedConfig(B=B, S=S, k_mode="schedule", schedule_slope=0.0,
                      k_fixed=k, k_min=k, k_max=k, exp_group=1,
                      basis_layout="row", quant_aware=True, signal_k=False,
                      cond_max=0.0, target_bits=(S + 4 + 4 * k) / B)


def _compress_one(spec):
    """One tensor at one seed width.  Columns are pre-sorted by the caller.

    ``dev`` empty selects the numpy SeedLM codec, which produced every published
    number in this project.  Anything else is a torch device string and selects
    the SenSeed codec configured to be SeedLM -- see ``_seedlm_equivalent_cfg``.
    """
    name, Wp, S, k, B, dev, bc, sc = spec
    t0 = time.time()
    W = np.ascontiguousarray(Wp, np.float32)
    if not dev:
        hat = np.asarray(decompress(compress(W, SeedLMConfig(C=B, P=k, K=S))),
                         np.float32)
    else:
        from senseed.codec.allocate import compress_senseed, decompress_senseed
        cfg = _seedlm_equivalent_cfg(S, k, B)
        # block_chunk and seed_chunk are NOT optional on a device.  Their
        # defaults (128, 4096) are numpy-sized; on a 47 GiB card resolve_devices
        # asks for 8192, a 64x larger tile.  Running the small one turns the
        # search into a launch-overhead benchmark: the first GPU run of this
        # script came out at 0.78x the CPU, not the ~100x the hardware gives,
        # purely because these two were left at their defaults.
        hat = np.asarray(decompress_senseed(
            compress_senseed(W, cfg, backend=dev,
                          block_chunk=bc, seed_chunk=sc)), np.float32)
    return name, hat.astype(np.float16), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="one sub-directory per arm")
    ap.add_argument("--cache", default="",
                    help="per-width compressions; default <out>/_cache. "
                         "Existing widths are reused, so the run is restartable "
                         "and extra arms cost nothing.")
    ap.add_argument("--seed-bits", default="14,12,10",
                    help="comma list of MEAN widths; each gives a scheduled arm "
                         "and a uniform arm at the same rate")
    ap.add_argument("--reference", type=int, default=16,
                    help="the uniform width everything is measured against")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--slope", type=float, default=4.0)
    ap.add_argument("--grid", default="",
                    help="widths to compress at; default is the reference plus "
                         "every mean, plus a floor at 8")
    ap.add_argument("--layers", default="")
    ap.add_argument("--cpu-workers", type=int,
                    default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--devices", default="",
                    help="comma list of torch devices, e.g. "
                         "cuda:0,cuda:1,cuda:2,cuda:3.  Empty (the default) "
                         "keeps the numpy SeedLM codec that produced every "
                         "published number here.  One worker per device, "
                         "tensors dealt round-robin.  A cache records which "
                         "backend built it and this refuses to mix them.")
    ap.add_argument("--arms", default="",
                    help="comma list of tag prefixes to write, e.g. 'sched'. "
                         "Default writes all of them.  Only the scheduled arms "
                         "depend on --slope, so re-running at a new slope needs "
                         "no rewrite of ref/uni -- each arm is a full copy of "
                         "the checkpoint, so skipping four of seven is ~54 GB "
                         "and a third of the assembly time on a 7B model.")
    ap.add_argument("--slope-scan", default="",
                    help="comma list of slopes.  Print the seed-width "
                         "distribution each would realise, for every "
                         "--seed-bits mean, then exit without compressing or "
                         "assembling.  Costs the gather phase only.  Worth "
                         "running before any 7B job: the slope is in units of "
                         "log-importance, whose spread is a property of the "
                         "checkpoint, so one slope does not mean one schedule "
                         "across models -- slope 1 moves 18% of Qwen2.5-0.5B's "
                         "blocks and 1% of Llama-2-7B's.")
    ap.add_argument("--scan-blocks", type=int, default=50_000,
                    help="blocks sampled per tensor during --slope-scan.  The "
                         "tally is a population statistic and 50k estimates it "
                         "to well under a percent, while the exact version is "
                         "unusable: schedule_S bisects 60 times, each pass an "
                         "argmin over (blocks x grid), and a 7B model has 810 "
                         "million blocks -- one slope took two hours.")
    ap.add_argument("--seed-chunk", type=int, default=8192,
                    help="seeds per search tile on a device (pipeline default)")
    ap.add_argument("--block-chunk", type=int, default=0,
                    help="blocks per search tile; 0 asks resolve_devices to "
                         "size it to the smallest card, which is what the "
                         "measured 7.46 us/weight/device was run with")
    args = ap.parse_args()

    model = os.path.expanduser(args.model)
    out = os.path.expanduser(args.out)
    cache = os.path.expanduser(args.cache) if args.cache else os.path.join(out, "_cache")
    os.makedirs(cache, exist_ok=True)
    sbars = [float(x) for x in args.seed_bits.split(",")]
    grid = sorted(set(int(x) for x in args.grid.split(",")) if args.grid else
                  {args.reference, 8} | {int(round(s)) for s in sbars})
    B, k = args.block, args.k
    rate = lambda s: (s + 4 * k + 4) / B                      # noqa: E731
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    block_chunk, seed_chunk = args.block_chunk, args.seed_chunk
    if devices:
        from senseed.codec.gpu import resolve_devices
        devices, block_chunk = resolve_devices(
            devices, seed_chunk, args.block_chunk, k_max=args.k, B=args.block)
    backend_tag = ",".join(devices) if devices else "numpy"
    global _POOL_CTX
    if devices:
        import multiprocessing as _mp
        _POOL_CTX = _mp.get_context("spawn")

    cfg_j = json.load(open(os.path.join(model, "config.json")))
    arch = ArchSpec.from_config(cfg_j)
    sh = Shards(model)
    todo = (set(int(x) for x in args.layers.split(",")) if args.layers
            else set(range(cfg_j["num_hidden_layers"])))

    # --- gather the tensors, pre-sorted by sensitivity -----------------------
    order_of, imp_of, Wp_of = {}, {}, {}
    for L in sorted(todo):
        av = sensitivity(sh, arch, L)
        if av is None:
            continue
        p = f"model.layers.{L}."
        for proj in PROJ:
            n = p + ("self_attn." if proj in ATTN else "mlp.") + proj + ".weight"
            W = sh.get(n)
            if W is None:
                continue
            W = np.asarray(W, np.float32)
            a = np.asarray(av[proj], np.float64)
            if len(a) != W.shape[1]:
                continue
            o = np.argsort(-a, kind="stable")
            order_of[n] = o
            Wp_of[n] = np.ascontiguousarray(W[:, o], np.float32)
            imp_of[n] = block_importance(a[o], W.shape[0], B)
    names = sorted(Wp_of)
    # Everything after phase 1 needs each tensor's shape, never its values.
    # Keep the metadata separately so the weights themselves can be dropped.
    size_of = {n: Wp_of[n].size for n in names}
    shape_of = {n: Wp_of[n].shape for n in names}
    total = sum(size_of[n] for n in names)
    print(f"{len(names)} linear tensors, {total/1e6:.0f}M weights", flush=True)
    print(f"reference S={args.reference} -> {rate(args.reference):.4f} bits/weight;"
          f" arms " + ", ".join(f"S_mean={s:g} -> {rate(s):.4f}b "
                                f"({(1-rate(s)/rate(args.reference))*100:.2f}% smaller)"
                                for s in sbars), flush=True)
    print(f"compression grid {tuple(grid)}  (cost goes as 2**S, so the top one "
          f"dominates)", flush=True)
    # Say the backend out loud.  A --devices flag that silently falls back to
    # numpy is exactly how a gated Mistral run spent twenty hours on the CPU
    # with four idle GPUs attached to it.
    print(f"backend: {backend_tag}"
          + (f"  ({len(devices)} worker(s), one per device, tile "
             f"{block_chunk} blocks x {seed_chunk} seeds)" if devices
             else f"  ({args.cpu_workers} CPU workers)") + "\n", flush=True)

    # One block-importance vector per tensor, trimmed to the codec's block
    # count.  Defined once so the scan below and the arms cannot drift apart.
    def imp_for(n):
        nb = -(-size_of[n] // B)
        return imp_of[n][:nb] if len(imp_of[n]) >= nb else imp_of[n]

    if args.slope_scan:
        print("slope scan -- no compression, no arms.\n"
              "'moved' is the share of blocks that leave the mean width; at 0% "
              "the scheduled\narm is the uniform arm and the comparison is "
              "vacuous.\n", flush=True)
        sub = np.random.default_rng(0)
        for sl in [float(x) for x in args.slope_scan.split(",")]:
            for s in sbars:
                tally = {}
                for n in names:
                    v = imp_for(n)
                    if len(v) > args.scan_blocks:      # estimate, do not enumerate
                        v = v[sub.integers(0, len(v), args.scan_blocks)]
                    S_i = schedule_S(v, s, sl, grid)
                    for S in sorted(set(int(v) for v in S_i)):
                        tally[S] = tally.get(S, 0.0) + \
                            float((S_i == S).mean()) / len(names)
                stay = tally.get(int(round(s)), 0.0)
                print(f"  slope {sl:6.2f}  S_mean={s:<4g} moved {(1-stay)*100:5.1f}%   "
                      + "  ".join(f"S{S}:{f*100:.1f}%"
                                  for S, f in sorted(tally.items())), flush=True)
        return

    # --- phase 1: compress once per width ------------------------------------
    import stbf as stn
    import safetensors.torch as stt
    import torch
    from safetensors import safe_open
    for S in grid:
        f = os.path.join(cache, f"S{S}.safetensors")
        if os.path.exists(f):
            with safe_open(f, framework="numpy") as h:
                was = (h.metadata() or {}).get("backend", "numpy")
            if was != backend_tag:
                raise SystemExit(
                    f"\ncache {f}\n  was built with backend {was!r}, this run "
                    f"asks for {backend_tag!r}.\n  The two backends agree on the "
                    f"error they achieve but can pick different seeds where two\n"
                    f"  candidates tie, so mixing them inside one grid puts an "
                    f"uncontrolled variable\n  in the comparison.  Point --cache "
                    f"somewhere else, or match the backend.")
            print(f"[cache] S={S} already there ({was})", flush=True)
            continue
        t0 = time.time()
        jobs = [(n, Wp_of[n], S, k, B,
                 devices[i % len(devices)] if devices else "",
                 block_chunk, seed_chunk)
                for i, n in enumerate(names)]
        workers = len(devices) if devices else args.cpu_workers
        got = {}
        # CUDA refuses to re-initialise inside a forked child, and this process
        # has already touched it: resolve_devices asks the driver how big the
        # cards are.  So the device path must spawn.  Spawn pickles each job's
        # weight array down a pipe instead of inheriting it copy-on-write --
        # one pass over the model's weights per grid point, seconds against
        # hours of search, and the only alternative is to not size the tile.
        # The CPU path keeps fork, which is what every published number used.
        with ProcessPoolExecutor(workers, mp_context=_POOL_CTX) as ex:
            for i, (n, hat, secs) in enumerate(ex.map(_compress_one, jobs), 1):
                got[n] = hat
                if i % 40 == 0 or i == len(jobs):
                    print(f"  S={S:2d} [{i}/{len(jobs)}] {(time.time()-t0)/60:5.1f} min",
                          flush=True)
        tmp = f + ".partial"
        stn.save_file(got, tmp, metadata={"format": "pt", "backend": backend_tag})
        os.replace(tmp, f)
        print(f"[cache] S={S} written in {(time.time()-t0)/60:.1f} min "
              f"({backend_tag})", flush=True)

    # --- phase 2: assemble every arm from the cache --------------------------
    # Phase 1 is over, and nothing below reads a weight value out of Wp_of --
    # only shapes, which size_of/shape_of already hold.  Keeping the arrays
    # costs 4 bytes per parameter of host RAM (27 GB on a 7B model) on top of
    # the bf16 source that write_arm loads, which is enough for the OOM killer
    # to take the process with no traceback, mid-arm.  Drop them here.
    if Wp_of:
        Wp_of.clear()
        gc.collect()

    handles = {S: safe_open(os.path.join(cache, f"S{S}.safetensors"), framework="numpy")
               for S in grid}
    src_files = [os.path.join(model, fn) for fn in sorted(os.listdir(model))
                 if fn.endswith(".safetensors")]

    want = [a.strip() for a in args.arms.split(",") if a.strip()]

    def write_arm(tag, pick, note):
        if want and not any(tag.startswith(w) for w in want):
            print(f"  {tag:22s} skipped (--arms)", flush=True)
            return
        d = os.path.join(out, tag)
        os.makedirs(d, exist_ok=True)
        src = {}
        for f in src_files:
            src.update(stt.load_file(f))      # torch: the source may be bf16
        tally = {}
        # Phase 1 prints every 40 tensors; this loop used to print nothing at
        # all, so a 7B arm was 10-20 silent minutes and the only way to tell
        # work from a hang was to attach py-spy.  Same cadence here.
        t_arm = time.time()
        for i_n, n in enumerate(names, 1):
            if i_n % 40 == 0 or i_n == len(names):
                print(f"    {tag} [{i_n}/{len(names)}] "
                      f"{(time.time()-t_arm)/60:5.1f} min", flush=True)
            S_i = pick(n)                       # array of widths, one per block
            nb = -(-size_of[n] // B)
            flat = np.zeros(nb * B, np.float32)
            for S in sorted(set(int(s) for s in np.atleast_1d(S_i))):
                m = (S_i == S) if np.ndim(S_i) else np.ones(nb, bool)
                tally[S] = tally.get(S, 0.0) + float(np.mean(m)) / len(names)
                h = handles[S].get_tensor(n).astype(np.float32).reshape(-1)
                if len(h) < nb * B:
                    h = np.concatenate([h, np.zeros(nb * B - len(h), np.float32)])
                flat.reshape(-1, B)[m] = h.reshape(-1, B)[m]
            inv = np.argsort(order_of[n])
            v = np.ascontiguousarray(
                flat[:size_of[n]].reshape(shape_of[n])[:, inv])
            src[n] = torch.from_numpy(v).to(src[n].dtype)
        stt.save_file(src, os.path.join(d, "model.safetensors"),
                      metadata={"format": "pt"})
        for fn in os.listdir(model):
            if fn.endswith(".safetensors") or fn.endswith(".index.json"):
                continue
            s = os.path.join(model, fn)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(d, fn))
        sm = sum(S * f for S, f in tally.items())
        json.dump(dict(tag=tag, note=note, k=k, B=B, seed_tally=tally,
                       mean_seed_bits=sm, bits_per_weight=rate(sm),
                       reference_bits=rate(args.reference), backend=backend_tag,
                       storage_saved=1 - rate(sm) / rate(args.reference)),
                  open(os.path.join(d, "arm.json"), "w"), indent=1)
        print(f"  {tag:22s} mean S {sm:5.2f}  {rate(sm):.4f} b/w  "
              f"{(1-rate(sm)/rate(args.reference))*100:5.2f}% smaller   "
              + " ".join(f"S{S}:{f*100:.0f}%" for S, f in sorted(tally.items())),
              flush=True)

    print("\nassembling arms:", flush=True)
    write_arm(f"ref-S{args.reference}",
              lambda n: float(args.reference), "uniform reference")
    for s in sbars:
        write_arm(f"sched-S{s:g}",
                  lambda n, s=s: schedule_S(imp_for(n), s, args.slope, grid),
                  f"scheduled, slope {args.slope}")
        write_arm(f"uni-S{s:g}", lambda n, s=s: float(int(round(s))),
                  "uniform at the same mean width")
    print(f"\narms written under {out}", flush=True)


if __name__ == "__main__":
    main()
