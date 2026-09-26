#!/usr/bin/env python3
"""Per-block (S, k) allocation on the measured damage surface.

Every arm measured so far is UNIFORM -- all blocks at one (S, k) -- which is
what the surface is for: with a global error table, a uniform arm's total
damage is ``e(S,k) * sum_i imp_i``, so the constant cancels and the measured
grid IS the ``e(S,k)`` the allocator needs.  It is not the answer to "what is
best at 4.000 b/w"; that question is about allocations, and uniform is only one
of them.

This does the allocation.  Each block solves

    min over grid points g of   imp_i * e(g) + lam * rate(g)

which, as ``lam/imp_i`` sweeps, selects exactly the points on the lower convex
hull of ``{(rate, e)}`` -- important blocks land at high-rate points, the rest
slide down.  ``lam`` is bisected so the mean rate hits the budget.  The hull is
shared and the table is a handful of global constants, so the decoder recomputes
the same assignment from gamma alone: still zero signalling.

Damage, not MSE.  MSE is not even monotone across k -- on Llama-2-7B (12,2)
beats (8,3) on MSE and loses to it on perplexity -- so a frontier built in MSE
would order the k axis backwards.

    python joint_allocate.py --target 4.0 --out ckpt/joint_4p0
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
from seed_schedule import (sensitivity, block_importance,          # noqa: E402
                           PROJ, ATTN)
from measure_actscale import Shards                                # noqa: E402
from senseed.sensitivity.actscale import ArchSpec                     # noqa: E402

rate = lambda S, k, B=8: (S + 4 * k + 4) / B                       # noqa: E731
parse = lambda t: (int(t.split("k")[0][1:]), int(t.split("k")[1]))  # noqa: E731


def load_surface(results, fp16_json, prefix="g05_q05_k"):
    """Measured damage per (S, k) from the uniform arms.

    ``prefix`` is everything up to the k digit; a file is
    ``<prefix><k>_<anything>-S<S>.json``.  0.5B is ``g05_q05_k``, 7B is
    ``ev7b_k``.

    The token filter is not cosmetic.  A Llama-2 tokenizer gives 339,968 tokens
    on this text and Qwen2.5 gives 299,008, and a surface that mixes the two is
    wrong in a way that still produces a plausible table.  Arms that disagree
    with the fp16 reference are dropped -- and if that drops ALL of them, the
    --fp16 flag is pointing at the wrong model, which is worth stopping for
    rather than handing back an empty surface.
    """
    fp = json.load(open(fp16_json))
    e, tok, seen = {}, fp["tokens"], 0
    for f in sorted(glob.glob(os.path.join(results, prefix + "*.json"))):
        b = os.path.basename(f)[len(prefix):-len(".json")]
        try:
            k, S = int(b.split("_")[0]), int(b.split("-S")[1])
        except (ValueError, IndexError):
            continue
        d = json.load(open(f))
        seen += 1
        if d.get("tokens") != tok:
            continue
        e[(S, k)] = d["nll"] - fp["nll"]
    if seen and not e:
        raise SystemExit(
            f"{seen} arms matched {prefix!r} but not one has tokens == {tok}, "
            f"the fp16 arm's.  Llama-2 gives 339,968 on this text and Qwen2.5 "
            f"gives 299,008 -- --fp16 is pointing at the wrong model.")
    return e


def lower_hull(e):
    """Points a block can ever be assigned; everything else is dominated.

    Two kinds of dominated point have to go BEFORE the convex hull is built.
    The hull alone keeps both, and each one silently breaks ``assign``:

    * **Equal rate.**  (16,4) and (12,5) both cost 4.500 b/w.  Sorted by rate
      they make a zero-width segment, its slope is -inf, ``lam/s`` stops being
      ascending, and ``searchsorted`` is then a binary search on unsorted
      input -- undefined, and in practice it hands the traffic to the WORSE
      member of the tie.  4.25, 4.50, 4.75 and 5.00 are all doubled on this
      surface, so every band in the 20-cell sweep ended on a tie.

    * **Higher rate, higher damage.**  S10k5 costs 4.250 and measures 0.2310;
      S16k3 costs 4.000 and measures 0.2289.  No block ever wants the first,
      but it is the rightmost point, so the hull keeps it, the closing slope
      comes out negative, and the budget bisection cannot reach the target at
      all -- those are the mean 4.250 / 3.859 / 3.869 rows.

    Keep the cheapest point per rate, then the running minimum in damage, then
    the convex hull of what survives.  ``r`` is now strictly increasing and the
    slopes strictly positive, which is what ``hull_slopes`` asserts.
    """
    best = {}
    for (S, k), v in e.items():
        r = rate(S, k)
        if r not in best or v < best[r][1]:
            best[r] = (r, v, (S, k))
    pts, lo = [], float("inf")
    for p in sorted(best.values()):
        if p[1] < lo:                 # strictly better than anything cheaper
            lo = p[1]
            pts.append(p)
    h = []
    for p in pts:
        while len(h) >= 2:
            (x1, y1, _), (x2, y2, _) = h[-2], h[-1]
            if (y2 - y1) * (p[0] - x1) >= (p[1] - y1) * (x2 - x1):
                h.pop()
            else:
                break
        h.append(p)
    return h


def hull_slopes(hull):
    """Rates, damages, and the switch thresholds between adjacent points.

    Block i minimises ``imp_i*e + lam*rate``; dividing through by ``imp_i``
    makes the choice a function of ``mu = lam/imp_i`` alone, and point j beats
    j+1 exactly when ``mu > s_j``.  The hull is convex so ``s`` decreases and
    ``lam/s`` is increasing -- which turns the per-block argmin into one
    searchsorted against a len(hull)-1 array.

    The obvious spelling builds an (n_blocks, n_points) cost matrix inside the
    bisection: 3.9 GB per trial lambda at 44.7M blocks, 200 trials, ~28 minutes.
    This is the same mistake ``schedule_S`` had -- a broadcast over every block
    inside a loop that only needs a scalar out of it.
    """
    r = np.array([p[0] for p in hull])
    v = np.array([p[1] for p in hull])
    dr = r[1:] - r[:-1]
    if not np.all(dr > 0):
        raise ValueError("hull has two points at one rate -- lower_hull should "
                         "have dropped the worse one; lam/s is not ascending "
                         "and searchsorted is undefined on it")
    s = (v[:-1] - v[1:]) / dr
    if not (np.all(s > 0) and np.all(np.diff(s) <= 0)):
        raise ValueError(f"hull slopes must be positive and non-increasing for "
                         f"lam/s to be ascending; got {s}")
    return r, v, s


def assign(imp, s, lam):
    """Hull index per block.  Verified identical to the argmin form."""
    return np.searchsorted(lam / s, imp, side="left")


def settle(imp, r, s, lo, hi, target):
    """Spend the budget exactly, not to within a tied group.

    ``block_importance`` is a function of the block's COLUMNS, so every row of
    a tensor sharing a column range carries the identical value: 44.7M blocks
    take only ~1e5 distinct importances, thousands of blocks at a time.  A
    threshold therefore does not cross one block as lambda moves, it crosses a
    whole tied group, and the bisection cannot land on the budget -- it stops
    at the last lambda ABOVE it.  Every arm built before this was a little over
    its stated rate, which is the wrong direction to be wrong in: it quietly
    spends more than the baseline it is being compared against.

    ``lo`` overshoots and ``hi`` undershoots by construction, and the two
    assignments differ only on the blocks that group crossing accounts for.
    Demote them one at a time, in block order, until the budget is spent.  The
    decoder walks blocks in the same order and needs one extra integer -- the
    count -- in the global table, so this is still zero signalling.
    """
    idx = assign(imp, s, lo)
    alt = assign(imp, s, hi)
    d = np.flatnonzero(idx != alt)
    if d.size:
        excess = float(r[idx].sum() - target * imp.size)
        give = r[idx[d]] - r[alt[d]]            # >= 0: alt is the cheaper side
        m = int(np.searchsorted(np.cumsum(give), excess, side="right"))
        idx[d[:m]] = alt[d[:m]]
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Llama-2-7b-hf")
    ap.add_argument("--cache-fmt", default="ckpt/grid_cache_k{k}",
                    help="cache dir per k; must contain S{S}.safetensors")
    ap.add_argument("--results", default="results")
    ap.add_argument("--fp16", default="results/fp16.json")
    ap.add_argument("--target", type=float, default=4.0, help="mean bits/weight")
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--surface-prefix", default="g05_q05_k",
                    help="filename prefix up to the k digit; 7B is 'ev7b_k'")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the allocation and stop, writing nothing")
    ap.add_argument("--e-table", default="",
                    help="JSON of per-block coefficients measured by "
                         "sparse_probe.py.  The surface's own values come from "
                         "UNIFORM arms, where every block sits at the point; "
                         "measured at f=0.05 the same points read 0.29-0.38 of "
                         "that, so damage is subadditive in the fraction "
                         "degraded and there is no single coefficient.  Points "
                         "the table does not cover keep their uniform value, "
                         "so a run with this flag mixes two calibrations -- the "
                         "listing below says which is which.")
    ap.add_argument("--e-select", default="random",
                    help="which sparse_probe selection to read.  'random' is "
                         "the one that measures e(g) itself; 'least'/'most' "
                         "fold the importance weighting into the coefficient "
                         "and must not be used here.")
    ap.add_argument("--points", default="",
                    help="restrict the candidate set, e.g. S14k4,S14k3.  The "
                         "Lagrangian on the full hull is only optimal if damage "
                         "really is linear in the per-block coefficients, and "
                         "that assumption has already failed once (3.250 b/w), "
                         "so a hand-picked pair is worth measuring against it.")
    a_ = ap.parse_args()

    model = os.path.expanduser(a_.model)
    out = os.path.expanduser(a_.out)
    e = load_surface(os.path.expanduser(a_.results), os.path.expanduser(a_.fp16),
                     a_.surface_prefix)
    corrected = set()
    if a_.e_table:
        tab = json.load(open(os.path.expanduser(a_.e_table)))
        for key, v in tab.items():
            pt, sel = key.split(":") if ":" in key else (key, "random")
            if sel != a_.e_select:
                continue
            g = parse(pt)
            if g in e:
                e[g] = v
                corrected.add(g)
        print(f"e-table: {len(corrected)} of {len(e)} points overridden "
              f"from {a_.e_table} ({a_.e_select})")
    if a_.points:
        want = {p.strip() for p in a_.points.split(",") if p.strip()}
        e = {(S, k): v for (S, k), v in e.items() if f"S{S}k{k}" in want}
        missing = want - {f"S{S}k{k}" for S, k in e}
        if missing:
            raise SystemExit(f"no measured arm for {sorted(missing)}")
        if len(e) < 2:
            raise SystemExit("need at least two points to allocate between")
        print(f"restricted to {sorted(want)}")
    hull = lower_hull(e)
    B = a_.block

    print(f"surface: {len(e)} measured points, {len(hull)} on the hull")
    for r, v, g in hull:
        src = "sparse f=0.05" if g in corrected else "uniform arm"
        print(f"   S{g[0]}k{g[1]:<2} {r:.3f} b/w   e {v:.4f}   [{src}]")
    off = [f"S{S}k{k}" for (S, k) in e if (S, k) not in [g for _, _, g in hull]]
    print(f"   dominated, never assigned: {off}\n")

    sh = Shards(model)
    arch = ArchSpec.from_config(json.load(open(os.path.join(model, "config.json"))))
    order_of, imp_of, shape_of, size_of = {}, {}, {}, {}
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
            W = np.asarray(W, np.float32)
            av_p = np.asarray(av[proj], np.float64)
            if len(av_p) != W.shape[1]:
                continue
            o = np.argsort(-av_p, kind="stable")
            order_of[n], shape_of[n], size_of[n] = o, W.shape, W.size
            imp_of[n] = block_importance(av_p[o], W.shape[0], B)
        L += 1
    names = sorted(order_of)
    nb_of = {n: -(-size_of[n] // B) for n in names}
    imp = np.concatenate([imp_of[n][:nb_of[n]] for n in names])
    print(f"{len(names)} tensors, {imp.size/1e6:.2f}M blocks\n")

    r_hull, v_hull, s = hull_slopes(hull)
    imps = np.sort(imp)                      # sorted once; the sweep is then free
    N = imps.size

    def mean_rate(lam):
        b = np.searchsorted(imps, lam / s, side="right")
        return float(np.diff(np.concatenate(([0], b, [N]))) @ r_hull) / N

    lo, hi = 1e-12, 1e6
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if mean_rate(mid) > a_.target:
            lo = mid
        else:
            hi = mid
    idx = settle(imp, r_hull, s, lo, hi, a_.target)
    got = r_hull[idx].mean()
    est = float((imp * v_hull[idx]).sum() / imp.sum())
    print(f"lambda {lo:.6g}   mean rate {got:.4f} b/w (target {a_.target})")
    print(f"predicted damage {est:.4f}   -- linear model, the arm decides\n")
    print("assignment:")
    for j, (r, v, g) in enumerate(hull):
        f = float((idx == j).mean())
        if f > 0:
            print(f"   S{g[0]}k{g[1]:<2} {r:.3f} b/w   {f*100:5.1f}% of blocks")

    uni = min((abs(rate(S, k) - a_.target), v, (S, k)) for (S, k), v in e.items())
    if uni[0] < 1e-9:
        print(f"\nbest uniform arm at this rate: S{uni[2][0]}k{uni[2][1]} "
              f"damage {uni[1]:.4f}   -> allocation predicts "
              f"{(1-est/uni[1])*100:+.1f}%")
    if a_.dry_run:
        return

    import stbf as stn                                   # noqa: F401
    import safetensors.torch as stt
    import torch
    os.makedirs(out, exist_ok=True)
    handles = {g: safe_open(os.path.join(
        os.path.expanduser(a_.cache_fmt.format(k=g[1])), f"S{g[0]}.safetensors"),
        framework="numpy") for _, _, g in hull}
    src = {}
    for f in sorted(glob.glob(os.path.join(model, "*.safetensors"))):
        src.update(stt.load_file(f))

    pos = 0
    for i_n, n in enumerate(names, 1):
        nb = nb_of[n]
        sub = idx[pos:pos + nb]
        pos += nb
        flat = np.zeros(nb * B, np.float32)
        for j in np.unique(sub):
            g = hull[j][2]
            m = sub == j
            h = handles[g].get_tensor(n).astype(np.float32).reshape(-1)
            if len(h) < nb * B:
                h = np.concatenate([h, np.zeros(nb * B - len(h), np.float32)])
            flat.reshape(-1, B)[m] = h.reshape(-1, B)[m]
        inv = np.argsort(order_of[n])
        v = np.ascontiguousarray(
            flat[:size_of[n]].reshape(shape_of[n])[:, inv])
        src[n] = torch.from_numpy(v).to(src[n].dtype)
        if i_n % 40 == 0 or i_n == len(names):
            print(f"   assembling [{i_n}/{len(names)}]", flush=True)

    stt.save_file(src, os.path.join(out, "model.safetensors"),
                  metadata={"format": "pt"})
    for fn in os.listdir(model):
        if fn.endswith(".safetensors") or fn.endswith(".index.json"):
            continue
        s = os.path.join(model, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(out, fn))
    json.dump(dict(target=a_.target, mean_rate=float(got), lam=float(lo),
                   predicted_damage=est,
                   hull=[[p[0], p[1], list(p[2])] for p in hull],
                   share={f"S{g[0]}k{g[1]}": float((idx == j).mean())
                          for j, (_, _, g) in enumerate(hull)}),
              open(os.path.join(out, "arm.json"), "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
