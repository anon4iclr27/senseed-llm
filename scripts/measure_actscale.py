#!/usr/bin/env python3
"""Measure the real per-channel input scale E[x_j^2] of every projection.

Ground truth for the data-free estimate in ``seedlm/actscale.py``.  Runs a
forward pass over a few hundred tokens and records, for every linear layer,
the mean square of each input channel -- plus the normalised vector ``n``
itself, which is what tests the one assumption the estimate really rests on::

    a_j = gamma_j^2 * E[n_j^2]  ~  gamma_j^2      requires  E[n_j^2] ~ 1

RMSNorm only guarantees that on average over channels.  If the per-channel
values are flat the proxy is exact up to the diagonal approximation; if a few
channels carry all the mass -- the "massive activations" reported for Llama-3 --
the proxy is wrong exactly where it matters most.

Numpy only.  No torch, no transformers, no network.  The checkpoint is memory
mapped and processed one layer at a time, so peak RSS is roughly one MLP matrix
in float32 (~270 MB for a 7B model) rather than the whole model.

    python measure_actscale.py --model /path/to/Qwen2.5-7B --out qwen_act.safetensors

Add ``--text-file some.txt`` to use your own text; the built-in sample is plain
English prose written for this script.  ``--tokens 512`` sets the sequence
length, ``--layers 0,14,27`` restricts to a few layers if you want it quick.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import re
import struct
import sys
import time

import numpy as np

# --------------------------------------------------------------------------
# safetensors: memory-mapped read, plain write
# --------------------------------------------------------------------------

_DT = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16,
       "I64": np.int64, "I32": np.int32, "U8": np.uint8, "BOOL": np.bool_}


class Shards:
    """All shards of a checkpoint, mapped, addressed by tensor name."""

    def __init__(self, path: str):
        self.path = path
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                weight_map = json.load(f)["weight_map"]
        else:
            single = "model.safetensors"
            if not os.path.exists(os.path.join(path, single)):
                # Say what *is* there.  This error has meant, at various times,
                # a download that had not finished, a --cache-dir layout where
                # the weights sit under models--Org--Name/snapshots/<sha>/, and
                # a plain typo -- and the bare message told them apart not at
                # all.
                try:
                    here = sorted(os.listdir(path))
                except OSError as e:
                    raise SystemExit(f"cannot read {path}: {e}")
                nested = [d for d in here
                          if os.path.isdir(os.path.join(path, d, "snapshots"))
                          or d == "snapshots"]
                hint = ""
                if nested:
                    hint = ("\nThis looks like a hub *cache* layout.  The "
                            "weights are under\n  "
                            + os.path.join(path, nested[0], "snapshots",
                                           "<commit>")
                            + "\nPoint --model there, or re-download with "
                              "--local-dir.")
                raise SystemExit(
                    f"no {single} and no model.safetensors.index.json in "
                    f"{path}\ncontents: "
                    + (", ".join(here[:20]) + (" ..." if len(here) > 20 else "")
                       if here else "(empty)") + hint)
            weight_map = None
        self._maps, self._hdr, self._where = {}, {}, {}
        files = sorted({v for v in weight_map.values()}) if weight_map else \
            ["model.safetensors"]
        for fn in files:
            full = os.path.join(path, fn)
            f = open(full, "rb")
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self._maps[fn] = (f, mm, 8 + n)
            self._hdr[fn] = hdr
            for k in hdr:
                if k != "__metadata__":
                    self._where[k] = fn

    def __contains__(self, name):
        return name in self._where

    def keys(self):
        return self._where.keys()

    def get(self, name, dtype=np.float32):
        """Tensor as ``dtype``; bf16 is widened by the shift trick."""
        fn = self._where.get(name)
        if fn is None:
            return None
        meta = self._hdr[fn][name]
        _f, mm, base = self._maps[fn]
        s, e = meta["data_offsets"]
        raw = np.frombuffer(mm, dtype=_DT[meta["dtype"]], count=(e - s) //
                            np.dtype(_DT[meta["dtype"]]).itemsize,
                            offset=base + s).reshape(meta["shape"])
        if meta["dtype"] == "BF16":
            out = np.zeros(raw.shape, np.uint32)
            out |= raw.astype(np.uint32) << 16
            return out.view(np.float32).astype(dtype, copy=False)
        return raw.astype(dtype, copy=False)

    def rows(self, name, idx, dtype=np.float32):
        """Only the requested rows -- for the embedding table."""
        fn = self._where.get(name)
        meta = self._hdr[fn][name]
        _f, mm, base = self._maps[fn]
        s, _e = meta["data_offsets"]
        dt = _DT[meta["dtype"]]
        cols = meta["shape"][1]
        item = np.dtype(dt).itemsize
        out = np.empty((len(idx), cols), dtype)
        for i, r in enumerate(idx):
            raw = np.frombuffer(mm, dtype=dt, count=cols,
                                offset=base + s + r * cols * item)
            if meta["dtype"] == "BF16":
                v = np.zeros(cols, np.uint32)
                v |= raw.astype(np.uint32) << 16
                out[i] = v.view(np.float32)
            else:
                out[i] = raw
        return out


def save_safetensors(path: str, tensors: dict):
    hdr, blobs, off = {}, [], 0
    for k, v in tensors.items():
        v = np.ascontiguousarray(v, np.float32)
        hdr[k] = {"dtype": "F32", "shape": list(v.shape),
                  "data_offsets": [off, off + v.nbytes]}
        off += v.nbytes
        blobs.append(v)
    raw = json.dumps(hdr).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b.tobytes())


# --------------------------------------------------------------------------
# byte-level BPE, enough of it to tokenise English correctly
# --------------------------------------------------------------------------

_PAT_UNICODE = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
                r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_PAT_ASCII = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\nA-Za-z0-9]?[A-Za-z]+|"
              r"[0-9]{1,3}| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def _byte_encoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class BPE:
    def __init__(self, tok_json: str):
        with open(tok_json, encoding="utf-8") as f:
            spec = json.load(f)
        m = spec["model"]
        self.vocab = m["vocab"]
        merges = m["merges"]
        if merges and isinstance(merges[0], list):
            merges = [" ".join(x) for x in merges]
        self.ranks = {tuple(x.split(" ")): i for i, x in enumerate(merges)}
        self.benc = _byte_encoder()
        try:
            import regex
            self.re = regex.compile(_PAT_UNICODE)
        except ImportError:
            self.re = re.compile(_PAT_ASCII)

    def _bpe(self, word: str):
        parts = list(word)
        while len(parts) > 1:
            pairs = [(self.ranks.get((parts[i], parts[i + 1]), 1 << 30), i)
                     for i in range(len(parts) - 1)]
            rank, i = min(pairs)
            if rank == 1 << 30:
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        return parts

    def encode(self, text: str):
        ids = []
        for piece in self.re.findall(text):
            s = "".join(self.benc[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(s):
                v = self.vocab.get(tok)
                if v is not None:
                    ids.append(v)
        return ids


SAMPLE = """
The compression of a neural network is, at bottom, an argument about which
errors are cheap. Every scheme has to decide where to spend the bits it has,
and every scheme that decides well is making a claim about the geometry of the
function it is approximating. A method that treats all weights alike is making
a claim too, namely that the geometry is flat, and that claim is almost never
right.

Consider a single linear layer. Its output is a sum over input channels, and
each channel arrives scaled by whatever the previous operation left behind. If
one channel routinely carries values a hundred times larger than its
neighbours, an error in the weights that read that channel is amplified a
hundredfold on its way to the output, while the same error on a quiet channel
is barely felt at all. The weights look identical in the file. They are not
identical in effect.

This is why calibration data became standard practice. Run a few hundred
sentences through the network, watch which channels light up, and weight the
compression accordingly. It works. It also costs something that is easy to
overlook: the data has to exist, it has to be representative, it has to be
shipped alongside the method, and the resulting compression is quietly
specialised to whatever the calibration set happened to contain.

There is a different question worth asking. How much of that structure is
already written down in the checkpoint itself? Normalisation layers carry an
explicit per-channel gain. Attention mixes across positions but not across
channels. The gate and the projection that follow it are both matrices whose
row norms say something about the size of what they produce. None of that
requires a single token of text to read.

The honest answer is that the checkpoint does not contain everything. It
contains the part of the scale that the architecture states outright, and it
misses the part that emerges from the data flowing through. How large the
missing part is happens to be a measurable quantity, and measuring it is the
point of this script.

Numbers first, interpretation second. A proxy that recovers most of the
variation is worth using, because it is free and it travels. A proxy that
recovers little should be abandoned quickly and loudly, before it is built
into anything. Both outcomes are useful. Only the untested version is not.

Consider what happens at the very first layer of a deep network, where the
residual stream has barely begun to accumulate structure. The gains there are
often close to uniform, or wildly non-uniform, and either case is informative.
Later layers tend toward a settled pattern in which a small number of channels
dominate. Whether that pattern is visible in the stored parameters or only in
the activations is exactly the distinction being drawn here.

A last practical note. Reading is cheap and writing is expensive, in silicon
as much as in software. A generator that produces weights on demand pays
almost nothing to produce them in a different order, or with a different
scale folded in, because the order and the scale are properties of the address
map rather than of the stored bits. That freedom is worth something, and it is
worth knowing precisely how much.
"""


# --------------------------------------------------------------------------
# the forward pass
# --------------------------------------------------------------------------

def sigmoid(x):
    """Overflow-free logistic; real activations do reach -100."""
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def rms_norm(h, gamma, eps):
    n = h / np.sqrt((h * h).mean(-1, keepdims=True) + eps)
    return n, n * gamma


# --------------------------------------------------------------------------
# the data-free estimate, computed here as well
#
# Mirrors ``seedlm/actscale.py`` deliberately: the full producer matrices are
# on this machine and nowhere else, so estimating here is the only way to get
# an estimate for o_proj and down_proj without shipping 270 MB per layer.
# ``tests/test_actscale.py`` asserts the two implementations agree.
# --------------------------------------------------------------------------

def unit_mean(a):
    m = a.mean()
    return a / m if m > 0 else np.ones_like(a)


def propagate(W, a_in, chunk=4096):
    """``a_out[i] = sum_j W[i,j]^2 a_in[j]``."""
    out = np.empty(W.shape[0], np.float64)
    for lo in range(0, W.shape[0], chunk):
        b = W[lo:lo + chunk].astype(np.float64)
        out[lo:lo + chunk] = (b * b) @ a_in
    return out


def silu_second_moment(sigma, n_quad=129):
    """``E[silu(g)^2]`` for ``g ~ N(0, sigma^2)``, Gauss-Hermite."""
    sigma = np.atleast_1d(np.asarray(sigma, np.float64))
    nodes, wts = np.polynomial.hermite_e.hermegauss(n_quad)
    wts = wts / np.sqrt(2.0 * np.pi)
    g = sigma[:, None] * nodes[None, :]
    s = g * sigmoid(g)
    return (s * s) @ wts


def rope_tables(T, head_dim, theta, dtype=np.float32):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64)
                           / head_dim))
    ang = np.outer(np.arange(T), inv)
    return np.cos(ang).astype(dtype), np.sin(ang).astype(dtype)


def apply_rope(x, cos, sin):
    """``x`` is (T, H, D); rotate-half exactly as HF does."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--layers", default="", help="e.g. 0,14,27 (default: all)")
    ap.add_argument("--text-file", default="")
    ap.add_argument("--skip-embed-stats", action="store_true",
                    help="do not read the whole embedding table")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    H = cfg["hidden_size"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim", H // nh)
    rep = nh // nkv
    eps = cfg.get("rms_norm_eps", 1e-5)
    theta = cfg.get("rope_theta", 10000.0)
    nl = cfg["num_hidden_layers"]
    want = [int(x) for x in args.layers.split(",")] if args.layers else list(range(nl))

    tok_json = os.path.join(args.model, "tokenizer.json")
    if not os.path.exists(tok_json):
        raise SystemExit(f"need {tok_json} (download it into the model dir)")
    text = open(args.text_file, encoding="utf-8").read() if args.text_file \
        else SAMPLE
    bpe = BPE(tok_json)
    ids = bpe.encode(text)
    bos = cfg.get("bos_token_id")
    if bos is not None:
        ids = [bos] + ids
    while len(ids) < args.tokens:                 # repeat rather than pad
        ids = ids + ids[1:]
    ids = ids[:args.tokens]
    T = len(ids)
    print(f"{T} tokens, layers {want[0]}..{want[-1]} of {nl}", flush=True)

    sh = Shards(args.model)
    out, t0 = {}, time.time()
    h = sh.rows("model.embed_tokens.weight", ids)          # (T, H)

    # A data-free stand-in for E[n_j^2] at layer 0.  The residual stream there
    # is exactly one embedding row, so averaging the squared embedding over the
    # *whole vocabulary* -- no text, no token distribution -- estimates the
    # channel structure entering layer 0.  It is the one place the
    # "E[n_j^2] = 1 per channel" assumption can be repaired without data, and
    # for Llama-3 layer 0 it is the place that assumption is most suspect.
    if not args.skip_embed_stats:
        E = sh.get("model.embed_tokens.weight")
        acc = np.zeros(E.shape[1], np.float64)
        for lo in range(0, E.shape[0], 8192):
            b = E[lo:lo + 8192].astype(np.float64)
            acc += (b * b).sum(0)
        acc /= E.shape[0]
        out["embed.uniform_vocab_ms"] = unit_mean(acc)      # data-free
        emb = h.astype(np.float64)
        out["embed.sampled_ms"] = unit_mean((emb * emb).mean(0))  # this text
        print(f"  embedding second moment over {E.shape[0]} vocab rows",
              flush=True)
        del E
    cos, sin = rope_tables(T, hd, theta)
    causal = np.triu(np.full((T, T), -np.inf, np.float32), 1)


    for L in range(nl):
        p = f"model.layers.{L}."
        g_in = sh.get(p + "input_layernorm.weight")
        n, x = rms_norm(h, g_in, eps)
        if L in want:
            out[f"L{L}.n_in"] = (n * n).mean(0)
            out[f"L{L}.a_qkv"] = (x * x).mean(0)

        if L in want:
            out[f"L{L}.est_qkv"] = unit_mean(g_in.astype(np.float64) ** 2)
        q = x @ sh.get(p + "self_attn.q_proj.weight").T
        W_v = sh.get(p + "self_attn.v_proj.weight")
        if L in want:                       # o_proj's estimate: one hop back
            a_v = propagate(W_v, out[f"L{L}.est_qkv"]).reshape(nkv, hd)
            out[f"L{L}.est_o"] = unit_mean(
                np.repeat(a_v, rep, axis=0).reshape(-1))
        k = x @ sh.get(p + "self_attn.k_proj.weight").T
        v = x @ W_v.T
        for nm, arr in (("q", q), ("k", k), ("v", v)):
            b = sh.get(p + f"self_attn.{nm}_proj.bias")
            if b is not None:
                arr += b
        q = apply_rope(q.reshape(T, nh, hd), cos, sin)
        k = apply_rope(k.reshape(T, nkv, hd), cos, sin)
        v = v.reshape(T, nkv, hd)
        k = np.repeat(k, rep, 1)
        v = np.repeat(v, rep, 1)
        logits = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd) + causal
        logits -= logits.max(-1, keepdims=True)
        w = np.exp(logits)
        w /= w.sum(-1, keepdims=True)
        z = np.einsum("hts,shd->thd", w, v).reshape(T, nh * hd)
        if L in want:
            out[f"L{L}.a_o"] = (z * z).mean(0)
        h = h + z @ sh.get(p + "self_attn.o_proj.weight").T

        g_po = sh.get(p + "post_attention_layernorm.weight")
        n2, x2 = rms_norm(h, g_po, eps)
        if L in want:
            out[f"L{L}.n_post"] = (n2 * n2).mean(0)
            out[f"L{L}.a_mlp"] = (x2 * x2).mean(0)
        W_g, W_u = sh.get(p + "mlp.gate_proj.weight"), \
            sh.get(p + "mlp.up_proj.weight")
        if L in want:                       # down_proj's estimate, two hops
            a_post = unit_mean(g_po.astype(np.float64) ** 2)
            out[f"L{L}.est_mlp"] = a_post
            s2g, s2u = propagate(W_g, a_post), propagate(W_u, a_post)
            out[f"L{L}.est_down"] = unit_mean(
                silu_second_moment(np.sqrt(s2g)) * s2u)
            out[f"L{L}.est_sig2_gate"] = s2g
            out[f"L{L}.est_sig2_up"] = s2u
        gt = x2 @ W_g.T
        up = x2 @ W_u.T
        d = (gt * sigmoid(gt)) * up
        if L in want:
            out[f"L{L}.a_down"] = (d * d).mean(0)
            out[f"L{L}.sig2_gate"] = (gt * gt).mean(0)
            out[f"L{L}.sig2_up"] = (up * up).mean(0)
        h = h + d @ sh.get(p + "mlp.down_proj.weight").T
        del gt, up, d
        print(f"  layer {L:3d}  |h| rms {float(np.sqrt((h*h).mean())):9.3f}  "
              f"{time.time()-t0:6.1f}s", flush=True)

    save_safetensors(args.out, out)
    print(f"wrote {args.out}  ({len(out)} vectors, "
          f"{os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
