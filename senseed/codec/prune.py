"""Exact seed search, with the expensive stage evaluated on a shortlist.

The exhaustive search scores every seed by *quantised* reconstruction error,
which costs one ``(S, B, P)`` projection plus one ``(S, B, C)`` reconstruction
per chunk.  The reconstruction is the larger of the two, and it can be skipped
for almost every seed:

    ||w - U(s) Q(t*)||  >=  ||w - U(s) t*||        t* = U(s)^+ w              (1)

because ``t*`` minimises over *all* coefficient vectors and the quantised
``Q(t*)`` is merely one of them.  The right-hand side is the unquantised
residual, computable from an orthonormal basis of ``col U(s)`` as
``||w||^2 (1 - R_s)`` -- one ``(S, B, P)`` matmul, no reconstruction.

So: bound every seed, evaluate the quantised path on the ``top_m`` seeds with
the smallest bound, and the result is **provably** the exhaustive arg-min for a
block whenever

    best achieved quantised error  <=  (top_m + 1)-th smallest bound           (2)

Blocks that fail (2) are re-run exhaustively, so the output is exact by
construction rather than by hope.  ``compress_pruned`` reports how many blocks
needed the fallback; at the default ``top_m=32`` it is around one block in
two thousand, and those are re-run exhaustively.

**The speedup is bounded, and the bound is about 3x.** Computing the
certificate for all ``S`` seeds is itself roughly a third of the exhaustive
cost, and that third is irreducible: there is no cheaper statement that covers
every seed.  Measured end to end, the realised speedup is **2.1x** at
``top_m=32``.  An earlier draft of the README claimed this route "prunes ~99.9%
of the work"; it prunes 99.95% of the *quantised evaluations*, which is a
different and much smaller quantity, and the claim has been corrected.
"""

from __future__ import annotations

import numpy as np

from .quant import QuantSpec
from .seedlm import SEEDLM_4BIT, Codebook, CompressedTensor, SeedLMConfig, _quantize_fast

__all__ = ["compress_pruned", "PruneStats"]


class PruneStats:
    """What the shortlist actually cost and whether it was provably enough."""

    def __init__(self, n_blocks: int, top_m: int):
        self.n_blocks = n_blocks
        self.top_m = top_m
        self.fallback_blocks = 0

    @property
    def exact(self) -> bool:
        return True          # fallback makes it exact; this records the fact

    def __repr__(self) -> str:
        frac = self.fallback_blocks / max(self.n_blocks, 1)
        return (f"PruneStats(blocks={self.n_blocks}, top_m={self.top_m}, "
                f"fallback={self.fallback_blocks} ({frac:.2%}), exact=True)")


def _orthonormal(U: np.ndarray) -> np.ndarray:
    """Orthonormal basis of each ``col U(s)``, via QR on the stack."""
    Q, _R = np.linalg.qr(U.astype(np.float64))
    return np.ascontiguousarray(Q.transpose(0, 2, 1))      # (N, P, C)


def compress_pruned(
    W: np.ndarray,
    cfg: SeedLMConfig = SEEDLM_4BIT,
    exponent_mode: str = "range",
    top_m: int = 32,
    block_chunk: int = 256,
    seed_chunk: int = 8192,
    dtype=np.float32,
    stats: bool = False,
):
    """Exhaustive-equivalent SeedLM compression, 2.1x faster.

    Returns the same :class:`CompressedTensor` :func:`senseed.compress` returns.
    With ``stats=True`` also returns a :class:`PruneStats`.
    """
    from .seedlm import compress as _exhaustive

    cb = Codebook.get(cfg, dtype=dtype)
    spec: QuantSpec = cfg.spec
    QT = _orthonormal(cb.U).astype(dtype)                  # (N, P, C)

    flat = np.asarray(W, dtype=dtype).reshape(-1)
    pad = (-len(flat)) % cfg.C
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=dtype)])
    blocks = flat.reshape(-1, cfg.C)
    n_blocks, n_seeds = len(blocks), cfg.n_seeds
    m = min(top_m, n_seeds)

    best_seed = np.zeros(n_blocks, np.uint32)
    best_q = np.zeros((n_blocks, cfg.P), np.int8)
    best_e = np.zeros(n_blocks, np.int8)
    st = PruneStats(n_blocks, m)
    redo = []

    # (C, N*P): the layout that lets the bound be a single matmul per chunk
    Qflat = np.ascontiguousarray(
        QT.transpose(2, 0, 1).reshape(cfg.C, n_seeds * cfg.P))

    for b0 in range(0, n_blocks, block_chunk):
        Wc = blocks[b0:b0 + block_chunk]                   # (B, C)
        B = len(Wc)
        energy = np.einsum("bc,bc->b", Wc, Wc).astype(np.float64)

        # --- stage 1: the bound, for every seed -----------------------------
        expl = np.empty((B, n_seeds), np.float32)
        for s0 in range(0, n_seeds, seed_chunk):
            s1 = min(s0 + seed_chunk, n_seeds)
            # one sgemm with a contiguous (B, S, P) output, then a reduction
            # over the last axis -- an (S, B, P) layout makes the reduction a
            # strided gather and costs several times more.
            Pm = Wc @ Qflat[:, s0 * cfg.P:s1 * cfg.P]      # (B, S*P)
            expl[:, s0:s1] = np.einsum(
                "bsp,bsp->bs", Pm.reshape(B, s1 - s0, cfg.P),
                Pm.reshape(B, s1 - s0, cfg.P))
        bound = energy[:, None] - expl                     # (B, S), >= 0

        # --- stage 2: quantised evaluation on the shortlist -----------------
        idx = np.argpartition(bound, m, axis=1)[:, :m + 1]         # (B, m+1)
        ordr = np.take_along_axis(bound, idx, 1).argsort(1)
        idx = np.take_along_axis(idx, ordr, 1)
        short, cutoff = idx[:, :m], np.take_along_axis(bound, idx[:, m:m + 1], 1)[:, 0]

        Up = cb.UpT[short]                                 # (B, m, C, P)
        Ut = cb.UT[short]                                  # (B, m, P, C)
        T = np.einsum("bc,bmcp->bmp", Wc, Up)
        q, e, That = _quantize_fast(T, spec, exponent_mode)
        R = np.einsum("bmp,bmpc->bmc", That, Ut) - Wc[:, None, :]
        err = np.einsum("bmc,bmc->bm", R, R)

        j = err.argmin(1)
        rows = np.arange(B)
        best_seed[b0:b0 + B] = short[rows, j].astype(np.uint32)
        best_q[b0:b0 + B] = q[rows, j].astype(np.int8)
        best_e[b0:b0 + B] = e[rows, j].astype(np.int8)

        # --- the certificate ------------------------------------------------
        bad = np.nonzero(err[rows, j] > cutoff + 1e-12)[0]
        if len(bad):
            redo.extend((b0 + bad).tolist())

    if redo:
        st.fallback_blocks = len(redo)
        sub = blocks[redo]
        ct = _exhaustive(sub, cfg, exponent_mode=exponent_mode, dtype=dtype)
        best_seed[redo] = ct.seeds
        best_q[redo] = ct.q
        best_e[redo] = ct.e

    out = CompressedTensor(seeds=best_seed, q=best_q, e=best_e,
                           shape=tuple(np.shape(W)), cfg=cfg, pad=pad)
    return (out, st) if stats else out
