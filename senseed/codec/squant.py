"""S-Quant: seed-based weight compression with adaptive basis count.

Reference:
    Wang, Zou, Yin, He, Yu. "S-Quant: Rethinking Weight Quantization with
    Seed-Based Generation." ICML 2026.

Implemented from the paper; the repository the paper advertises
(github.com/wmz-max/S-Quant) is empty.

Relationship to SeedLM
----------------------
The generator is the same: an LFSR sequence normalised by
``(Z - 2**(R-1)) / (2**(R-1) - 1)`` -- S-Quant's Eq. 3 is SeedLM's
normalisation verbatim -- and each block is reconstructed as a linear
combination of basis vectors read off that sequence, with coefficients from a
least-squares projection.  Three things differ:

1. **Adaptive basis count.** SeedLM fixes the latent dimension ``P``.  S-Quant
   starts at ``k = 2`` and raises ``k`` until the *explained energy ratio*
   ``R_k = ||P_k w||^2 / ||w||^2`` clears a threshold ``R_th`` (Algorithm 1),
   so easy blocks spend fewer bases than hard ones.
2. **Coefficient format.** SeedLM stores 4-bit mantissas with a 4-bit
   power-of-two exponent per block.  S-Quant stores int8 coefficients with one
   FP16 scale shared across ``G`` consecutive blocks.
3. **Basis layout.** S-Quant takes "K consecutive basis elements", i.e. the
   columns of ``U`` are consecutive length-``B`` chunks of the sequence.
   SeedLM fills its matrix row-wise instead.  Both are read from the same
   cached cycle; ``basis_layout`` selects between them.

Storage per block is Eq. 13, ``S + 8k + 16/G`` bits, so bits per weight is that
over ``B``.  Because ``k`` varies per block, the rate is data-dependent -- the
headline "3.8 bits" is a mean, not a guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lfsr import full_cycle, normalize_states

__all__ = ["SQuantConfig", "SQuantCodebook", "SQuantResult", "compress_squant",
           "decompress_squant", "explained_energy_scan"]


@dataclass(frozen=True)
class SQuantConfig:
    """The paper's four-dimensional configuration ``<B, S, G, R_th>`` (Eq. 14)."""

    B: int = 16          # block size
    S: int = 16          # LFSR seed length in bits
    G: int = 8           # blocks sharing one FP16 coefficient scale
    R_th: float = 0.90   # target explained energy ratio
    k_min: int = 2       # Algorithm 1 starts here
    k_max: int = 12      # cap, so an unreachable threshold cannot loop forever
    coeff_bits: int = 8  # int8 coefficients
    scale_bits: int = 16 # FP16 shared scale

    @property
    def n_seeds(self) -> int:
        return (1 << self.S) - 1

    def bits_per_element(self, mean_k: float) -> float:
        """Eq. 13 divided by the block size."""
        return (self.S + self.coeff_bits * mean_k + self.scale_bits / self.G) / self.B

    def __repr__(self) -> str:
        return (f"SQuantConfig(B={self.B}, S={self.S}, G={self.G}, "
                f"R_th={self.R_th})")


class SQuantCodebook:
    """Cached bases and their orthonormalisations, for all ``2**S - 1`` seeds.

    The orthonormal form is what makes the adaptive search affordable.  For an
    orthonormal ``Q``, the explained energy of the first ``k`` columns is
    ``sum_{j<=k} (q_j . w)^2``, so a single projection onto ``Q`` yields
    ``R_k`` for *every* ``k`` at once via a cumulative sum.  Searching k = 2, 3,
    4, ... then costs no more than searching one fixed k.
    """

    _cache: dict[tuple, "SQuantCodebook"] = {}

    def __init__(self, cfg: SQuantConfig, dtype=np.float32,
                 basis_layout: str = "column"):
        self.cfg = cfg
        self.basis_layout = basis_layout
        B, k = cfg.B, cfg.k_max
        cycle = full_cycle(cfg.S)
        u = normalize_states(cycle, cfg.S, dtype=np.float64)
        n = len(u)

        ext = np.concatenate([u, u[: B * k]])
        win = np.lib.stride_tricks.sliding_window_view(ext, B * k)[:n]
        flat = np.ascontiguousarray(win)
        if basis_layout == "column":
            # U[:, j] is the j-th consecutive length-B chunk -- "K consecutive
            # basis elements" in the paper's words.
            U = flat.reshape(n, k, B).transpose(0, 2, 1)
        elif basis_layout == "row":
            U = flat.reshape(n, B, k)          # SeedLM's row-major fill
        else:
            raise ValueError(basis_layout)
        self.U = np.ascontiguousarray(U.astype(dtype))       # (n, B, k_max)

        Q, _ = np.linalg.qr(self.U.astype(np.float64))       # (n, B, k_max)
        # QR may return fewer columns if B < k_max; pad so indexing is uniform.
        if Q.shape[2] < k:
            Q = np.concatenate(
                [Q, np.zeros((n, B, k - Q.shape[2]))], axis=2)
        self.Q = np.ascontiguousarray(Q.astype(dtype))
        self.QT = np.ascontiguousarray(self.Q.transpose(0, 2, 1))

    @classmethod
    def get(cls, cfg: SQuantConfig, dtype=np.float32,
            basis_layout: str = "column") -> "SQuantCodebook":
        key = (cfg.B, cfg.S, cfg.k_max, np.dtype(dtype).str, basis_layout)
        if key not in cls._cache:
            cls._cache[key] = cls(cfg, dtype, basis_layout)
        return cls._cache[key]

    @property
    def nbytes(self) -> int:
        return self.U.nbytes + self.Q.nbytes + self.QT.nbytes


@dataclass
class SQuantResult:
    seeds: np.ndarray        # (n_blocks,)   chosen seed per block
    ks: np.ndarray           # (n_blocks,)   adaptive basis count per block
    coeffs: np.ndarray       # (n_blocks, k_max) int8, zero-padded past k
    scales: np.ndarray       # (n_groups,)   FP16 scale per group of G blocks
    R: np.ndarray            # (n_blocks,)   achieved explained energy ratio
    reached: np.ndarray      # (n_blocks,)   bool: R_th met within k_max
    shape: tuple
    cfg: SQuantConfig
    pad: int = 0

    @property
    def mean_k(self) -> float:
        return float(self.ks.mean())

    def bits_per_element(self) -> float:
        return self.cfg.bits_per_element(self.mean_k)

    def storage_bits(self) -> int:
        c = self.cfg
        n_groups = int(np.ceil(len(self.seeds) / c.G))
        return (len(self.seeds) * c.S + int(self.ks.sum()) * c.coeff_bits
                + n_groups * c.scale_bits)


def _quantize_groups(a: np.ndarray, ks: np.ndarray, cfg: SQuantConfig):
    """int8 coefficients with one FP16 scale per group of G blocks.

    Only the first ``k_i`` entries of row ``i`` are live; the padding must not
    influence the group's scale.
    """
    n, kmax = a.shape
    live = np.arange(kmax)[None, :] < ks[:, None]
    a = np.where(live, a, 0.0)

    n_groups = int(np.ceil(n / cfg.G))
    qmax = (1 << (cfg.coeff_bits - 1)) - 1        # 127
    q = np.zeros_like(a, dtype=np.int16)
    scales = np.zeros(n_groups, dtype=np.float32)

    for g in range(n_groups):
        sl = slice(g * cfg.G, min((g + 1) * cfg.G, n))
        blk = a[sl]
        amax = np.abs(blk).max() if blk.size else 0.0
        s = np.float16(amax / qmax) if amax > 0 else np.float16(1.0)
        s = np.float32(s)
        if s <= 0 or not np.isfinite(s):
            s = np.float32(1.0)
        scales[g] = s
        q[sl] = np.clip(np.rint(blk / s), -qmax - 1, qmax).astype(np.int16)

    return q.astype(np.int8), scales


def explained_energy_scan(blocks, cfg, cb, block_chunk=128, seed_chunk=4096):
    """Best explained-energy ratio over all seeds, for every k at once.

    Returns ``(best_R, best_seed)``, both ``(n_blocks, k_max)``: entry ``[i, k-1]``
    is the largest ``R_k`` any seed achieves on block ``i``, and the seed that
    achieves it.

    This is the whole search.  Because ``Q`` is orthonormal, ``R_k`` is a
    cumulative sum along the projection, so one pass over the seeds yields the
    answer for *every* ``k`` -- Algorithm 1's ``while`` loop costs nothing extra.
    It also means a sweep over ``R_th`` is free: run this once per ``(B, S)``
    and read off every threshold.
    """
    kmax = cfg.k_max
    n = len(blocks)
    energy = np.einsum("nb,nb->n", blocks, blocks).astype(np.float64)
    energy[energy == 0] = 1.0

    best_seed = np.zeros((n, kmax), dtype=np.int32)
    best_R = np.zeros((n, kmax), dtype=np.float64)

    for b0 in range(0, n, block_chunk):
        Wc = blocks[b0 : b0 + block_chunk]
        nb = len(Wc)
        loc_R = np.zeros((nb, kmax))
        loc_s = np.zeros((nb, kmax), dtype=np.int32)
        for s0 in range(0, cfg.n_seeds, seed_chunk):
            s1 = min(s0 + seed_chunk, cfg.n_seeds)
            QT = cb.QT[s0:s1]                            # (S, kmax, B)
            proj = Wc @ QT.transpose(0, 2, 1)            # (S, nb, kmax)
            np.square(proj, out=proj)
            cum = np.cumsum(proj, axis=2)                # explained energy, all k
            j = np.argmax(cum, axis=0)                   # (nb, kmax)
            take = np.take_along_axis(cum, j[None], axis=0)[0]
            upd = take > loc_R
            loc_R = np.where(upd, take, loc_R)
            loc_s = np.where(upd, s0 + j, loc_s)
        sl = slice(b0, b0 + nb)
        best_R[sl] = loc_R / energy[sl, None]
        best_seed[sl] = loc_s
    return best_R, best_seed


def compress_squant(
    W: np.ndarray,
    cfg: SQuantConfig = SQuantConfig(),
    dtype=np.float32,
    basis_layout: str = "column",
    block_chunk: int = 128,
    seed_chunk: int = 4096,
) -> SQuantResult:
    """Algorithm 1: adaptive basis selection, then grouped int8 quantisation.

    Follows the paper's ordering exactly -- the seed and ``k`` are chosen on the
    *unquantised* explained-energy ratio, and coefficients are quantised only
    afterwards.  (SeedLM instead scores every candidate seed on its quantised
    reconstruction, which is the stricter thing to do.)
    """
    cb = SQuantCodebook.get(cfg, dtype=dtype, basis_layout=basis_layout)
    B, kmax = cfg.B, cfg.k_max

    flat = np.asarray(W, dtype=dtype).reshape(-1)
    pad = (-len(flat)) % B
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=dtype)])
    blocks = flat.reshape(-1, B)
    n = len(blocks)

    best_R, best_seed = explained_energy_scan(
        blocks, cfg, cb, block_chunk=block_chunk, seed_chunk=seed_chunk)

    # Algorithm 1's while-loop: smallest k in [k_min, k_max] clearing R_th.
    ok = best_R >= cfg.R_th
    ok[:, : cfg.k_min - 1] = False
    reached = ok.any(axis=1)
    ks = np.where(reached, ok.argmax(axis=1) + 1, kmax)
    ks = np.maximum(ks, cfg.k_min)
    idx = np.arange(n)
    seeds = best_seed[idx, ks - 1]
    R = best_R[idx, ks - 1]

    # Least-squares coefficients on the chosen (seed, k).
    coeffs = np.zeros((n, kmax), dtype=np.float64)
    for k in np.unique(ks):
        m = ks == k
        U = cb.U[seeds[m]][:, :, :k].astype(np.float64)   # (m, B, k)
        w = blocks[m].astype(np.float64)[:, :, None]
        # SVD-based pinv in float64, for the same conditioning reason as SeedLM:
        # neighbouring seeds are overlapping windows, so some bases are close to
        # collinear and normal equations lose them in float32.
        coeffs[m, :k] = (np.linalg.pinv(U) @ w)[:, :, 0]

    q, scales = _quantize_groups(coeffs, ks, cfg)
    return SQuantResult(seeds=seeds.astype(np.uint32), ks=ks.astype(np.int16),
                        coeffs=q, scales=scales, R=R, reached=reached,
                        shape=tuple(W.shape), cfg=cfg, pad=pad)


def decompress_squant(res: SQuantResult, dtype=np.float32,
                      basis_layout: str = "column") -> np.ndarray:
    """Rebuild the tensor: ``sum_k a_k B_k`` with the dequantised coefficients."""
    cfg = res.cfg
    cb = SQuantCodebook.get(cfg, dtype=dtype, basis_layout=basis_layout)
    n = len(res.seeds)
    gi = np.arange(n) // cfg.G
    a = res.coeffs.astype(np.float64) * res.scales[gi][:, None]
    live = np.arange(cfg.k_max)[None, :] < res.ks[:, None]
    a = np.where(live, a, 0.0)
    U = cb.U[res.seeds.astype(np.int64)].astype(np.float64)   # (n, B, k_max)
    out = np.einsum("nbk,nk->nb", U, a).reshape(-1)
    if res.pad:
        out = out[: -res.pad]
    return out.reshape(res.shape).astype(dtype)
