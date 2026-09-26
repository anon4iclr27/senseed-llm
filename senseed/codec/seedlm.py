"""SeedLM compression: Algorithm 3 (search) and Algorithm 2 (reconstruction).

Reference:
    Shafipour et al., "SeedLM: Compressing LLM Weights into Seeds of
    Pseudo-Random Generators", ICLR 2025.

A weight tensor is flattened and cut into blocks of ``C`` contiguous elements.
For each block ``w in R^C`` we search all ``N = 2**K - 1`` candidate
pseudo-random bases ``U(s) in R^{C x P}``, take the least-squares coefficients
``t = U(s)^+ w``, quantise them, and keep the seed whose quantised
reconstruction ``U(s) t_hat`` is closest to ``w`` in squared error.

Storage per block is ``K + 4 + 4P`` bits (Eq. 3), independent of the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .lfsr import build_U_all, full_cycle, lfsr_state_for_offset
from .quant import DEFAULT_SPEC, QuantSpec, dequantize, shared_exponent

__all__ = [
    "SeedLMConfig",
    "SEEDLM_4BIT",
    "SEEDLM_3BIT",
    "Codebook",
    "CompressedTensor",
    "compress",
    "decompress",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SeedLMConfig:
    """Block size ``C``, latent dimension ``P``, LFSR length ``K``."""

    C: int
    P: int
    K: int
    mantissa_bits: int = 4
    exponent_bits: int = 4

    @property
    def n_seeds(self) -> int:
        return (1 << self.K) - 1

    @property
    def bits_per_block(self) -> int:
        """K bits of seed + one shared exponent + P mantissas."""
        return self.K + self.exponent_bits + self.mantissa_bits * self.P

    @property
    def bits_per_element(self) -> float:
        """Eq. 3:  M = (K + 4 + 4P) / C."""
        return self.bits_per_block / self.C

    @property
    def spec(self) -> QuantSpec:
        return QuantSpec(self.mantissa_bits, self.exponent_bits)

    def __repr__(self) -> str:
        return (
            f"SeedLMConfig(C={self.C}, P={self.P}, K={self.K}"
            f" -> {self.bits_per_element:g} bits/element)"
        )


# Table 1 of the paper.
SEEDLM_4BIT = SeedLMConfig(C=8, P=3, K=16)
SEEDLM_3BIT = SeedLMConfig(C=12, P=4, K=16)


# ---------------------------------------------------------------------------
# Codebook: the cached bases and their pseudo-inverses
# ---------------------------------------------------------------------------
class Codebook:
    """Caches ``U(s)`` and ``U(s)^+`` for every seed, per Section 3.3.

    The paper notes that caching the pseudo-inverses costs "at most 6.3MB"; see
    :attr:`nbytes` for the figure this implementation actually uses.
    """

    _cache: dict[tuple, "Codebook"] = {}

    def __init__(self, cfg: SeedLMConfig, dtype=np.float32):
        self.cfg = cfg
        self.dtype = dtype
        self.cycle = full_cycle(cfg.K)
        self.U = build_U_all(cfg.K, cfg.C, cfg.P, dtype=dtype)  # (N, C, P)
        self.Upinv = self._pseudo_inverse(self.U)               # (N, P, C)
        # Contiguous transposes, so the search loop never re-strides a 12MB
        # array on every one of its several hundred iterations.
        self.UT = np.ascontiguousarray(self.U.transpose(0, 2, 1))       # (N, P, C)
        self.UpT = np.ascontiguousarray(self.Upinv.transpose(0, 2, 1))  # (N, C, P)

    @staticmethod
    def _pseudo_inverse(U: np.ndarray) -> np.ndarray:
        """Batched Moore-Penrose pseudo-inverse, computed by SVD in float64.

        The SVD route matters here.  Consecutive seeds index overlapping
        windows of the same LFSR sequence, and a small fraction of those
        windows are badly conditioned -- for ``K=16, C=8, P=3`` the condition
        number of ``U(s)`` is 2.6 at the median but reaches 5e5 in the tail.
        Forming ``(U^T U)^-1 U^T`` squares that condition number, which in
        float32 destroys the coefficients for roughly 0.8% of seeds.  Doing the
        decomposition in float64 and casting down afterwards costs well under a
        second for all 65535 seeds and avoids the problem entirely.
        """
        out = np.linalg.pinv(U.astype(np.float64))
        return np.ascontiguousarray(out.astype(U.dtype))

    @property
    def nbytes(self) -> int:
        return self.U.nbytes + self.Upinv.nbytes

    @classmethod
    def get(cls, cfg: SeedLMConfig, dtype=np.float32) -> "Codebook":
        key = (cfg.C, cfg.P, cfg.K, np.dtype(dtype).str)
        if key not in cls._cache:
            cls._cache[key] = cls(cfg, dtype=dtype)
        return cls._cache[key]


# ---------------------------------------------------------------------------
# Fast inner-loop quantiser
# ---------------------------------------------------------------------------
def _quantize_fast(
    t: np.ndarray, spec: QuantSpec, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantise along the last axis; return ``(q, e, t_hat)``.

    Numerically identical to :func:`seedlm.quant.quantize` -- it calls the same
    :func:`seedlm.quant.shared_exponent`, so the two cannot disagree -- but
    written for the search loop, where it runs on arrays of shape
    ``(n_seeds_chunk, n_blocks_chunk, P)`` and dominates the runtime:

    * the max-abs reduction is folded over ``P`` slices so the ``(S, B, P)``
      absolute-value temporary is never materialised;
    * ``np.ldexp`` replaces ``exp2`` + reciprocal + multiply, since scaling by
      a power of two is just an exponent-field add;
    * every elementwise step writes in place.

    Together these are about 3x faster than the straightforward formulation.
    """
    amax = np.abs(t[..., 0])
    for i in range(1, t.shape[-1]):
        np.maximum(amax, np.abs(t[..., i]), out=amax)

    zero = amax == 0
    if zero.any():
        amax = np.where(zero, np.asarray(1, dtype=t.dtype), amax)

    e = shared_exponent(amax, spec, mode)
    np.clip(e, spec.e_min, spec.e_max, out=e)
    en = e[..., None]

    q = np.ldexp(t, -en)
    np.rint(q, out=q)
    np.clip(q, spec.q_min, spec.q_max, out=q)
    t_hat = np.ldexp(q, en)

    if zero.any():
        q[zero] = 0
        t_hat[zero] = 0
        e = np.where(zero, spec.e_min, e)
    return q, e, t_hat


# ---------------------------------------------------------------------------
# Compressed representation
# ---------------------------------------------------------------------------
@dataclass
class CompressedTensor:
    """The complete stored payload for one weight tensor."""

    seeds: np.ndarray        # (n_blocks,) uint32 -- cached-mode offsets
    q: np.ndarray            # (n_blocks, P) int8 -- mantissas
    e: np.ndarray            # (n_blocks,) int8   -- shared exponents
    shape: tuple[int, ...]
    cfg: SeedLMConfig
    pad: int = 0             # elements of zero padding on the final block

    @property
    def n_blocks(self) -> int:
        return len(self.seeds)

    @property
    def n_elements(self) -> int:
        return int(np.prod(self.shape))

    def storage_bits(self) -> int:
        """Actual payload size, ignoring any container overhead."""
        return self.n_blocks * self.cfg.bits_per_block

    def bits_per_element(self) -> float:
        return self.storage_bits() / self.n_elements

    def hardware_seeds(self, codebook: "Codebook | None" = None) -> np.ndarray:
        """Seeds re-expressed as LFSR register states.

        The search works in cached-offset space (Algorithm 2); a hardware
        decoder needs the register state that generates the same window.
        """
        cb = codebook or Codebook.get(self.cfg)
        return np.array(
            [lfsr_state_for_offset(cb.cycle, int(s)) for s in self.seeds],
            dtype=np.uint32,
        )


# ---------------------------------------------------------------------------
# Algorithm 3: search
# ---------------------------------------------------------------------------
def compress(
    W: np.ndarray,
    cfg: SeedLMConfig = SEEDLM_4BIT,
    exponent_mode: str = "range",
    n_candidate_seeds: int | None = None,
    block_chunk: int = 256,
    seed_chunk: int = 8192,
    dtype=np.float32,
    progress=None,
) -> CompressedTensor:
    """Compress ``W`` with SeedLM.

    Parameters
    ----------
    W
        Weight tensor of any shape; flattened in C order and zero-padded up to
        a multiple of ``cfg.C``.
    n_candidate_seeds
        Restrict the search to the first this-many seeds instead of all
        ``2**K - 1``.  Only for ablations -- the paper searches all of them.
    block_chunk, seed_chunk
        Tiling of the ``(block, seed)`` search grid.  Peak memory is roughly
        ``block_chunk * seed_chunk * (C + P) * 4`` bytes.
    progress
        Optional callable invoked as ``progress(done, total)`` per block chunk.
    """
    cb = Codebook.get(cfg, dtype=dtype)
    spec = cfg.spec

    flat = np.asarray(W, dtype=dtype).reshape(-1)
    pad = (-len(flat)) % cfg.C
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=dtype)])
    blocks = flat.reshape(-1, cfg.C)
    n_blocks = len(blocks)

    n_seeds = cfg.n_seeds if n_candidate_seeds is None else min(n_candidate_seeds, cfg.n_seeds)

    best_seed = np.zeros(n_blocks, dtype=np.uint32)
    best_q = np.zeros((n_blocks, cfg.P), dtype=np.int8)
    best_e = np.zeros(n_blocks, dtype=np.int8)
    best_err = np.full(n_blocks, np.inf, dtype=np.float64)

    for b0 in range(0, n_blocks, block_chunk):
        Wc = blocks[b0 : b0 + block_chunk]                      # (B, C)
        B = len(Wc)
        loc_err = np.full(B, np.inf, dtype=dtype)
        loc_seed = np.zeros(B, dtype=np.int64)
        loc_q = np.zeros((B, cfg.P), dtype=dtype)
        loc_e = np.zeros(B, dtype=np.int32)

        for s0 in range(0, n_seeds, seed_chunk):
            s1 = min(s0 + seed_chunk, n_seeds)  # cap: n_seeds may be < a full tile
            UpT = cb.UpT[s0:s1]                                  # (S, C, P)
            UT = cb.UT[s0:s1]                                    # (S, P, C)

            # Step 2: t = U^+ w, for every (seed, block) pair at once.
            T = Wc @ UpT                                         # (S, B, P)

            # Step 3: quantise the coefficients.
            q, e, That = _quantize_fast(T, spec, exponent_mode)

            # Step 4: reconstruction error with the *quantised* coefficients.
            # The subtraction is in place: materialising (R - Wc) as a fresh
            # (S, B, C) temporary costs more than the matmul that produced R.
            R = That @ UT                                        # (S, B, C)
            R -= Wc
            err = np.einsum("sbc,sbc->sb", R, R)                 # (S, B)

            # Step 5: running arg-min over seeds.
            j = np.argmin(err, axis=0)                           # (B,)
            cand = err[j, np.arange(B)]
            take = cand < loc_err
            if take.any():
                idx = np.nonzero(take)[0]
                loc_err[idx] = cand[idx]
                loc_seed[idx] = s0 + j[idx]
                loc_q[idx] = q[j[idx], idx]
                loc_e[idx] = e[j[idx], idx]

        sl = slice(b0, b0 + B)
        best_seed[sl] = loc_seed.astype(np.uint32)
        best_q[sl] = loc_q.astype(np.int8)
        best_e[sl] = loc_e.astype(np.int8)
        best_err[sl] = loc_err
        if progress is not None:
            progress(min(b0 + block_chunk, n_blocks), n_blocks)

    return CompressedTensor(
        seeds=best_seed,
        q=best_q,
        e=best_e,
        shape=tuple(W.shape),
        cfg=cfg,
        pad=pad,
    )


# ---------------------------------------------------------------------------
# Algorithm 2: reconstruction
# ---------------------------------------------------------------------------
def decompress(ct: CompressedTensor, dtype=np.float32) -> np.ndarray:
    """Rebuild the weight tensor from seeds and quantised coefficients."""
    cfg = ct.cfg
    cb = Codebook.get(cfg, dtype=dtype)
    t = dequantize(ct.q, ct.e, dtype=dtype)                      # (n_blocks, P)
    U = cb.U[ct.seeds.astype(np.int64)]                          # (n_blocks, C, P)
    blocks = np.einsum("ncp,np->nc", U, t)
    flat = blocks.reshape(-1)
    if ct.pad:
        flat = flat[: -ct.pad]
    return flat.reshape(ct.shape).astype(dtype)


def relative_error(W: np.ndarray, What: np.ndarray) -> float:
    """``||W - What||_F / ||W||_F``."""
    W = np.asarray(W, dtype=np.float64)
    What = np.asarray(What, dtype=np.float64)
    return float(np.linalg.norm(W - What) / np.linalg.norm(W))
