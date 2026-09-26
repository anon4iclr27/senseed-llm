"""SenSeed: seed-based weight compression with rate-optimal basis allocation.

Built from three measured deficits in SeedLM (ICLR 2025) and S-Quant (ICML
2026), each toggleable so the contribution of each can be isolated.

The three changes
-----------------

**1. Allocation by marginal gain, not explained energy** (``k_mode="marginal"``).

S-Quant raises ``k`` on each block until its *explained energy ratio* clears a
global threshold ``R_th``.  That equalises a **relative** quantity.  For
minimising total squared error under a bit budget the correct rule equalises the
**marginal** gain per bit: allocate the next basis to whichever block gains most
absolute error reduction from it.  Formally, with cost ``c`` bits per basis,
the optimum is a water-filling threshold ``lambda`` and

    k_i = max { k : Delta_i(k) / c >= lambda },
    Delta_i(k) = ||w_i||^2 * (R_i(k) - R_i(k-1)) * imp_i

Equalising relative energy systematically over-serves low-energy blocks: a block
with small ``||w||`` gets bases that would have removed more total error
elsewhere.  ``imp_i`` is an optional per-block importance weight -- pass Fisher
diagonals here to get sensitivity-weighted allocation, but note that *needs
calibration data* and therefore forfeits the data-free property that is
SeedLM's and S-Quant's headline claim.  Default is 1 (data-free).

**2. Shared exponent across blocks** (``exp_group > 1``).

SeedLM spends 4 bits per block on a private exponent.  At ``B=8`` that is 0.5
bits per weight -- an eighth of a 4-bit budget -- to buy per-block dynamic
range that the Llama-3-70B measurements showed is mostly not needed (removing
the exponent floor entirely changed error by 0.4%).  Sharing one exponent over
``G`` blocks costs ``4/G`` bits and frees the difference for bases, which are
what actually buy accuracy.

**3. Condition-filtered candidate set** (``cond_max``).

Neighbouring seeds are overlapping windows of one LFSR sequence, so a minority
of bases are close to collinear -- cond(U) is ~2.6 at the median but reaches
5e5.  Those produce least-squares coefficients thousands of times the weights
they encode.  With a *per-block* exponent that is merely wasteful; with a
*shared* exponent one such block destroys its whole group, because a 4-bit
mantissa spans 16 levels and can only absorb ~8x of spread.  Dropping
ill-conditioned seeds costs nothing (65535 candidates is far more than needed)
and is what makes a shared exponent viable at 4-bit mantissas at all.

**4. Quantisation-aware seed selection** (``quant_aware=True``).

S-Quant's Algorithm 1 picks the seed on the *unquantised* explained-energy
ratio and quantises afterwards, so it can choose a basis whose coefficients do
not survive quantisation (measured: coefficients reaching 2029 on blocks whose
weights are ~0.02).  Scoring candidates on their quantised reconstruction, as
SeedLM does, removes that failure mode at no rate cost.

Rate accounting
---------------

    bits/weight = (S + mantissa_bits * mean_k + exp_bits/exp_group + sig) / B

``sig`` is the cost of signalling ``k`` to the decoder, ``ceil(log2(k_max -
k_min + 1))`` bits per block.  S-Quant's Eq. 13 omits this; it is charged here,
because a decoder genuinely cannot know how many coefficients to read without
it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .seedlm import _quantize_fast
from .quant import QuantSpec, dequantize
from .squant import SQuantCodebook, explained_energy_scan

__all__ = ["SenSeedConfig", "compress_senseed", "decompress_senseed", "SenSeedResult"]


@dataclass(frozen=True)
class SenSeedConfig:
    B: int = 8                 # block size
    S: int = 16                # LFSR seed bits
    mantissa_bits: int = 4     # per coefficient
    exp_bits: int = 4          # shared exponent width
    exp_group: int = 8         # blocks sharing one exponent (1 = SeedLM)
    k_mode: str = "marginal"   # "fixed" | "energy" | "marginal" | "schedule"
    k_fixed: int = 3
    k_min: int = 2
    k_max: int = 6
    R_th: float = 0.90         # for k_mode="energy" (S-Quant's rule)
    target_bits: float = 4.0   # for k_mode="marginal", water-fill to this rate
    signal_k: bool = True
    schedule_slope: float = 0.0  # k_mode="schedule": bases per e-fold of importance
    quant_aware: bool = True
    basis_layout: str = "row"   # "row" = SeedLM's fill, "column" = S-Quant's
    cond_max: float = 0.0       # 0 = keep every seed; else drop cond(U_k) above this

    @property
    def n_seeds(self) -> int:
        return (1 << self.S) - 1

    @property
    def sig_bits(self) -> float:
        # "schedule" needs no per-block signalling by construction: the decoder
        # recomputes k from the same importance vector the encoder used.
        if self.k_mode in ("fixed", "schedule") or not self.signal_k:
            return 0.0
        return float(math.ceil(math.log2(self.k_max - self.k_min + 1)))

    def bits_per_element(self, mean_k: float) -> float:
        return (self.S + self.mantissa_bits * mean_k
                + self.exp_bits / self.exp_group + self.sig_bits) / self.B

    @property
    def min_bits(self) -> float:
        """Floor rate: every block still needs ``k_min`` bases."""
        return self.bits_per_element(self.k_min)

    @property
    def max_bits(self) -> float:
        return self.bits_per_element(self.k_max)

    def feasible(self) -> bool:
        return (self.k_mode not in ("marginal", "schedule")
                or self.target_bits >= self.min_bits - 1e-9)

    def max_mean_k(self) -> float:
        """Largest mean k affordable at ``target_bits``."""
        budget = self.target_bits * self.B - self.S - self.exp_bits / self.exp_group \
            - self.sig_bits
        return budget / self.mantissa_bits


@dataclass
class SenSeedResult:
    seeds: np.ndarray
    ks: np.ndarray
    q: np.ndarray            # (n, k_max) int8 mantissas
    e: np.ndarray            # (n_groups,) shared exponents
    shape: tuple
    cfg: SenSeedConfig
    pad: int = 0

    @property
    def mean_k(self) -> float:
        return float(self.ks.mean())

    def bits_per_element(self) -> float:
        return self.cfg.bits_per_element(self.mean_k)


def _concave_envelope(R: np.ndarray) -> np.ndarray:
    """Make each block's R(k) curve concave in k.

    Water-filling on marginal gains is only valid when the gains are
    non-increasing.  The k-th basis of a *searched* seed is not guaranteed to
    help less than the (k-1)-th, so take the upper concave envelope; the
    resulting allocation is then the exact greedy optimum over the envelope,
    and never claims a gain the real curve cannot deliver.
    """
    R = R.copy()
    n, kmax = R.shape
    for k in range(1, kmax):
        # enforce non-increasing increments by pulling later gains down
        prev_gain = R[:, k - 1] - (R[:, k - 2] if k >= 2 else 0.0)
        gain = R[:, k] - R[:, k - 1]
        bad = gain > prev_gain
        if bad.any():
            R[bad, k] = R[bad, k - 1] + prev_gain[bad]
    return R


def _allocate(R, energy, cfg, importance=None):
    """Choose k per block.  Returns an int array in [k_min, k_max]."""
    n, kmax = R.shape
    if cfg.k_mode == "fixed":
        return np.full(n, cfg.k_fixed, dtype=np.int16)

    if cfg.k_mode == "energy":                      # S-Quant's rule
        ok = R >= cfg.R_th
        ok[:, : cfg.k_min - 1] = False
        ks = np.where(ok.any(1), ok.argmax(1) + 1, kmax)
        return np.maximum(ks, cfg.k_min).astype(np.int16)

    if cfg.k_mode == "schedule":
        return _schedule_alloc(n, cfg, importance)

    if cfg.k_mode != "marginal":
        raise ValueError(cfg.k_mode)

    # Water-filling on absolute marginal gain per basis.
    Rc = _concave_envelope(R)
    imp = np.ones(n) if importance is None else np.asarray(importance, float)
    gain = np.diff(Rc, axis=1, prepend=0.0) * (energy * imp)[:, None]   # (n, kmax)

    target = np.clip(cfg.max_mean_k(), cfg.k_min, cfg.k_max)
    lo, hi = 0.0, float(gain.max()) * 1.000001
    for _ in range(60):                              # bisect the threshold
        lam = 0.5 * (lo + hi)
        take = gain >= lam
        take[:, : cfg.k_min] = True
        take[:, cfg.k_max :] = False
        ks = take.sum(1)
        if ks.mean() > target:
            lo = lam
        else:
            hi = lam
    take = gain >= hi
    take[:, : cfg.k_min] = True
    take[:, cfg.k_max :] = False
    ks = np.clip(take.sum(1), cfg.k_min, cfg.k_max)
    return ks.astype(np.int16)


def _schedule_alloc(n, cfg, importance):
    """Allocate ``k`` from importance alone -- so the decoder can recompute it.

    Every other allocation rule here needs the *weights* to decide: S-Quant's
    threshold reads the achieved explained-energy ratio, marginal water-filling
    reads block energies.  The decoder has neither before it decodes, so the
    choice has to be transmitted -- ``ceil(log2(k_max-k_min+1))`` bits per
    block, 2 bits at B=8, which is 0.25 bits/element.  That overhead has cost
    more than adaptivity has returned on every real tensor measured.

    Importance derived from ``gamma`` is different in kind: the decoder already
    holds ``gamma``.  If ``k`` is a fixed monotone function of importance, the
    decoder recomputes the whole allocation from the norm vector plus a handful
    of numbers per layer, and the per-block signalling disappears.  For a source
    whose distortion falls geometrically in ``k``, reverse water-filling makes
    that function affine in ``log`` importance::

        k_i = round( kbar + slope * (log imp_i - mean log imp) )

    ``slope = 0`` is uniform ``k`` -- exactly SeedLM -- so the schedule is a
    strict generalisation and the comparison at matched rate is honest.  The
    offset is bisected so the mean survives clipping and the bit budget is hit.
    """
    kbar = float(np.clip(cfg.max_mean_k(), cfg.k_min, cfg.k_max))
    if importance is None or cfg.schedule_slope == 0.0:
        base = np.full(n, kbar)
    else:
        imp = np.asarray(importance, float).ravel()
        if len(imp) != n:
            raise ValueError(f"importance has {len(imp)} entries, "
                             f"{n} blocks")
        z = np.log(np.maximum(imp, 1e-300))
        base = kbar + cfg.schedule_slope * (z - z.mean())

    lo, hi = -float(cfg.k_max), float(cfg.k_max)
    for _ in range(60):                      # bisect the offset onto the budget
        mid = 0.5 * (lo + hi)
        ks = np.clip(np.round(base + mid), cfg.k_min, cfg.k_max)
        if ks.mean() > kbar:
            hi = mid
        else:
            lo = mid
    ks = np.clip(np.round(base + lo), cfg.k_min, cfg.k_max)
    return ks.astype(np.int16)


def _quantize_shared(a, ks, cfg):
    """Mantissas with one power-of-two exponent per ``exp_group`` blocks."""
    spec = QuantSpec(cfg.mantissa_bits, cfg.exp_bits)
    n, kmax = a.shape
    live = np.arange(kmax)[None, :] < ks[:, None]
    a = np.where(live, a, 0.0).astype(np.float32)

    G = cfg.exp_group
    ng = int(np.ceil(n / G))
    pad = ng * G - n
    ap = np.concatenate([a, np.zeros((pad, kmax), np.float32)]) if pad else a
    grouped = ap.reshape(ng, G * kmax)
    q, e, _ = _quantize_fast(grouped, spec, "range")   # one exponent per row
    q = q.reshape(ng * G, kmax)[:n]
    return q.astype(np.int8), e.astype(np.int8)


def _dequantize_shared(q, e, cfg):
    gi = np.arange(len(q)) // cfg.exp_group
    return q.astype(np.float64) * np.exp2(e[gi].astype(np.float64))[:, None]


def compress_senseed(W, cfg: SenSeedConfig = SenSeedConfig(), importance=None,
                  dtype=np.float32, block_chunk=128, seed_chunk=4096,
                  top_m: int = 24, backend: str = "numpy") -> SenSeedResult:
    """Compress ``W``.  ``importance`` is an optional per-block weight.

    ``backend`` selects who runs the quantisation-aware seed search, which is
    ~90% of the work: ``"numpy"`` (the default, and what every published number
    in this repository was produced with) or a torch device string such as
    ``"cuda"`` or ``"cuda:1"``.  The two agree on the reconstruction they
    achieve; they can disagree on *which* seed they pick where two seeds tie,
    because the reduction order in a batched matmul differs.  See
    ``tests/test_gpu.py``.
    """
    from .squant import SQuantConfig
    if not cfg.feasible():
        raise ValueError(
            f"target_bits={cfg.target_bits} is below the floor rate "
            f"{cfg.min_bits:.3f} implied by k_min={cfg.k_min} at B={cfg.B}, "
            f"S={cfg.S} (every block needs at least k_min bases). "
            f"Lower k_min, raise B, or raise target_bits.")
    qcfg = SQuantConfig(B=cfg.B, S=cfg.S, k_max=cfg.k_max)
    cb = SQuantCodebook.get(qcfg, dtype=dtype, basis_layout=cfg.basis_layout)
    mask = _condition_mask(cb, cfg)

    flat = np.asarray(W, dtype=dtype).reshape(-1)
    pad = (-len(flat)) % cfg.B
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=dtype)])
    blocks = flat.reshape(-1, cfg.B)
    n = len(blocks)
    energy = np.einsum("nb,nb->n", blocks, blocks).astype(np.float64)
    energy[energy == 0] = 1.0

    # The explained-energy scan exists to (a) drive the allocator and (b) seed
    # the initial choice.  Under `k_mode="schedule"` the allocator reads only
    # the importance vector, and under `quant_aware` every seed it picks is
    # overwritten by the quantisation-aware pass below.  So for that
    # combination the scan is dead work -- and it is a full pass over all
    # 65535 seeds, which measured 43% of this function's runtime.  Skipped
    # only when nothing downstream can observe the difference; the
    # `mask is None` guard keeps the condition-filtered path on the old route,
    # where a fully masked seed chunk can leave a block un-updated.
    skip_scan = (cfg.k_mode == "schedule" and cfg.quant_aware and mask is None)
    if skip_scan:
        ks = _schedule_alloc(n, cfg, importance)
        seeds = np.zeros(n, dtype=np.int64)
    else:
        R, seed_at_k = _scan_masked(blocks, qcfg, cb, mask,
                                    block_chunk=block_chunk,
                                    seed_chunk=seed_chunk)
        ks = _allocate(R, energy, cfg, importance)
        seeds = seed_at_k[np.arange(n), ks - 1].astype(np.int64)

    # Exact coefficients at the chosen (seed, k).  Also overwritten by the
    # quantisation-aware pass, so skipped on the same condition.
    coeffs = np.zeros((n, cfg.k_max))
    if not skip_scan:
        for k in np.unique(ks):
            m = ks == k
            U = cb.U[seeds[m]][:, :, :k].astype(np.float64)
            coeffs[m, :k] = (np.linalg.pinv(U)
                             @ blocks[m].astype(np.float64)[:, :, None])[:, :, 0]
    q, e = _quantize_shared(coeffs, ks, cfg)

    if cfg.quant_aware:
        if backend == "numpy":
            _quant_aware_numpy(blocks, cb, cfg, mask, ks, seeds, coeffs,
                               block_chunk, seed_chunk)
        else:
            from .gpu import quant_aware_torch
            quant_aware_torch(blocks, cb, cfg, mask, ks, seeds, coeffs,
                              block_chunk, seed_chunk, device=backend)
        q, e = _quantize_shared(coeffs, ks, cfg)

    return SenSeedResult(seeds=seeds.astype(np.uint32), ks=ks, q=q, e=e,
                      shape=tuple(W.shape), cfg=cfg, pad=pad)


def _quant_aware_numpy(blocks, cb, cfg, mask, ks, seeds, coeffs,
                       block_chunk, seed_chunk):
    """Exhaustive quantisation-aware re-selection, per distinct k, in numpy.

    A shortlist is not enough: the seed with the best *unquantised* explained
    energy is frequently not the one whose *quantised* reconstruction is best,
    and scoring only a few dozen candidates leaves most of SeedLM's advantage
    on the table.  This is SeedLM's own search, run once per k group -- and it
    is ~90% of the runtime of a whole-checkpoint compression, which is why
    ``senseed/codec/gpu.py`` exists.

    ``seeds`` and ``coeffs`` are updated in place.
    """
    spec = QuantSpec(cfg.mantissa_bits, cfg.exp_bits)
    for k in np.unique(ks):
        grp = np.nonzero(ks == k)[0]
        # Tile over blocks as well as seeds.  The intermediates here are
        # (seed_chunk, blocks, k) and (seed_chunk, blocks, B); tiling only
        # the seed axis makes the second one scale with the whole tensor,
        # which reaches 6 GB for a single worker on a 4864x896 MLP matrix
        # and OOMs long before it finishes.  Chunking the block axis bounds
        # it at seed_chunk * block_chunk * B and changes nothing else: the
        # arg-min is taken per block, so blocks never interact.
        for g0 in range(0, len(grp), block_chunk):
            m = grp[g0:g0 + block_chunk]
            Wm = blocks[m]
            best = np.full(len(m), np.inf, dtype=np.float64)
            for s0 in range(0, cfg.n_seeds, seed_chunk):
                s1 = min(s0 + seed_chunk, cfg.n_seeds)
                if mask is not None and not mask[s0:s1, k - 1].any():
                    continue
                Uk = cb.U[s0:s1, :, :k]
                Up = _pinv_cache(cb, s0, s1, k)
                T = Wm @ Up.transpose(0, 2, 1)            # (S, m, k)
                q_, e_, That = _quantize_fast(T, spec, "range")
                Rc = That @ Uk.transpose(0, 2, 1)         # (S, m, B)
                Rc -= Wm
                err = np.einsum("smb,smb->sm", Rc, Rc).astype(np.float64)
                if mask is not None:
                    err = np.where(mask[s0:s1, k - 1][:, None], err, np.inf)
                j = np.argmin(err, axis=0)
                cand = err[j, np.arange(len(m))]
                upd = cand < best
                if upd.any():
                    idx = np.nonzero(upd)[0]
                    best[idx] = cand[idx]
                    seeds[m[idx]] = s0 + j[idx]
                    coeffs[m[idx], :k] = That[j[idx], idx]
                    coeffs[m[idx], k:] = 0.0


_PINV: dict = {}


def _pinv_all(cb, k):
    """Pseudo-inverses of ``U[:, :, :k]`` for *every* seed, computed in float64.

    Cached per ``k`` rather than per seed chunk.  The old key included the
    chunk bounds, which at ``seed_chunk=4096`` over 65535 seeds means 16
    entries per ``k`` and up to 80 live entries -- past the 64-entry cap, so
    the cache cleared itself and every chunk recomputed a batched SVD on every
    pass.  One entry per ``k`` is both correct and smaller: 12.6 MB at k=6
    against 63 MB for the chunked form.
    """
    key = (id(cb), k)
    if key not in _PINV:
        if len(_PINV) > 16:
            _PINV.clear()
        _PINV[key] = np.ascontiguousarray(
            np.linalg.pinv(cb.U[:, :, :k].astype(np.float64)).astype(cb.U.dtype))
    return _PINV[key]


def _pinv_cache(cb, s0, s1, k):
    """``pinv(U[s0:s1, :, :k])`` -- a slice of the per-``k`` cache above."""
    return _pinv_all(cb, k)[s0:s1]


def _shortlist(blocks, cb, k, top_m, seed_chunk, mask=None):
    """Top-``top_m`` seeds per block by unquantised explained energy at ``k``."""
    n = len(blocks)
    nseed = len(cb.U)
    best_v = np.full((n, top_m), -np.inf)
    best_s = np.zeros((n, top_m), dtype=np.int64)
    for s0 in range(0, nseed, seed_chunk):
        s1 = min(s0 + seed_chunk, nseed)
        QT = cb.QT[s0:s1, :k]                       # (S, k, B)
        p = blocks @ QT.transpose(0, 2, 1)          # (S, n, k)
        v = np.einsum("snk,snk->sn", p, p).T        # (n, S)
        if mask is not None:
            v = np.where(mask[s0:s1, k - 1][None, :], v, -np.inf)
        merged_v = np.concatenate([best_v, v], axis=1)
        merged_s = np.concatenate([best_s, np.broadcast_to(
            np.arange(s0, s1), (n, s1 - s0))], axis=1)
        idx = np.argpartition(-merged_v, top_m - 1, axis=1)[:, :top_m]
        best_v = np.take_along_axis(merged_v, idx, 1)
        best_s = np.take_along_axis(merged_s, idx, 1)
    return best_s


_COND_CACHE: dict = {}


def _condition_mask(cb, cfg):
    """Boolean ``(n_seeds, k_max)``: is U(s)[:, :k] well enough conditioned?"""
    if not cfg.cond_max:
        return None
    key = (cfg.B, cfg.S, cfg.k_max, cfg.basis_layout, cfg.cond_max)
    if key in _COND_CACHE:
        return _COND_CACHE[key]
    U = cb.U.astype(np.float64)
    m = np.ones((len(U), cfg.k_max), dtype=bool)
    for k in range(2, cfg.k_max + 1):
        sv = np.linalg.svd(U[:, :, :k], compute_uv=False)
        cond = sv[:, 0] / np.maximum(sv[:, -1], 1e-300)
        m[:, k - 1] = cond <= cfg.cond_max
    m[:, 0] = True
    _COND_CACHE[key] = m
    return m


def _scan_masked(blocks, qcfg, cb, mask, block_chunk=128, seed_chunk=4096):
    """explained_energy_scan, restricted to the allowed seeds at each k."""
    if mask is None:
        return explained_energy_scan(blocks, qcfg, cb,
                                     block_chunk=block_chunk,
                                     seed_chunk=seed_chunk)
    kmax = qcfg.k_max
    n = len(blocks)
    energy = np.einsum("nb,nb->n", blocks, blocks).astype(np.float64)
    energy[energy == 0] = 1.0
    best_seed = np.zeros((n, kmax), dtype=np.int32)
    best_R = np.full((n, kmax), -np.inf)
    for b0 in range(0, n, block_chunk):
        Wc = blocks[b0 : b0 + block_chunk]
        nb = len(Wc)
        loc_R = np.full((nb, kmax), -np.inf)
        loc_s = np.zeros((nb, kmax), dtype=np.int32)
        for s0 in range(0, qcfg.n_seeds, seed_chunk):
            s1 = min(s0 + seed_chunk, qcfg.n_seeds)
            QT = cb.QT[s0:s1]
            proj = Wc @ QT.transpose(0, 2, 1)
            np.square(proj, out=proj)
            cum = np.cumsum(proj, axis=2)
            cum = np.where(mask[s0:s1][:, None, :], cum, -np.inf)
            j = np.argmax(cum, axis=0)
            take = np.take_along_axis(cum, j[None], axis=0)[0]
            upd = take > loc_R
            loc_R = np.where(upd, take, loc_R)
            loc_s = np.where(upd, s0 + j, loc_s)
        sl = slice(b0, b0 + nb)
        best_R[sl] = np.where(np.isfinite(loc_R), loc_R / energy[sl, None], 0.0)
        best_seed[sl] = loc_s
    return best_R, best_seed


def decompress_senseed(res: SenSeedResult, dtype=np.float32) -> np.ndarray:
    cfg = res.cfg
    from .squant import SQuantConfig
    if not cfg.feasible():
        raise ValueError(
            f"target_bits={cfg.target_bits} is below the floor rate "
            f"{cfg.min_bits:.3f} implied by k_min={cfg.k_min} at B={cfg.B}, "
            f"S={cfg.S} (every block needs at least k_min bases). "
            f"Lower k_min, raise B, or raise target_bits.")
    qcfg = SQuantConfig(B=cfg.B, S=cfg.S, k_max=cfg.k_max)
    cb = SQuantCodebook.get(qcfg, dtype=dtype, basis_layout=cfg.basis_layout)
    a = _dequantize_shared(res.q, res.e, cfg)
    live = np.arange(cfg.k_max)[None, :] < res.ks[:, None]
    a = np.where(live, a, 0.0)
    U = cb.U[res.seeds.astype(np.int64)].astype(np.float64)
    out = np.einsum("nbk,nk->nb", U, a).reshape(-1)
    if res.pad:
        out = out[: -res.pad]
    return out.reshape(res.shape).astype(dtype)
