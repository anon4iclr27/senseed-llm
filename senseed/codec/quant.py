"""Coefficient quantisation for SeedLM (Section 3.2).

Each latent coefficient vector ``t in R^P`` is stored as ``P`` 4-bit two's
complement mantissas sharing a single 4-bit exponent::

    t_i ~= q_i * 2**e,   q_i in [-8, 7],   e in [-8, 7]

which is where the paper's quoted dynamic range ``[-8 * 2**-8, 7 * 2**7]``
comes from.  Keeping the scale a power of two is deliberate: on the decode side
it is a shift, not a multiply.

Three rules for choosing the shared exponent are provided.

``paper``
    The literal formula printed in Section 3.2, ``e = max_i floor(log2 |t_i|)``.
    Reproduced for fidelity, but note what it implies: it places
    ``max_i |t_i| / 2**e`` in ``[1, 2)``, so rounding can only ever return
    ``{-2, -1, 0, 1, 2}`` -- 5 of the 16 available codes -- and most of the
    mantissa budget is wasted.

``range`` (default)
    ``e = ceil(log2(max_i |t_i| / 7.5))``, which places the largest magnitude in
    ``(3.75, 7.5]``.  This is the largest scale that keeps round-to-nearest
    inside the mantissa range except on an exact tie, where ``rint(7.5) == 8``
    and is clipped to 7; the code ``-8`` is correspondingly reachable only on
    the matching tie, so 15 of the 16 codes appear in practice.  This is what
    actually uses the 4 bits, and is almost certainly what the authors ran.

``search``
    Try several exponents around the ``range`` choice and keep whichever gives
    the smallest squared error after rounding.  Slightly better than ``range``
    at a proportional increase in search cost.

The exponent is derived with :func:`numpy.frexp` rather than :func:`numpy.log2`.
That is not a micro-optimisation: on float32 input ``log2`` carries ~1e-7 of
error, which straddles the integer boundary and returns an exponent one too
large for values just under a power of two (e.g. ``amax = 2**-7 - 1ulp`` gives
-7 instead of -8).  A one-off exponent doubles the scale of every mantissa in
the vector, so the boundary cases have to be exact.  ``frexp`` reads the
exponent field directly and cannot be wrong.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "QuantSpec",
    "shared_exponent",
    "quantize",
    "dequantize",
    "quantize_dequantize",
]


class QuantSpec:
    """Bit widths for the mantissa/exponent pair."""

    def __init__(self, mantissa_bits: int = 4, exponent_bits: int = 4):
        self.mantissa_bits = mantissa_bits
        self.exponent_bits = exponent_bits

    @property
    def q_min(self) -> int:
        return -(1 << (self.mantissa_bits - 1))

    @property
    def q_max(self) -> int:
        return (1 << (self.mantissa_bits - 1)) - 1

    @property
    def e_min(self) -> int:
        return -(1 << (self.exponent_bits - 1))

    @property
    def e_max(self) -> int:
        return (1 << (self.exponent_bits - 1)) - 1

    def __repr__(self) -> str:
        return (
            f"QuantSpec(mantissa_bits={self.mantissa_bits}, "
            f"exponent_bits={self.exponent_bits})"
        )


DEFAULT_SPEC = QuantSpec()


def shared_exponent(amax: np.ndarray, spec: QuantSpec, mode: str) -> np.ndarray:
    """Shared exponent for a coefficient vector, from its max absolute value.

    Computed with ``frexp``, which splits ``x = m * 2**E`` with ``m`` in
    ``[0.5, 1)`` by reading the exponent field -- exact at every boundary,
    unlike a float32 ``log2``.  This is the single implementation used by both
    the readable quantiser here and the vectorised one in the search loop, so
    the two cannot drift apart.

    ``paper``:  ``floor(log2 amax) = E - 1``.
    ``range``:  ``ceil(log2(amax / (q_max + 0.5)))``, which is ``E`` unless the
    scaled value is an exact power of two (``m == 0.5``), where it is ``E - 1``.
    """
    if mode == "paper":
        _, E = np.frexp(amax)
        return (E - 1).astype(np.int32)
    if mode == "range":
        m, E = np.frexp(amax / (spec.q_max + 0.5))
        return (E - (m == 0.5)).astype(np.int32)
    raise ValueError(f"unknown exponent mode {mode!r}")


def quantize(
    t: np.ndarray,
    spec: QuantSpec = DEFAULT_SPEC,
    mode: str = "range",
    search_span: tuple[int, ...] = (0, 1, -1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """Quantise ``t`` along its last axis.

    Parameters
    ----------
    t
        Array of shape ``(..., P)``.
    mode
        ``"paper"``, ``"range"`` or ``"search"`` -- see the module docstring.
    search_span
        Exponent offsets tried in ``"search"`` mode, relative to the ``range``
        choice.

    Returns
    -------
    (q, e)
        ``q`` has shape ``t.shape`` and integer dtype; ``e`` has shape
        ``t.shape[:-1]`` and integer dtype.  Both are already clamped to the
        representable range implied by ``spec``.
    """
    t = np.asarray(t)
    amax = np.abs(t).max(axis=-1)
    zero = amax == 0

    safe = np.where(zero, np.asarray(1, dtype=amax.dtype), amax)
    if mode == "search":
        e = shared_exponent(safe, spec, "range")
        offsets: tuple[int, ...] = search_span
    else:
        e = shared_exponent(safe, spec, mode)
        offsets = (0,)

    base = np.clip(e, spec.e_min, spec.e_max)

    best_q = None
    best_e = None
    best_err = None
    for off in offsets:
        cand_e = np.clip(base + off, spec.e_min, spec.e_max)
        scale = np.exp2(cand_e.astype(np.float64))[..., None]
        cand_q = np.clip(np.rint(t / scale), spec.q_min, spec.q_max)
        err = np.sum((t - cand_q * scale) ** 2, axis=-1)
        if best_err is None:
            best_q, best_e, best_err = cand_q, cand_e, err
        else:
            take = err < best_err
            best_q = np.where(take[..., None], cand_q, best_q)
            best_e = np.where(take, cand_e, best_e)
            best_err = np.where(take, err, best_err)

    # An all-zero block needs no exponent; pin it to e_min for determinism.
    best_e = np.where(zero, spec.e_min, best_e)
    best_q = np.where(zero[..., None], 0.0, best_q)

    return best_q.astype(np.int8), best_e.astype(np.int8)


def dequantize(q: np.ndarray, e: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Inverse of :func:`quantize`: ``t_hat = q * 2**e``."""
    scale = np.exp2(np.asarray(e).astype(np.float64))[..., None]
    return (np.asarray(q).astype(np.float64) * scale).astype(dtype)


def quantize_dequantize(
    t: np.ndarray, spec: QuantSpec = DEFAULT_SPEC, mode: str = "range", **kw
) -> np.ndarray:
    """Convenience round-trip used inside the seed search."""
    q, e = quantize(t, spec=spec, mode=mode, **kw)
    return dequantize(q, e, dtype=t.dtype)
