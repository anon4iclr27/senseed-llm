"""Bit-exact serialisation of a compressed tensor.

Eq. 3 claims ``K + 4 + 4P`` bits per block.  This module actually lays those
fields out end to end so the claim can be checked against a byte count rather
than taken on trust::

    [ K bits: seed ][ 4 bits: shared exponent ][ 4 bits x P: mantissas ]

Fields are packed MSB-first, blocks run back to back with no padding between
them, and signed fields use two's complement.  For the 4-bit configuration a
block is exactly 32 bits; for the 3-bit configuration it is 36, so blocks
straddle byte boundaries and only the whole stream is byte-aligned.
"""

from __future__ import annotations

import numpy as np

from .seedlm import CompressedTensor, SeedLMConfig

__all__ = ["pack", "unpack", "packed_nbytes"]


def packed_nbytes(ct: CompressedTensor) -> int:
    """Size of the serialised payload in bytes (Eq. 3, rounded up)."""
    return (ct.n_blocks * ct.cfg.bits_per_block + 7) // 8


def _put(bits: np.ndarray, values: np.ndarray, width: int, offset: int) -> None:
    """Write ``values`` as ``width``-bit big-endian fields at column ``offset``."""
    v = values.astype(np.int64) & ((1 << width) - 1)
    for i in range(width):
        bits[:, offset + i] = (v >> (width - 1 - i)) & 1


def _get(bits: np.ndarray, width: int, offset: int) -> np.ndarray:
    v = np.zeros(len(bits), dtype=np.int64)
    for i in range(width):
        v = (v << 1) | bits[:, offset + i]
    return v


def _sign_extend(v: np.ndarray, width: int) -> np.ndarray:
    """Interpret an unsigned ``width``-bit field as two's complement."""
    half = 1 << (width - 1)
    return (v ^ half) - half


def pack(ct: CompressedTensor) -> bytes:
    """Serialise seeds, exponents and mantissas into a dense bitstream."""
    cfg = ct.cfg
    mb, eb = cfg.mantissa_bits, cfg.exponent_bits
    bits = np.zeros((ct.n_blocks, cfg.bits_per_block), dtype=np.uint8)

    _put(bits, ct.seeds, cfg.K, 0)
    _put(bits, ct.e, eb, cfg.K)
    for p in range(cfg.P):
        _put(bits, ct.q[:, p], mb, cfg.K + eb + p * mb)

    return np.packbits(bits.reshape(-1)).tobytes()


def unpack(
    buf: bytes, shape: tuple[int, ...], cfg: SeedLMConfig, pad: int | None = None
) -> CompressedTensor:
    """Inverse of :func:`pack`."""
    mb, eb = cfg.mantissa_bits, cfg.exponent_bits
    n_elem = int(np.prod(shape))
    if pad is None:
        pad = (-n_elem) % cfg.C
    n_blocks = (n_elem + pad) // cfg.C

    flat = np.unpackbits(np.frombuffer(buf, dtype=np.uint8))
    need = n_blocks * cfg.bits_per_block
    if len(flat) < need:
        raise ValueError(f"buffer holds {len(flat)} bits, need {need}")
    bits = flat[:need].reshape(n_blocks, cfg.bits_per_block)

    seeds = _get(bits, cfg.K, 0).astype(np.uint32)
    e = _sign_extend(_get(bits, eb, cfg.K), eb).astype(np.int8)
    q = np.stack(
        [
            _sign_extend(_get(bits, mb, cfg.K + eb + p * mb), mb)
            for p in range(cfg.P)
        ],
        axis=1,
    ).astype(np.int8)

    return CompressedTensor(seeds=seeds, q=q, e=e, shape=tuple(shape), cfg=cfg, pad=pad)
