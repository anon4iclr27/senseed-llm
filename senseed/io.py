"""Reading and writing safetensors without a torch dependency.

The two scripts under ``scripts/`` deliberately carry their own copies of this
logic: they run on the machine holding a checkpoint, where this package may not
be installed, and their whole point is needing nothing but the standard library
(``extract_weights.py``) or numpy (``measure_actscale.py``).  The duplication is
the feature; ``tests/test_sensitivity.py`` pins the two implementations of the
estimator against each other so they cannot drift.
"""

from __future__ import annotations

import json
import struct

import numpy as np

__all__ = ["load_safetensors", "save_safetensors"]


def load_safetensors(path):
    """Minimal reader; handles F16/BF16/F32 without a torch dependency."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        raw, dt = data[a:b], meta["dtype"]
        if dt == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dt == "BF16":
            # bfloat16 is the top 16 bits of a float32
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32) if u.dtype == np.uint32 else u.astype(np.uint32).view(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).copy()
        else:
            raise ValueError(f"unsupported dtype {dt} for {name}")
        out[name] = arr.reshape(meta["shape"])
    return out


def to_bfloat16(v):
    """float32 -> the raw uint16 of bfloat16, rounded half to even.

    bfloat16 is the top 16 bits of a float32, so the conversion is a shift --
    but a bare shift *truncates*, which biases every weight towards zero.  The
    rounding term below adds half an ulp, plus one more when the surviving low
    bit is odd, which is what hardware does.
    """
    u = np.ascontiguousarray(v, np.float32).view(np.uint32)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def save_safetensors(path: str, tensors: dict, dtypes=None) -> None:
    """Write a safetensors file.  Round-trips with the loader above.

    ``dtypes`` optionally maps a tensor name to ``"F32"``, ``"F16"`` or
    ``"BF16"``; anything unnamed is written as float32.  Preserving the source
    dtype matters when the output is meant to be loaded as a checkpoint: a
    model saved in bf16 that comes back as f32 is four times the size and no
    more accurate.
    """
    header, blobs, off = {}, [], 0
    for k, v in tensors.items():
        dt = (dtypes or {}).get(k, "F32")
        if dt == "F16":
            b = np.ascontiguousarray(v, np.float16)
        elif dt == "BF16":
            b = to_bfloat16(v)
        elif dt == "F32":
            b = np.ascontiguousarray(v, np.float32)
        else:
            raise ValueError(f"unsupported dtype {dt} for {k}")
        header[k] = {"dtype": dt, "shape": list(np.shape(v)),
                     "data_offsets": [off, off + b.nbytes]}
        off += b.nbytes
        blobs.append(b)
    raw = json.dumps(header).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b.tobytes())
