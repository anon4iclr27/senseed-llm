"""safetensors load/save with a numpy interface that survives bfloat16.

``safetensors.numpy.load_file`` raises "data type 'bfloat16' not understood"
on any bf16 checkpoint, which is most current HF releases.  The cause is not
safetensors: numpy has no bfloat16 dtype at all, so there is nothing for it to
hand back.  Going through safetensors.torch fixes it; torch has the type.

Drop-in -- a caller only changes its import:

    import safetensors.numpy as stn      # before
    import stbf as stn                   # after

Tensors come back as ordinary numpy arrays with bf16 widened to float32, which
is exact (float32 has strictly more mantissa and the same exponent range).  The
on-disk dtype of every tensor is remembered and save_file narrows it back, so a
checkpoint that arrives in bf16 leaves in bf16, replaced tensors included --
a compressed tensor is stored at the source's precision rather than silently
promoted to f32 and doubling the file.  The dtype memory is global and keyed by
name, which is enough for a script that loads one checkpoint and writes one.
"""
from __future__ import annotations

import numpy as np
import torch
from safetensors.torch import load_file as _t_load, save_file as _t_save

__all__ = ["load_file", "save_file", "original_dtypes"]

original_dtypes: dict = {}
_WIDEN = {torch.bfloat16: torch.float32}


def load_file(path, device="cpu"):
    """Like safetensors.numpy.load_file, but bf16 arrives as float32."""
    out = {}
    for name, t in _t_load(path, device=device).items():
        original_dtypes[name] = t.dtype
        if t.dtype in _WIDEN:
            t = t.to(_WIDEN[t.dtype])
        out[name] = t.numpy()
    return out


def save_file(tensors, path, metadata=None):
    """Like safetensors.numpy.save_file, narrowing back to the source dtype."""
    td = {}
    for name, a in tensors.items():
        t = a.detach().cpu() if isinstance(a, torch.Tensor) \
            else torch.from_numpy(np.ascontiguousarray(a))
        want = original_dtypes.get(name)
        if want is not None and t.dtype != want:
            t = t.to(want)
        td[name] = t.contiguous()
    _t_save(td, path, metadata=metadata)
