"""The quantisation-aware seed search, on a GPU.

The search is ~90% of the cost of compressing a checkpoint: for every block of
``B`` weights it projects onto all 65535 candidate bases, quantises the
coefficients, reconstructs, and keeps the argmin.  On 12 CPU cores that is 16.5
hours for Qwen-2.5-0.5B and about twelve days for a 7B model, which is what
stops the experiment from scaling.

It is also the easiest possible workload to move: a batched matmul, an
elementwise quantiser, a squared-error reduction, and an argmin, with no
branching and no dependence between blocks.  The numpy version is bandwidth
bound -- it materialises ``(seeds, blocks, k)`` and ``(seeds, blocks, B)``
intermediates in DRAM and gets *slower* per core as workers are added, because
they contend for the same memory bus.  A GPU has both far more bandwidth and
far more arithmetic, so this is the one place where the hardware maps onto the
problem exactly.

**Numerics.**  Every step here is the same operation as its numpy counterpart,
including ``frexp`` for the shared exponent (an exponent-field read, exact at
every boundary, not a float ``log2``) and round-half-to-even for the mantissa.
What is *not* guaranteed is bit-identical seed selection: the reduction order
inside a batched matmul differs between backends, so two seeds whose quantised
reconstruction error agrees to the last ulp can swap places.  The test suite
therefore asserts that the two backends achieve the same reconstruction error,
not that they choose the same seed -- see ``tests/test_gpu.py``.  Pass
``dtype=torch.float64`` to make them agree exactly, at roughly half the speed
on a consumer card and 1/32 on a datacentre one.

**Choosing the chunk sizes.**  Peak device memory is about
``seed_chunk * block_chunk * (k + B) * 4`` bytes for the intermediates plus the
basis and its pseudo-inverse.  On a 48 GB card ``seed_chunk=8192`` and
``block_chunk=4096`` is comfortable and keeps the GPU saturated; the defaults
below are deliberately smaller so that a first run on an unknown card does not
fail with an allocator error.
"""
from __future__ import annotations

import numpy as np

__all__ = ["have_torch", "device_report", "resolve_devices", "peak_bytes",
           "set_full_precision_matmul", "quant_aware_torch"]


def have_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def device_report() -> str:
    """One line per visible device -- what a run should print before it starts."""
    try:
        import torch
    except ImportError:
        return "torch is not installed; the numpy backend is the only option"
    if not torch.cuda.is_available():
        return (f"torch {torch.__version__}, no CUDA device visible "
                f"(the torch backend will run on the CPU, which is slower "
                f"than the numpy one -- use backend='numpy')")
    out = [f"torch {torch.__version__}, {torch.cuda.device_count()} device(s)"]
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out.append(f"  cuda:{i}  {p.name}  {p.total_memory / 2**30:.0f} GiB  "
                   f"sm_{p.major}{p.minor}")
    return "\n".join(out)


def set_full_precision_matmul():
    """Turn off every reduced-precision matmul path, and say what was on.

    Returns a dict of the settings as they now stand, so a run can record what
    arithmetic produced its numbers.  Written defensively: the flags have moved
    around between torch versions and a missing one must not stop a run.
    """
    import torch
    was = {}
    for mod, attr in (("torch.backends.cuda.matmul", "allow_tf32"),
                      ("torch.backends.cudnn", "allow_tf32"),
                      ("torch.backends.cuda.matmul",
                       "allow_fp16_reduced_precision_reduction"),
                      ("torch.backends.cuda.matmul",
                       "allow_bf16_reduced_precision_reduction")):
        obj = torch
        for part in mod.split(".")[1:]:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is None or not hasattr(obj, attr):
            continue
        was[f"{mod}.{attr}"] = getattr(obj, attr)
        try:
            setattr(obj, attr, False)
        except (RuntimeError, AttributeError):       # pragma: no cover
            pass
    try:
        # Read before writing.  This is the modern knob -- "high"/"medium"
        # enable TF32 regardless of the legacy allow_tf32 flag -- so reporting
        # the value being set instead of the one found would make the selftest
        # claim to have measured something it never looked at.
        was["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
    except (AttributeError, ValueError):             # pragma: no cover
        pass
    return was


def peak_bytes(seed_chunk, block_chunk, k_max=6, B=8, itemsize=4):
    """Device memory the search tile needs, counting every live intermediate.

    Per (seed, block) the loop holds, at once: ``T`` (k), the quantiser's
    ``q`` and ``t_hat`` (k each), ``Rc`` (B), the ``amax``/exponent/scale
    scalars (about 5), and ``err`` in float64 (2 slots).  An earlier version of
    this counted only ``T`` and ``Rc`` and under-reserved by 2.3x.
    """
    per_pair = (3 * k_max + B + 5) * itemsize + 8
    return seed_chunk * block_chunk * per_pair


def resolve_devices(devices, seed_chunk, block_chunk=0, k_max=6, B=8,
                    fraction=0.15):
    """Validate a device list and size the block tile to the smallest card.

    Two failures this prevents, both of which happen on the first run on a new
    machine.  Asking for ``cuda:0,cuda:1,cuda:2,cuda:3`` on a single-GPU box
    used to start four workers and crash three of them several minutes in, with
    an error about an invalid ordinal rather than about the device list.  And a
    tile sized for a 48 GB card is an allocator failure on an 11 GB one, so the
    default is derived from the hardware instead of guessed: peak device memory
    for the search is about ``seed_chunk * block_chunk * (k_max + B) * 4``
    bytes, and taking ``fraction`` of the smallest card leaves ample room for
    the basis, its pseudo-inverse and the allocator's own slack.

    Returns ``(devices, block_chunk)``.  ``"auto"`` anywhere in the list
    expands to every visible CUDA device.
    """
    import torch
    if any(d == "auto" for d in devices):
        n = torch.cuda.device_count()
        if n == 0:
            raise SystemExit(
                "--devices auto found no CUDA device.  Run with --devices ''"
                " to use the numpy backend, or check nvidia-smi.")
        devices = [f"cuda:{i}" for i in range(n)]

    have = torch.cuda.device_count()
    want = sorted({int(d.split(":")[1]) for d in devices
                   if d.startswith("cuda:")})
    missing = [i for i in want if i >= have]
    if missing:
        listed = "\n".join(
            f"  cuda:{i}  {torch.cuda.get_device_properties(i).name}  "
            f"{torch.cuda.get_device_properties(i).total_memory / 2**30:.0f} GiB"
            for i in range(have))
        raise SystemExit(
            f"asked for cuda:{', cuda:'.join(str(i) for i in missing)} but "
            f"this machine has {have} CUDA device(s):\n{listed}\n"
            f"Pass --devices auto, or name only the devices that exist.")

    if block_chunk:
        return devices, block_chunk
    mem = min([torch.cuda.get_device_properties(i).total_memory for i in want]
              or [8 * 2 ** 30])
    bc = int(fraction * mem / (peak_bytes(seed_chunk, 1, k_max, B)))
    bc = max(256, min(16384, 1 << (max(bc, 1).bit_length() - 1)))
    return devices, bc


def _quantize_fast_t(t, spec):
    """``seedlm._quantize_fast(t, spec, "range")``, in torch.

    Returns ``t_hat`` only: the search needs the dequantised coefficients to
    reconstruct with, and the winning block's mantissas are recomputed on the
    host afterwards from the same code path the numpy backend uses.
    """
    import torch
    amax = t.abs().amax(dim=-1)
    zero = amax == 0
    amax = torch.where(zero, torch.ones_like(amax), amax)

    # shared_exponent(..., "range"): ceil(log2(amax / (q_max + 0.5))), read off
    # the exponent field.  frexp gives x = m * 2**E with m in [0.5, 1), so the
    # ceiling is E except where the scaled value is an exact power of two.
    mant, E = torch.frexp(amax / (spec.q_max + 0.5))
    e = E - (mant == 0.5).to(E.dtype)
    e = e.clamp(spec.e_min, spec.e_max)

    # Scale by an explicit power of two rather than through `torch.ldexp`.
    # `ldexp(x, n)` is implemented as `x * pow(2, n)`, and with an *integer*
    # exponent tensor that inner `pow` takes the integral path, where negative
    # exponents are undefined -- on the CPU it happens to work, on CUDA it
    # faulted with an illegal address.  `e` is bounded by the exponent width
    # (here [-8, 7]), so `exp2` in the working dtype is exact and the whole
    # step stays bit-identical to numpy's ldexp.
    scale = torch.exp2(e.to(t.dtype)).unsqueeze(-1)

    q = torch.round(t / scale).clamp(spec.q_min, spec.q_max)  # half to even
    t_hat = q * scale
    if bool(zero.any()):
        t_hat = torch.where(zero.unsqueeze(-1), torch.zeros_like(t_hat), t_hat)
    return t_hat


def quant_aware_torch(blocks, cb, cfg, mask, ks, seeds, coeffs,
                      block_chunk=4096, seed_chunk=8192, device="cuda",
                      dtype=None):
    """Drop-in replacement for ``allocate._quant_aware_numpy``.

    ``seeds`` and ``coeffs`` are numpy arrays and are updated in place, so the
    caller does not know or care which backend ran.
    """
    import torch
    from .allocate import _pinv_all

    # No reduced-precision matmul.  This search is an arg-min over 65535
    # candidates whose quantised reconstruction errors are often equal to
    # within a few ulps, so which one wins is decided in the last bits of a
    # float32 matmul.  TF32 keeps 10 mantissa bits instead of 24; on an Ada
    # card that silently moved the winner on a small fraction of blocks, which
    # showed up downstream as a perplexity that agreed with the numpy pipeline
    # on six windows out of eight and drifted on the other two.  Equal-error
    # ties may still break either way -- that is inherent -- but they should
    # break on the same arithmetic the reference uses.
    set_full_precision_matmul()

    if dtype is None:
        dtype = torch.float32
    dev = torch.device(device)
    spec = _Spec(cfg.mantissa_bits, cfg.exp_bits)

    blocks_t = torch.as_tensor(np.ascontiguousarray(blocks), device=dev,
                               dtype=dtype)
    mask_t = None if mask is None else torch.as_tensor(
        np.ascontiguousarray(mask), device=dev)

    for k in np.unique(ks):
        k = int(k)
        grp_np = np.nonzero(ks == k)[0]
        # The basis and its pseudo-inverse are shared by every block at this k,
        # so they are uploaded once per k rather than once per tile.  At k=6
        # and 65535 seeds that is 12.6 MB each -- nothing on a 48 GB card, and
        # it removes the transfer from the inner loop entirely.
        Uk = torch.as_tensor(np.ascontiguousarray(cb.U[:, :, :k]), device=dev,
                             dtype=dtype)                       # (S, B, k)
        Up = torch.as_tensor(np.ascontiguousarray(_pinv_all(cb, k)), device=dev,
                             dtype=dtype)                       # (S, k, B)
        UkT = Uk.transpose(1, 2).contiguous()                   # (S, k, B)
        UpT = Up.transpose(1, 2).contiguous()                   # (S, B, k)

        for g0 in range(0, len(grp_np), block_chunk):
            m_np = grp_np[g0:g0 + block_chunk]
            Wm = blocks_t[torch.as_tensor(m_np, device=dev)]     # (m, B)
            nm = len(m_np)
            best = torch.full((nm,), float("inf"), device=dev, dtype=torch.float64)
            best_seed = torch.zeros(nm, device=dev, dtype=torch.long)
            best_that = torch.zeros((nm, k), device=dev, dtype=dtype)

            for s0 in range(0, cfg.n_seeds, seed_chunk):
                s1 = min(s0 + seed_chunk, cfg.n_seeds)
                if mask_t is not None and not bool(mask_t[s0:s1, k - 1].any()):
                    continue
                T = Wm @ UpT[s0:s1]                              # (S, m, k)
                That = _quantize_fast_t(T, spec)
                Rc = That @ UkT[s0:s1]                           # (S, m, B)
                Rc -= Wm
                err = (Rc * Rc).sum(-1).to(torch.float64)        # (S, m)
                if mask_t is not None:
                    err = torch.where(mask_t[s0:s1, k - 1].unsqueeze(1), err,
                                      torch.full_like(err, float("inf")))
                cand, j = err.min(dim=0)                         # (m,)
                upd = cand < best
                if bool(upd.any()):
                    best = torch.where(upd, cand, best)
                    best_seed = torch.where(upd, j + s0, best_seed)
                    picked = That[j, torch.arange(nm, device=dev)]   # (m, k)
                    best_that = torch.where(upd.unsqueeze(-1), picked,
                                            best_that)

            seeds[m_np] = best_seed.cpu().numpy()
            coeffs[m_np, :k] = best_that.to(torch.float64).cpu().numpy()
            coeffs[m_np, k:] = 0.0

        del Uk, Up, UkT, UpT
        if dev.type == "cuda":
            torch.cuda.empty_cache()


class _Spec:
    """The three constants ``_quantize_fast_t`` needs from a ``QuantSpec``."""

    def __init__(self, mantissa_bits, exponent_bits):
        self.q_min = -(1 << (mantissa_bits - 1))
        self.q_max = (1 << (mantissa_bits - 1)) - 1
        self.e_min = -(1 << (exponent_bits - 1))
        self.e_max = (1 << (exponent_bits - 1)) - 1
