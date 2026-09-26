"""Data-free saliency for weight blocks.

The quantity that actually matters when compressing a linear layer is the loss
increase a weight error causes:

    dL  =  E || dW x ||^2  =  tr( dW . E[x x^T] . dW^T )

so sensitivity is governed by ``E[x x^T]``, the second moment of the layer's
*input activations*.  That is a property of the data, not of the weights, which
is why Fisher, AWQ and GPTQ all spend calibration data estimating it.  It cannot
be recovered from a checkpoint in general.

In Llama/Qwen-style architectures a large part of it can be, because of what
sits immediately before every linear layer.  The input is

    x = gamma * RMSNorm(h)

and RMSNorm forces the normalised vector ``n`` to unit RMS *overall*
(``mean_j E[n_j^2] = 1``).  Therefore

    E[x_j^2] = gamma_j^2 * E[n_j^2]

If ``n`` were isotropic this would be exact and ``E[x x^T] ~ diag(gamma^2)``.
It is not isotropic -- the massive-activation channels documented in Llama-3 are
precisely a violation -- so ``gamma^2`` captures the explicit architectural
scaling and misses whatever channel structure survives normalisation.  It is a
proxy, and ``validate_against_activations`` exists to say how good a one.

The output side needs no approximation.  An error in output channel ``i`` of a
layer is read by its consumer weighted by that consumer's column ``i``:

    out_saliency[i] = || W_next[:, i] ||^2

which is exactly computable from the checkpoint.

Combining the two gives a rank-1 saliency map over the weight matrix,

    sens[i, j] ~ out_saliency[i] * gamma_j^2

aggregated to one importance value per block for
:func:`seedlm.senseed.compress_senseed`.

**Nothing here needs data.**  Calibration data appears only in
``validate_against_activations``, which is a research check on the proxy, not a
step in the method.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "blocks_from_scale",
    "gamma_for_layer",
    "consumer_for",
    "output_saliency",
    "block_importance",
    "gain_informativeness",
]


# Which RMSNorm feeds each projection, and which layer consumes its output.
_NORM_OF = {
    "self_attn.q_proj": "input_layernorm",
    "self_attn.k_proj": "input_layernorm",
    "self_attn.v_proj": "input_layernorm",
    "self_attn.o_proj": None,          # input is the attention output, not normed
    "mlp.gate_proj": "post_attention_layernorm",
    "mlp.up_proj": "post_attention_layernorm",
    "mlp.down_proj": None,             # input is the gated hidden, not normed
}

# Whose columns read this layer's output channels.
_CONSUMER_OF = {
    "self_attn.v_proj": "self_attn.o_proj",
    "mlp.up_proj": "mlp.down_proj",
    "mlp.gate_proj": "mlp.down_proj",
    # q/k feed the attention logits, o_proj and down_proj feed the residual
    # stream; for those the consumer is not a single matrix, so the output
    # factor falls back to uniform.
}


def _layer_of(name: str) -> int | None:
    p = name.split(".")
    return int(p[2]) if len(p) > 2 and p[1] == "layers" else None


def _proj_of(name: str) -> str:
    return name.split(".", 3)[3].replace(".weight", "")


def gamma_for_layer(tensors: dict, name: str) -> np.ndarray | None:
    """RMSNorm gain feeding ``name``, or None if its input is not normed."""
    L, proj = _layer_of(name), _proj_of(name)
    norm = _NORM_OF.get(proj)
    if L is None or norm is None:
        return None
    return tensors.get(f"model.layers.{L}.{norm}.weight")


def consumer_for(tensors: dict, name: str) -> np.ndarray | None:
    """The weight matrix whose columns read ``name``'s output channels."""
    L, proj = _layer_of(name), _proj_of(name)
    cons = _CONSUMER_OF.get(proj)
    if L is None or cons is None:
        return None
    return tensors.get(f"model.layers.{L}.{cons}.weight")


def output_saliency(W_next: np.ndarray | None, n_out: int) -> np.ndarray:
    """``|| W_next[:, i] ||^2`` per output channel, normalised to mean 1."""
    if W_next is None or W_next.shape[1] != n_out:
        return np.ones(n_out)
    s = np.einsum("ij,ij->j", W_next, W_next).astype(np.float64)
    m = s.mean()
    return s / m if m > 0 else np.ones(n_out)


def block_importance(W, gamma=None, W_next=None, B=8, mode="both"):
    """Per-block importance for ``compress_senseed(..., importance=)``.

    ``W`` is ``(out, in)``; blocks run along the flattened row-major order, so a
    block of ``B`` contiguous elements sits inside one output row and spans
    ``B`` consecutive input channels.  Returns one value per block, normalised
    to mean 1 (the allocator only uses relative values).
    """
    W = np.asarray(W, np.float64)
    n_out, n_in = W.shape

    g2 = np.ones(n_in)
    if mode in ("both", "gamma") and gamma is not None:
        g = np.asarray(gamma, np.float64).ravel()
        if len(g) == n_in:
            g2 = g * g
            g2 = g2 / g2.mean() if g2.mean() > 0 else np.ones(n_in)

    o = np.ones(n_out)
    if mode in ("both", "output"):
        o = output_saliency(W_next, n_out)

    # sens[i, j] = o_i * g2_j ; aggregate over each block of B input channels
    pad = (-n_in) % B
    gp = np.concatenate([g2, np.zeros(pad)]) if pad else g2
    g_blk = gp.reshape(-1, B).mean(1)                 # (blocks per row,)
    imp = (o[:, None] * g_blk[None, :]).ravel()
    m = imp.mean()
    return imp / m if m > 0 else np.ones_like(imp)


def blocks_from_scale(a, rows: int, B: int = 8) -> np.ndarray:
    """Per-block importance for ``compress_senseed(..., importance=)``.

    ``a`` is one value per *input channel*.  Blocks are ``B`` contiguous
    elements in row-major order, so a block sits inside one output row and
    spans ``B`` consecutive input channels: the per-row pattern is the mean of
    ``a`` over each group of ``B``, and it repeats for every row.  Normalised
    to mean 1, since the allocator only reads relative values.
    """
    a = np.asarray(a, np.float64).ravel()
    n = len(a) - len(a) % B
    m = a[:n].reshape(-1, B).mean(1)
    m = m / m.mean() if m.mean() > 0 else np.ones_like(m)
    return np.tile(m, rows)


def gain_informativeness(gamma) -> dict:
    """Is this gain vector worth using at all?

    Equivalent to ``metrics.informativeness(gamma ** 2)`` -- every statistic
    here is a ratio, so the unit-mean normalisation does not matter.  Kept as a
    separate entry point because callers here hold ``gamma``, not ``a``;
    ``tests/test_sensitivity.py`` pins the two against each other.

    """
    g = np.asarray(gamma, np.float64).ravel()
    g2 = g * g
    if g2.sum() <= 0:
        return dict(n=len(g), p99_over_median=1.0, top1pct_share=0.01, cv=0.0)
    s = np.sort(g2)[::-1]
    k = max(1, len(s) // 100)
    return dict(
        n=len(g),
        p99_over_median=float(np.percentile(g2, 99) / max(np.median(g2), 1e-30)),
        max_over_median=float(g2.max() / max(np.median(g2), 1e-30)),
        top1pct_share=float(s[:k].sum() / s.sum()),
        cv=float(g2.std() / max(g2.mean(), 1e-30)),
    )
