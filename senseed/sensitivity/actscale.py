"""Data-free estimate of every linear layer's input activation scale.

The loss increase caused by a weight error is

    dL  =  E || dW x ||^2  =  tr( dW . E[x x^T] . dW^T )

and under the usual diagonal approximation this is

    dL  ~  sum_j  a_j * || dW[:, j] ||^2 ,      a_j = E[x_j^2]           (1)

so ``a`` is the whole of sensitivity.  Fisher, AWQ and GPTQ all spend
calibration data estimating it.  This module estimates it **from the
checkpoint alone**, by walking the same graph the activations walk.

Five of the seven projections in a Llama/Qwen block read a normalised tensor::

    x = gamma * RMSNorm(h)        ->   a_j = gamma_j^2 * E[n_j^2] ~ gamma_j^2

RMSNorm pins ``mean_j E[n_j^2] = 1`` exactly, so ``gamma^2`` is the explicit
architectural part of ``a`` and the residual is whatever channel structure
survives normalisation.

The other two read *unnormalised* tensors, and this is where the checkpoint
still has something to say.  ``o_proj`` reads the attention output, which is a
convex combination over tokens of ``v`` -- and ``v = W_v x`` with ``x`` already
covered above.  ``down_proj`` reads ``silu(gate(x)) * up(x)``, both of whose
arguments are also covered.  So the scale *propagates*:

    a_out = rowwise( W^2 ) @ a_in                                        (2)

which is (1) applied to the producing layer, one hop upstream.  Two hops of
that cover the whole block with no data anywhere.

What the estimate assumes, stated once so it can be checked:

* ``E[x x^T]`` is diagonal.  Same assumption ``gamma^2``, AWQ's per-channel
  scale and every diagonal-Fisher method make.
* ``E[n_j^2] = 1`` per channel, not just on average.  This is the one that the
  massive-activation channels in Llama-3 violate, and the reason
  ``measure_actscale.py`` exists.
* Attention mixes tokens, not channels, so it rescales ``E[v^2]`` by a factor
  that is constant within a head.  Taken as constant across heads, it drops out
  of a *relative* importance and is ignored.
* ``silu(g)`` and ``u`` are uncorrelated, so the product's second moment
  factorises.  ``silu``'s own second moment is not scale-free -- it interpolates
  between ``sigma^2/4`` (small ``sigma``, silu ~ g/2) and ``sigma^2/2`` (large
  ``sigma``, silu ~ relu) -- so it is integrated exactly rather than assumed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sigmoid",
    "silu_second_moment",
    "propagate",
    "normed_scale",
    "attn_out_scale",
    "mlp_act_scale",
    "block_scales",
    "residual_saliency",
    "ArchSpec",
]


class ArchSpec:
    """The few config fields the propagation needs."""

    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads,
                 intermediate_size, **_ignored):
        self.hidden = int(hidden_size)
        self.n_heads = int(num_attention_heads)
        self.n_kv = int(num_key_value_heads)
        self.inter = int(intermediate_size)
        self.head_dim = self.hidden // self.n_heads
        if self.n_heads % self.n_kv:
            raise ValueError("num_attention_heads must be a multiple of "
                             "num_key_value_heads")
        self.rep = self.n_heads // self.n_kv

    @classmethod
    def from_config(cls, cfg: dict) -> "ArchSpec":
        return cls(**{k: cfg[k] for k in (
            "hidden_size", "num_attention_heads", "num_key_value_heads",
            "intermediate_size")})


# --------------------------------------------------------------------------
# silu
# --------------------------------------------------------------------------

def silu_second_moment(sigma, n_quad: int = 129) -> np.ndarray:
    """``E[ silu(g)^2 ]`` for ``g ~ N(0, sigma^2)``, by Gauss-Hermite.

    Not proportional to ``sigma^2``: silu is scale-dependent, so the ratio
    ``E[silu^2] / sigma^2`` runs from 1/4 at small ``sigma`` to 1/2 at large
    ``sigma``.  Assuming either end would misweight the MLP by up to 2x.
    """
    sigma = np.atleast_1d(np.asarray(sigma, np.float64))
    nodes, wts = np.polynomial.hermite_e.hermegauss(n_quad)   # weight exp(-z^2/2)
    wts = wts / np.sqrt(2.0 * np.pi)                          # normalise to a pdf
    g = sigma[:, None] * nodes[None, :]                       # (n, q)
    s = g * sigmoid(g)                                        # silu
    return (s * s) @ wts


def sigmoid(x):
    """Overflow-free logistic: ``exp(-x)`` alone blows up for x << 0."""
    x = np.asarray(x, np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# --------------------------------------------------------------------------
# one propagation hop
# --------------------------------------------------------------------------

def propagate(W, a_in=None, chunk: int = 4096) -> np.ndarray:
    """``a_out[i] = sum_j W[i, j]^2 a_in[j]`` -- equation (2).

    Chunked over rows so a 28k x 8k matrix does not need a float64 copy.
    """
    W = np.asarray(W)
    n_out, n_in = W.shape
    a_in = np.ones(n_in, np.float64) if a_in is None else \
        np.asarray(a_in, np.float64).ravel()
    if len(a_in) != n_in:
        raise ValueError(f"a_in has {len(a_in)} entries, W wants {n_in}")
    out = np.empty(n_out, np.float64)
    for lo in range(0, n_out, chunk):
        blk = W[lo:lo + chunk].astype(np.float64)
        out[lo:lo + chunk] = (blk * blk) @ a_in
    return out


def _unit_mean(a):
    a = np.asarray(a, np.float64)
    m = a.mean()
    return a / m if m > 0 else np.ones_like(a)


# --------------------------------------------------------------------------
# the three input kinds
# --------------------------------------------------------------------------

def normed_scale(gamma) -> np.ndarray:
    """``a`` for a projection whose input is ``gamma * RMSNorm(h)``."""
    g = np.asarray(gamma, np.float64).ravel()
    return _unit_mean(g * g)


def attn_out_scale(W_v, gamma_in, arch: ArchSpec) -> np.ndarray:
    """``a`` for ``o_proj``: one hop back through ``W_v``, then GQA expand.

    ``W_v`` must have all ``n_kv * head_dim`` rows; a row-sampled slice cannot
    give the full input scale and raises.
    """
    W_v = np.asarray(W_v)
    want = arch.n_kv * arch.head_dim
    if W_v.shape[0] != want:
        raise ValueError(
            f"attn_out_scale needs all {want} rows of v_proj, got "
            f"{W_v.shape[0]}.  Extract v_proj whole for this layer.")
    a_v = propagate(W_v, normed_scale(gamma_in))          # (n_kv*head_dim,)
    # repeat_kv: query head q reads kv head q // rep
    a_v = a_v.reshape(arch.n_kv, arch.head_dim)
    a_o = np.repeat(a_v, arch.rep, axis=0).reshape(-1)    # (n_heads*head_dim,)
    return _unit_mean(a_o)


def mlp_act_scale(W_gate, W_up, gamma_post) -> np.ndarray:
    """``a`` for ``down_proj``: ``E[silu(g)^2] * E[u^2]``, both propagated."""
    a_in = normed_scale(gamma_post)
    sig2_g = propagate(W_gate, a_in)
    sig2_u = propagate(W_up, a_in)
    return _unit_mean(silu_second_moment(np.sqrt(sig2_g)) * sig2_u)


# --------------------------------------------------------------------------
# whole block
# --------------------------------------------------------------------------

_NORMED = {
    "self_attn.q_proj": "input_layernorm",
    "self_attn.k_proj": "input_layernorm",
    "self_attn.v_proj": "input_layernorm",
    "mlp.gate_proj": "post_attention_layernorm",
    "mlp.up_proj": "post_attention_layernorm",
}


def block_scales(tensors: dict, layer: int, arch: ArchSpec,
                 strict: bool = False) -> dict:
    """``{proj_name: a}`` for every projection of one transformer block.

    Projections whose upstream matrices are missing (or row-sampled) are
    skipped, with the reason in ``result['_skipped']`` -- this runs against
    partial extractions, and silently substituting a uniform ``a`` there would
    be indistinguishable from the method having nothing to say.
    """
    pre = f"model.layers.{layer}."
    get = lambda n: tensors.get(pre + n + ".weight")           # noqa: E731
    out, skipped = {}, {}

    for proj, norm in _NORMED.items():
        g = get(norm)
        if g is None:
            skipped[proj] = f"missing {norm}"
        else:
            out[proj] = normed_scale(g)

    g_in = get("input_layernorm")
    W_v = get("self_attn.v_proj")
    if g_in is None or W_v is None:
        skipped["self_attn.o_proj"] = "missing input_layernorm or v_proj"
    else:
        try:
            out["self_attn.o_proj"] = attn_out_scale(W_v, g_in, arch)
        except ValueError as e:
            skipped["self_attn.o_proj"] = str(e).split(".")[0]

    g_post, W_g, W_u = get("post_attention_layernorm"), get("mlp.gate_proj"), \
        get("mlp.up_proj")
    if g_post is None or W_g is None or W_u is None:
        skipped["mlp.down_proj"] = "missing post_attention_layernorm, gate or up"
    elif W_g.shape[0] != arch.inter or W_u.shape[0] != arch.inter:
        skipped["mlp.down_proj"] = (
            f"needs all {arch.inter} rows of gate/up, got "
            f"{W_g.shape[0]}/{W_u.shape[0]}")
    else:
        out["mlp.down_proj"] = mlp_act_scale(W_g, W_u, g_post)

    if strict and skipped:
        raise ValueError(f"layer {layer}: {skipped}")
    out["_skipped"] = skipped
    return out


# --------------------------------------------------------------------------
# the objective itself
# --------------------------------------------------------------------------

def residual_saliency(norms: dict, layer: int, n_layers: int,
                      include_final: bool = True) -> np.ndarray | None:
    """How much a residual-stream channel matters to everything downstream.

    ``o_proj`` and ``down_proj`` write into the residual stream, so their output
    channels have no single consumer -- every later block reads them, and reads
    them through its own RMSNorm gain.  An error in residual channel ``i`` after
    layer ``L`` reaches layer ``L'`` scaled by ``gamma_{L'}[i]``, so

        out_saliency[i]  =  sum_{L' > L}  ( gamma_in[L'][i]^2
                                          + gamma_post[L'][i]^2 )

    The RMSNorm denominator is shared by all channels and cancels in a relative
    importance.  Computed from the norm vectors alone -- a few hundred kB of a
    checkpoint -- and it is the piece that lets the two residual-writing layers
    be weighted at all.

    Returns None if no downstream norm is present (the last layer).
    """
    acc, seen = None, 0
    for L in range(layer + 1, n_layers):
        for which in ("input_layernorm", "post_attention_layernorm"):
            g = norms.get(f"model.layers.{L}.{which}.weight")
            if g is None:
                continue
            g2 = np.asarray(g, np.float64).ravel() ** 2
            acc = g2 if acc is None else acc + g2
            seen += 1
    if include_final:
        g = norms.get("model.norm.weight")
        if g is not None:
            g2 = np.asarray(g, np.float64).ravel() ** 2
            acc = g2 if acc is None else acc + g2
            seen += 1
    return None if seen == 0 else _unit_mean(acc)
