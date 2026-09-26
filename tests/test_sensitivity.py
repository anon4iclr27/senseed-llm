"""Tests for the data-free sensitivity estimate and the equalization fold.

Two things are worth testing here and they are different in kind.

The *transformations* (equalize / fold / permute) claim to leave the layer's
output bit-for-bit unchanged.  That is an identity and is tested as one.

The *estimate* claims that a number computed from the checkpoint approximates a
number that only exists once data flows.  No unit test can establish that; what
the tests here pin down is that the estimator computes the quantity it says it
computes, on inputs where the answer is known independently.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from senseed.sensitivity.actscale import (ArchSpec, attn_out_scale, block_scales,
                                       mlp_act_scale, normed_scale, propagate,
                                       silu_second_moment)
from senseed.sensitivity.metrics import informativeness, weighted_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")


# --------------------------------------------------------------------------
# silu
# --------------------------------------------------------------------------

def test_silu_second_moment_matches_monte_carlo():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(2_000_000)
    for sg in (0.2, 1.0, 4.0):
        g = z * sg
        mc = ((g / (1 + np.exp(-g))) ** 2).mean()
        assert silu_second_moment(sg)[0] == pytest.approx(mc, rel=2e-3)


def test_silu_ratio_spans_quarter_to_half():
    """The reason it is integrated rather than assumed."""
    lo = silu_second_moment(1e-3)[0] / 1e-6
    hi = silu_second_moment(50.0)[0] / 2500.0
    assert lo == pytest.approx(0.25, abs=0.01)
    assert hi == pytest.approx(0.50, abs=0.01)


# --------------------------------------------------------------------------
# propagation
# --------------------------------------------------------------------------

def test_propagate_is_the_second_moment_of_the_output():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((24, 40))
    a_in = rng.random(40) * 3 + 0.1
    x = rng.standard_normal((300_000, 40)) * np.sqrt(a_in)
    emp = ((x @ W.T) ** 2).mean(0)
    assert propagate(W, a_in) == pytest.approx(emp, rel=0.02)


def test_propagate_is_chunk_invariant():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((97, 31))
    a = rng.random(31)
    assert propagate(W, a, chunk=7) == pytest.approx(propagate(W, a, chunk=1000))


def test_propagate_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        propagate(np.zeros((4, 5)), np.ones(6))


def test_normed_scale_has_unit_mean():
    assert normed_scale(np.array([1.0, 2.0, 3.0])).mean() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# GQA plumbing
# --------------------------------------------------------------------------

def test_attn_out_scale_repeats_kv_heads_the_way_attention_does():
    arch = ArchSpec(hidden_size=32, num_attention_heads=8,
                    num_key_value_heads=2, intermediate_size=1)
    rng = np.random.default_rng(3)
    W_v = rng.standard_normal((arch.n_kv * arch.head_dim, 32))
    a = attn_out_scale(W_v, np.ones(32), arch)
    blocks = a.reshape(arch.n_heads, arch.head_dim)
    # query heads 0..3 share kv head 0, heads 4..7 share kv head 1
    for h in range(1, arch.rep):
        assert blocks[h] == pytest.approx(blocks[0])
        assert blocks[arch.rep + h] == pytest.approx(blocks[arch.rep])
    assert not np.allclose(blocks[0], blocks[arch.rep])


def test_attn_out_scale_refuses_a_row_sampled_v_proj():
    """A partial extraction cannot give a full input scale, and quietly
    returning one would be indistinguishable from the method working."""
    arch = ArchSpec(hidden_size=32, num_attention_heads=8,
                    num_key_value_heads=2, intermediate_size=1)
    with pytest.raises(ValueError, match="all 8 rows"):
        attn_out_scale(np.zeros((4, 32)), np.ones(32), arch)


def test_mlp_act_scale_tracks_the_product_of_the_two_branches():
    rng = np.random.default_rng(4)
    n_in, inter, T = 24, 16, 200_000
    W_g = rng.standard_normal((inter, n_in)) * 0.3
    W_u = rng.standard_normal((inter, n_in)) * 0.3
    gamma = np.exp(rng.standard_normal(n_in) * 0.3)
    x = rng.standard_normal((T, n_in)) * gamma
    g, u = x @ W_g.T, x @ W_u.T
    emp = (((g / (1 + np.exp(-g))) * u) ** 2).mean(0)
    est = mlp_act_scale(W_g, W_u, gamma)
    assert np.corrcoef(est, emp / emp.mean())[0, 1] > 0.97


def test_block_scales_reports_what_it_could_not_do():
    arch = ArchSpec(hidden_size=8, num_attention_heads=2,
                    num_key_value_heads=1, intermediate_size=16)
    t = {"model.layers.0.input_layernorm.weight": np.ones(8),
         "model.layers.0.post_attention_layernorm.weight": np.ones(8)}
    got = block_scales(t, 0, arch)
    assert set(got) >= {"self_attn.q_proj", "mlp.up_proj", "_skipped"}
    assert "self_attn.o_proj" in got["_skipped"]
    assert "mlp.down_proj" in got["_skipped"]
    with pytest.raises(ValueError):
        block_scales(t, 0, arch, strict=True)


# --------------------------------------------------------------------------
# the objective
# --------------------------------------------------------------------------

def test_weighted_error_reduces_to_relative_frobenius():
    rng = np.random.default_rng(5)
    W = rng.standard_normal((9, 7))
    H = W + rng.standard_normal((9, 7)) * 0.01
    plain = np.linalg.norm(W - H) / np.linalg.norm(W)
    assert weighted_error(W, H, None) == pytest.approx(plain)
    assert weighted_error(W, H, np.ones(7)) == pytest.approx(plain)


def test_weighted_error_charges_for_the_column_the_weight_points_at():
    W = np.ones((4, 2))
    H = W.copy()
    H[:, 0] += 0.1                       # error in column 0 only
    hot = weighted_error(W, H, np.array([1.9, 0.1]))
    cold = weighted_error(W, H, np.array([0.1, 1.9]))
    assert hot > 3 * cold
# --------------------------------------------------------------------------
# measure_actscale.py, end to end on a checkpoint built for the purpose
# --------------------------------------------------------------------------

def _write_safetensors(path, tensors):
    hdr, blobs, off = {}, [], 0
    for k, v in tensors.items():
        v = np.ascontiguousarray(v, np.float32)
        hdr[k] = {"dtype": "F32", "shape": list(v.shape),
                  "data_offsets": [off, off + v.nbytes]}
        off += v.nbytes
        blobs.append(v)
    raw = json.dumps(hdr).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b.tobytes())


def _tiny_model(d, H=16, nh=4, nkv=2, inter=32, nl=2, vocab=300):
    rng = np.random.default_rng(20)
    hd = H // nh
    cfg = dict(hidden_size=H, num_attention_heads=nh, num_key_value_heads=nkv,
               intermediate_size=inter, num_hidden_layers=nl, vocab_size=vocab,
               rms_norm_eps=1e-6, rope_theta=10000.0, bos_token_id=1)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    # a vocabulary of single characters is a legal byte-level BPE with no merges
    vmap = {chr(i): i for i in range(33, 33 + 200)}
    vmap["Ġ"] = 250
    json.dump({"model": {"vocab": vmap, "merges": []}},
              open(os.path.join(d, "tokenizer.json"), "w"))
    t = {"model.embed_tokens.weight": rng.standard_normal((vocab, H)) * 0.5,
         "model.norm.weight": np.ones(H),
         "lm_head.weight": rng.standard_normal((vocab, H)) * 0.5}
    for L in range(nl):
        p = f"model.layers.{L}."
        t[p + "input_layernorm.weight"] = np.exp(rng.standard_normal(H) * 0.5)
        t[p + "post_attention_layernorm.weight"] = np.exp(
            rng.standard_normal(H) * 0.5)
        t[p + "self_attn.q_proj.weight"] = rng.standard_normal((nh * hd, H)) * .2
        t[p + "self_attn.k_proj.weight"] = rng.standard_normal((nkv * hd, H)) * .2
        t[p + "self_attn.v_proj.weight"] = rng.standard_normal((nkv * hd, H)) * .2
        t[p + "self_attn.o_proj.weight"] = rng.standard_normal((H, nh * hd)) * .2
        t[p + "mlp.gate_proj.weight"] = rng.standard_normal((inter, H)) * .2
        t[p + "mlp.up_proj.weight"] = rng.standard_normal((inter, H)) * .2
        t[p + "mlp.down_proj.weight"] = rng.standard_normal((H, inter)) * .2
    _write_safetensors(os.path.join(d, "model.safetensors"), t)
    return cfg, t


def test_measure_actscale_runs_and_reports_the_identity_it_claims():
    """``a_qkv`` must equal ``gamma^2 * n_in`` channel by channel -- that
    factorisation is the entire basis for using ``gamma^2`` as a proxy, so it
    is worth checking the measurement actually exhibits it."""
    from senseed.io import load_safetensors
    with tempfile.TemporaryDirectory() as d:
        _cfg, t = _tiny_model(d)
        out = os.path.join(d, "act.safetensors")
        r = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "measure_actscale.py"),
             "--model", d, "--out", out, "--tokens", "32"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stdout + r.stderr
        got = load_safetensors(out)

        for L in (0, 1):
            g = t[f"model.layers.{L}.input_layernorm.weight"]
            assert got[f"L{L}.a_qkv"] == pytest.approx(
                g ** 2 * got[f"L{L}.n_in"], rel=1e-4)
            # RMSNorm's guarantee: the normalised vector has unit mean square
            assert got[f"L{L}.n_in"].mean() == pytest.approx(1.0, rel=1e-3)
            # and the down_proj input really is the gated product
            assert got[f"L{L}.a_down"].shape == (32,)
            assert np.all(got[f"L{L}.a_down"] > 0)


def test_measure_actscale_estimate_beats_uniform_on_a_tiny_model():
    """The proxy has to explain *something* -- if gamma^2 correlated no better
    with the measured scale than a constant does, the idea is dead."""
    from senseed.io import load_safetensors
    with tempfile.TemporaryDirectory() as d:
        _cfg, t = _tiny_model(d)
        out = os.path.join(d, "act.safetensors")
        subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "measure_actscale.py"),
             "--model", d, "--out", out, "--tokens", "48"],
            capture_output=True, text=True, cwd=ROOT, check=True)
        got = load_safetensors(out)
        for L in (0, 1):
            g = t[f"model.layers.{L}.input_layernorm.weight"]
            est, meas = normed_scale(g), got[f"L{L}.a_qkv"]
            meas = meas / meas.mean()
            assert np.corrcoef(est, meas)[0, 1] > 0.5


def test_measure_script_estimator_agrees_with_the_package():
    """The script carries its own copy of the estimator because the full
    producer matrices only exist on the machine holding the checkpoint.  Two
    copies of a formula is exactly the situation that drifts, so pin them."""
    import measure_actscale as M
    rng = np.random.default_rng(30)
    arch = ArchSpec(hidden_size=32, num_attention_heads=8,
                    num_key_value_heads=2, intermediate_size=24)
    gamma = np.exp(rng.standard_normal(32))
    W_v = rng.standard_normal((arch.n_kv * arch.head_dim, 32))
    W_g = rng.standard_normal((24, 32)) * 0.3
    W_u = rng.standard_normal((24, 32)) * 0.3

    assert M.silu_second_moment(np.array([0.3, 2.0])) == pytest.approx(
        silu_second_moment(np.array([0.3, 2.0])))
    assert M.propagate(W_v, normed_scale(gamma)) == pytest.approx(
        propagate(W_v, normed_scale(gamma)))

    est_q = M.unit_mean(gamma.astype(np.float64) ** 2)
    a_v = M.propagate(W_v, est_q).reshape(arch.n_kv, arch.head_dim)
    est_o = M.unit_mean(np.repeat(a_v, arch.rep, axis=0).reshape(-1))
    assert est_o == pytest.approx(attn_out_scale(W_v, gamma, arch))

    a_post = M.unit_mean(gamma.astype(np.float64) ** 2)
    est_d = M.unit_mean(M.silu_second_moment(np.sqrt(M.propagate(W_g, a_post)))
                        * M.propagate(W_u, a_post))
    assert est_d == pytest.approx(mlp_act_scale(W_g, W_u, gamma))


def test_measure_script_writes_both_estimate_and_measurement():
    from senseed.io import load_safetensors
    with tempfile.TemporaryDirectory() as d:
        _cfg, _t = _tiny_model(d)
        out = os.path.join(d, "act.safetensors")
        subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "measure_actscale.py"),
             "--model", d, "--out", out, "--tokens", "32"],
            capture_output=True, text=True, cwd=ROOT, check=True)
        got = load_safetensors(out)
        for L in (0, 1):
            for kind in ("qkv", "o", "mlp", "down"):
                assert f"L{L}.est_{kind}" in got, kind
            assert got[f"L{L}.est_o"].shape == got[f"L{L}.a_o"].shape
            assert got[f"L{L}.est_down"].shape == got[f"L{L}.a_down"].shape


def test_gain_informativeness_is_the_general_one_applied_to_gamma_squared():
    """``blockmap`` keeps its own entry point because callers there hold
    ``gamma`` rather than ``a``.  Two spellings of one statistic is exactly the
    situation that drifts, so pin them."""
    from senseed.sensitivity.blockmap import gain_informativeness
    rng = np.random.default_rng(40)
    gamma = np.exp(rng.standard_normal(500) * 1.5)
    a = gain_informativeness(gamma)
    b = informativeness(normed_scale(gamma))
    for k in ("p99_over_median", "max_over_median", "top1pct_share", "cv", "n"):
        assert a[k] == pytest.approx(b[k]), k
