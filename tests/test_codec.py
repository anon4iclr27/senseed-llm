"""Validation of the SeedLM replication against the published paper.

Every test names the part of the paper it checks.  Run with::

    python -m pytest tests/ -v
    python tests/test_seedlm.py        # no pytest required
"""

from __future__ import annotations

import numpy as np
import pytest

from senseed import (
    LFSR_TAPS,
    SEEDLM_3BIT,
    SEEDLM_4BIT,
    Codebook,
    CompressedTensor,
    SeedLMConfig,
    build_U_all,
    build_V_cached,
    build_V_direct,
    compress,
    decompress,
    dequantize,
    full_cycle,
    is_maximal_length,
    lfsr_sequence,
    lfsr_state_for_offset,
    normalize_states,
    quantize,
    relative_error,
)
from senseed.codec.seedlm import _quantize_fast
from senseed.codec.bitstream import pack, packed_nbytes, unpack
from senseed.codec.quant import QuantSpec


# ===========================================================================
# Section 3.1 / Appendix A.1 / A.3 -- the LFSR itself
# ===========================================================================
def test_figure4_worked_example():
    """Appendix A.2, Figure 4: V(4) for K=3, C=4, P=2 is printed in the paper."""
    expected = np.array([[2, 5], [6, 7], [3, 1], [4, 2]])
    assert np.array_equal(build_V_direct(K=3, seed=4, C=4, P=2), expected)


def test_algorithm1_first_states_by_hand():
    """Algorithm 1 step-by-step for K=3, taps (0,1), seed 4.

    4 = 0b100 -> feedback 0^0 = 0, shift right -> 0b010 = 2
    2 = 0b010 -> feedback 0^1 = 1, shift right -> 0b101 = 5
    5 = 0b101 -> feedback 1^0 = 1, shift right -> 0b110 = 6
    """
    assert lfsr_sequence(3, 4, 6).tolist() == [2, 5, 6, 7, 3, 1]


def test_sequence_starts_after_the_seed():
    """results[0] is the successor of the seed, not the seed itself."""
    seq = lfsr_sequence(16, seed=12345, length=4)
    assert seq[0] != 12345


@pytest.mark.parametrize("K", [k for k in sorted(LFSR_TAPS) if k <= 18])
def test_table6_taps_are_maximal_length(K):
    """Appendix A.1 claims every tabulated tap set gives a maximal-length LFSR."""
    assert is_maximal_length(K), f"K={K} taps {LFSR_TAPS[K]} are not maximal length"


def test_cycle_returns_to_start():
    """A maximal-length cycle wraps back to state 1."""
    for K in (3, 8, 12, 16):
        assert full_cycle(K)[-1] == 1


def test_all_zero_state_never_occurs():
    """Section 3.1: the all-zero state is absorbing and must be excluded."""
    for K in (3, 8, 16):
        assert (full_cycle(K) != 0).all()


# ===========================================================================
# Section 3.2 -- normalisation of U(s)
# ===========================================================================
@pytest.mark.parametrize("K", [3, 8, 12, 16])
def test_normalisation_formula(K):
    """U(s) = (V(s) - 2**(K-1)) / (2**(K-1) - 1)."""
    cyc = full_cycle(K)
    half = 1 << (K - 1)
    expected = (cyc.astype(np.float64) - half) / (half - 1)
    assert np.allclose(normalize_states(cyc, K, dtype=np.float64), expected)


@pytest.mark.parametrize("K", [3, 8, 12, 16])
def test_normalisation_range_is_pm_one(K):
    """Section 3.2: normalisation puts the entries of U inside [-1, 1]."""
    u = normalize_states(full_cycle(K), K, dtype=np.float64)
    assert u.min() >= -1.0 and u.max() <= 1.0
    # The full nonzero state range is used, so the bounds are attained.
    assert np.isclose(u.min(), -1.0) and np.isclose(u.max(), 1.0)
    assert abs(u.mean()) < 1e-9


# ===========================================================================
# Figure 4 vs Algorithm 2: the two seed conventions
# ===========================================================================
def test_direct_and_cached_yield_the_same_candidate_set():
    """The paper specifies two different seed->matrix maps.

    Figure 4 runs the LFSR from the seed; Algorithm 2 step 5 indexes a cached
    cycle at ``s % length``.  They label rotations differently but expose the
    same set of candidate matrices, so the achievable reconstruction error --
    and therefore every accuracy number -- is unaffected.
    """
    K, C, P = 8, 4, 2
    cyc = full_cycle(K)
    n = len(cyc)
    direct = {build_V_direct(K, s, C, P).tobytes() for s in range(1, n + 1)}
    cached = {build_V_cached(cyc, o, C, P).tobytes() for o in range(n)}
    assert len(direct) == len(cached) == n
    assert direct == cached


def test_offset_to_hardware_state_is_exact():
    """A cached-mode offset must be translatable to a real register state."""
    K, C, P = 8, 4, 2
    cyc = full_cycle(K)
    for off in range(len(cyc)):
        state = lfsr_state_for_offset(cyc, off)
        assert np.array_equal(
            build_V_cached(cyc, off, C, P), build_V_direct(K, state, C, P)
        )


def test_window_wraps_around_end_of_cycle():
    """Algorithm 2 step 6: 'if the slice exceeds length, cycle through states'."""
    K, C, P = 3, 4, 2  # C*P = 8 > 2**3 - 1 = 7, so wrapping is forced
    cyc = full_cycle(K)
    V = build_V_cached(cyc, offset=0, C=C, P=P)
    assert V.ravel()[7] == V.ravel()[0]


# ===========================================================================
# Section 3.2 -- coefficient quantisation
# ===========================================================================
def test_quantised_values_fit_their_bit_widths():
    spec = QuantSpec(4, 4)
    rng = np.random.default_rng(0)
    for scale in (1e-6, 1e-3, 1.0, 1e3):
        t = rng.normal(0, scale, (500, 3)).astype(np.float32)
        q, e = quantize(t, spec, mode="range")
        assert q.min() >= -8 and q.max() <= 7
        assert e.min() >= -8 and e.max() <= 7


def test_dequantisation_is_exactly_q_times_two_to_the_e():
    spec = QuantSpec(4, 4)
    rng = np.random.default_rng(1)
    t = rng.normal(0, 0.02, (200, 3)).astype(np.float32)
    q, e = quantize(t, spec, mode="range")
    assert np.array_equal(
        dequantize(q, e, dtype=np.float64),
        q.astype(np.float64) * 2.0 ** e.astype(np.float64)[:, None],
    )


@pytest.mark.parametrize("mode", ["paper", "range"])
def test_search_quantiser_matches_reference_quantiser(mode):
    """The optimised in-loop quantiser must be bit-identical to the readable one."""
    spec = QuantSpec(4, 4)
    rng = np.random.default_rng(2)
    for scale in (1e-6, 1e-3, 1e-2, 1.0, 10.0, 1e3):
        t = rng.normal(0, scale, (2000, 3)).astype(np.float32)
        q_ref, e_ref = quantize(t, spec, mode=mode)
        q_fast, e_fast, t_hat = _quantize_fast(t, spec, mode)
        assert np.array_equal(q_ref.astype(np.int32), q_fast.astype(np.int32))
        assert np.array_equal(e_ref.astype(np.int32), e_fast.astype(np.int32))
        assert np.allclose(t_hat, dequantize(q_ref, e_ref))


@pytest.mark.parametrize("mode", ["paper", "range"])
def test_exponent_is_exact_at_power_of_two_boundaries(mode):
    """Gaussian samples never probe the case where the exponent rule can break.

    The shared exponent is a floor/ceil of a log, so the only inputs that can
    go wrong are the ones within an ULP of a power of two -- which random data
    hits with probability ~1e-7 and a test therefore never sees.  Constructing
    them deliberately: computing the exponent with a float32 ``log2`` gets
    these wrong (returning -7 where -8 is correct, which doubles the scale of
    every mantissa in the vector); ``frexp`` does not.
    """
    spec = QuantSpec(4, 4)
    edges = []
    for k in range(-12, 8):
        p = np.float32(2.0) ** np.float32(k)
        edges += [p, np.nextafter(p, np.float32(0)), np.nextafter(p, np.float32(1e30))]
        for m in (spec.q_max + 0.5, spec.q_max, 1.0):
            v = np.float32(p * m)
            edges += [v, np.nextafter(v, np.float32(0)),
                      np.nextafter(v, np.float32(1e30))]
    t = np.array([[v, v / 2, -v / 4] for v in edges], dtype=np.float32)

    q_ref, e_ref = quantize(t, spec, mode=mode)
    q_fast, e_fast, _ = _quantize_fast(t, spec, mode)
    assert np.array_equal(e_ref.astype(np.int32), e_fast.astype(np.int32))
    assert np.array_equal(q_ref.astype(np.int32), q_fast.astype(np.int32))

    # ...and both must agree with an exact float64 evaluation of the rule.
    amax = np.abs(t).max(axis=-1).astype(np.float64)
    if mode == "paper":
        exact = np.floor(np.log2(amax))
    else:
        exact = np.ceil(np.log2(amax / (spec.q_max + 0.5)))
    exact = np.clip(exact, spec.e_min, spec.e_max)
    assert np.array_equal(e_fast.astype(np.int64), exact.astype(np.int64))


def test_paper_exponent_rule_wastes_mantissa_bits():
    """Section 3.2's literal rule e = max_i floor(log2|t_i|) cannot be what ran.

    It forces max|t| / 2**e into [1, 2), so rounding can only return
    {-2, -1, 0, 1, 2}: 5 of the 16 codes are reachable and the coefficient
    error is several times larger than necessary.  Recorded here because it is
    the one place the replication has to deviate from the text to reproduce the
    reported accuracy.
    """
    spec = QuantSpec(4, 4)
    rng = np.random.default_rng(3)
    t = rng.normal(0, 0.02, (20000, 3)).astype(np.float32)

    q_paper, e_paper = quantize(t, spec, mode="paper")
    q_range, e_range = quantize(t, spec, mode="range")

    assert set(np.unique(q_paper).tolist()) <= {-2, -1, 0, 1, 2}
    assert set(np.unique(q_range).tolist()) == set(range(-7, 8))

    err_paper = np.linalg.norm(t - dequantize(q_paper, e_paper))
    err_range = np.linalg.norm(t - dequantize(q_range, e_range))
    assert err_paper > 3 * err_range


def test_zero_vector_quantises_to_zero():
    spec = QuantSpec(4, 4)
    t = np.zeros((5, 3), dtype=np.float32)
    q, e = quantize(t, spec, mode="range")
    assert (q == 0).all()
    assert np.array_equal(dequantize(q, e), t)


# ===========================================================================
# Section 3.3 -- the pseudo-inverse cache
# ===========================================================================
def test_pseudo_inverse_satisfies_moore_penrose_conditions():
    cfg = SeedLMConfig(C=8, P=3, K=10)
    cb = Codebook(cfg, dtype=np.float64)
    idx = np.random.default_rng(0).integers(0, cfg.n_seeds, 200)
    U, Up = cb.U[idx], cb.Upinv[idx]
    assert np.allclose(U @ Up @ U, U, atol=1e-8)
    assert np.allclose(Up @ U @ Up, Up, atol=1e-8)
    assert np.allclose((U @ Up).transpose(0, 2, 1), U @ Up, atol=1e-8)
    assert np.allclose((Up @ U).transpose(0, 2, 1), Up @ U, atol=1e-8)


def test_pseudo_inverse_survives_ill_conditioned_bases():
    """Overlapping windows make a small fraction of the K=16 bases ill conditioned.

    Consecutive seeds are sliding windows over one LFSR sequence, so some of
    them are nearly collinear: the condition number of U(s) is ~2.6 at the
    median but reaches 5e5 in the tail.  Forming ``(U^T U)^-1 U^T`` squares
    that, which in float32 destroys those seeds' coefficients.  The SVD route
    used by :class:`Codebook` must stay accurate.
    """
    cb = Codebook.get(SEEDLM_4BIT)
    U64 = cb.U.astype(np.float64)
    cond = np.linalg.cond(U64)
    assert cond.max() > 1e4, "expected an ill-conditioned tail"

    worst = np.argsort(cond)[-50:]
    ref = np.linalg.pinv(U64[worst])  # accurate pinv of the *stored* bases

    svd_rel = np.linalg.norm(ref - cb.Upinv[worst]) / np.linalg.norm(ref)

    Ut = U64[worst].transpose(0, 2, 1).astype(np.float32)
    Uw = cb.U[worst]
    normal_eq = np.linalg.inv(Ut @ Uw) @ Ut
    neq_rel = np.linalg.norm(ref - normal_eq) / np.linalg.norm(ref)

    assert svd_rel < 1e-6, f"SVD pinv lost accuracy: {svd_rel:.2e}"
    assert neq_rel > 1e-3, "expected float32 normal equations to be inaccurate"
    assert neq_rel > 100 * svd_rel


# ===========================================================================
# Eq. 3 -- the bit budget
# ===========================================================================
def test_published_configs_hit_their_bit_budgets():
    """Table 1: (C,P,K) = (8,3,16) -> 4 bits, (12,4,16) -> 3 bits."""
    assert SEEDLM_4BIT.bits_per_element == 4.0
    assert SEEDLM_3BIT.bits_per_element == 3.0
    assert SEEDLM_4BIT.bits_per_block == 32
    assert SEEDLM_3BIT.bits_per_block == 36


def test_eq3_matches_actual_payload():
    """The stored arrays must really occupy (K + 4 + 4P) bits per block."""
    rng = np.random.default_rng(0)
    for cfg in (SEEDLM_4BIT, SEEDLM_3BIT):
        W = rng.normal(0, 0.02, (16, cfg.C * 4)).astype(np.float32)
        ct = compress(W, cfg, n_candidate_seeds=64)
        assert ct.bits_per_element() == cfg.bits_per_element
        # seed field must be wide enough, mantissas and exponent must fit
        assert ct.seeds.max() < 2 ** cfg.K
        assert ct.q.min() >= -8 and ct.q.max() <= 7
        assert ct.e.min() >= -8 and ct.e.max() <= 7


# ===========================================================================
# Algorithm 3 -- the search
# ===========================================================================
def _reference_search(w, cfg, cb, mode="range"):
    """Algorithm 3 transcribed literally, one block at a time."""
    best_j, best_q, best_e, best_norm = -1, None, None, np.inf
    for j in range(cfg.n_seeds):
        t = cb.Upinv[j] @ w
        q, e = quantize(t, cfg.spec, mode=mode)
        r = w - cb.U[j] @ dequantize(q, e, dtype=w.dtype)
        norm = float(r @ r)
        if norm < best_norm:
            best_j, best_q, best_e, best_norm = j, q, e, norm
    return best_j, best_q, best_e, best_norm


def test_vectorised_search_matches_literal_algorithm3():
    """The chunked implementation must achieve what the pseudocode achieves.

    The invariant is the *reconstruction*, not the seed index.  Adjacent seeds
    are overlapping windows of one LFSR sequence, so the candidate set contains
    exact ties (see ``test_adjacent_seeds_are_structurally_degenerate``); which
    member of a tied group wins is decided by summation order.  Asserting seed
    equality would be asserting a round-off outcome.
    """
    cfg = SeedLMConfig(C=8, P=3, K=9)  # 511 seeds -- small enough to brute force
    cb = Codebook.get(cfg)
    rng = np.random.default_rng(7)
    W = rng.normal(0, 0.02, (12, cfg.C)).astype(np.float32)

    ct = compress(W, cfg, block_chunk=5, seed_chunk=37)  # deliberately ragged tiles
    t_hat = dequantize(ct.q, ct.e)
    for i, w in enumerate(W):
        j, q, e, norm = _reference_search(w, cfg, cb)
        mine = cb.U[int(ct.seeds[i])] @ t_hat[i]
        assert np.allclose(np.sum((w - mine) ** 2), norm, rtol=1e-5, atol=1e-12)
        theirs = cb.U[j] @ dequantize(q, e, dtype=np.float32)
        assert np.allclose(mine, theirs, atol=1e-6)


def test_adjacent_seeds_are_structurally_degenerate():
    """The 2**K candidate bases are far from 2**K independent bases.

    U(s) is a sliding window over one LFSR sequence, so column p of U(s) equals
    column p-1 of U(s+1).  A coefficient vector whose last entry is zero under
    seed s therefore produces exactly the same reconstruction as a shifted
    coefficient vector under seed s+1.  This caps how much genuine diversity
    the seed search has to work with, and it is why the arg-min over seeds is
    not unique.
    """
    cfg = SeedLMConfig(C=8, P=3, K=10)
    cb = Codebook.get(cfg)
    s = 123
    # Columns overlap by construction.
    assert np.array_equal(cb.U[s][:, 1:], cb.U[s + 1][:, :-1])

    # ...so these two (seed, coefficient) pairs are the same weight block.
    a = cb.U[s] @ np.array([0.0, 2.0, -3.0], dtype=np.float32)
    b = cb.U[s + 1] @ np.array([2.0, -3.0, 0.0], dtype=np.float32)
    assert np.allclose(a, b, atol=1e-7)


def test_result_is_independent_of_tiling():
    """Chunk sizes are a performance knob and must not change the answer."""
    cfg = SeedLMConfig(C=8, P=3, K=10)
    rng = np.random.default_rng(8)
    W = rng.normal(0, 0.02, (40, cfg.C)).astype(np.float32)
    a = compress(W, cfg, block_chunk=7, seed_chunk=13)
    b = compress(W, cfg, block_chunk=64, seed_chunk=4096)
    assert np.array_equal(a.seeds, b.seeds)
    assert np.array_equal(a.q, b.q)
    assert np.array_equal(a.e, b.e)


def test_reported_error_survives_the_round_trip():
    """Decompressing must reproduce exactly the block the search scored."""
    cfg = SEEDLM_4BIT
    cb = Codebook.get(cfg)
    rng = np.random.default_rng(9)
    W = rng.normal(0, 0.02, (32, cfg.C)).astype(np.float32)
    ct = compress(W, cfg, n_candidate_seeds=2048)
    What = decompress(ct)

    t = dequantize(ct.q, ct.e)
    for i in range(len(W)):
        manual = cb.U[int(ct.seeds[i])] @ t[i]
        assert np.allclose(manual, What[i], atol=1e-6)


def test_searching_more_seeds_never_hurts():
    """Monotonicity: the arg-min is over a growing candidate set."""
    cfg = SEEDLM_4BIT
    rng = np.random.default_rng(10)
    W = rng.normal(0, 0.02, (64, cfg.C)).astype(np.float32)
    errs = []
    for n in (1, 16, 256, 4096):
        ct = compress(W, cfg, n_candidate_seeds=n)
        errs.append(np.linalg.norm(W - decompress(ct)))
    assert all(errs[i] >= errs[i + 1] - 1e-9 for i in range(len(errs) - 1))
    assert errs[0] > errs[-1], "the seed search should actually be buying something"


# ===========================================================================
# Plumbing
# ===========================================================================
def test_padding_for_sizes_that_are_not_a_multiple_of_C():
    cfg = SEEDLM_4BIT
    rng = np.random.default_rng(11)
    W = rng.normal(0, 0.02, (7, 5)).astype(np.float32)  # 35 elements, C = 8
    ct = compress(W, cfg, n_candidate_seeds=256)
    assert ct.pad == 5 and ct.n_blocks == 5
    assert decompress(ct).shape == W.shape


def test_shape_and_dtype_are_preserved():
    cfg = SEEDLM_4BIT
    rng = np.random.default_rng(12)
    for shape in [(16, 16), (4, 8, 8), (64,)]:
        W = rng.normal(0, 0.02, shape).astype(np.float32)
        What = decompress(compress(W, cfg, n_candidate_seeds=128))
        assert What.shape == W.shape and What.dtype == np.float32


def test_packed_payload_is_exactly_eq3_bits():
    """Serialise for real and check the byte count against (K + 4 + 4P)/C."""
    rng = np.random.default_rng(14)
    for cfg in (SEEDLM_4BIT, SEEDLM_3BIT):
        W = rng.normal(0, 0.02, (cfg.C * 300,)).astype(np.float32)
        ct = compress(W, cfg, n_candidate_seeds=512)
        buf = pack(ct)
        assert len(buf) == packed_nbytes(ct)
        assert len(buf) * 8 >= ct.n_blocks * cfg.bits_per_block
        assert len(buf) * 8 - ct.n_blocks * cfg.bits_per_block < 8  # only tail pad
        assert 8 * len(buf) / W.size == pytest.approx(
            cfg.bits_per_element, abs=8 / W.size
        )


def test_pack_unpack_round_trip_is_lossless():
    rng = np.random.default_rng(15)
    for cfg in (SEEDLM_4BIT, SEEDLM_3BIT):
        W = rng.normal(0, 0.02, (37, 11)).astype(np.float32)
        ct = compress(W, cfg, n_candidate_seeds=512)
        back = unpack(pack(ct), ct.shape, cfg, pad=ct.pad)
        assert np.array_equal(back.seeds, ct.seeds)
        assert np.array_equal(back.q, ct.q)
        assert np.array_equal(back.e, ct.e)
        assert np.array_equal(decompress(back), decompress(ct))


def test_pack_survives_extreme_field_values():
    """Two's complement fields must round-trip at their limits, not just typically."""
    cfg = SEEDLM_4BIT
    n = 6
    ct = CompressedTensor(
        seeds=np.array([0, 1, 2**16 - 2, 2**16 - 2, 7, 9], dtype=np.uint32),
        q=np.array([[-8, 7, 0], [7, -8, -1], [0, 0, 0],
                    [-8, -8, -8], [7, 7, 7], [1, -1, 2]], dtype=np.int8),
        e=np.array([-8, 7, 0, -1, 3, -8], dtype=np.int8),
        shape=(n * cfg.C,),
        cfg=cfg,
    )
    back = unpack(pack(ct), ct.shape, cfg, pad=0)
    assert np.array_equal(back.seeds, ct.seeds)
    assert np.array_equal(back.q, ct.q)
    assert np.array_equal(back.e, ct.e)


def test_hardware_seed_translation():
    cfg = SeedLMConfig(C=8, P=3, K=9)
    cb = Codebook.get(cfg)
    rng = np.random.default_rng(13)
    W = rng.normal(0, 0.02, (8, cfg.C)).astype(np.float32)
    ct = compress(W, cfg)
    hw = ct.hardware_seeds(cb)
    assert (hw >= 1).all() and (hw <= cfg.n_seeds).all()
    for off, state in zip(ct.seeds, hw):
        assert np.array_equal(
            build_V_cached(cb.cycle, int(off), cfg.C, cfg.P),
            build_V_direct(cfg.K, int(state), cfg.C, cfg.P),
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))


# ===========================================================================
# S-Quant (Wang et al., ICML 2026) -- separate paper, same LFSR generator
# ===========================================================================
def test_squant_normalisation_matches_seedlm():
    """S-Quant Eq. 3 is SeedLM's normalisation verbatim, so the cycles agree."""
    from senseed.codec.squant import SQuantConfig, SQuantCodebook
    cfg = SQuantConfig(B=8, S=10, k_max=4)
    cb = SQuantCodebook.get(cfg)
    assert cb.U.min() >= -1.0 and cb.U.max() <= 1.0
    assert np.isclose(cb.U.min(), -1.0) and np.isclose(cb.U.max(), 1.0)


def test_squant_orthonormal_basis_spans_the_same_subspace():
    """The cumulative-sum search is only valid if Q spans exactly U's subspace."""
    from senseed.codec.squant import SQuantConfig, SQuantCodebook
    cfg = SQuantConfig(B=16, S=10, k_max=8)
    cb = SQuantCodebook.get(cfg)
    for i in (0, 137, 512):
        for k in (2, 4, 8):
            U = cb.U[i][:, :k].astype(np.float64)
            Q = cb.Q[i][:, :k].astype(np.float64)
            assert np.allclose(Q.T @ Q, np.eye(k), atol=1e-6)
            assert np.allclose(U @ np.linalg.pinv(U), Q @ Q.T, atol=1e-6)


def test_squant_explained_energy_cumsum_equals_direct_projection():
    """R_k from the cumulative sum must equal an explicit least-squares fit."""
    from senseed.codec.squant import SQuantConfig, SQuantCodebook
    cfg = SQuantConfig(B=16, S=10, k_max=8)
    cb = SQuantCodebook.get(cfg)
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.02, 16)
    i = 321
    cum = np.cumsum((w @ cb.Q[i].astype(np.float64)) ** 2) / (w @ w)
    for k in range(1, 9):
        U = cb.U[i][:, :k].astype(np.float64)
        direct = 1 - np.linalg.norm(w - U @ np.linalg.pinv(U) @ w) ** 2 / (w @ w)
        assert np.isclose(cum[k - 1], direct, atol=1e-8)


def test_squant_meets_its_energy_threshold():
    """Algorithm 1's contract: every block reaches R_th, or reports it could not."""
    from senseed.codec.squant import SQuantConfig, compress_squant
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.02, (32, 128)).astype(np.float32)
    for rth in (0.80, 0.90):
        cfg = SQuantConfig(B=16, S=12, G=8, R_th=rth, k_max=12)
        r = compress_squant(W, cfg)
        assert (r.R[r.reached] >= rth - 1e-9).all()
        assert (r.ks >= cfg.k_min).all() and (r.ks <= cfg.k_max).all()


def test_squant_k_adapts_and_rate_follows_eq13():
    from senseed.codec.squant import SQuantConfig, compress_squant
    rng = np.random.default_rng(2)
    W = rng.normal(0, 0.02, (32, 128)).astype(np.float32)
    prev = 0.0
    for rth in (0.80, 0.90, 0.95):
        cfg = SQuantConfig(B=16, S=12, G=8, R_th=rth, k_max=14)
        r = compress_squant(W, cfg)
        assert r.ks.min() < r.ks.max(), "k should vary across blocks"
        assert r.mean_k > prev, "a higher threshold must buy more bases"
        prev = r.mean_k
        expected = (cfg.S + 8 * r.mean_k + 16 / cfg.G) / cfg.B
        assert np.isclose(r.bits_per_element(), expected)


def test_squant_error_floor_is_set_by_the_threshold():
    """rel.err >= sqrt(1 - R) by definition: R_th caps accuracy before quantising.

    This is why the threshold, not the coefficient format, decides S-Quant's
    accuracy -- and why its int8 coefficients turn out to be nearly free.
    """
    from senseed.codec.squant import SQuantConfig, compress_squant, decompress_squant
    rng = np.random.default_rng(3)
    W = rng.normal(0, 0.02, (64, 128)).astype(np.float32)
    cfg = SQuantConfig(B=16, S=12, G=8, R_th=0.90, k_max=14)
    r = compress_squant(W, cfg)
    err = relative_error(W, decompress_squant(r))
    floor = np.sqrt(1 - np.median(r.R))
    assert err <= floor * 1.15, "error should sit close to the energy floor"
    assert err >= floor * 0.6


def test_squant_round_trip_shapes_and_padding():
    from senseed.codec.squant import SQuantConfig, compress_squant, decompress_squant
    rng = np.random.default_rng(4)
    cfg = SQuantConfig(B=16, S=10, G=4, R_th=0.85, k_max=8)
    for shape in ((7, 5), (64,), (4, 8, 8)):
        W = rng.normal(0, 0.02, shape).astype(np.float32)
        r = compress_squant(W, cfg)
        out = decompress_squant(r)
        assert out.shape == W.shape and out.dtype == np.float32
        assert np.isfinite(out).all()


# ===========================================================================
# SenSeed -- our own method, assembled from the two papers' measured deficits
# ===========================================================================
def test_senseed_reduces_to_seedlm():
    """With every improvement disabled, SenSeed must match SeedLM.

    This is the load-bearing test: it is what licenses reading any difference
    elsewhere as the effect of a specific change rather than of a reimplementation.
    """
    from senseed.codec.allocate import SenSeedConfig, compress_senseed, decompress_senseed
    rng = np.random.default_rng(0)
    w = (rng.normal(0, 1, 8192) * 0.0166).astype(np.float32)
    ref = relative_error(w, decompress(compress(w, SEEDLM_4BIT)))
    cfg = SenSeedConfig(B=8, S=16, k_mode="fixed", k_fixed=3, exp_group=1,
                     signal_k=False, basis_layout="row", k_min=2, k_max=6)
    got = relative_error(w, decompress_senseed(compress_senseed(w, cfg)))
    assert cfg.bits_per_element(3.0) == 4.0
    assert abs(got / ref - 1) < 0.05, f"SenSeed {got:.4f} vs SeedLM {ref:.4f}"


def test_senseed_rate_accounting_charges_for_signalling_k():
    """Adaptive k is not free: the decoder must be told k."""
    from senseed.codec.allocate import SenSeedConfig
    fixed = SenSeedConfig(B=8, S=16, k_mode="fixed", k_fixed=3, exp_group=1)
    adapt = SenSeedConfig(B=8, S=16, k_mode="marginal", k_min=2, k_max=4, exp_group=1)
    assert fixed.sig_bits == 0.0
    assert adapt.sig_bits == 2.0          # ceil(log2(3)) values -> 2 bits
    assert adapt.bits_per_element(3.0) > fixed.bits_per_element(3.0)


def test_senseed_marginal_allocation_respects_the_budget():
    from senseed.codec.allocate import SenSeedConfig, compress_senseed
    rng = np.random.default_rng(1)
    nb = 512
    s = np.exp(rng.normal(0, 1.2, nb)) * 0.0166        # heterogeneous blocks
    w = (rng.normal(0, 1, (nb, 8)) * s[:, None]).astype(np.float32).ravel()
    for target in (4.0, 4.4):
        cfg = SenSeedConfig(B=8, S=16, k_mode="marginal", target_bits=target,
                         k_min=2, k_max=4, exp_group=1, basis_layout="row")
        assert cfg.feasible()
        r = compress_senseed(w, cfg)
        assert r.bits_per_element() <= target + 1e-6
        assert r.ks.min() < r.ks.max(), "allocation should not be uniform here"

    # Below the floor rate the request is impossible, not merely tight:
    # every block needs k_min bases, so the rate cannot go lower.
    bad = SenSeedConfig(B=8, S=16, k_mode="marginal", target_bits=3.5,
                     k_min=2, k_max=4, exp_group=1, basis_layout="row")
    assert not bad.feasible()
    assert bad.min_bits == pytest.approx(3.75)
    with pytest.raises(ValueError, match="floor rate"):
        compress_senseed(w, bad)


def test_senseed_concave_envelope_is_non_increasing():
    """Water-filling is only valid on non-increasing marginal gains."""
    from senseed.codec.allocate import _concave_envelope
    rng = np.random.default_rng(2)
    R = np.sort(rng.random((50, 6)), axis=1)
    Rc = _concave_envelope(R)
    g = np.diff(Rc, axis=1, prepend=0.0)
    assert (np.diff(g, axis=1) <= 1e-12).all()
    assert (Rc <= R + 1e-12).all(), "envelope must never overstate the real gain"


def test_senseed_condition_filter_excludes_bad_bases():
    from senseed.codec.allocate import SenSeedConfig, _condition_mask
    from senseed.codec.squant import SQuantConfig, SQuantCodebook
    cfg = SenSeedConfig(B=8, S=12, k_max=4, cond_max=10.0, basis_layout="row")
    cb = SQuantCodebook.get(SQuantConfig(B=8, S=12, k_max=4),
                            basis_layout="row")
    m = _condition_mask(cb, cfg)
    assert m is not None and m.shape == (cb.U.shape[0], 4)
    for k in (2, 3, 4):
        U = cb.U[:, :, :k].astype(np.float64)
        cond = np.linalg.cond(U)
        assert (cond[m[:, k - 1]] <= 10.0 + 1e-6).all()
        assert m[:, k - 1].mean() > 0.5, "filter should keep most seeds"


def test_senseed_adaptive_pays_only_when_blocks_differ():
    """The measured conditional result, pinned as a regression test.

    Adaptive allocation moves bases between blocks; if every block is
    statistically identical there is nothing to move and the bits spent
    signalling k are pure loss.
    """
    from senseed.codec.allocate import SenSeedConfig, compress_senseed, decompress_senseed
    rng = np.random.default_rng(3)
    cfg = SenSeedConfig(B=8, S=16, k_mode="marginal", target_bits=4.0,
                     k_min=2, k_max=4, exp_group=1, basis_layout="row",
                     cond_max=10)

    iid = (rng.normal(0, 1, 16384) * 0.0166).astype(np.float32)
    nb = 2048
    s = np.exp(rng.normal(0, 1.2, nb)) * 0.0166
    het = (rng.normal(0, 1, (nb, 8)) * s[:, None]).astype(np.float32).ravel()

    for w, adaptive_should_win in ((iid, False), (het, True)):
        sl = relative_error(w, decompress(compress(w, SEEDLM_4BIT)))
        bb = relative_error(w, decompress_senseed(compress_senseed(w, cfg)))
        assert (bb < sl) == adaptive_should_win, (
            f"expected adaptive {'win' if adaptive_should_win else 'loss'}; "
            f"SeedLM {sl:.4f} SenSeed {bb:.4f}")


# ===========================================================================
# Data-free sensitivity from RMSNorm gains
# ===========================================================================
def test_gain_informativeness_detects_a_flat_gain():
    """A uniform gain carries no signal; the metric must say so."""
    from senseed.sensitivity.blockmap import gain_informativeness
    flat = gain_informativeness(np.ones(1000))
    assert flat["top1pct_share"] == pytest.approx(0.01, abs=1e-3)
    assert flat["cv"] == pytest.approx(0.0, abs=1e-9)
    assert flat["p99_over_median"] == pytest.approx(1.0, abs=1e-9)

    spiky = np.ones(1000)
    spiky[:10] = 30.0
    s = gain_informativeness(spiky)
    assert s["top1pct_share"] > 0.85
    assert s["cv"] > 5


def test_block_importance_shape_and_normalisation():
    from senseed.sensitivity.blockmap import block_importance
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.02, (16, 64)).astype(np.float32)
    g = np.abs(rng.normal(1, 0.3, 64))
    imp = block_importance(W, gamma=g, B=8, mode="gamma")
    assert imp.shape == (16 * 64 // 8,)
    assert imp.mean() == pytest.approx(1.0)
    assert (imp > 0).all()


def test_block_importance_is_uniform_without_signal():
    """No gamma and no consumer matrix must reduce to uniform weighting."""
    from senseed.sensitivity.blockmap import block_importance
    W = np.random.default_rng(1).normal(0, 0.02, (8, 64)).astype(np.float32)
    imp = block_importance(W, gamma=None, W_next=None, B=8, mode="both")
    assert np.allclose(imp, 1.0)
    flat = block_importance(W, gamma=np.ones(64), B=8, mode="gamma")
    assert np.allclose(flat, 1.0)


def test_block_importance_tracks_gamma():
    """Blocks over high-gamma input channels must get higher importance."""
    from senseed.sensitivity.blockmap import block_importance
    g = np.ones(64)
    g[:8] = 10.0                       # first block of every row is salient
    W = np.random.default_rng(2).normal(0, 0.02, (4, 64)).astype(np.float32)
    imp = block_importance(W, gamma=g, B=8, mode="gamma").reshape(4, 8)
    assert (imp[:, 0] > imp[:, 1:].max()).all()


def test_norm_routing_matches_the_architecture():
    """q/k/v read input_layernorm; gate/up read post_attention_layernorm;
    o_proj and down_proj read un-normalised inputs and must return None."""
    from senseed.sensitivity.blockmap import gamma_for_layer
    T = {"model.layers.3.input_layernorm.weight": np.ones(4),
         "model.layers.3.post_attention_layernorm.weight": np.full(4, 2.0)}
    for p, expect in (("self_attn.q_proj", 1.0), ("self_attn.k_proj", 1.0),
                      ("self_attn.v_proj", 1.0), ("mlp.gate_proj", 2.0),
                      ("mlp.up_proj", 2.0)):
        g = gamma_for_layer(T, f"model.layers.3.{p}.weight")
        assert g is not None and g[0] == expect
    for p in ("self_attn.o_proj", "mlp.down_proj"):
        assert gamma_for_layer(T, f"model.layers.3.{p}.weight") is None


# ---------------------------------------------------------------------------
# the pruned search must be an accelerator, not an approximation
# ---------------------------------------------------------------------------

def test_pruned_search_returns_the_exhaustive_reconstruction():
    """A shortlist is only legitimate if it provably contains the winner.  The
    certificate is checked per block and violations are re-run exhaustively, so
    the two paths must agree exactly -- on outlier-heavy blocks too, which are
    where the bound is loosest and the fallback actually fires."""
    from senseed.codec.prune import compress_pruned

    rng = np.random.default_rng(77)
    for label, W in (
        ("gaussian", rng.normal(0, 0.02, (4, 256)).astype(np.float32)),
        ("outliers", (rng.normal(0, 0.02, (4, 256)) *
                      np.where(rng.random((4, 256)) < 0.02, 50, 1)
                      ).astype(np.float32)),
        ("tiny-norm", (rng.normal(0, 1e-4, (4, 256))).astype(np.float32)),
    ):
        exact = decompress(compress(W, SEEDLM_4BIT))
        fast, st = compress_pruned(W, SEEDLM_4BIT, stats=True)
        assert np.allclose(exact, decompress(fast), atol=1e-6), label
        assert st.exact, label


def test_pruned_search_rate_is_unchanged():
    from senseed.codec.prune import compress_pruned
    W = np.random.default_rng(78).normal(0, 0.02, (4, 128)).astype(np.float32)
    assert compress_pruned(W, SEEDLM_4BIT).bits_per_element() == 4.0


def test_a_shortlist_of_one_still_certifies_or_falls_back():
    """top_m=1 makes the certificate fail constantly; the result must still be
    exhaustive-exact, which is what proves the fallback path is wired up."""
    from senseed.codec.prune import compress_pruned
    W = np.random.default_rng(79).normal(0, 0.02, (4, 128)).astype(np.float32)
    exact = decompress(compress(W, SEEDLM_4BIT))
    fast, st = compress_pruned(W, SEEDLM_4BIT, top_m=1, stats=True)
    assert st.fallback_blocks > 0
    assert np.allclose(exact, decompress(fast), atol=1e-6)
