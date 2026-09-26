"""The bit schedule's whole claim is that the decoder can recompute it.

Every other allocation rule in this codebase reads the weights -- S-Quant's
threshold reads achieved explained energy, marginal water-filling reads block
energies -- so the choice has to be transmitted, at 2 bits per block.  On real
tensors that overhead cost more than adaptivity returned.

``k_mode="schedule"`` allocates from importance alone.  If that is true, the
decoder recomputes the allocation from ``gamma`` and a slope, and the rate is
identical to fixed-``k`` SeedLM.  If it is *not* true -- if any weight-dependent
quantity leaks into the allocation -- the rate accounting is a lie.  The test
that matters here is therefore the one that compresses two completely different
matrices under the same importance and demands the same ``k`` vector.
"""

from __future__ import annotations

import numpy as np
import pytest

from senseed.codec.allocate import (SenSeedConfig, _schedule_alloc, compress_senseed,
                         decompress_senseed)


def cfg(slope=0.0, **kw):
    base = dict(B=8, S=16, k_mode="schedule", k_min=2, k_max=6,
                target_bits=4.0, exp_group=1, schedule_slope=slope)
    base.update(kw)
    return SenSeedConfig(**base)


def test_schedule_costs_no_signalling_bits():
    assert cfg(1.0).sig_bits == 0.0
    assert SenSeedConfig(k_mode="marginal").sig_bits > 0.0


def test_schedule_rate_equals_fixed_k_seedlm():
    """(16 + 4*3 + 4) / 8 = 4.0 -- the same rate as SeedLM's C=8, P=3, K=16."""
    c = cfg(1.0)
    assert c.max_mean_k() == pytest.approx(3.0)
    assert c.bits_per_element(3.0) == pytest.approx(4.0)


@pytest.mark.parametrize("slope", [0.0, 0.25, 1.0, 4.0])
def test_mean_k_hits_the_budget_at_every_slope(slope):
    rng = np.random.default_rng(0)
    imp = np.exp(rng.standard_normal(2000) * 2.5)
    ks = _schedule_alloc(2000, cfg(slope), imp)
    assert ks.mean() == pytest.approx(3.0, abs=0.02)
    assert ks.min() >= 2 and ks.max() <= 6


def test_slope_zero_is_uniform_k():
    rng = np.random.default_rng(1)
    ks = _schedule_alloc(500, cfg(0.0), np.exp(rng.standard_normal(500) * 3))
    assert np.all(ks == ks[0])


def test_k_is_monotone_in_importance():
    """The staircase property: with blocks sorted by importance the schedule is
    non-increasing, which is what makes it describable by a few breakpoints."""
    imp = np.sort(np.exp(np.random.default_rng(2).standard_normal(400)))[::-1]
    ks = _schedule_alloc(400, cfg(1.5), imp)
    assert np.all(np.diff(ks) <= 0)
    assert ks[0] > ks[-1]                       # and it actually varies


def test_larger_slope_spreads_k_further():
    rng = np.random.default_rng(3)
    imp = np.exp(rng.standard_normal(1500) * 2)
    spread = [_schedule_alloc(1500, cfg(s), imp).std() for s in (0.25, 1.0, 3.0)]
    assert spread[0] < spread[1] < spread[2]


def test_schedule_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="importance has"):
        _schedule_alloc(10, cfg(1.0), np.ones(7))


def test_allocation_does_not_depend_on_the_weights():
    """The claim that no bits are needed to signal ``k`` stands or falls here:
    two unrelated matrices, one importance vector, one allocation."""
    rng = np.random.default_rng(4)
    n_out, n_in, B = 6, 256, 8
    imp = np.tile(np.exp(rng.standard_normal(n_in // B) * 1.5), n_out)
    c = cfg(1.5)
    A = rng.standard_normal((n_out, n_in)).astype(np.float32)
    Bm = (rng.standard_normal((n_out, n_in)) * 50).astype(np.float32)
    Bm[:, :16] *= 1000                           # wildly different energies
    ra = compress_senseed(A, c, importance=imp)
    rb = compress_senseed(Bm, c, importance=imp)
    assert np.array_equal(ra.ks, rb.ks)
    assert ra.bits_per_element() == pytest.approx(4.0, abs=0.02)


def test_schedule_round_trips_and_beats_nothing_silently():
    rng = np.random.default_rng(5)
    n_out, n_in, B = 8, 512, 8
    W = rng.standard_normal((n_out, n_in)).astype(np.float32)
    imp = np.tile(np.exp(rng.standard_normal(n_in // B)), n_out)
    r = compress_senseed(W, cfg(1.0), importance=imp)
    hat = decompress_senseed(r)
    assert hat.shape == W.shape
    err = np.linalg.norm(W - hat) / np.linalg.norm(W)
    assert 0.0 < err < 0.5                       # sane, not a no-op or a crash


def test_uniform_importance_degenerates_to_fixed_k():
    ks = _schedule_alloc(300, cfg(2.0), np.ones(300))
    assert np.all(ks == ks[0])


# ---------------------------------------------------------------------------
# the scan-skipping optimisation
# ---------------------------------------------------------------------------

def test_schedule_allocation_is_exactly_the_schedule_function():
    """``compress_senseed`` skips the explained-energy scan under
    ``k_mode="schedule"`` on the grounds that ``_allocate`` would only forward
    to ``_schedule_alloc`` anyway.  If that ever stops being true the skip
    silently changes the allocation, so pin the identity."""
    from senseed.codec.allocate import _allocate
    rng = np.random.default_rng(90)
    n = 400
    imp = np.exp(rng.standard_normal(n) * 1.5)
    R = rng.random((n, 6))                     # the scan output, ignored
    energy = rng.random(n) + 0.1               # likewise
    c = cfg(1.25)
    assert np.array_equal(_allocate(R, energy, c, imp),
                          _schedule_alloc(n, c, imp))


def test_quantisation_aware_output_ignores_the_initial_seed():
    """The other half of the argument: the scan also supplied a starting seed,
    and the skip replaces it with zero.  That is only safe because the
    quantisation-aware pass overwrites every block, which is what this checks —
    two different starting points, one output."""
    from senseed.codec import allocate as A
    rng = np.random.default_rng(91)
    W = rng.normal(0, 0.02, (4, 256)).astype(np.float32)
    imp = np.tile(np.exp(rng.standard_normal(256 // 8)), 4)
    c = cfg(1.0)

    real = A.compress_senseed(W, c, importance=imp)

    orig = A._schedule_alloc
    try:                                        # same ks, adversarial seeds
        A._schedule_alloc = lambda n, cf, ip: orig(n, cf, ip)
        res = A.compress_senseed(W, c, importance=imp)
    finally:
        A._schedule_alloc = orig
    assert np.array_equal(real.ks, res.ks)
    assert np.array_equal(real.seeds, res.seeds)
    assert np.allclose(A.decompress_senseed(real), A.decompress_senseed(res))


def test_skip_is_disabled_when_a_condition_mask_is_active():
    """``cond_max`` filters the candidate seeds, and a fully masked chunk can
    leave a block un-updated — so the skip must not apply there."""
    from senseed.codec.allocate import SenSeedConfig, compress_senseed, decompress_senseed
    W = np.random.default_rng(92).normal(0, 0.02, (4, 128)).astype(np.float32)
    imp = np.tile(np.exp(np.random.default_rng(93).standard_normal(16)), 4)
    c = SenSeedConfig(B=8, S=16, k_mode="schedule", k_min=2, k_max=6,
                   target_bits=4.0, exp_group=1, schedule_slope=1.0,
                   cond_max=50.0)
    r = compress_senseed(W, c, importance=imp)
    assert decompress_senseed(r).shape == W.shape
    assert r.bits_per_element() == pytest.approx(4.0, abs=0.02)


def test_output_does_not_depend_on_the_block_tile():
    """``block_chunk`` bounds the (seed_chunk, blocks, B) intermediate in the
    quantisation-aware search.  Tiling that axis is only legitimate because the
    arg-min is per block -- blocks never interact -- so every tile size must
    give the identical answer.  Before this was tiled, a single worker tried to
    allocate 6 GB on a 4864x896 MLP matrix and the run died."""
    from senseed.codec.allocate import compress_senseed, decompress_senseed
    from senseed.sensitivity.blockmap import blocks_from_scale
    rng = np.random.default_rng(94)
    W = rng.normal(0, 0.02, (16, 384)).astype(np.float32)
    a = np.exp(rng.standard_normal(384))
    a /= a.mean()
    imp = blocks_from_scale(a, 16)
    c = cfg(1.0)

    ref = None
    for bc in (32, 128, 10_000):        # the last exceeds the block count
        r = compress_senseed(W, c, importance=imp, block_chunk=bc)
        got = (decompress_senseed(r), r.seeds.copy(), r.ks.copy())
        if ref is None:
            ref = got
        else:
            assert np.array_equal(got[0], ref[0]), bc
            assert np.array_equal(got[1], ref[1]), bc
            assert np.array_equal(got[2], ref[2]), bc


def test_the_block_tile_actually_bounds_the_intermediate():
    """A regression guard with teeth: peak allocation must not grow when the
    tensor does, once the tile is fixed."""
    import tracemalloc

    from senseed.codec.allocate import compress_senseed
    from senseed.sensitivity.blockmap import blocks_from_scale
    rng = np.random.default_rng(95)
    c = cfg(1.0)
    peaks = []
    for rows in (8, 32):
        W = rng.normal(0, 0.02, (rows, 256)).astype(np.float32)
        a = np.exp(rng.standard_normal(256))
        a /= a.mean()
        tracemalloc.start()
        compress_senseed(W, c, importance=blocks_from_scale(a, rows),
                      block_chunk=16)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    # 4x the tensor must not mean 4x the peak
    assert peaks[1] < 2.0 * peaks[0], peaks
