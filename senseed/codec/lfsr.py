"""Linear Feedback Shift Register machinery for SeedLM.

Implements Section 3.1, Appendix A.1 (tap table), Appendix A.2 (state sequence
illustration) and Appendix A.3 (Algorithm 1) of

    Shafipour et al., "SeedLM: Compressing LLM Weights into Seeds of
    Pseudo-Random Generators", ICLR 2025.

Two ways of turning a seed into the C x P matrix U(s) are provided, because the
paper specifies two that are *not* literally the same map (see
``build_V_direct`` vs ``build_V_cached`` and the note in ``README.md``):

``direct``
    Run the LFSR forward from the seed itself, as in Figure 4 / Algorithm 1.
    This is what real hardware does: the stored seed *is* the register state.

``cached``
    Generate the whole 2**K - 1 state cycle once (starting from state 1) and
    index into it at offset ``s % length``, as in Algorithm 2 step 5.  This is
    the formulation that makes the exhaustive seed search tractable, since every
    candidate matrix is then a sliding window over one cached array.

The two induce the *same set* of candidate matrices -- they differ only in which
integer labels which rotation.  ``lfsr_state_for_offset`` converts a cached-mode
offset into the register state a hardware LFSR would need to be loaded with.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "LFSR_TAPS",
    "lfsr_sequence",
    "full_cycle",
    "normalize_states",
    "build_V_direct",
    "build_V_cached",
    "build_U_all",
    "lfsr_state_for_offset",
    "is_maximal_length",
]


# ---------------------------------------------------------------------------
# Appendix A.1, Table 6: indices j for which alpha_j == 1.  All others are zero.
# ---------------------------------------------------------------------------
LFSR_TAPS: dict[int, tuple[int, ...]] = {
    2: (0, 1),
    3: (0, 1),
    4: (0, 1),
    5: (0, 2),
    6: (0, 1),
    7: (0, 1),
    8: (0, 2, 3, 4),
    9: (0, 4),
    10: (0, 3),
    11: (0, 2),
    12: (0, 1, 2, 8),
    13: (0, 1, 2, 5),
    14: (0, 1, 2, 12),
    15: (0, 1),
    16: (0, 1, 3, 12),
    17: (0, 3),
    18: (0, 7),
    19: (0, 1, 2, 5),
    20: (0, 3),
    21: (0, 2),
    22: (0, 1),
    23: (0, 5),
    24: (0, 1, 2, 7),
}


def lfsr_sequence(
    K: int,
    seed: int,
    length: int,
    taps: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Algorithm 1 (Appendix A.3), transcribed literally.

    Starting from ``seed``, repeatedly:
      * XOR the bits at the tap positions to form the feedback bit ``bk``;
      * shift the state right by one and insert ``bk`` at the leftmost position.

    Note that ``results[0]`` is the state *after* one update, not the seed
    itself -- this matches both the pseudocode (the assignment on line 8 follows
    the update on line 7) and the worked example in Figure 4.

    Returns an ``int64`` array of ``length`` successive states.
    """
    if K < 2:
        raise ValueError("K must be >= 2")
    if taps is None:
        if K not in LFSR_TAPS:
            raise KeyError(f"no tap set tabulated for K={K}")
        taps = LFSR_TAPS[K]
    if not 1 <= seed <= (1 << K) - 1:
        raise ValueError(f"seed must lie in [1, 2**{K}-1], got {seed}")

    out = np.empty(length, dtype=np.int64)
    state = int(seed)
    for i in range(length):
        bk = 0
        for tap in taps:
            bk ^= (state >> tap) & 1
        state = (state >> 1) | (bk << (K - 1))
        out[i] = state
    return out


def full_cycle(K: int, taps: tuple[int, ...] | None = None) -> np.ndarray:
    """Algorithm 2 step 2: all ``2**K - 1`` states, generated from state 1.

    ``full_cycle(K)[0]`` is the successor of state 1, and the array wraps back
    around to 1 at its final entry when the polynomial is primitive.
    """
    return lfsr_sequence(K, seed=1, length=(1 << K) - 1, taps=taps)


def is_maximal_length(K: int, taps: tuple[int, ...] | None = None) -> bool:
    """True iff the tap set visits every nonzero state exactly once."""
    cyc = full_cycle(K, taps)
    return len(np.unique(cyc)) == (1 << K) - 1


def normalize_states(states: np.ndarray, K: int, dtype=np.float32) -> np.ndarray:
    """Section 3.2 normalisation, mapping raw states into [-1, 1].

        U(s) = (V(s) - 2**(K-1)) / (2**(K-1) - 1)

    Raw states span [1, 2**K - 1], and both endpoints map to exactly -1 and +1
    (the offset and the divisor differ by one, which is what makes the range
    tight rather than merely approximate).  The result is exactly zero-centred.
    """
    half = 1 << (K - 1)
    return ((states.astype(np.float64) - half) / (half - 1)).astype(dtype)


def build_V_direct(
    K: int, seed: int, C: int, P: int, taps: tuple[int, ...] | None = None
) -> np.ndarray:
    """Figure 4 semantics: run the LFSR forward from ``seed``, fill row-wise.

    Reproduces the paper's worked example exactly::

        >>> build_V_direct(K=3, seed=4, C=4, P=2)
        array([[2, 5],
               [6, 7],
               [3, 1],
               [4, 2]])
    """
    return lfsr_sequence(K, seed, C * P, taps).reshape(C, P)


def build_V_cached(cycle: np.ndarray, offset: int, C: int, P: int) -> np.ndarray:
    """Algorithm 2 steps 5-7: slice the cached cycle at ``offset % length``.

    Wraps around the end of the cycle when ``C * P`` exceeds the remaining
    states, exactly as the pseudocode's "if the slice exceeds length, cycle
    through states" instructs.
    """
    length = len(cycle)
    idx = (offset % length + np.arange(C * P)) % length
    return cycle[idx].reshape(C, P)


def lfsr_state_for_offset(cycle: np.ndarray, offset: int) -> int:
    """Convert a cached-mode offset to the LFSR register state that produces it.

    A hardware decoder loads a *state*, not an index into a table, so a seed
    found by the cached-mode search has to be translated before it can be
    written into a bitstream.  The window at ``offset`` begins with
    ``cycle[offset]``; the state whose successor is ``cycle[offset]`` is
    ``cycle[offset - 1]``, or 1 when ``offset == 0`` (since ``full_cycle``
    starts from state 1).
    """
    length = len(cycle)
    offset %= length
    return 1 if offset == 0 else int(cycle[offset - 1])


def build_U_all(K: int, C: int, P: int, dtype=np.float32) -> np.ndarray:
    """All ``2**K - 1`` candidate matrices U(s), shape ``(N, C, P)``.

    Built from a single normalised cycle via a sliding window, so cost is one
    LFSR pass plus one gather rather than N independent LFSR runs.
    """
    cycle = full_cycle(K)
    u = normalize_states(cycle, K, dtype=dtype)
    n = len(u)
    ext = np.concatenate([u, u[: C * P]])
    win = np.lib.stride_tricks.sliding_window_view(ext, C * P)[:n]
    return np.ascontiguousarray(win).reshape(n, C, P)
