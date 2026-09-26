"""SenSeed -- data-free sensitivity for on-chip weight generation.

Weights are compressed into LFSR seeds, as in SeedLM and S-Quant, but the bits
are allocated by *sensitivity* rather than by energy -- and the sensitivity is
recovered from the checkpoint alone, with no calibration data and no extra
stored bits.

Three subpackages, in the order the method uses them:

``senseed.sensitivity``
    The contribution.  ``actscale`` estimates each layer's input activation
    scale ``E[x_j^2]`` from the checkpoint (``gamma^2`` where a norm precedes
    the projection, one propagation hop where none does).  ``metrics`` scores a
    compression against it and says, in advance and for free, whether a given
    tensor is worth applying it to.  ``equalize`` holds the exact identities
    that spend it -- channel scaling folded back into ``gamma``, or into the
    producing layer's rows where there is no ``gamma``.

``senseed.codec``
    The seed generator and the three codecs built on it: ``seedlm`` and
    ``squant`` reimplemented from their papers, and ``allocate`` -- SenSeed's
    schedule, in which ``k`` follows a fixed function of the importance the
    decoder can recompute, so adaptivity costs no signalling bits.

``senseed.baselines``
    Block floating point, the data-free baseline neither paper runs.

    >>> import numpy as np
    >>> from senseed import SenSeedConfig, compress_senseed, decompress_senseed, normed_scale
    >>> W = np.random.default_rng(0).normal(0, 0.02, (16, 256)).astype(np.float32)
    >>> gamma = np.exp(np.random.default_rng(1).normal(0, 1.0, 256))
    >>> a = normed_scale(gamma)                       # data-free sensitivity
    >>> cfg = SenSeedConfig(k_mode="schedule", schedule_slope=1.0, exp_group=1)
    >>> cfg.bits_per_element(3.0)
    4.0
"""

from .codec.allocate import (
    SenSeedConfig,
    SenSeedResult,
    compress_senseed,
    decompress_senseed,
)
from .codec.bitstream import pack, packed_nbytes, unpack
from .codec.lfsr import (
    LFSR_TAPS,
    build_U_all,
    build_V_cached,
    build_V_direct,
    full_cycle,
    is_maximal_length,
    lfsr_sequence,
    lfsr_state_for_offset,
    normalize_states,
)
from .codec.quant import (
    QuantSpec,
    dequantize,
    quantize,
    quantize_dequantize,
    shared_exponent,
)
from .codec.seedlm import (
    SEEDLM_3BIT,
    SEEDLM_4BIT,
    Codebook,
    CompressedTensor,
    SeedLMConfig,
    compress,
    decompress,
    relative_error,
)
from .codec.squant import (
    SQuantConfig,
    SQuantResult,
    compress_squant,
    decompress_squant,
)

from .sensitivity.actscale import (
    ArchSpec,
    attn_out_scale,
    block_scales,
    mlp_act_scale,
    normed_scale,
    propagate,
    residual_saliency,
    silu_second_moment,
)
from .sensitivity.blockmap import (block_importance, blocks_from_scale,
                                   consumer_for, gamma_for_layer,
                                   output_saliency)
from .sensitivity.metrics import informativeness, weighted_error

from .io import load_safetensors, save_safetensors

__version__ = "0.2.0"

__all__ = [
    # SenSeed
    "SenSeedConfig", "SenSeedResult", "compress_senseed", "decompress_senseed",
    # sensitivity
    "ArchSpec", "normed_scale", "attn_out_scale", "mlp_act_scale",
    "block_scales", "propagate", "residual_saliency", "silu_second_moment",
    "weighted_error", "informativeness",
    "blocks_from_scale", "block_importance", "gamma_for_layer",
    "consumer_for", "output_saliency",
    # baselines reimplemented from their papers
    "SeedLMConfig", "SEEDLM_3BIT", "SEEDLM_4BIT", "Codebook",
    "CompressedTensor", "compress", "decompress", "relative_error",
    "SQuantConfig", "SQuantResult", "compress_squant", "decompress_squant",
    # generator internals
    "LFSR_TAPS", "lfsr_sequence", "full_cycle", "is_maximal_length",
    "normalize_states", "build_V_direct", "build_V_cached", "build_U_all",
    "lfsr_state_for_offset",
    "QuantSpec", "shared_exponent", "quantize", "dequantize",
    "quantize_dequantize",
    "pack", "unpack", "packed_nbytes",
    # io
    "load_safetensors", "save_safetensors",
]
