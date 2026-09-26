"""The LFSR generator and the three codecs built on it.

``lfsr``      Algorithm 1, Table 6 taps, and both seed -> matrix conventions
              the SeedLM paper defines (they differ; see the README).
``quant``     4-bit mantissas with one shared exponent, and the exponent rule.
``bitstream`` real serialisation, so the rate claims are checkable rather than
              arithmetic.
``seedlm``    SeedLM (Shafipour et al., ICLR 2025), from the paper text.
``squant``    S-Quant (Wang et al., ICML 2026), from the paper text.
``allocate``  SenSeed -- the schedule that makes adaptive ``k`` cost no bits.
"""
