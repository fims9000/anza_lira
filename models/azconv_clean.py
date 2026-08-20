"""Clean ANZA candidate with independent bounded fuzzy memberships.

Legacy :mod:`models.azconv` remains unchanged.  This class gives the independent
fuzzy formulation a stable method name for the Connectivity/Diffusion study.
"""

from __future__ import annotations

from .azconv_affinity import IndependentFuzzyAZConv2d


class CleanANZA2d(IndependentFuzzyAZConv2d):
    """ANZA pair geometry with independent sigmoid membership degrees."""

