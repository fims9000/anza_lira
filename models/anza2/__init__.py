"""ANZA-2 Hyperbolic Fuzzy Reachability components.

This namespace is deliberately separate from :mod:`models.azconv`.  The latter
is the frozen LegacyANZA implementation used by historical experiments.
"""

from .affinity import ANZA2StructuralAffinity, GenericAffinityCombiner
from .aggregation import ANZA2Aggregation, aggregate_modes
from .block import ANZA2Block
from .field import ANZA2Field, ANZA2FieldConfig, ANZA2FieldOutput

__all__ = [
    "ANZA2Aggregation",
    "ANZA2Block",
    "ANZA2Field",
    "ANZA2FieldConfig",
    "ANZA2FieldOutput",
    "ANZA2StructuralAffinity",
    "GenericAffinityCombiner",
    "aggregate_modes",
]
