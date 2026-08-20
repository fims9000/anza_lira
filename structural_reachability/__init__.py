"""Frozen structural-reachability probes for ANZA geometry."""

from .geometry import (
    compute_axial_consistency,
    compute_directed_anisotropic_factor,
    compute_fuzzy_compatibility,
    compute_scale_compatibility,
    symmetrize_affinity,
)
from .metrics import evaluate_low_fpr_curve, section_paired_bootstrap

__all__ = [
    "compute_axial_consistency",
    "compute_directed_anisotropic_factor",
    "compute_fuzzy_compatibility",
    "compute_scale_compatibility",
    "symmetrize_affinity",
    "evaluate_low_fpr_curve",
    "section_paired_bootstrap",
]
