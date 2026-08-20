"""Exact F0--F9 oracle/learned ANZA-2 component replacement matrix."""

from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np
import torch

from models.anza2.field import ANZA2FieldOutput


REFERENCE_BASE_SCALE = 0.95
REFERENCE_HYPERBOLICITY = 0.85
ORACLE_GEOMETRY_SEMANTICS = (
    "Phase-2B branch-fixture reference constants; the v4 generator does not "
    "define per-pixel ell/h ground truth, so they are replacement controls, not supervision targets."
)

COMPONENT_MATRIX = {
    "F0_full_oracle": ("oracle", "oracle", "oracle", "oracle"),
    "F1_full_learned": ("learned", "learned", "learned", "learned"),
    "F2_learned_membership_only": ("learned", "oracle", "oracle", "oracle"),
    "F3_learned_orientation_only": ("oracle", "learned", "oracle", "oracle"),
    "F4_learned_base_scale_only": ("oracle", "oracle", "learned", "oracle"),
    "F5_learned_hyperbolicity_only": ("oracle", "oracle", "oracle", "learned"),
    "F6_learned_membership_orientation": ("learned", "learned", "oracle", "oracle"),
    "F7_learned_scale_hyperbolicity": ("oracle", "oracle", "learned", "learned"),
    "F8_learned_geometry_oracle_membership": ("oracle", "learned", "learned", "learned"),
    # The packet intentionally repeats F2 as F9. Preserve that duplicate as an
    # exactness check rather than silently inventing another configuration.
    "F9_learned_membership_oracle_geometry": ("learned", "oracle", "oracle", "oracle"),
}


def _field(
    membership: torch.Tensor,
    orientation: torch.Tensor,
    base_scale: torch.Tensor,
    hyperbolicity: torch.Tensor,
) -> ANZA2FieldOutput:
    sigma_parallel = base_scale * torch.exp(hyperbolicity)
    sigma_perpendicular = base_scale * torch.exp(-hyperbolicity)
    return ANZA2FieldOutput(
        membership, orientation, base_scale, hyperbolicity,
        sigma_parallel, sigma_perpendicular,
    )


def oracle_field_from_sample(sample: dict[str, Any], *, device: torch.device) -> tuple[ANZA2FieldOutput, torch.Tensor]:
    """Construct the generator-defined mode/axis oracle plus frozen geometry controls."""

    theta = torch.as_tensor(np.asarray(sample["gt_theta_set"]), dtype=torch.float32, device=device).unsqueeze(0)
    valid = torch.as_tensor(np.asarray(sample["gt_theta_valid"]), dtype=torch.bool, device=device).unsqueeze(0)
    modes, height, width = theta.shape[1:]
    membership = torch.where(valid, torch.full_like(theta, 0.98), torch.full_like(theta, 0.005))
    orientation = torch.stack((torch.cos(2.0 * theta), torch.sin(2.0 * theta)), dim=2)
    # Invalid target slots get deterministic axial basis vectors. Their very low
    # membership keeps them inert while making all tensors finite and testable.
    basis = torch.arange(modes, dtype=torch.float32, device=device) * torch.pi / modes
    basis_orientation = torch.stack((torch.cos(2 * basis), torch.sin(2 * basis)), dim=1)
    basis_orientation = basis_orientation.view(1, modes, 2, 1, 1).expand(1, modes, 2, height, width)
    orientation = torch.where(valid.unsqueeze(2), orientation, basis_orientation)
    base = torch.full_like(membership, REFERENCE_BASE_SCALE)
    hyper = torch.full_like(membership, REFERENCE_HYPERBOLICITY)
    return _field(membership, orientation, base, hyper), valid


def _gather_modes(tensor: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Gather a BxR[xC]xHxW tensor with a per-pixel oracle->learned mapping."""

    if tensor.ndim == 4:
        return torch.gather(tensor, 1, mapping)
    if tensor.ndim == 5:
        return torch.gather(tensor, 1, mapping.unsqueeze(2).expand(-1, -1, tensor.shape[2], -1, -1))
    raise ValueError("mode tensor must be rank 4 or 5")


def align_learned_field(
    learned: ANZA2FieldOutput,
    oracle: ANZA2FieldOutput,
    oracle_valid: torch.Tensor,
) -> tuple[ANZA2FieldOutput, torch.Tensor]:
    """Permutation-align local learned modes to generator target-axis slots."""

    modes = learned.num_modes
    if modes != oracle.num_modes or modes > 6:
        raise ValueError("oracle and learned mode counts must match and remain small")
    similarity = torch.einsum("brchw,bkchw->brkhw", learned.orientation, oracle.orientation)
    permutation_table = torch.tensor(list(permutations(range(modes))), device=similarity.device)
    scores = []
    for permutation in permutation_table:
        selected = torch.stack([similarity[:, int(permutation[k]), k] for k in range(modes)], dim=1)
        scores.append((selected * oracle_valid.to(selected.dtype)).sum(dim=1))
    best = torch.stack(scores, dim=1).argmax(dim=1)
    # permutation_table[best] is BxHxWxR; gather expects BxRxHxW.
    mapping = permutation_table[best].permute(0, 3, 1, 2)
    return _field(
        _gather_modes(learned.membership, mapping),
        _gather_modes(learned.orientation, mapping),
        _gather_modes(learned.base_scale, mapping),
        _gather_modes(learned.hyperbolicity, mapping),
    ), mapping


def component_replacements(
    oracle: ANZA2FieldOutput,
    learned_aligned: ANZA2FieldOutput,
) -> dict[str, ANZA2FieldOutput]:
    components = {
        "membership": (oracle.membership, learned_aligned.membership),
        "orientation": (oracle.orientation, learned_aligned.orientation),
        "base": (oracle.base_scale, learned_aligned.base_scale),
        "hyper": (oracle.hyperbolicity, learned_aligned.hyperbolicity),
    }
    output = {}
    for name, sources in COMPONENT_MATRIX.items():
        values = [components[key][source == "learned"] for key, source in zip(components, sources, strict=True)]
        output[name] = _field(*values)
    return output
