"""Differentiable dense implementation of the frozen K1 feature families."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from anza_ks.benchmark.static_signature import measurement_rows
from anza_ks.constants import FEATURE_WIDTH, ORIENTATION_COUNT
from anza_ks.features import itineraries_for
from anza_ks.koopman_probes import fixed_probes
from anza_ks.orientation_bank import unstable_angle
from anza_ks.partitions import PARTITION_COUNT
from anza_ks.torus import CAT_MAP, SHEAR_MAP, permutation_indices


PATCH_SIZE = 17
METHODS = ("static", "shear_ks", "cat_raw", "cat_ks")


@lru_cache(maxsize=8)
def _numpy_indices(dynamics: str) -> dict[str, object]:
    matrix = CAT_MAP if dynamics == "cat" else SHEAR_MAP
    itinerary = itineraries_for(dynamics, PATCH_SIZE)
    permutation = {}
    for lag in range(-4, 5):
        y, x = permutation_indices(PATCH_SIZE, matrix, power=lag)
        permutation[lag] = (y * PATCH_SIZE + x).ravel()
    return {
        "permutation": permutation,
        "partition": itinerary.symbols_by_lag[0].ravel(),
        "forward": {length: value.ravel() for length, value in itinerary.forward_word_ids.items()},
        "backward": {length: value.ravel() for length, value in itinerary.backward_word_ids.items()},
        "past": itinerary.predictive_past_ids.ravel(),
        "future": itinerary.predictive_future_ids.ravel(),
        "joint": itinerary.predictive_joint_ids.ravel(),
    }


def _index(value: np.ndarray, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.long, device=reference.device)


def _symbolic_probability(density: torch.Tensor, ids: np.ndarray, count: int) -> torch.Tensor:
    result = density.new_zeros(*density.shape[:-1], count)
    index = _index(ids, density).view(*([1] * (density.ndim - 1)), -1).expand_as(density)
    result.scatter_add_(-1, index, density)
    return result / result.sum(dim=-1, keepdim=True).clamp_min(1e-30)


def _entropy(probability: torch.Tensor) -> torch.Tensor:
    value = -(probability * torch.log(probability + 1e-12)).sum(dim=-1)
    return value.clamp_min(0.0)


def _density(flat: torch.Tensor) -> torch.Tensor:
    median = flat.median(dim=-1, keepdim=True).values
    std = flat.std(dim=-1, correction=0, keepdim=True)
    scaled = ((flat - median) / (std + 1e-9)) / 0.5
    positive = F.softplus(scaled) + 1e-9
    return positive / positive.sum(dim=-1, keepdim=True)


def _raw_dynamic(flat: torch.Tensor, dynamics: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
    data = _numpy_indices(dynamics)
    density = _density(flat)
    masses = []
    coarse_entropy = []
    for lag in range(-4, 5):
        # Frozen NumPy uses exact_permutation(density, power=-lag).
        transported = flat.new_empty(density.shape)
        transported.copy_(density.index_select(-1, _index(data["permutation"][-lag], flat)))
        probability = _symbolic_probability(transported, data["partition"], PARTITION_COUNT)
        masses.append(probability)
        coarse_entropy.append(_entropy(probability))

    centered = flat - flat.mean(dim=-1, keepdim=True)
    normalized = centered / (torch.linalg.vector_norm(centered, dim=-1, keepdim=True) + 1e-12)
    probes = torch.as_tensor(fixed_probes(PATCH_SIZE).reshape(4, -1), dtype=flat.dtype, device=flat.device)
    correlations = []
    for lag in range(-4, 5):
        transported = normalized.index_select(-1, _index(data["permutation"][lag], flat))
        correlations.append(torch.einsum("...n,pn->...p", transported, probes))
    return torch.cat((*masses, *correlations), dim=-1), coarse_entropy


def _information(flat: torch.Tensor, dynamics: str, coarse_entropy: list[torch.Tensor]) -> torch.Tensor:
    data = _numpy_indices(dynamics)
    density = _density(flat)
    forward = []
    backward = []
    for length in range(1, 5):
        count = PARTITION_COUNT**length
        forward.append(_entropy(_symbolic_probability(density, data["forward"][length], count)))
        backward.append(_entropy(_symbolic_probability(density, data["backward"][length], count)))
    forward_tensor = torch.stack(forward, dim=-1)
    backward_tensor = torch.stack(backward, dim=-1)
    forward_conditional = torch.diff(forward_tensor, dim=-1)
    backward_conditional = torch.diff(backward_tensor, dim=-1)
    past = _entropy(_symbolic_probability(density, data["past"], 16))
    future = _entropy(_symbolic_probability(density, data["future"], 16))
    joint = _entropy(_symbolic_probability(density, data["joint"], 256))
    predictive = (past + future - joint).clamp_min(0.0)
    asymmetry_h = torch.abs(forward_tensor - backward_tensor).mean(dim=-1)
    asymmetry_conditional = torch.abs(forward_conditional - backward_conditional).mean(dim=-1)
    entropy_production = torch.diff(torch.stack(coarse_entropy, dim=-1), dim=-1)
    return torch.cat(
        (
            forward_tensor,
            backward_tensor,
            forward_conditional,
            backward_conditional,
            predictive[..., None],
            asymmetry_h[..., None],
            asymmetry_conditional[..., None],
            entropy_production,
        ),
        dim=-1,
    )


def _static(flat: torch.Tensor) -> torch.Tensor:
    rows = torch.as_tensor(measurement_rows(PATCH_SIZE), dtype=flat.dtype, device=flat.device)
    raw = torch.einsum("...n,rn->...r", flat, rows)
    mean = flat.mean(dim=-1, keepdim=True)
    variance = flat.var(dim=-1, correction=0, keepdim=True)
    energy = (flat * flat).sum(dim=-1, keepdim=True)
    return torch.cat((raw, mean, variance, energy), dim=-1)


def features_from_patches(patches: torch.Tensor, method: str) -> torch.Tensor:
    """Map ``(...,17,17)`` patches to the frozen 104-dimensional signature."""

    if method not in METHODS:
        raise ValueError(f"unknown dense feature method: {method}")
    if patches.shape[-2:] != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError("K2 feature patches must be 17x17")
    flat = patches.reshape(*patches.shape[:-2], PATCH_SIZE * PATCH_SIZE)
    if method == "static":
        values = _static(flat)
    else:
        dynamics = "shear" if method == "shear_ks" else "cat"
        raw, coarse_entropy = _raw_dynamic(flat, dynamics)
        values = raw if method == "cat_raw" else torch.cat((raw, _information(flat, dynamics, coarse_entropy)), dim=-1)
    if values.shape[-1] > FEATURE_WIDTH:
        raise ValueError("dense signature exceeds frozen width")
    return F.pad(values, (0, FEATURE_WIDTH - values.shape[-1]))


def extract_reflect_patches(structural_map: torch.Tensor) -> torch.Tensor:
    if structural_map.ndim != 4 or structural_map.shape[1] != 1:
        raise ValueError("structural_map must be Bx1xHxW")
    padded = F.pad(structural_map, (8, 8, 8, 8), mode="reflect")
    unfolded = F.unfold(padded, kernel_size=PATCH_SIZE)
    batch, _, locations = unfolded.shape
    return unfolded.transpose(1, 2).reshape(batch, locations, PATCH_SIZE, PATCH_SIZE)


def _rotate(patches: torch.Tensor, angle: float) -> torch.Tensor:
    if abs(angle) < 1e-14:
        return patches
    leading = patches.shape[:-2]
    values = patches.reshape(-1, 1, PATCH_SIZE, PATCH_SIZE)
    cosine, sine = math.cos(angle), math.sin(angle)
    theta = values.new_tensor([[cosine, -sine, 0.0], [sine, cosine, 0.0]])[None].expand(len(values), -1, -1)
    grid = F.affine_grid(theta, values.shape, align_corners=True)
    rotated = F.grid_sample(values, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
    return rotated.reshape(*leading, PATCH_SIZE, PATCH_SIZE)


def dense_orientation_features(structural_map: torch.Tensor, method: str) -> torch.Tensor:
    """Return BxHxWxMx104 fixed features for one learned scalar map."""

    batch, _, height, width = structural_map.shape
    patches = extract_reflect_patches(structural_map)
    branch = []
    base = unstable_angle()
    for mode in range(ORIENTATION_COUNT):
        angle = base - mode * math.pi / ORIENTATION_COUNT
        branch.append(_rotate(patches, angle))
    values = features_from_patches(torch.stack(branch, dim=-3), method)
    return values.reshape(batch, height, width, ORIENTATION_COUNT, FEATURE_WIDTH)
