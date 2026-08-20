"""ANZA-derived edge conductance and deterministic bottleneck paths."""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from models.azconv_affinity import LOCAL8_OFFSETS, _shift_tensor
from synthetic.structural_metrics import skeletonize_mask
from trace_extraction.graph import extract_trace_graph
from path_completion.widest_path import EndpointPair


def image_conductance(image: torch.Tensor, *, contrast_scale: float = 0.15) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3 or float(contrast_scale) <= 0:
        raise ValueError("image conductance expects Bx3xHxW and positive scale")
    rows = []
    for dx, dy in LOCAL8_OFFSETS:
        neighbor, valid = _shift_tensor(image, dx, dy)
        difference = (image - neighbor).abs().mean(dim=1, keepdim=True)
        rows.append(torch.exp(-difference / float(contrast_scale)) * valid)
    return torch.cat(rows, dim=1)


def legacy_anza_geometry_conductance(spatial: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Evaluate the published legacy membership-times-anisotropy relation."""

    if not hasattr(spatial, "gate_conv") or not hasattr(spatial, "geometry_conv"):
        raise TypeError("legacy ANZA spatial operator required")
    logits = spatial.gate_conv(image)
    memberships = torch.softmax(logits / float(spatial.cfg.fuzzy_temperature), dim=1)
    theta, raw_base, raw_hyper = torch.chunk(spatial.geometry_conv(image), 3, dim=1)
    base = F.softplus(raw_base) + 1e-4
    hyper = F.softplus(raw_hyper).clamp_max(float(spatial.cfg.max_hyperbolicity))
    rows = []
    for dx, dy in LOCAL8_OFFSETS:
        mu_q, valid = _shift_tensor(memberships, dx, dy)
        theta_q, _ = _shift_tensor(theta, dx, dy)
        base_q, _ = _shift_tensor(base, dx, dy)
        hyper_q, _ = _shift_tensor(hyper, dx, dy)
        c2 = torch.cos(2 * theta) + torch.cos(2 * theta_q)
        s2 = torch.sin(2 * theta) + torch.sin(2 * theta_q)
        pair_theta = 0.5 * torch.atan2(s2, c2)
        sigma_u = 0.5 * (base + base_q) * torch.exp(0.5 * (hyper + hyper_q))
        sigma_s = 0.5 * (base + base_q) * torch.exp(-0.5 * (hyper + hyper_q))
        projection_u = torch.cos(pair_theta) * float(dx) + torch.sin(pair_theta) * float(dy)
        projection_s = -torch.sin(pair_theta) * float(dx) + torch.cos(pair_theta) * float(dy)
        kernel = torch.exp(-(projection_u.square() / sigma_u.square()) - (projection_s.square() / sigma_s.square()))
        rows.append(((memberships * mu_q * kernel).sum(dim=1, keepdim=True) * valid).clamp(0, 1))
    result = torch.cat(rows, dim=1)
    if not torch.isfinite(result).all() or torch.any((result < 0) | (result > 1)):
        raise ValueError("ANZA conductance must be finite in [0,1]")
    return result


def anza_guided_conductance(spatial: torch.nn.Module, image: torch.Tensor, *, contrast_scale: float = 0.15) -> torch.Tensor:
    return image_conductance(image, contrast_scale=contrast_scale) * legacy_anza_geometry_conductance(spatial, image)


def filtered_endpoint_pairs(
    mask: np.ndarray,
    *,
    d_min: float,
    d_max: float,
    min_branch_length: float,
    border_margin: int = 5,
) -> tuple[EndpointPair, ...]:
    graph = extract_trace_graph(skeletonize_mask(np.asarray(mask, dtype=bool)), border_margin=border_margin)
    eligible = set()
    endpoint_count = len(graph.endpoints)
    for endpoint_id, (point, truncated) in enumerate(zip(graph.endpoints, graph.endpoint_border_truncated)):
        if truncated:
            continue
        lengths = [
            segment.pixel_length
            for segment in graph.segments
            if segment.start_node == endpoint_id or segment.end_node == endpoint_id
        ]
        if lengths and max(lengths) >= float(min_branch_length):
            eligible.add(tuple(point))
    pairs = []
    endpoints = sorted(eligible)
    for first_index, first in enumerate(endpoints):
        for second in endpoints[first_index + 1 :]:
            distance = math.dist(first, second)
            if float(d_min) <= distance <= float(d_max):
                pairs.append(EndpointPair(first, second, distance))
    return tuple(sorted(pairs, key=lambda pair: (pair.distance, pair.first, pair.second)))


def corridor_relation(relation: np.ndarray, pair: EndpointPair, *, margin: int = 12) -> np.ndarray:
    edges = np.asarray(relation, dtype=np.float32).copy()
    height, width = edges.shape[1:]
    y0 = max(0, min(pair.first[0], pair.second[0]) - int(margin))
    y1 = min(height, max(pair.first[0], pair.second[0]) + int(margin) + 1)
    x0 = max(0, min(pair.first[1], pair.second[1]) - int(margin))
    x1 = min(width, max(pair.first[1], pair.second[1]) + int(margin) + 1)
    allowed = np.zeros((height, width), dtype=bool)
    allowed[y0:y1, x0:x1] = True
    for channel, (dx, dy) in enumerate(LOCAL8_OFFSETS):
        destination = np.zeros_like(allowed)
        sy0, sy1 = max(0, -dy), min(height, height - dy)
        sx0, sx1 = max(0, -dx), min(width, width - dx)
        destination[sy0:sy1, sx0:sx1] = allowed[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx]
        edges[channel] *= allowed & destination
    return edges


def widest_path_tiebroken(relation: np.ndarray, pair: EndpointPair) -> tuple[float, tuple[tuple[int, int], ...], dict[str, float]]:
    """Maximize bottleneck, then minimize anisotropic cost, curvature, length."""

    edges = corridor_relation(relation, pair)
    height, width = edges.shape[1:]
    # State includes incoming offset, so curvature is a real secondary cost.
    start_state = (pair.first[0], pair.first[1], -1)
    best: dict[tuple[int, int, int], tuple[float, float, float, int]] = {start_state: (1.0, 0.0, 0.0, 0)}
    parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    queue = [(-1.0, 0.0, 0.0, 0, *start_state)]
    goal_state = None
    while queue:
        negative_bottleneck, cost, curvature, length, y, x, incoming = heapq.heappop(queue)
        state = (y, x, incoming)
        record = (-negative_bottleneck, cost, curvature, length)
        if best.get(state) != record:
            continue
        if (y, x) == pair.second:
            goal_state = state
            break
        for channel, (dx, dy) in enumerate(LOCAL8_OFFSETS):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            edge = float(edges[channel, y, x])
            if edge <= 0:
                continue
            turn = 0.0
            if incoming >= 0:
                pdx, pdy = LOCAL8_OFFSETS[incoming]
                dot = (pdx * dx + pdy * dy) / (math.hypot(pdx, pdy) * math.hypot(dx, dy))
                turn = math.acos(max(-1.0, min(1.0, dot)))
            candidate = (min(record[0], edge), cost - math.log(max(edge, 1e-12)), curvature + turn, length + 1)
            next_state = (ny, nx, channel)
            previous = best.get(next_state)
            candidate_rank = (-candidate[0], candidate[1], candidate[2], candidate[3])
            previous_rank = None if previous is None else (-previous[0], previous[1], previous[2], previous[3])
            if previous_rank is None or candidate_rank < previous_rank:
                best[next_state] = candidate
                parent[next_state] = state
                heapq.heappush(queue, (*candidate_rank, *next_state))
    if goal_state is None:
        return 0.0, (), {"anisotropic_cost": float("inf"), "curvature": float("inf"), "length": 0.0}
    path = [(goal_state[0], goal_state[1])]
    state = goal_state
    while state != start_state:
        state = parent[state]
        path.append((state[0], state[1]))
    bottleneck, cost, curvature, length = best[goal_state]
    return float(bottleneck), tuple(reversed(path)), {"anisotropic_cost": float(cost), "curvature": float(curvature), "length": float(length)}

