"""Tensorization of shared TraceGraph scenes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import generate_scene
from .tracelets import endpoints


K_MAX = 8


class RelationDataset(Dataset):
    def __init__(self, split: str, indices: list[int]) -> None:
        self.split = split; self.indices = list(indices)

    def __len__(self) -> int: return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        scene = generate_scene(self.split, self.indices[item]); source = scene["source_endpoint"]; source_tracelet = scene["tracelets"][0]
        destination_points = np.zeros((K_MAX, 2), dtype=np.float32); destination_tangents = np.zeros((K_MAX, 2), dtype=np.float32); destination_tokens = np.zeros((K_MAX, 16), dtype=np.float32); geometry = np.zeros((K_MAX, 8), dtype=np.float32); mask = np.zeros(K_MAX, dtype=bool)
        for rank, candidate in enumerate(scene["candidates"]):
            tracelet = scene["tracelets"][candidate.endpoint.tracelet_id]; endpoint = candidate.endpoint; mask[rank] = True
            destination_points[rank] = endpoint.point_yx; destination_tangents[rank] = endpoint.outgoing_tangent_yx; destination_tokens[rank] = scene["tracelet_tokens"][tracelet.tracelet_id]
            delta = np.asarray(endpoint.point_yx) - np.asarray(source.point_yx); along = np.asarray(source.outgoing_tangent_yx); lateral = abs(float(delta[0] * along[1] - delta[1] * along[0]))
            geometry[rank] = [candidate.distance / 68.0, candidate.tangent_error / (math.pi / 2), lateral / 32.0, source_tracelet.length / 96.0, tracelet.length / 96.0, source.confidence, endpoint.confidence, rank / 7.0]
        # Valid in-bounds coordinates for padded grid_sample entries; mask removes scores.
        destination_points[~mask] = np.asarray(source.point_yx, dtype=np.float32) + np.asarray([0.0, 8.0], dtype=np.float32)
        label = int(scene["target_index"] if scene["target_index"] >= 0 else K_MAX)
        return {
            "dense": torch.from_numpy(scene["dense"]), "source_point": torch.tensor(source.point_yx, dtype=torch.float32), "source_tangent": torch.tensor(source.outgoing_tangent_yx, dtype=torch.float32),
            "source_token": torch.from_numpy(scene["tracelet_tokens"][0]), "destination_points": torch.from_numpy(destination_points), "destination_tangents": torch.from_numpy(destination_tangents), "destination_tokens": torch.from_numpy(destination_tokens), "geometry": torch.from_numpy(geometry), "candidate_mask": torch.from_numpy(mask), "label": label, "scene_type": scene["scene_type"], "index": int(scene["index"]),
        }
