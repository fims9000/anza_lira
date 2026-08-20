"""Frozen segmentation source and predicted-tracelet TG1 audit.

The relation benchmark generator supplies only the image and relation truth.
Fault probability, orientation, tracelets, endpoints, and candidates are derived
from the already-frozen ANZA-KIR R0 segmentation checkpoint.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

from anza_kir.model import build_kir_model
from anza_kir.training import load_base_state

from .candidates import generate_candidates
from .protocol import PROTOCOL
from .tracelets import endpoints, extract_tracelets, tracelet_token


ROOT = Path(__file__).resolve().parents[1]
BASE_CHECKPOINT = ROOT.parent / "_wip_backups/anza_lira/anza_kir_checkpoints/IR1-base-d7695ee995a7ec56.pt"
DENSE_CHECKPOINT = ROOT.parent / "_wip_backups/anza_lira/anza_kir_checkpoints/R0_static_residual-d7695ee995a7ec56.pt"
FEATURE_NORM = ROOT / "results/anza_kir/ir2/freeze/feature_norm.json"
DENSE_THRESHOLD = 0.35
MATCH_TOLERANCE_PX = 6.0
FORCED_GAP_X = (35, 50)


def load_frozen_source(device: str) -> torch.nn.Module:
    norm = json.loads(FEATURE_NORM.read_text())["methods"]
    model = build_kir_model("R0_static_residual", load_base_state(BASE_CHECKPOINT, device), norm).to(device)
    payload = torch.load(DENSE_CHECKPOINT, map_location=device, weights_only=False)
    if payload.get("variant") != "R0_static_residual":
        raise ValueError("TraceGraph dense checkpoint variant drift")
    model.load_state_dict(payload["model"])
    return model.eval()


def infer_dense(model: torch.nn.Module, images: np.ndarray, *, device: str) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        output = model(torch.from_numpy(images).to(device), return_aux=True)
        probability = torch.sigmoid(output["visible_logits"])
        orientation = F.interpolate(output["orientation_logits"], size=images.shape[-2:], mode="bilinear", align_corners=False)
        orientation = torch.softmax(orientation, dim=1)
    return probability[:, 0].cpu().numpy().astype(np.float32), orientation.cpu().numpy().astype(np.float32)


def predicted_relation_scene(raw: dict[str, Any], probability: np.ndarray, orientation_bank: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(probability) >= DENSE_THRESHOLD
    # Relation isolation uses predicted support on both sides while enforcing the
    # benchmark's known unobserved corridor; no latent gap pixels become input.
    mask[:, FORCED_GAP_X[0] : FORCED_GAP_X[1]] = False
    tracelets = extract_tracelets(mask, probability, raw["dense"][0], min_length=int(PROTOCOL["tracelets"]["min_length"]))
    endpoint_rows = [endpoint for tracelet in tracelets for endpoint in endpoints(tracelet, int(PROTOCOL["tracelets"]["tangent_points"]))]
    source_truth = np.asarray(raw["source_endpoint"].point_yx)
    source_options = [endpoint for endpoint in endpoint_rows if endpoint.point_yx[1] < FORCED_GAP_X[1] - 8]
    if not source_options:
        return {"source_available": False, "candidate_recalled": False, "candidate_count": 0, "target_index": -1, "raw": raw}
    source = min(source_options, key=lambda endpoint: float(np.linalg.norm(np.asarray(endpoint.point_yx) - source_truth)))
    destinations = [endpoint for endpoint in endpoint_rows if endpoint.tracelet_id != source.tracelet_id and endpoint.point_yx[1] > FORCED_GAP_X[0] + 10]
    settings = PROTOCOL["candidates"]
    candidates = generate_candidates(
        source,
        destinations,
        k_max=int(settings["k_max"]),
        min_distance=float(settings["min_distance"]),
        max_distance=float(settings["max_distance"]),
        max_tangent_error=math.radians(float(settings["max_tangent_mismatch_degrees"])),
    )
    target_index = -1
    target_distance = None
    if raw["has_valid_continuation"]:
        true_endpoint = np.asarray(endpoints(raw["tracelets"][1], int(PROTOCOL["tracelets"]["tangent_points"]))[0].point_yx)
        distances = [float(np.linalg.norm(np.asarray(candidate.endpoint.point_yx) - true_endpoint)) for candidate in candidates]
        if distances:
            target_distance = min(distances)
            if target_distance <= MATCH_TOLERANCE_PX:
                target_index = int(np.argmin(distances))
    angles = np.arange(orientation_bank.shape[0], dtype=np.float32) * (math.pi / orientation_bank.shape[0])
    c2 = np.tensordot(np.cos(2 * angles), orientation_bank, axes=(0, 0))
    s2 = np.tensordot(np.sin(2 * angles), orientation_bank, axes=(0, 0))
    confidence = orientation_bank.max(axis=0)
    centerline = gaussian_filter(probability, 0.7)
    dense = np.concatenate((raw["dense"][:3], probability[None], centerline[None], c2[None], s2[None], confidence[None]), axis=0).astype(np.float32)
    tokens = np.stack([tracelet_token(tracelet, probability, raw["dense"][0], probability.shape) for tracelet in tracelets])
    return {
        "source_available": True,
        "dense": dense,
        "tracelets": tracelets,
        "tracelet_tokens": tokens,
        "source_endpoint": source,
        "candidates": candidates,
        "target_index": target_index,
        "target_match_distance": target_distance,
        "has_valid_continuation": bool(raw["has_valid_continuation"]),
        "candidate_recalled": bool(not raw["has_valid_continuation"] or target_index >= 0),
        "candidate_count": len(candidates),
        "scene_type": raw["scene_type"],
        "split": raw["split"],
        "index": raw["index"],
    }


def iter_predicted_scenes(raw_scenes: Iterable[dict[str, Any]], *, device: str, batch_size: int = 64) -> Iterable[dict[str, Any]]:
    model = load_frozen_source(device)
    local: list[dict[str, Any]] = []
    for raw in raw_scenes:
        local.append(raw)
        if len(local) == batch_size:
            probability, orientation = infer_dense(model, np.stack([item["dense"][:3] for item in local]), device=device)
            yield from (predicted_relation_scene(item, p, o) for item, p, o in zip(local, probability, orientation))
            local = []
    if local:
        probability, orientation = infer_dense(model, np.stack([item["dense"][:3] for item in local]), device=device)
        yield from (predicted_relation_scene(item, p, o) for item, p, o in zip(local, probability, orientation))

