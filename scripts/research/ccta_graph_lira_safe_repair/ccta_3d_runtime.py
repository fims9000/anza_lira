"""Experimental full-volume Graph-LIRA runtime for a 3D CCTA + broken mask.

The frozen CT28 HGB/CT scorer is the default inference engine, not a substitute
heuristic. The source mask is a *predicted* segmentation, not ground truth.
This pipeline is a research prototype, not clinically validated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol
from collections import Counter
import json

import numpy as np
from scipy import ndimage

import ct28_path_stage as path_stage
from ct28_mask_reconstruction import RepairProposal, apply_approved_paths
from ccta_3d_endpoints import (
    CONNECTIVITY, Candidate, Endpoint, Scene,
    extract_endpoints, build_candidate_scenes,
)
from ccta_3d_k10 import ScenePrediction


HELD_OUT_CT28_IDS = {"954", "958", "972", "973", "980", "984"}


@dataclass(frozen=True)
class RuntimeConfig:
    radius_mm: float
    max_new_voxels: int
    min_component_voxels: int = 12
    trace_mm: float = 3.0
    max_endpoints: int = 256
    max_scenes: int = 128

    def __post_init__(self):
        if not np.isfinite(self.radius_mm) or self.radius_mm < 0:
            raise ValueError("radius_mm must be finite and nonnegative")
        if not isinstance(self.max_new_voxels, int) or self.max_new_voxels <= 0:
            raise ValueError("max_new_voxels must be positive integer")
        if self.min_component_voxels < 2 or self.trace_mm <= 0:
            raise ValueError("Invalid endpoint extraction protocol")
        if self.max_endpoints < 2 or self.max_scenes < 1:
            raise ValueError("Invalid resource caps")


class Predictor(Protocol):
    def predict(
        self, scene: Scene, endpoints: list[Endpoint],
        ct: np.ndarray, affine: np.ndarray,
    ) -> ScenePrediction: ...


def frozen_path_protocol(path: Path) -> tuple[path_stage.PathConfig, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data.get("config", {})
    if data.get("protocol") != "ct28_path_protocol_v2":
        raise ValueError("Path protocol is not the selected CT28 v2")
    if data.get("bottleneck_slack_fraction") != 0.05:
        raise ValueError("Frozen path v2 requires 5% bottleneck slack")
    if data.get("avoid_unselected_scene_endpoints") is not True:
        raise ValueError("Frozen path must avoid unselected endpoints")
    known = {
        "corridor_radius_mm": 2.0,
        "center_refine_mm": 2.0,
        "hu_difference_scale": 200.0,
        "vessel_scale_floor": 80.0,
        "background_scale_floor": 40.0,
        "extra_endpoint_clearance_mm": 0.75,
    }
    if config != known:
        raise ValueError("Frozen path v2 parameters changed")
    return path_stage.PathConfig(**known), 0.05


def validate_ct_inputs(ct: np.ndarray, mask: np.ndarray, affine: np.ndarray) -> None:
    if ct.ndim != 3 or ct.shape != mask.shape:
        raise ValueError("CCTA and mask must share the same 3D voxel grid")
    if mask.dtype != bool:
        raise ValueError("Expected boolean predicted mask")
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Expected finite 4x4 affine")
    if np.min(np.linalg.svd(affine[:3, :3], compute_uv=False)) <= 0:
        raise ValueError("Singular CT affine")
    if not np.isfinite(ct).all():
        raise ValueError("CCTA contains non-finite values")
    if not np.allclose(affine[3], [0, 0, 0, 1]):
        raise ValueError("Invalid affine bottom row")


def _candidate_for_id(scene: Scene, candidate_id: str | None) -> Candidate | None:
    if candidate_id is None:
        return None
    matches = [x for x in scene.candidates if x.candidate_id == candidate_id]
    if len(matches) != 1:
        raise ValueError("Selected candidate missing or duplicated in scene")
    return matches[0]


def _paths_for_decision(
    prediction: ScenePrediction,
    scene: Scene,
    endpoints: list[Endpoint],
    ct: np.ndarray,
    affine: np.ndarray,
    path_config: path_stage.PathConfig,
    slack_fraction: float,
    solver: Callable | None,
) -> tuple[list[RepairProposal], tuple[str, ...]]:
    lookup = {e.endpoint_id: e for e in endpoints}
    chosen = [
        _candidate_for_id(scene, prediction.pair_candidate_id),
        _candidate_for_id(scene, prediction.junction_candidate_id),
    ]
    candidates = [candidate for candidate in chosen if candidate is not None]
    if len(candidates) not in (1, 2):
        raise ValueError("Accepted prediction has no valid selected candidates")
    actual = ("BOTH" if len(candidates) == 2 else candidates[0].kind)
    if actual != prediction.selected_relation:
        raise ValueError("Accepted prediction relation does not match candidates")
    chosen_ids = tuple(
        identifier for candidate in candidates for identifier in candidate.endpoint_ids
    )
    if len(chosen_ids) != len(set(chosen_ids)):
        raise ValueError("Selected relation reuses an endpoint")
    selected_components = [lookup[i].component for i in chosen_ids]
    if len(selected_components) != len(set(selected_components)):
        raise ValueError("Selected relation reuses one original component")
    other_points = [
        e.position_lps for e in endpoints if e.endpoint_id not in chosen_ids
    ]
    forbidden = np.asarray(other_points, dtype=float).reshape(-1, 3)

    result = []
    for candidate in candidates:
        positions = np.asarray([lookup[i].position_lps for i in candidate.endpoint_ids])
        tangents = np.asarray([lookup[i].tangent_lps for i in candidate.endpoint_ids])
        if solver is None:
            route = path_stage.solve_path_with_bottleneck_slack(
                ct, affine, positions, tangents, path_config,
                slack_fraction, forbidden_lps=forbidden,
            )
        else:
            route = solver(ct, affine, positions, tangents, path_config,
                           slack_fraction, forbidden)
        if route.status != "ok":
            raise ValueError("CT path solver: " + route.status)
        # Uses exactly the same 26-neighbor, endpoint and junction contract
        # checked by the train-selected CT28 path evaluation.
        _validate_route(route, positions, affine)
        result.append(RepairProposal(candidate.kind, route.paths_voxel))
    return result, chosen_ids


def _validate_route(
    route: path_stage.PathResult, positions: np.ndarray, affine: np.ndarray
) -> None:
    snapped = np.rint(path_stage.lps_to_voxel(positions, np.linalg.inv(affine))).astype(int)
    expected = 1 if len(positions) == 2 else 3
    if len(route.paths_voxel) != expected:
        raise ValueError("Path count does not match selected degree")
    for start, path in zip(snapped, route.paths_voxel):
        coordinates = np.asarray(path, dtype=int)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) < 2:
            raise ValueError("Path does not contain a valid voxel polyline")
        if not np.array_equal(coordinates[0], start):
            raise ValueError("Route does not start at selected endpoint")
        if len(set(path)) != len(path):
            raise ValueError("Route visits a voxel more than once")
        if not np.all(np.max(np.abs(np.diff(coordinates, axis=0)), axis=1) == 1):
            raise ValueError("Route violates 26-connectivity")
    if len(positions) == 2:
        if not np.array_equal(route.paths_voxel[0][-1], snapped[1]):
            raise ValueError("PAIR route misses second endpoint")
    elif len({arm[-1] for arm in route.paths_voxel}) != 1:
        raise ValueError("JUNCTION routes do not share one common node")


def run_volume(
    ct: np.ndarray,
    predicted_mask: np.ndarray,
    affine: np.ndarray,
    *,
    patient_label: str,
    engine: Predictor | None,
    params: RuntimeConfig,
    path_config: path_stage.PathConfig,
    slack_fraction: float = 0.05,
    solver: Callable | None = None,
) -> tuple[np.ndarray, dict]:
    """Run candidate inspection (engine=None) or frozen model inference.

    Complete inference may still abstain everywhere when real-mask candidates
    differ from controlled CT28 scenes. Never infer a clinically low false
    repair risk from successful algorithm execution.
    """
    if str(patient_label).strip() in HELD_OUT_CT28_IDS:
        raise ValueError("Held-out CT28 patients remain blocked before protocol freeze")
    if not patient_label or any(c in patient_label for c in "/\\:"):
        raise ValueError("Patient label must be a nonempty safe identifier")
    validate_ct_inputs(ct, predicted_mask, affine)
    endpoints, extraction = extract_endpoints(
        predicted_mask, affine,
        min_component_voxels=params.min_component_voxels,
        trace_mm=params.trace_mm, max_endpoints=params.max_endpoints,
    )
    scenes = build_candidate_scenes(
        endpoints, patient_label=patient_label, max_scenes=params.max_scenes,
    )
    report = {
        "protocol": "experimental_ccta_3d_runtime_v1",
        "status": "candidate_inspection_only" if engine is None else "experimental_inference",
        "patient_label": patient_label,
        "held_out_graph_lira_test_accessed": False,
        "input_shape": list(ct.shape),
        "input_nonzero_voxels": int(predicted_mask.sum()),
        "extraction": extraction,
        "endpoint_count": len(endpoints),
        "scene_count": len(scenes),
        "candidate_count": sum(len(scene.candidates) for scene in scenes),
        "endpoints": [
            {
                "endpoint_id": endpoint.endpoint_id,
                "original_component_id": endpoint.component,
                "voxel": list(endpoint.voxel),
                "position_lps_mm": list(endpoint.position_lps),
                "outward_tangent_lps": list(endpoint.tangent_lps),
            }
            for endpoint in endpoints
        ],
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "endpoint_ids": list(scene.endpoint_ids),
                "pair_candidates": sum(c.kind == "PAIR" for c in scene.candidates),
                "junction_candidates": sum(c.kind == "JUNCTION" for c in scene.candidates),
                "candidates": [
                    {"candidate_id": c.candidate_id, "type": c.kind,
                     "endpoint_ids": list(c.endpoint_ids)}
                    for c in scene.candidates
                ],
            }
            for scene in scenes
        ],
        "model_pack_sha256": getattr(engine, "pack_sha", None),
        "radius_mm": params.radius_mm,
        "max_new_voxels_per_decision": params.max_new_voxels,
        "path_slack_fraction": slack_fraction,
        "decisions": [],
        "warnings": [
            "The K10/15 relation head was trained on controlled five-endpoint CT28 scenes; full-mask windows may be out of distribution.",
            "No held-out or patient-independent clinical safety evidence is established.",
            "3D ANZA encoder is not part of the frozen HGB model.",
        ],
    }
    result = predicted_mask.copy()
    if engine is None or not scenes:
        report["output_nonzero_voxels"] = int(result.sum())
        report["accepted_decisions"] = 0
        report["added_voxels"] = 0
        report["abstention_reasons"] = (
            {"insufficient_5_endpoint_scenes": 1} if not scenes else {}
        )
        return result, report

    scored = []
    for scene in scenes:
        try:
            prediction = engine.predict(scene, endpoints, ct, affine)
            if prediction.scene_id != scene.scene_id:
                raise ValueError("Model prediction returned an unrelated scene ID")
            if prediction.accepted and (not np.isfinite(prediction.confidence) or
                                        not 0 <= prediction.consistency <= 1):
                raise ValueError("Invalid accepted model confidence/consistency")
            scored.append((scene, prediction))
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            report["decisions"].append({
                "scene_id": scene.scene_id, "accepted": False,
                "reason": "feature_or_model_error", "detail": str(exc)[:250],
            })
    scored.sort(key=lambda x: (
        -float(x[1].confidence * x[1].consistency),
        -x[1].identity_log_utility,
        x[0].scene_id,
    ))
    used_components: set[int] = set()
    used_endpoint_ids: set[str] = set()
    accepted = 0
    added_voxels = 0
    for scene, prediction in scored:
        item = {
            "scene_id": scene.scene_id,
            "raw_relation": prediction.raw_relation,
            "proposed_relation": prediction.proposed_relation,
            "selected_relation": prediction.selected_relation,
            "confidence": prediction.confidence,
            "consistency": prediction.consistency,
            "accepted_by_K10": prediction.accepted,
            "candidate_pair": prediction.pair_candidate_id,
            "candidate_junction": prediction.junction_candidate_id,
            "accepted": False,
        }
        if not prediction.accepted:
            item["reason"] = "K10_abstain"
            report["decisions"].append(item)
            continue
        try:
            selection = [
                _candidate_for_id(scene, prediction.pair_candidate_id),
                _candidate_for_id(scene, prediction.junction_candidate_id),
            ]
            selected_ids = [
                identifier for c in selection if c is not None for identifier in c.endpoint_ids
            ]
            endpoint_lookup = {e.endpoint_id: e for e in endpoints}
            selected_components = {endpoint_lookup[i].component for i in selected_ids}
            if len(selected_components) != len(selected_ids):
                raise ValueError("selected_endpoint_component_conflict")
            if used_endpoint_ids & set(selected_ids):
                raise ValueError("globally_reused_endpoint")
            if not selected_components:
                raise ValueError("no_selected_components")
            proposals, chosen_ids = _paths_for_decision(
                prediction, scene, endpoints, ct, affine,
                path_config, slack_fraction, solver,
            )
            reconstructed = apply_approved_paths(
                result, affine, proposals,
                radius_mm=params.radius_mm,
                max_new_voxels=params.max_new_voxels,
            )
            if not reconstructed.accepted:
                raise ValueError("mask_gate:" + reconstructed.reason)
            result = reconstructed.mask
            used_components.update(selected_components)
            used_endpoint_ids.update(selected_ids)
            accepted += 1
            added_voxels += reconstructed.added_voxels
            item["accepted"] = True
            item["reason"] = "ok"
            item["added_voxels"] = reconstructed.added_voxels
            item["connected_original_components"] = sorted(selected_components)
            item["selected_endpoints"] = list(chosen_ids)
            item["components_before"] = reconstructed.original_components
            item["components_after"] = reconstructed.final_components
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            item["reason"] = str(exc)[:250]
        report["decisions"].append(item)
    report["accepted_decisions"] = accepted
    report["added_voxels"] = added_voxels
    report["output_nonzero_voxels"] = int(result.sum())
    report["repaired_original_component_count"] = len(used_components)
    report["output_components_26"] = int(ndimage.label(
        result, structure=CONNECTIVITY
    )[1])
    report["abstention_reasons"] = dict(Counter(
        item["reason"] for item in report["decisions"] if not item["accepted"]
    ))
    return result, report
