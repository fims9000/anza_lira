"""Synthetic masks only; no reference medical scans or held-out subjects."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_endpoints import extract_endpoints, build_candidate_scenes


def synthetic_mask():
    mask = np.zeros((42, 55, 45), dtype=bool)
    for index in range(5):
        mask[10:18, 10 + 6 * index, 20] = True
    return mask


def test_extraction_from_predicted_3d_mask_and_candidate_scenes():
    mask = synthetic_mask()
    endpoints, stats = extract_endpoints(mask, np.eye(4), min_component_voxels=5)
    assert stats["components"] == 5
    assert stats["raw_endpoints"] == 10
    assert len(endpoints) == 10
    assert len({e.voxel for e in endpoints}) == 10
    assert len({e.component for e in endpoints}) == 5
    for endpoint in endpoints:
        assert np.linalg.norm(endpoint.tangent_lps) == pytest.approx(1)
    scenes = build_candidate_scenes(endpoints, patient_label="train_demo")
    assert scenes
    for scene in scenes:
        assert len(scene.endpoint_ids) == 5
        assert len(set(scene.endpoint_ids)) == 5
        for candidate in scene.candidates:
            assert len(candidate.endpoint_ids) == (2 if candidate.kind == "PAIR" else 3)
            assert len({next(e.component for e in endpoints if e.endpoint_id == identifier)
                        for identifier in candidate.endpoint_ids}) == len(candidate.endpoint_ids)


def test_ct_affine_lps_and_tangent_millimeters():
    mask = synthetic_mask()
    affine = np.eye(4)
    affine[0, 0] = 1.5
    affine[1, 1] = 2
    affine[2, 2] = 3
    affine[:3, 3] = [12, 18, -4]
    endpoints, _ = extract_endpoints(mask, affine, min_component_voxels=5, trace_mm=5)
    point = endpoints[0]
    ras = affine[:3, :3] @ np.asarray(point.voxel) + affine[:3, 3]
    assert np.allclose(point.position_lps, ras * [-1, -1, 1])


def test_limits_never_silently_drop_endpoints_or_scenes():
    endpoints, _ = extract_endpoints(synthetic_mask(), np.eye(4),
                                     min_component_voxels=5)
    with pytest.raises(ValueError, match="Too many endpoints"):
        extract_endpoints(synthetic_mask(), np.eye(4),
                          min_component_voxels=5, max_endpoints=5)
    with pytest.raises(ValueError, match="Too many candidate windows"):
        build_candidate_scenes(endpoints, patient_label="test", max_scenes=1)


def test_empty_mask_and_fewer_than_five_endpoints_abstain():
    mask = np.zeros((15, 15, 15), dtype=bool)
    endpoints, stats = extract_endpoints(mask, np.eye(4))
    assert not endpoints
    assert stats["components"] == 0
    mask[3:10, 5, 5] = True
    endpoints, _ = extract_endpoints(mask, np.eye(4), min_component_voxels=5)
    assert len(endpoints) == 2
    assert build_candidate_scenes(endpoints, patient_label="small") == []
