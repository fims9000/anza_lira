from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from trace_extraction.export import traces_to_geojson, write_geojson
from trace_extraction.geometry import (
    axial_distance,
    combine_axial_geometry,
    edge_geometry_confidence,
    local_pca_orientation,
)
from trace_extraction.graph import extract_trace_graph
from trace_extraction.metrics import compute_trace_metrics


def test_orientation_is_pi_periodic() -> None:
    theta = np.array([0.17, 0.73, 1.4])
    assert np.allclose(axial_distance(theta, theta + math.pi), 0.0, atol=1e-7)


def test_membership_weighted_axial_geometry_is_finite_and_coherent() -> None:
    memberships = np.array([[[0.75]], [[0.25]]], dtype=np.float64)
    theta = np.array([[[0.0]], [[math.pi]]], dtype=np.float64)
    sigma_u = np.full_like(theta, 2.0)
    sigma_s = np.full_like(theta, 0.5)
    geometry = combine_axial_geometry(memberships, theta, sigma_u, sigma_s)

    assert abs(float(geometry.orientation[0, 0])) < 1e-7
    assert np.allclose(geometry.coherence, 1.0)
    assert 0.0 < float(geometry.anisotropy[0, 0]) <= 1.0


def test_local_pca_finds_horizontal_axis() -> None:
    skeleton = np.zeros((21, 21), dtype=bool)
    skeleton[10, 3:18] = True
    orientation = local_pca_orientation(skeleton, radius=5)
    assert math.degrees(float(axial_distance(orientation[10, 10], 0.0))) < 5.0


def test_local_pca_finds_vertical_axis() -> None:
    skeleton = np.zeros((21, 21), dtype=bool)
    skeleton[3:18, 10] = True
    orientation = local_pca_orientation(skeleton, radius=5)
    assert math.degrees(float(axial_distance(orientation[10, 10], math.pi / 2))) < 5.0


def test_geometry_and_edge_confidence_are_finite_and_bounded() -> None:
    rng = np.random.default_rng(42)
    shape = (3, 8, 9)
    geometry = combine_axial_geometry(
        rng.random(shape),
        rng.uniform(-math.pi, math.pi, size=shape),
        rng.uniform(0.1, 3.0, size=shape),
        rng.uniform(0.1, 3.0, size=shape),
    )
    probability = rng.random((8, 9))
    score = edge_geometry_confidence(
        (3, 3), (3, 4), probability, geometry.orientation, geometry.coherence, geometry.anisotropy
    )
    for array in (probability, geometry.coherence, geometry.anisotropy):
        assert np.isfinite(array).all()
        assert np.all((0 <= array) & (array <= 1))
    assert 0 <= score <= 1
    assert np.array_equal(geometry.rho, geometry.coherence)


def test_geometry_rejects_nan_and_inf() -> None:
    maps = np.ones((3, 3), dtype=np.float64)
    broken = maps.copy()
    broken[1, 1] = np.nan
    with np.testing.assert_raises_regex(ValueError, "finite"):
        edge_geometry_confidence((1, 1), (1, 2), maps, broken, maps, maps)
    memberships = np.ones((2, 1, 1), dtype=np.float64)
    theta = memberships.copy()
    theta[0, 0, 0] = np.inf
    with np.testing.assert_raises_regex(ValueError, "finite"):
        combine_axial_geometry(memberships, theta, memberships, memberships)


def test_trace_metrics_are_exact_for_identical_lines() -> None:
    skeleton = np.zeros((21, 21), dtype=bool)
    skeleton[10, 3:18] = True
    orientation = local_pca_orientation(skeleton, radius=5)
    metrics = compute_trace_metrics(skeleton, skeleton, pred_orientation=orientation, tolerance=2.0)

    assert metrics["trace_precision"] == 1.0
    assert metrics["trace_recall"] == 1.0
    assert metrics["trace_f1"] == 1.0
    assert metrics["endpoint_f1"] == 1.0
    assert metrics["symmetric_skeleton_distance"] == 0.0
    assert metrics["orientation_error_mean_deg"] < 1e-6
    assert metrics["trace_length_error"] == 0.0


def test_geojson_round_trip_contains_linestring_and_provenance(tmp_path: Path) -> None:
    skeleton = np.zeros((9, 9), dtype=bool)
    skeleton[4, 2:7] = True
    graph = extract_trace_graph(skeleton)
    payload = traces_to_geojson(
        graph.segments,
        source_image_id="DJI_0001",
        patch_id="DJI_0001_patch1",
        model="az_thesis",
        seed=42,
    )
    output = tmp_path / "traces.geojson"
    write_geojson(output, payload)
    parsed = json.loads(output.read_text(encoding="utf-8"))

    assert parsed["type"] == "FeatureCollection"
    assert len(parsed["features"]) == 1
    feature = parsed["features"][0]
    assert feature["geometry"]["type"] == "LineString"
    assert feature["properties"]["source_image_id"] == "DJI_0001"
    assert feature["properties"]["model"] == "az_thesis"
    assert feature["properties"]["seed"] == 42
    assert feature["properties"]["border_truncated"] is True


def test_border_nodes_are_counted_but_excluded_from_primary_f1() -> None:
    target = np.zeros((41, 41), dtype=bool)
    target[20, 0:31] = True
    predicted = target.copy()
    predicted[20, 0:3] = False
    metrics = compute_trace_metrics(predicted, target, border_margin=5)
    assert metrics["target_endpoint_total"] == 2
    assert metrics["target_endpoint_evaluable"] == 1
    assert metrics["target_endpoint_border_truncated"] == 1
    assert metrics["endpoint_f1"] == 1.0


def test_empty_trace_metric_convention_is_finite_and_perfect() -> None:
    empty = np.zeros((17, 17), dtype=bool)
    metrics = compute_trace_metrics(empty, empty)
    assert metrics["trace_precision"] == 1.0
    assert metrics["trace_recall"] == 1.0
    assert metrics["trace_f1"] == 1.0
    assert metrics["endpoint_f1"] == 1.0
    assert metrics["junction_f1"] == 1.0
    assert metrics["orientation_error_mean_deg"] == 0.0
    assert all(np.isfinite(float(value)) for value in metrics.values())
