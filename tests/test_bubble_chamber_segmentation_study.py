import json
import numpy as np
import pytest

from bubble_chamber.pilot import PilotConfig, analyze_frame
from bubble_chamber.ridges import frangi_dark_ridges
from bubble_chamber.segmentation_study import (
    SegmentationProtocol, generate_scene, visible_track_metrics,
    run_segmentation_study,
)


def test_ridge_segmentation_keeps_default_contrast_unchanged():
    clean = generate_scene(1234, "clean_proxy")
    p0, mask0, _ = analyze_frame(clean["image"], PilotConfig())
    assert p0["pipeline"] == "bubble_classical_p0_v1"
    ridge, mask1, _ = analyze_frame(clean["image"], PilotConfig(segmentation_method="frangi"))
    assert ridge["pipeline"] == "bubble_p2_multiscale_frangi_v1"
    assert mask0.shape == mask1.shape
    assert ridge["automatic_join_count"] == 0
    assert ridge["source_sha256_grayscale_pixels"] == p0["source_sha256_grayscale_pixels"]


def test_new_ridge_detector_rejects_invalid_configuration_and_preserves_empty():
    with pytest.raises(ValueError, match="response_fraction"):
        frangi_dark_ridges(np.ones((20,20), dtype=np.uint8), response_fraction=-1)
    with pytest.raises(ValueError, match="image must"):
        frangi_dark_ridges(np.ones((20,20), dtype=np.float32))
    mask, response = frangi_dark_ridges(np.full((30, 30), 130, dtype=np.uint8))
    assert mask.sum() == 0
    assert response.shape == mask.shape


def test_proxy_generator_geometry_changes_by_seed_and_image_is_not_identity_label():
    a = generate_scene(1234, "clean_proxy")
    b = generate_scene(1235, "clean_proxy")
    c = generate_scene(1234, "clutter_proxy")
    assert a["case"] != b["case"]
    assert not np.array_equal(a["observed_truth"], b["observed_truth"])
    assert not np.array_equal(a["image"], c["image"])
    assert a["observed_truth"].sum() > 0 and c["observed_truth"].sum() > 0
    assert a["image"].dtype == np.uint8


def test_metrics_measure_surviving_fragments_not_raw_mask():
    mask = np.zeros((80, 80), dtype=bool)
    reference = np.zeros_like(mask)
    mask[20, 15:21] = True  # too short to count as a surviving track
    reference[20, 15:21] = True
    scores = visible_track_metrics(mask, reference, radius_px=3, min_length_px=28)
    assert scores["accepted_centerline_px"] == 0
    assert scores["recall"] == 0
    assert scores["surviving_fragments"] == 0


def test_validation_selection_and_test_partition_separation():
    p = SegmentationProtocol(val_scenes_per_domain=2, test_scenes_per_domain=2)
    result = run_segmentation_study(p)
    assert result["status"] == "SYNTHETIC_SEGMENTATION_ONLY"
    for method, entry in result["studies"].items():
        assert entry["selected_parameter"] in (p.contrast_thresholds if method == "contrast" else p.ridge_fractions)
        assert set(entry["test"]) == {"clean_proxy", "clutter_proxy"}
        for table in entry["test"].values():
            assert table["macro"]["scene_count"] == 2
            assert all(case["automatic_merges"] == 0 for case in table["by_case"])
