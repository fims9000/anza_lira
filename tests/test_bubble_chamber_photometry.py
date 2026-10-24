from pathlib import Path
import cv2
import numpy as np
import pytest
from bubble_chamber.photometry import sample_track_profiles
from bubble_chamber.pilot import PilotConfig, analyze_frame


def test_thicker_optical_line_has_wider_fwhm_without_particle_id_claim():
    samples = np.column_stack((np.linspace(25, 155, 14), np.full(14, 75.0)))
    widths = []
    for thickness in (1, 3, 6):
        image = np.full((150, 180), 225, dtype=np.uint8)
        cv2.line(image, (20, 75), (160, 75), 75, thickness, cv2.LINE_8)
        records = sample_track_profiles(image, samples)
        valid = [p["width_fwhm_px"] for p in records if p["status"] == "valid"]
        assert len(valid) == 14
        widths.append(np.median(valid))
    assert widths[0] < widths[1] < widths[2]


def test_low_contrast_and_frame_edge_are_abstained():
    plain = np.full((100, 150), 220, np.uint8)
    x = np.linspace(20, 120, 10)
    pts = np.column_stack((x, np.full(10, 50)))
    records = sample_track_profiles(plain, pts)
    assert all(p["status"] == "low_local_contrast" and p["width_fwhm_px"] is None for p in records)
    top = np.column_stack((x, np.zeros(10)))
    edge = sample_track_profiles(plain, top)
    assert all(p["status"] == "border_truncated" for p in edge)


def test_frangi_pipeline_outputs_explicit_uncalibrated_profile_in_global_coordinates():
    from bubble_chamber.synthetic import generate_case
    image, _, _ = generate_case("continuous")
    r, mask, _ = analyze_frame(image, PilotConfig(segmentation_method="frangi"), roi_xyxy=(30, 90, 350, 300))
    assert r["fragment_count"] > 0
    assert r["automatic_join_count"] == 0
    for s in r["segments"]:
        assert s["profile_status"] == "IMAGE_SPACE_ONLY_NOT_CALIBRATED_DEDX"
        assert len(s["transverse_profile_samples"]) == len(s["measurement_xy_px"]) == 14
        for p, xy in zip(s["transverse_profile_samples"], s["measurement_xy_px"]):
            assert abs(p["x_source_px"] - xy[0]) < 1e-5
            assert abs(p["y_source_px"] - xy[1]) < 1e-5
            assert p["status"] in ("valid", "low_local_contrast", "border_truncated", "unbounded_profile", "degenerate_tangent")
