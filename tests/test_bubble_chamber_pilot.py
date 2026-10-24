import math
import numpy as np
import pytest

from bubble_chamber.pilot import PilotConfig, analyze_frame, arc_sample, extract_skeleton_segments, fit_quadratic, propose_joins
from bubble_chamber.synthetic import centerline_scores, generate_case


def test_fit_quadratic_pca_and_measured_endpoints():
    x = np.linspace(20, 130, 120)
    y = 50 + 0.1 * (x - 80) + 0.002 * (x - 80) ** 2
    points = np.column_stack((y, x))
    fit = fit_quadratic(points)
    assert fit["residual_rms_px"] < 0.6
    assert np.allclose(fit["endpoint_A_xy_px"], [20, y[0]])
    assert np.allclose(fit["endpoint_B_xy_px"], [130, y[-1]])
    arc = arc_sample(points, 14)
    assert arc.shape == (14, 2)
    assert np.allclose(arc[0], points[0])
    assert np.allclose(arc[-1], points[-1])


def test_simple_collinear_gap_is_proposed_but_never_merged():
    a = np.asarray([[30.0, x] for x in range(20, 81)])
    b = np.asarray([[30.0, x] for x in range(94, 151)])
    c = propose_joins([a, b], max_gap_px=25)
    assert len(c) == 1
    assert c[0]["gap_px"] == pytest.approx(14)
    assert c[0]["status"] == "needs_review"
    assert c[0]["automatic_merge"] is False


def test_ambiguous_two_continuations_are_flagged():
    source = np.asarray([[100.0, x] for x in range(20, 81)])
    b = np.asarray([[100.0, x] for x in range(90, 130)])
    c = np.asarray([[103.0, x] for x in range(90, 130)])
    p = propose_joins([source, b, c], max_gap_px=26)
    assert any(x["ambiguous"] for x in p)
    assert all(not x["automatic_merge"] for x in p)


def test_crossing_skeleton_is_split_not_crossed_as_one_track():
    sk = np.zeros((40, 40), dtype=bool)
    sk[20, 6:35] = True
    sk[6:35, 20] = True
    segments = extract_skeleton_segments(sk, min_length_px=5)
    assert len(segments) >= 4
    assert max(len(x) for x in segments) < 25


@pytest.mark.parametrize("case", ["continuous", "gap", "crossing", "parallel"])
def test_synthetic_has_visible_track_fragments(case):
    image, truth, _ = generate_case(case)
    out, mask, overlay = analyze_frame(image, PilotConfig(threshold=18, min_segment_length_px=35))
    scores = centerline_scores(mask, truth)
    assert mask.shape == image.shape
    assert overlay.shape == (*image.shape, 3)
    assert out["fragment_count"] > 0
    assert scores["recall_tolerant"] > 0.45
    assert out["automatic_join_count"] == 0
    assert out["physical_result_status"] == "NOT_RECONSTRUCTED"
    assert all(not p["automatic_merge"] for p in out["join_proposals"])


def test_invalid_roi_is_rejected():
    with pytest.raises(ValueError, match="invalid ROI"):
        analyze_frame(np.ones((20, 20), dtype=np.uint8), roi_xyxy=(0, 0, 22, 12))


def test_weak_noisy_case_is_measured_without_claim_of_accuracy():
    image, truth, _ = generate_case("weak_noisy")
    output, mask, _ = analyze_frame(image)
    score = centerline_scores(mask, truth)
    assert 0.0 <= score["recall_tolerant"] <= 1.0
    assert output["automatic_join_count"] == 0
    assert output["no_ground_truth"] is True


def test_cropped_frame_measurements_keep_original_pixel_reference():
    image, _, _ = generate_case("continuous")
    output, mask, _ = analyze_frame(image, roi_xyxy=(40, 50, 350, 320))
    assert output["roi_xyxy"] == [40, 50, 350, 320]
    assert mask.shape == (270, 310)
    assert all(40 <= xy[0] < 350 and 50 <= xy[1] < 320
               for seg in output["segments"] for xy in seg["measurement_xy_px"])


def test_archive_extraction_preserves_lineage_without_splitting(tmp_path):
    import cv2
    from zipfile import ZipFile
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/research/bubble_chamber/prepare_reference.py"
    spec = importlib.util.spec_from_file_location("bc_prepare_reference", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = np.full((1744, 912), 220, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr("unknown_name/action.jpg", encoded.tobytes())
    meta = module.prepare_archive(archive, tmp_path / "subset")
    assert len(meta["samples"]) == 4
    assert meta["shape_yx"] == [1744, 912]
    assert all(item["independent_sample"] is False for item in meta["samples"])
    assert meta["stereo_views_available"] is False
