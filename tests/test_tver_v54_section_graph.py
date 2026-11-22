"""V54 whole-section graph exporter unit tests."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from scripts.tver_v54_section_graph import export_section_graph


def test_section_export_produces_graphs_and_label_free_links(tmp_path):
    prob = np.zeros((96, 128), dtype=np.float32)
    prob[48, 12:44] = .95
    prob[48, 53:88] = .95
    labels = np.zeros_like(prob, dtype=np.uint8)
    labels[48, 12:88] = 1
    prob_file = tmp_path / "segmentation.npy"
    label_file = tmp_path / "reference.npy"
    np.save(prob_file, prob)
    np.save(label_file, labels)

    report = export_section_graph(
        prob_file, tmp_path / "result",
        expert_mask_path=label_file, thresholds=(.65, .35),
        max_gap_px=20.,
    )
    assert report["no_expert_gap_rectangle_required"] is True
    assert len(report["layers"]) == 2
    first = report["layers"][0]
    assert first["n_trace_segments"] == 2
    assert first["n_unaccepted_link_candidates"] == 1
    assert 0 < first["annotated_centerline_coverage_2px"] < 1
    out = tmp_path / "result"
    for threshold in ("0p65", "0p35"):
        skeleton = np.load(out / f"V54_skeleton_{threshold}.npy")
        assert skeleton.shape == prob.shape
        with (out / f"V54_candidate_links_{threshold}.csv").open() as fh:
            assert len(list(csv.DictReader(fh))) == 1
    saved = json.loads((out / "V54_SECTION_TRACE_REPORT.json").read_text())
    assert saved["layers"] == report["layers"]


def test_section_export_does_not_require_reference_and_rejects_wrong_shape(tmp_path):
    array = np.zeros((96, 128), np.float32)
    array[48, 12:44] = .95
    array[48, 53:88] = .95
    p = tmp_path / "seg.npy"
    np.save(p, array)
    report = export_section_graph(p, tmp_path / "out", thresholds=(.65,))
    assert report["layers"][0]["annotated_centerline_coverage_2px"] is None

    wrong = tmp_path / "wrong.npy"
    np.save(wrong, np.ones((10, 10), np.uint8))
    with pytest.raises(ValueError, match="shape differs"):
        export_section_graph(p, tmp_path / "out2",
                             expert_mask_path=wrong, thresholds=(.65,))
