"""Fixtures exercise V52 paired audit, without downloading geological data."""
import csv
import json

import pytest

from scripts import tver_v52_thebe_comparison as m


def fake_sources():
    scenes, v50 = [], []
    for section in range(60):
        for idx in range(2):
            gid = str(idx)
            flat = section * 2 + idx
            good_pair = flat < 45
            v50_correct = flat < 18
            link = {"section": section, "id": gid, "kind": "LINK",
                    "proposals": [{"source": (20, 45), "target": (70, 47), "y": int(good_pair)}]}
            no_link = {"section": section, "id": gid, "kind": "NO_LINK",
                       "proposals": [{"source": (20, 45), "target": (70, 47), "y": 0}]}
            scenes.extend([link, no_link])
            v50.append({"section": section, "gap_id": gid,
                        "correct_top1_pair": int(good_pair),
                        "successful_learned_ribbon_gap_path": int(v50_correct)})
    return scenes, {"case_rows": v50}


def fake_decode(_model, scene, proposal):
    # Never consume expert centerlines or labels in the decoder.
    assert "source" in proposal and "target" in proposal
    if scene["kind"] == "NO_LINK" or scene["section"] > 30:
        return None, {"status": "ABSTAIN_NO_OBSERVED_PAIR"}
    return ((20, 45), (21, 46)), {"status": "JOINT_OBSERVED_ANCHORS", "n_hypotheses": 2}


def fake_select(scene, _scale, _ranker):
    return scene["proposals"][0]


def fake_score(_scene, _path):
    return True


def test_full_paired_report_and_case_receipts(tmp_path):
    scene, baseline = fake_sources()
    result = m.evaluate_paired_cal(
        scene, None, None, None, baseline, tmp_path,
        scorer=fake_score, decoder=fake_decode, select=fake_select)
    assert result["n_LINK"] == 120
    assert result["n_synthetic_NO_LINK"] == 120
    assert result["n_original_v50_correct_paths"] == 18
    assert result["n_v52_correct_paths"] == 45
    assert result["paired_wins_v52_not_v50"] == 27
    assert result["paired_wins_v50_not_v52"] == 0
    assert result["n_v52_synthetic_NO_LINK_paths_emitted"] == 0
    assert result["source_cluster_bootstrap"]["n_clusters"] == 60
    assert result["no_TEST7_access"] is True
    saved = json.loads((tmp_path / "V52_REAL_THEBE_REL_CAL_RESULT.json").read_text())
    assert saved["n_v52_correct_paths"] == 45
    with (tmp_path / "V52_REAL_THEBE_REL_CAL_CASES.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 240


def test_repeated_bootstrap_is_deterministic():
    scenes, baseline = fake_sources()
    rows = [{"section": s["section"], "world": s["kind"],
             "v50_path_correct": 0, "v52_path_correct": 1} for s in scenes]
    a = m._descriptive_source_cluster_ci(rows)
    b = m._descriptive_source_cluster_ci(rows)
    assert a == b
    assert a["observed_success_rate_delta_percentage_points"] == 100.


def test_mismatched_baseline_fails_closed(tmp_path):
    scenes, baseline = fake_sources()
    baseline["case_rows"][50]["correct_top1_pair"] = 1
    with pytest.raises(RuntimeError, match="top1 candidate differs"):
        m.evaluate_paired_cal(scenes, None, None, None, baseline, tmp_path,
                              scorer=fake_score, decoder=fake_decode, select=fake_select)


def test_wrong_path_or_missing_anchor_is_never_counted_as_success(tmp_path):
    scenes, baseline = fake_sources()
    result = m.evaluate_paired_cal(scenes, None, None, None, baseline, tmp_path,
                                   scorer=lambda *_: False,
                                   decoder=fake_decode, select=fake_select)
    assert result["n_v52_correct_paths"] == 0
    assert result["n_v52_wrong_LINK_paths_emitted"] > 0
    assert result["n_v52_abstain_NO_LINK"] == 120