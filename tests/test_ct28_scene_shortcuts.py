"""Read-only checks of the current scene shortcut diagnostic."""
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts/research/ccta_graph_lira_safe_repair"))
from audit_ct28_scene_shortcuts import audit_scene_protocol


def test_current_controlled_scene_shortcuts_are_explicitly_recorded():
    report = audit_scene_protocol(
        _ROOT / "artifacts/varvara/ct28_graph_lira/scene_plan_train_val.csv"
    )
    assert report["train_scenes"] == 1036
    assert report["validation_scenes"] == 368
    count = report["count_only_train_lookup"]
    assert count["covered"] == 367
    assert count["correct"] == 270
    val = report["endpoint_identity_patterns"]
    assert val["val/pair_only"]["patterns"] == [
        {"pair": "E00|E01", "junction": "NONE", "n": 134}
    ]
    assert val["val/junction_only"]["patterns"] == [
        {"pair": "NONE", "junction": "E00|E01|E02", "n": 36}
    ]
    assert val["val/mixed"]["patterns"] == [
        {"pair": "E00|E01", "junction": "E02|E03|E04", "n": 36}
    ]
