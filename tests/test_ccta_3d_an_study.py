"""Synthetic receipts verify comparable 3D CCTA train-only study outcomes."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_an_study import summarize,ensure_pilot,_verified_cache


def write(path,meta):
    path.write_text(json.dumps(meta))


def metrics(delta):
    return {
        "patient_id":"953",
        "before_sha256":"same_damaged_mask",
        "reference_sha256":"same_truth",
        "dice_before":.70,
        "dice_after":.70+delta,
        "dice_delta":delta,
        "cldice_before":.50,
        "cldice_after":.50+delta,
        "cldice_delta":delta,
        "reference_unsupported_added_voxels":1,
    }


def test_three_model_full_mask_comparison_requires_same_input_and_gt(tmp_path):
    pilot=tmp_path/"pilot_953"
    pilot.mkdir()
    write(pilot/"posthoc_metrics.json",metrics(.12))
    write(pilot/"decisions.json",{
        "patient_label":"953","accepted_decisions":2,
        "added_voxels":9,
    })
    for variant,delta in (("conv3d",.08),("anza3d",.09)):
        write(tmp_path/f"{variant}_posthoc.json",metrics(delta))
        write(tmp_path/f"{variant}_decisions.json",{
            "patient_label":"953",
            "experimental_evidence":{"variant":variant},
            "accepted_decisions":1,"added_voxels":7,
        })
    out=summarize(tmp_path,pilot=pilot,patient=953)
    assert set(out["models"])=={"K10","conv3d","anza3d"}
    assert out["same_ct_mask_and_reference_for_all"]
    assert out["models"]["anza3d"]["delta_dice"]==pytest.approx(.09)
    assert (tmp_path/"train_953_three_model_comparison.md").is_file()
    with pytest.raises(FileExistsError):
        summarize(tmp_path,pilot=pilot,patient=953)


def test_neural_volume_comparison_does_not_mix_different_masks(tmp_path):
    pilot=tmp_path/"pilot_953"
    pilot.mkdir()
    write(pilot/"posthoc_metrics.json",metrics(.1))
    write(pilot/"decisions.json",{
        "patient_label":"953","accepted_decisions":0,"added_voxels":0,
    })
    for name in ("conv3d","anza3d"):
        meta=metrics(.1)
        if name=="anza3d":
            meta["before_sha256"]="not_the_same_mask"
        write(tmp_path/f"{name}_posthoc.json",meta)
        write(tmp_path/f"{name}_decisions.json",{
            "patient_label":"953","accepted_decisions":0,"added_voxels":0,
            "experimental_evidence":{"variant":name},
        })
    with pytest.raises(ValueError,match="same broken"):
        summarize(tmp_path,pilot=pilot,patient=953)


def test_study_refuses_heldout_or_validation_patient_before_any_data(tmp_path):
    for patient in (954,957,961,973):
        with pytest.raises(ValueError,match="train"):
            ensure_pilot(tmp_path,patient=patient)


def test_partial_patch_cache_is_not_silently_reused(tmp_path):
    (tmp_path/"patches_953.json").write_text("{}")
    with pytest.raises(ValueError,match="Partial"):
        _verified_cache(tmp_path,953)
