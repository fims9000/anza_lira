"""Summarize only matched 17-train/5-val structure comparisons, no CT access."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from summarize_ct28_structure_seeds import aggregate


PATIENTS=(957,961,965,966,974)
CNN="conv3d_sep_structure"
ANZA="anza3d_sep_structure"


def fake_report(tmp_path,seed:int,*,extra_false=0,version="frozen-sha",complete=True,identity="neural_pair"):
    rows=[]
    for patient in PATIENTS:
        row={
            "patient_id":patient,
            "status":"complete" if complete else "missing_"+ANZA,
            "K10_dice_delta":.01,
            "K10_cldice_delta":.02,
            "ANZA_minus_CNN_dice_delta":.001,
            "ANZA_minus_CNN_cldice_delta":.002,
        }
        for variant in (CNN,ANZA):
            row.update({
                f"{variant}_dice_delta":.01,
                f"{variant}_cldice_delta":.02,
                f"{variant}_gt_false_bridge_pairs":extra_false if variant==ANZA else 0,
                f"{variant}_unsupported_added_voxels":0,
            })
        rows.append(row)
    report={
        "status":"complete_five_patient_matched_CCTA_dev_val",
        "patient_completed":5,"patient_failed_or_missing":0,
        "held_out_test_accessed":False,
        "pair_identity_source":identity,
        "relation_train_protocol":"experimental_patient_OOF_15copy_augmented_relation_head",
        "variant_pair":[CNN,ANZA],
        "tip_encoding":"separate",
        "noaxial_control":False,
        "protocol_sha256":version,
        "per_patient":rows,
    }
    path=tmp_path/f"{seed}.json"
    path.write_text(json.dumps(report))
    return path


def test_three_seed_summary_flags_extra_false_bridges_not_just_positive_cldice(tmp_path):
    runs={seed:fake_report(tmp_path,seed,extra_false=1 if seed==3 else 0)
          for seed in (1,2,3)}
    summary=aggregate(runs)
    assert summary["status"]=="three_seed_five_patient_development_only"
    assert summary["ANZA_better_cldice_seed_count"]==3
    assert summary["ANZA_no_extra_false_bridges_seed_count"]==2
    assert summary["ANZA_better_cldice_and_no_extra_false_bridge_seed_count"]==2
    assert summary["held_out_CT28_test_accessed"] is False


def test_multi_seed_aggregation_rejects_partial_or_incompatible_evidence(tmp_path):
    runs={seed:fake_report(tmp_path,seed) for seed in (11,12,13)}
    with pytest.raises(ValueError,match="Three distinct"):
        aggregate({11:runs[11],12:runs[12]})
    runs[13]=fake_report(tmp_path,13,version="tampered-freeze")
    with pytest.raises(ValueError,match="incompatible"):
        aggregate(runs)
    runs[13]=fake_report(tmp_path,13,complete=False)
    with pytest.raises(ValueError,match="five development"):
        aggregate(runs)


def test_neural_junction_identity_policy_must_match_across_three_patient_seeds(tmp_path):
    three={s:fake_report(tmp_path,s,identity="neural_pair_junction")
           for s in (11,12,13)}
    result=aggregate(three)
    assert result["candidate_identity_policy"]=="neural_pair_junction"
    assert result["held_out_CT28_test_accessed"] is False
    three[12]=fake_report(tmp_path,12,identity="neural_pair")
    with pytest.raises(ValueError,match="incompatible"):
        aggregate(three)
