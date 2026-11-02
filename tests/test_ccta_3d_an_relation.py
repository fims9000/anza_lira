"""Synthetic patient-OOF 3D CCTA stream substitution for 84-feature relation head."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_an_relation import substitute_neural_relation_stream


def fixture():
    plans=pd.DataFrame([
        {"scene_id":"s_train","scan_id":953,"split":"train",
         "scene_family":"pair_only","relation_class":"PAIR"},
        {"scene_id":"s_val","scan_id":957,"split":"val",
         "scene_family":"junction_only","relation_class":"JUNCTION"},
    ])
    candidates=pd.DataFrame([
        {"scene_id":sid,"candidate_id":f"{sid}_{kind}",
         "candidate_type":kind,"endpoint_ids":"E00|E01","y":int(kind=="PAIR")}
        for sid in ("s_train","s_val")
        for kind in ("PAIR","JUNCTION")
    ])
    baseline=pd.DataFrame([
        {"scene_id":r.scene_id,"candidate_id":r.candidate_id,
         "candidate_type":r.candidate_type,"geometry_score":.7,
         "ct_score":.1,"geometry_plus_ct_score":.5}
        for r in candidates.itertuples()
    ])
    learned=pd.DataFrame([
        {"scene_id":r.scene_id,"candidate_id":r.candidate_id,
         "candidate_type":r.candidate_type,
         "scan_id":953 if r.scene_id=="s_train" else 957,
         "split":"train" if r.scene_id=="s_train" else "val",
         "score":.8 if r.candidate_type=="PAIR" else .2,
         "oof_fold":0 if r.scene_id=="s_train" else -1}
        for r in candidates.itertuples()
    ])
    return plans,candidates,baseline,learned


def test_original_geometry_and_identity_scores_frozen_neural_ct_only():
    plans,candidates,original,learned=fixture()
    features,columns,output=substitute_neural_relation_stream(
        plans,candidates,original,learned
    )
    assert len(columns)==84
    assert len(features)==2
    assert set(output["ct_score"])=={.2,.8}
    assert set(original["ct_score"])=={.1}
    assert set(output["geometry_score"])=={.7}
    assert set(output["geometry_plus_ct_score"])=={.5}
    assert any(col.endswith("_candidate_count") for col in columns)


def test_missing_or_non_oof_train_predictions_fail_closed():
    plans,candidates,original,learned=fixture()
    with pytest.raises(ValueError,match="cover every frozen"):
        substitute_neural_relation_stream(
            plans,candidates,original,learned.iloc[:-1]
        )
    wrong=learned.copy()
    wrong.loc[wrong.split.eq("train"),"oof_fold"]=-1
    with pytest.raises(ValueError,match="OOF"):
        substitute_neural_relation_stream(plans,candidates,original,wrong)
