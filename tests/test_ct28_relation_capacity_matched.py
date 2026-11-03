"""Historical CT28 relation ablation: same 84 features across CT controls."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0,str(Path(__file__).resolve().parents[1]/
          "scripts/research/ccta_graph_lira_safe_repair")
)
from ct28_relation_capacity_matched import (
    transform_scores,checked_inputs,run_study,
    MODEL,SCORES,CANDIDATES,PLAN,
)


def test_permutation_preserves_score_marginals_and_hgb_identity():
    fake=pd.DataFrame({
        "scene_id":["a","a","b","b","c","c","d"],
        "candidate_id":["a1","a2","b1","b2","c1","c2","d1"],
        "candidate_type":["PAIR"]*4+["JUNCTION"]*2+["PAIR"],
        "scan_id":[953]*6+[957],
        "split":["train"]*6+["val"],
        "ct_score":[.10,.20,.30,.40,.50,.60,.70],
        "geometry_score":[.70,.60,.50,.40,.30,.20,.10],
        "geometry_plus_ct_score":[.1,.2,.3,.4,.5,.6,.7],
    })
    before=fake.copy(deep=True)
    for mode in ("real_CT","geometry_duplicate","constant_CT","sham_CT"):
        after=transform_scores(fake,mode,seed=42)
        assert np.array_equal(after.geometry_score,before.geometry_score)
        assert np.array_equal(after.geometry_plus_ct_score,before.geometry_plus_ct_score)
        assert after[["scene_id","candidate_id","scan_id","split"]].equals(
            before[["scene_id","candidate_id","scan_id","split"]]
        )
        assert fake.equals(before)
        if mode=="geometry_duplicate":
            assert np.array_equal(after.ct_score,after.geometry_score)
        if mode=="constant_CT":
            assert np.all(after.ct_score==.5)
        if mode=="real_CT":
            assert after.ct_score.equals(before.ct_score)
        if mode=="sham_CT":
            for (_,_,_),part in before.groupby(["split","scan_id","candidate_type"]):
                assert np.allclose(
                    np.sort(after.loc[part.index,"ct_score"]),
                    np.sort(part.ct_score)
                )
    with pytest.raises(ValueError,match="Unknown"):
        transform_scores(fake,"fabricated_variant")


def test_loaded_real_local_OOF_scores_have_no_test_patient():
    tr,val,candidates,scores,model=checked_inputs(
        model_file=MODEL,score_file=SCORES,
        candidate_file=CANDIDATES,plan_file=PLAN,
    )
    assert len(tr)==1036
    assert len(val)==368
    assert len(scores)==15092
    assert len(candidates)==15092
    assert len(model["ct_conditioned_relation_features"])==84
    assert set(scores.loc[scores.split.eq("val"),"scan_id"].astype(int))=={
        957,961,965,966,974
    }


def test_three_repeat_capacity_matched_study_is_patient_safe_and_paired():
    report,table,scenes=run_study(sham_repetitions=3,bootstrap=110,seed=19)
    assert report["train_scenes"]==1036
    assert report["val_scenes"]==368
    assert report["same_feature_count_all"]==84
    assert report["held_out_test_accessed"] is False
    assert len(table)==6
    assert len(scenes)==368*6
    for metric in ("real_CT","geometry_duplicate","constant_CT"):
        r=report["variants"][metric]
        assert r["relation_features"]==84
        assert r["train_scenes"]==1036
        assert r["val_scenes"]==368
        assert 0<r["raw_accuracy"]<=1
        assert r["log_loss"]>0
    assert set(report["paired_full_CT_vs_controls"])=={
        "geometry_duplicate","constant_CT"
    }
    for v in report["paired_full_CT_vs_controls"].values():
        assert len(v["per_patient"])==5
        assert v["patient_bootstrap_draws"]==110
