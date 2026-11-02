"""Integration contract against actual checked-in CT28 TRAIN scene CSVs.

Uses a tiny synthetic CT aligned around the real *train patient* endpoint
coordinates; no raw ImageCAS CT, patient masks or test annotations are opened.
This catches schema drift hidden by hand-authored miniature CSV fixtures.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_an_patch_cache import build_patient_cache,SCENES
from ccta_3d_an_patches import PatchConfig


def _synthetic_volume_covering_train_endpoints():
    # This is FAKE homogeneous HU data, not a patient CT.
    # Affine matches the physical region of the real, checked-in patient 953
    # scene endpoint coordinates; the entire grid is a compact 64-cubed array.
    ct=np.full((64,64,64),225,dtype=np.float32)
    affine=np.diag([2.5,2.5,2.5,1.])
    affine[:3,3]=[-100.,80.,120.]
    return ct,affine


def test_real_ct28_scene_csvs_join_correctly_and_emit_label_separated_patches(tmp_path):
    plan=pd.read_csv(SCENES/"scene_plan_train_val.csv")
    endpoints=pd.read_csv(SCENES/"scene_endpoints_train_val.csv")
    candidates=pd.read_csv(SCENES/"scene_candidates_train_val.csv")
    selected_plan=plan[(plan.scan_id==953)&plan.split.eq("train")]
    selected_candidates=candidates[candidates.scene_id.isin(selected_plan.scene_id)]
    assert len(selected_plan)==79
    assert len(selected_candidates)==748
    assert selected_candidates.y.sum()>0
    assert set(selected_candidates.candidate_type)=={"PAIR","JUNCTION"}
    assert set(endpoints.columns)>={
        "scene_id","endpoint_id","x_lps","y_lps","z_lps"
    }
    output=tmp_path/"cache"
    meta=build_patient_cache(
        patient=953,scenes_dir=SCENES,output_dir=output,
        config=PatchConfig(side=8,voxel_mm=2.),
        load_volume=lambda patient:_synthetic_volume_covering_train_endpoints(),
    )
    assert meta["split"]=="train"
    assert meta["patient_id"]==953
    assert meta["n_candidates"]==748
    assert meta["n_positives"]==int(selected_candidates.y.sum())
    assert meta["n_pair"]==int((selected_candidates.candidate_type=="PAIR").sum())
    assert meta["n_junction"]==int((selected_candidates.candidate_type=="JUNCTION").sum())
    assert meta["outside_fraction_over_20pct"]==0
    assert meta["held_out_test_accessed"] is False
    assert meta["model_input_contains_ground_truth"] is False

    with np.load(output/"patches_953.npz",allow_pickle=False) as z:
        assert z["patches"].shape==(748,2,8,8,8)
        assert z["y"].shape==(748,)
        assert len(set(zip(z["scene_id"],z["candidate_id"])))==748
        assert np.isfinite(z["patches"]).all()
        assert z["inside_fraction"].min()>.8
        keys={(str(scene),str(candidate),int(y))
              for scene,candidate,y in zip(z["scene_id"],z["candidate_id"],z["y"])}
        expected={(str(row.scene_id),str(row.candidate_id),int(row.y))
                  for row in selected_candidates.itertuples(index=False)}
        assert keys==expected


def test_real_ct28_scene_csvs_never_allow_heldout_patch_generation(tmp_path):
    with pytest.raises(ValueError,match="train"):
        build_patient_cache(
            patient=954,scenes_dir=SCENES,output_dir=tmp_path,
            load_volume=lambda patient: (_ for _ in ()).throw(
                AssertionError("Held-out data requested before firewall")
            ),
        )
