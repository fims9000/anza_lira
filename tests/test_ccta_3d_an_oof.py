"""Synthetic patient-disjoint OOF score test for volumetric candidate encoders."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_an_oof import build_oof


def write_fixture(cache_dir,pid,split):
    cache_dir.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(pid)
    path=cache_dir/f"patches_{pid}.npz"
    np.savez_compressed(
        path,
        patches=rng.standard_normal((4,2,8,8,8)).astype(np.float16),
        y=np.asarray([0,1,0,1],dtype=np.uint8),
        candidate_type=np.asarray(["PAIR"]*4),
        scene_id=np.asarray([f"scene_{pid}"]*4),
        candidate_id=np.asarray([f"pair_{pid}_{i}" for i in range(4)]),
    )
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    (cache_dir/f"patches_{pid}.json").write_text(json.dumps({
        "patient_id":pid,"split":split,
        "cache_sha256":checksum,
        "model_input_contains_ground_truth":False,
        "patch_protocol":{"side":8,"voxel_mm":1.0,"endpoint_sigma_mm":1.5},
    }))


def test_oof_folds_never_train_on_the_scored_patient(tmp_path):
    cache=tmp_path/"cache"
    for pid in (953,955,956):
        write_fixture(cache,pid,"train")
    write_fixture(cache,957,"val")
    out=tmp_path/"oof"
    report=build_oof(
        cache_dir=cache,out_dir=out,candidate_type="PAIR",
        variant="conv3d",train_patients=[953,955,956],
        val_patients=[957],folds=2,epochs=1,
        batch_size=2,allow_subset=True,seed=42,
    )
    assert report["n_train_patients"]==3
    assert report["n_validation_patients"]==1
    assert len(report["fold_assignments"])==2
    for f in report["fold_assignments"]:
        assert not set(f["train_patients"])&set(f["held_patients"])
    path=out/"PAIR_conv3d_oof_val.csv"
    with path.open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==16
    assert len({(r["scene_id"],r["candidate_id"]) for r in rows})==16
    assert all(int(x["oof_fold"])>=0 for x in rows if x["split"]=="train")
    assert all(int(x["oof_fold"])==-1 for x in rows if x["split"]=="val")
    assert (out/"PAIR_conv3d_fulltrain.pt").exists()
    assert report["held_out_test_accessed"] is False
    with pytest.raises(FileExistsError):
        build_oof(cache_dir=cache,out_dir=out,candidate_type="PAIR",
                  variant="conv3d",train_patients=[953,955,956],
                  val_patients=[957],folds=2,epochs=1,allow_subset=True)


def test_oof_firewall_prevents_val_as_train(tmp_path):
    with pytest.raises(ValueError,match="split"):
        build_oof(cache_dir=tmp_path,out_dir=tmp_path/"output",
                  candidate_type="PAIR",variant="anza3d",
                  train_patients=[953,957],val_patients=[961],
                  allow_subset=True)
