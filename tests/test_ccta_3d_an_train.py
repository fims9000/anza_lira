"""Synthetic 3D ANZA/CNN training smoke; no real patient CT or weights."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_an_train import fit_comparison


def fake_cache(root,patient,split):
    root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(patient)
    data=rng.normal(size=(4,2,8,8,8)).astype(np.float16)
    labels=np.array([0,1,0,1],dtype=np.uint8)
    path=root/f"patches_{patient}.npz"
    np.savez_compressed(
        path,patches=data,y=labels,
        candidate_type=np.array(["PAIR"]*4),
        candidate_id=np.array([f"PAIR:e{i}+e{i+1}" for i in range(4)]),
        scene_id=np.array([f"scene_{patient}"]*4),
    )
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    (root/f"patches_{patient}.json").write_text(json.dumps({
        "patient_id":patient,"split":split,"cache_sha256":sha,
        "model_input_contains_ground_truth":False,
        "patch_protocol":{"side":8,"voxel_mm":1.0,
                          "endpoint_sigma_mm":1.5,
                          "hu_min":-300,"hu_max":1200},
    }))


def test_two_3d_backbones_train_on_exact_same_patient_isolated_patches(tmp_path):
    caches=tmp_path/"cache"
    fake_cache(caches,953,"train")
    fake_cache(caches,957,"val")
    out=tmp_path/"comparison"
    result=fit_comparison(
        cache_dir=caches,output_dir=out,candidate_type="PAIR",
        train_patients=[953],val_patients=[957],
        epochs=1,batch_size=2,seed=7,
        device="cpu",allow_subset=True,
    )
    assert result["status"]=="exploratory_candidate_ablation_not_frozen_K10"
    assert result["matched_capacity"]["matched_parameter_budget"]
    assert set(result["results"])=={"conv3d","anza3d"}
    for name in ("conv3d","anza3d"):
        metrics=result["results"][name]["val_metrics"]
        assert metrics["n"]==4
        assert 0<=metrics["auroc"]<=1
        assert 0<=metrics["auprc"]<=1
        assert (out/f"PAIR_{name}.pt").exists()
        assert (out/f"PAIR_{name}_val_scores.csv").exists()
    assert (out/"experiment.json").exists()
    with pytest.raises(FileExistsError):
        fit_comparison(
            cache_dir=caches,output_dir=out,candidate_type="PAIR",
            train_patients=[953],val_patients=[957],
            epochs=1,batch_size=2,allow_subset=True,
        )


def test_training_firewall_and_exact_cohort_requirement(tmp_path):
    with pytest.raises(ValueError,match="firewall"):
        fit_comparison(
            cache_dir=tmp_path,output_dir=tmp_path/"out",
            candidate_type="PAIR",train_patients=[953,954],
            val_patients=[957],allow_subset=True,
        )
    with pytest.raises(ValueError,match="17 train"):
        fit_comparison(
            cache_dir=tmp_path,output_dir=tmp_path/"out",
            candidate_type="PAIR",train_patients=[953],
            val_patients=[957],allow_subset=False,
        )
