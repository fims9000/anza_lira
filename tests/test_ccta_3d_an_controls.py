"""CT28 candidate image reliance controls: exact labels, same endpoint geometry."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_an_controls import make_control_cache,digest
from ccta_3d_an_train import load_type


def create_source(root,patient=953):
    root.mkdir(exist_ok=True)
    file=root/f"patches_{patient}.npz"
    images=np.arange(4*2*8*8*8,dtype=np.float16).reshape((4,2,8,8,8))
    np.savez_compressed(
        file,patches=images,y=np.asarray([0,1,1,0],dtype=np.uint8),
        candidate_type=np.array(["PAIR","PAIR","JUNCTION","JUNCTION"]),
        scene_id=np.asarray([f"s{patient}"]*4),
        candidate_id=np.asarray([f"PAIR:{patient}_{i}" for i in range(4)]),
    )
    meta={
        "patient_id":patient,
        "split":"train" if patient==953 else "val",
        "cache_sha256":digest(file),
        "model_input_contains_ground_truth":False,
        "patch_protocol":{"side":8,"voxel_mm":1.,"endpoint_sigma_mm":1.5},
    }
    (root/f"patches_{patient}.json").write_text(json.dumps(meta))
    return images


@pytest.mark.parametrize("mode",["endpoint_only","ct_only","scrambled_ct"])
def test_controls_keep_labels_candidate_ids_and_valid_sha(mode,tmp_path):
    raw=tmp_path/"raw"
    original=create_source(raw)
    out=tmp_path/mode
    result=make_control_cache(
        source_dir=raw,output_dir=out,mode=mode,
        patients=[953],seed=21,
    )
    assert result["held_out_test_accessed"] is False
    with np.load(raw/"patches_953.npz",allow_pickle=False) as source:
        with np.load(out/"patches_953.npz",allow_pickle=False) as transformed:
            for key in ("y","candidate_id","scene_id","candidate_type"):
                assert np.array_equal(source[key],transformed[key])
            patch=transformed["patches"]
            if mode=="endpoint_only":
                assert np.all(patch[:,0]==0)
                assert np.array_equal(patch[:,1],original[:,1])
            elif mode=="ct_only":
                assert np.array_equal(patch[:,0],original[:,0])
                assert np.all(patch[:,1]==0)
            else:
                assert np.array_equal(patch[:,1],original[:,1])
                assert not np.array_equal(patch[:,0],original[:,0])
                for i in range(4):
                    assert np.array_equal(
                        np.sort(patch[i,0].reshape(-1)),
                        np.sort(original[i,0].reshape(-1)),
                    )
    manifest=json.loads((out/"patches_953.json").read_text())
    assert manifest["input_control_mode"]==mode
    assert manifest["cache_sha256"]==digest(out/"patches_953.npz")
    assert manifest["model_input_contains_ground_truth"] is False
    with pytest.raises(FileExistsError):
        make_control_cache(source_dir=raw,output_dir=out,mode=mode,patients=[953])


def test_validation_firewall_and_mixed_mode_detection(tmp_path):
    raw=tmp_path/"raw"
    create_source(raw)
    with pytest.raises(ValueError,match="17 train / 5 validation"):
        make_control_cache(source_dir=raw,output_dir=tmp_path/"forbidden",
                           mode="endpoint_only",patients=[954])
    derived=tmp_path/"derived"
    make_control_cache(source_dir=raw,output_dir=derived,
                       mode="endpoint_only",patients=[953])
    x,y,examples,receipts=load_type(
        derived,[953],split="train",candidate_type="PAIR"
    )
    assert x.shape==(2,2,8,8,8)
    assert list(y)==[0.,1.]
    assert len(examples)==2
