"""V36 independent audit contract: catch model, patch and CAL score drift."""
import copy
import json

from scripts.tver_v36_compare_independent_runner import compare

def receipt():
    one={"model_tensor_state_SHA256":"0123","train_loss_first150":1.2,
         "train_loss_last150":.8,"pixel_CAL_score_receipt":{"0.5":{"tp":23}},
         "original_model_parameters":14194}
    return {"status":"V35_THEBE_DETERMINISTIC_WITHIN_ONE_RUN_TEST7_UNOPENED",
            "external_TEST7_bytes_masks_scores_NEVER_OPENED":True,
            "training_patches_SHA256":"train",
            "SEG_CAL_patches_SHA256":"cal",
            "train_normalization":{"low":-.1,"high":.1},
            "num_CPU_threads":1,"num_interop_threads":1,
            "MKLDNN_disabled":True,
            "torch_deterministic_algorithms_enabled":True,
            "torch_version":"2.5.1+cpu","numpy_version":"2.4.6",
            "model_results":{model:{"both_replays":[copy.deepcopy(one),copy.deepcopy(one)]} for model in ("unet","anza_v1")}}

def run(tmp_path,a,b):
    pa=tmp_path/"v35.json";pb=tmp_path/"v36.json"
    pa.write_text(json.dumps(a));pb.write_text(json.dumps(b))
    return compare(pa,pb,tmp_path/"out.json")

def test_identical_receipts_pass(tmp_path):
    a=receipt()
    r=run(tmp_path,a,copy.deepcopy(a))
    assert r["status"]=="V36_CROSS_RUNNER_BITWISE_REPRODUCIBILITY_PASS"
    assert not r["mismatches"]

def test_model_tensor_divergence_is_not_silent(tmp_path):
    a=receipt();b=copy.deepcopy(a)
    b["model_results"]["anza_v1"]["both_replays"][0]["model_tensor_state_SHA256"]="changed"
    r=run(tmp_path,a,b)
    assert r["status"]=="V36_CROSS_RUNNER_REPRODUCIBILITY_FAILED"
    assert any("anza_v1.repeat1.model_tensor_state_SHA256"==k for k in r["mismatches"])

def test_mismatched_source_patches_detected(tmp_path):
    a=receipt();b=copy.deepcopy(a);b["training_patches_SHA256"]="wrong"
    assert "environment/source mismatch: training_patches_SHA256" in run(tmp_path,a,b)["mismatches"]

def test_CAL_difference_or_TEST7_access_must_fail(tmp_path):
    a=receipt();b=copy.deepcopy(a)
    b["external_TEST7_bytes_masks_scores_NEVER_OPENED"]=False
    b["model_results"]["unet"]["both_replays"][1]["pixel_CAL_score_receipt"]["0.5"]["tp"]=25
    r=run(tmp_path,a,b)
    assert r["status"]=="V36_CROSS_RUNNER_REPRODUCIBILITY_FAILED"
    assert any("TEST7 embargo" in s for s in r["mismatches"])
    assert any("unet.repeat2.pixel_CAL_score_receipt"==s for s in r["mismatches"])
