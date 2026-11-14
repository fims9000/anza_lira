"""Independent V36 comparison: original V35 versus second GitHub runner.

No model, Thebe source, sklearn, torch, or TEST7 access: only frozen
compact JSON receipts. Require exact raw states, patch hashes and metrics.
Report failure as failure; never retune experimental code.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

MODELS=("unet","anza_v1")
COMPARE=("training_patches_SHA256","SEG_CAL_patches_SHA256",
         "train_normalization","num_CPU_threads","num_interop_threads",
         "MKLDNN_disabled","torch_deterministic_algorithms_enabled",
         "torch_version","numpy_version")
REPEAT=("model_tensor_state_SHA256","train_loss_first150",
        "train_loss_last150","pixel_CAL_score_receipt",
        "original_model_parameters")


def compare(original:Path,second:Path,out:Path):
    oa=original.read_bytes();ob=second.read_bytes()
    a=json.loads(oa);b=json.loads(ob)
    issues=[]
    if a.get("status")!="V35_THEBE_DETERMINISTIC_WITHIN_ONE_RUN_TEST7_UNOPENED":
        issues.append("original V35 status not within-run deterministic")
    if b.get("status")!="V35_THEBE_DETERMINISTIC_WITHIN_ONE_RUN_TEST7_UNOPENED":
        issues.append("V36 second runner status not within-run deterministic")
    if not a.get("external_TEST7_bytes_masks_scores_NEVER_OPENED") or not b.get(
        "external_TEST7_bytes_masks_scores_NEVER_OPENED"):
        issues.append("TEST7 embargo violation")
    for k in COMPARE:
        if a.get(k)!=b.get(k):issues.append("environment/source mismatch: "+k)
    compared={}
    for model in MODELS:
        orig=a.get("model_results",{}).get(model,{})
        new=b.get("model_results",{}).get(model,{})
        x=orig.get("both_replays",[])
        y=new.get("both_replays",[])
        if len(x)!=2 or len(y)!=2:
            issues.append("wrong model repeat count: "+model);continue
        compared[model]={}
        for i in range(2):
            compared[model][f"repeat{i+1}"]={}
            for k in REPEAT:
                same=x[i].get(k)==y[i].get(k)
                compared[model][f"repeat{i+1}"][k]=same
                if not same:issues.append(f"{model}.repeat{i+1}.{k}")
    result={
        "status":("V36_CROSS_RUNNER_BITWISE_REPRODUCIBILITY_PASS" if not issues
                  else "V36_CROSS_RUNNER_REPRODUCIBILITY_FAILED"),
        "original_V35_JSON_SHA256":hashlib.sha256(oa).hexdigest(),
        "fresh_V36_JSON_SHA256":hashlib.sha256(ob).hexdigest(),
        "separate_independent_runner_comparison":True,
        "source_and_experimental_CI_conditions":{k:a.get(k)==b.get(k) for k in COMPARE},
        "model_comparisons":compared,
        "mismatches":issues,
        "external_TEST7_still_UNOPENED":bool(
          a.get("external_TEST7_bytes_masks_scores_NEVER_OPENED") and
          b.get("external_TEST7_bytes_masks_scores_NEVER_OPENED")),
        "proof_scope":"bitwise repeat of one fixed seed per architecture across 2 runners, not learned superiority or geological link safety",
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"mismatches":issues},indent=2))
    return result


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--original",type=Path,required=True)
    ap.add_argument("--second",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    compare(a.original,a.second,a.out)
