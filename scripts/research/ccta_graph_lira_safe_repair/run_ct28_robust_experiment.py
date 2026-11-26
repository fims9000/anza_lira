#!/usr/bin/env python3
"""One-command CT28 robustness experiment, with immutable output and resume.

Use windows-only + --reuse-oof-root first to diagnose/fix HGB without training
new CNN/ANZA weights. Use combined for matched fresh jitter training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from ccta_3d_an_study import ensure_oof, ensure_relations, verify_upstream_ranking_protocol, _verified_oof
from ccta_3d_val_an_compare import check_val_pilot
from ccta_3d_val_freeze import verify_validation_freeze, sha
from ccta_3d_val_pilot import DEFAULT_OUTPUT
from ccta_3d_train_gap_simulator import VAL_CT28_IDS

HERE=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reuse_oof(source, target, variants, seed, rank):
    for variant in variants:
        for kind in ("PAIR","JUNCTION"):
            name=f"oof_{kind.lower()}_{variant}"
            src=source/name
            if not _verified_oof(src,kind,variant):
                raise ValueError(f"Missing verified existing weights: {src}")
            info=read(src/"oof_protocol.json")
            hp=info["fixed_train_hyperparameters"]
            if (hp["seed"]!=seed or hp.get("ranking_weight",0.)!=rank or
                hp.get("endpoint_jitter_mm",0.)!=0 or hp["epochs"]!=8 or
                hp["batch_size"]!=2 or info["fold_count"]!=5):
                raise ValueError(f"Existing OOF training configuration differs: {src}")
            cfg=info["train_patch_sources"][0]["patch_protocol"]
            if cfg.get("frame_mode")!="fixed_lps" or cfg.get("endpoint_encoding")!="separate":
                raise ValueError(f"Existing OOF does not use fixed-LPS separate-tip patches: {src}")
            dst=target/name
            if dst.exists():
                if (not _verified_oof(dst,kind,variant) or
                    sha(dst/"oof_protocol.json")!=sha(src/"oof_protocol.json")):
                    raise ValueError(f"Existing target OOF differs: {dst}")
            else:
                shutil.copytree(src,dst)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze",type=Path,required=True)
    p.add_argument("--pilot-root",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--cache-dir",type=Path)
    p.add_argument("--reuse-oof-root",type=Path)
    p.add_argument("--arm",choices=("windows-only","jitter-only","combined"),default="windows-only")
    p.add_argument("--seed",type=int,choices=(20261102,20261103,20261104),default=20261102)
    p.add_argument("--ranking-weight",type=float,choices=(0.,.5),default=0.)
    p.add_argument("--deepsets",action="store_true")
    p.add_argument("--device",default="cpu")
    p.add_argument("--window-audit",action="store_true",help="Also run all four perturbation conditions on full-mask windows")
    a=p.parse_args()
    if a.reuse_oof_root and a.arm!="windows-only":
        p.error("Existing clean encoders can only be reused for windows-only")
    if not a.reuse_oof_root and not a.cache_dir:
        p.error("Provide --cache-dir for new training, or --reuse-oof-root for windows-only")
    variants=("conv3d_deepsets_sep_structure","anza3d_deepsets_sep_structure") if a.deepsets else ("conv3d_sep_structure","anza3d_sep_structure")
    jitter=.8 if a.arm in ("jitter-only","combined") else 0.
    windows=a.arm in ("windows-only","combined")
    frozen=verify_validation_freeze(a.freeze)
    for patient in sorted(VAL_CT28_IDS):
        check_val_pilot(pilot_dir=a.pilot_root/f"patient_{patient}",patient=patient,
                        freeze_path=a.freeze,frozen=frozen)
    study=a.root/"study"
    validation=a.root/"validation"
    receipt=a.root/"experiment.json"
    config={"arm":a.arm,"seed":a.seed,"variants":list(variants),
            "ranking_weight":a.ranking_weight,"endpoint_jitter_mm":jitter,
            "runtime_window_training":windows,"freeze_sha256":sha(a.freeze),
            "device":a.device,"epochs":8,"folds":5,"batch_size":2,
            "window_audit":bool(a.window_audit),
            "cache_dir":str(a.cache_dir.resolve()) if a.cache_dir else None,
            "reuse_oof_root":str(a.reuse_oof_root.resolve()) if a.reuse_oof_root else None,
            "pilot_root":str(a.pilot_root.resolve()),"held_out_test_accessed":False}
    a.root.mkdir(parents=True,exist_ok=True)
    if receipt.exists():
        if read(receipt)!=config:
            raise ValueError("Cannot resume with a changed experiment configuration")
    else:
        if any(a.root.iterdir()):
            raise ValueError("Choose a new root; never overwrite historical evidence")
        with receipt.open("x",encoding="utf-8") as f:
            json.dump(config,f,indent=2)
    study.mkdir(exist_ok=True)
    if a.reuse_oof_root:
        reuse_oof(a.reuse_oof_root,study,variants,a.seed,a.ranking_weight)
    else:
        ensure_oof(a.cache_dir,study,epochs=8,device=a.device,folds=5,batch_size=2,
                   frame_mode="fixed_lps",seed=a.seed,variants=variants,endpoint_encoding="separate",
                   ranking_weight=a.ranking_weight,endpoint_jitter_mm=jitter)
    verify_upstream_ranking_protocol(study,variants,a.ranking_weight,jitter)
    ensure_relations(study,mode="augmented_15",device=a.device,variants=variants,
                     runtime_window_training=windows)
    matched=validation/"matched_val_comparison.json"
    if not matched.exists():
        cmd=[sys.executable,str(HERE/"ccta_3d_val_an_compare.py"),
             "--run","--structure-models","--separate-endpoints",
             "--freeze",str(a.freeze),"--pilot-root",str(a.pilot_root),
             "--study-root",str(study),"--output-root",str(validation),
             "--device",a.device,"--pair-identity-source","neural_pair_junction",
             "--relation-training","augmented_15"]
        if a.deepsets:
            cmd.append("--deepsets")
        subprocess.run(cmd,check=True)
    summary=read(matched)
    expected={"ranking_weight":a.ranking_weight,"endpoint_jitter_mm":jitter,
              "runtime_window_training":windows}
    if (summary["status"]!="complete_five_patient_matched_CCTA_dev_val" or
        summary.get("training_signature")!=expected or summary["variant_pair"]!=list(variants) or
        summary["protocol_sha256"]!=config["freeze_sha256"]):
        raise ValueError("Incomplete or mismatched validation evidence")
    audit=a.root/"repair_audit.json"
    if not audit.exists():
        cmd=[sys.executable,str(HERE/"audit_ct28_runtime_repair.py"),
             "--freeze",str(a.freeze),"--pilot-root",str(a.pilot_root),
             "--validation-root",str(validation),"--output",str(audit),"--device",a.device]
        if a.window_audit:
            cmd.extend(["--study-root",str(study)])
        if a.deepsets:
            cmd.append("--deepsets")
        subprocess.run(cmd,check=True)
    report=read(audit)
    print(json.dumps({"arm":a.arm,"matched_report":str(matched),"audit":str(audit),
        "oracle_restored":report["restored_with_oracle"],"total_gaps":report["total_gaps"],
        "actual_repair_totals":report["actual_repair_totals"],
        "held_out_test_accessed":False},indent=2))

if __name__=="__main__":
    main()
