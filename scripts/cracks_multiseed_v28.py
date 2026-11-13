#!/usr/bin/env python3
"""V28 exploratory repeated-training real CRACKS: matched Conv2d vs real AZConv2d.

Models see crowd certain classes only; held-out expert section masks are opened
after all predicted segmentations, links and paths for that section. The 40 expert
slices are from one previously inspected F3 cube: not external confirmation.
No historical ANZA/LIRA checkpoint, natural fault instance labels or physical
false-link FPR is asserted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.cracks_real_neural_v26 import (
    CompactDetector, sample_batch, masked_binary_loss, dense_probs, calibrate,
    crowd_records, TRAIN_STEPS, TRAIN_LR, TRAIN_IDS, CAL_IDS
)
from scripts.cracks_real_auto_run_v23 import (
    zip_index, load_image, load_mask, _train, evaluate_mask, sha_file, _overlay
)
from scripts.cracks_real_auto_core_v23 import anisotropic_features
from scripts.cracks_real_region_v25 import run_regions

SEEDS = (11, 23, 37)
METHODS = ("conv", "az")
POLICIES = ("raw", "region_hybrid")


def fit_seed(variant, seed, records, out):
    """Equal 180-step budgets and identical patch RNG stream for both variants."""
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    net = CompactDetector(variant)
    optimizer = torch.optim.Adam(net.parameters(), lr=TRAIN_LR, weight_decay=1e-5)
    losses = []
    net.train()
    for step in range(TRAIN_STEPS):
        for attempt in range(5):
            x, y, known = sample_batch(records, rng)
            z = net(x)
            try:
                loss = masked_binary_loss(z, y, known)
            except ValueError:
                continue
            break
        else:
            raise RuntimeError("five sampled batches without both explicit classes")
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite objective for " + variant)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2.)
        optimizer.step()
        if step % 30 == 0 or step == TRAIN_STEPS - 1:
            losses.append({"model": variant, "seed": seed, "step": step,
                           "loss": float(loss.detach()),
                           "known": int((known > 0).sum())})
    net.eval()
    ckpt = out / f"{variant}_seed{seed}.pt"
    torch.save({"architecture": variant, "seed": seed, "steps": TRAIN_STEPS,
                "state_dict": net.state_dict()}, ckpt)
    return net, losses, sha_file(ckpt)


def summarize(rows, method, seed, policy):
    subset = [r for r in rows
              if r["method"] == method and r["seed"] == seed
              and r["policy"] == policy]
    if len(subset) != 40:
        raise AssertionError(f"incomplete evaluation {method} {seed} {policy}")
    def total(k): return sum(int(r[k]) for r in subset)
    blue = total("n_blue")
    orange = total("n_orange")
    positive = total("connected")
    eligible = total("eligible")
    return {
        "sections": len(subset), "eligible_paths": eligible,
        "connected": positive,
        "connectivity": positive / eligible,
        "blue_recall_tol2": total("blue_recalled") / blue,
        "orange_false_fraction": total("orange_false") / orange,
        "accepted_links": total("links"),
        "N_blue": blue, "N_orange": orange
    }


def section_bootstrap(rows, seed, policy="region_hybrid", repeats=3000):
    selected = [r for r in rows if r["seed"] == seed and r["policy"] == policy]
    by = {(r["section"], r["method"]): r for r in selected}
    section_ids = sorted({r["section"] for r in selected})
    if len(section_ids) != 40:
        raise AssertionError("not 40 sections")
    numerator = np.array([
        int(by[(s, "az")]["connected"]) - int(by[(s, "conv")]["connected"])
        for s in section_ids], float)
    denom = np.array([int(by[(s, "az")]["eligible"])
                      for s in section_ids], float)
    if any(int(by[(s,"az")]["eligible"]) != int(by[(s,"conv")]["eligible"])
           for s in section_ids):
        raise AssertionError("architecture comparison uses different anchor denominator")
    rng = np.random.default_rng(100000 + seed)
    choice = rng.integers(0, len(section_ids), size=(repeats, len(section_ids)))
    changes = numerator[choice].sum(axis=1)
    counts = denom[choice].sum(axis=1)
    samples = np.divide(changes, counts, out=np.zeros_like(changes),
                        where=counts>0)
    return {"difference": float(numerator.sum()/denom.sum()),
            "descriptive_interval95": np.quantile(samples, [.025,.975]).tolist(),
            "resample_unit": "F3_section", "N_sections": 40,
            "not_external_or_prospective": True}


def experiment(images_path, annotations_path, output):
    torch.set_num_threads(2)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    all_rows, training, calibration, checkpoints = [], [], {}, {}
    with zipfile.ZipFile(images_path) as zi, zipfile.ZipFile(annotations_path) as za:
        images, annotations = zip_index(zi, za)
        train = crowd_records(zi, za, images, annotations, TRAIN_IDS)
        cal = crowd_records(zi, za, images, annotations, CAL_IDS)
        if len(train)<20 or len(cal)<4:
            raise RuntimeError("insufficient crowd TRAIN/CAL sections")
        expert_ids = sorted(s for s in images if s in annotations
                            and "expert" in annotations[s])
        if len(expert_ids) != 40:
            raise RuntimeError(f"expected 40 expert sections, found {len(expert_ids)}")
        if set(expert_ids) & (set(r["section"] for r in train+cal)):
            raise RuntimeError("expert leakage into TRAIN/CAL")

        for seed in SEEDS:
            models={}
            for method in METHODS:
                model, losses, sha = fit_seed(method, seed, train, output)
                tau, curve = calibrate(model, cal)
                models[method] = (model, tau)
                training.extend(losses)
                calibration[f"{method}:{seed}"]={"threshold":tau,"curve":curve}
                checkpoints[f"{method}:{seed}"]=sha
                print("V28 TRAIN",method,"seed",seed,"tau",tau,flush=True)
            for idx, sid in enumerate(expert_ids):
                im = load_image(zi, images[sid])
                inferred = {}
                for method in METHODS:
                    model, tau = models[method]
                    p = dense_probs(model, im)
                    inferred[method] = run_regions(p, tau, mode="region_hybrid")
                # Expert annotations are not passed to prediction or ranking.
                expert = load_mask(za, annotations[sid]["expert"])
                for method in METHODS:
                    pred = inferred[method]
                    for policy in POLICIES:
                        chosen = pred["raw_mask"] if policy=="raw" else pred["completed_mask"]
                        m = evaluate_mask(chosen, expert)
                        all_rows.append({
                            "section":sid,"seed":seed,"method":method,"policy":policy,
                            "n_blue":m["N_expert_blue"],
                            "n_orange":m["N_explicit_orange"],
                            "blue_recalled":m["expert_blue_recalled_tol2"],
                            "orange_false":m["explicit_orange_false_pixels"],
                            "eligible":m["eligible_paths"],
                            "both_visible":m["both_gold_anchors_detected"],
                            "connected":m["connected_gold_anchor_pairs"],
                            "links":0 if policy=="raw" else len(pred["selected"]),
                        })
                    if idx == 0:
                        _overlay(im, pred["raw_mask"], pred["completed_mask"],
                                 output/f"{method}_seed{seed}_section_{sid:03d}.png")
                if idx%10==0:
                    print("V28 TEST",seed,sid,flush=True)

    with (output/"per_section_v28.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader();writer.writerows(all_rows)
    with (output/"training_v28.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(training[0]))
        writer.writeheader();writer.writerows(training)
    by_seed={str(seed):{method:{policy:summarize(all_rows,method,seed,policy)
                                 for policy in POLICIES} for method in METHODS}
             for seed in SEEDS}
    bootstraps={str(seed):section_bootstrap(all_rows,seed) for seed in SEEDS}
    report={
        "status":"EXECUTED_EXPLORATORY_IF_ALL_FITS_COMPLETE",
        "protocol":"CRACKS_V28_MATCHED_AZCONV_CONV_THREE_SEEDS",
        "actual_repository_AZConv2d":True,
        "original_historical_ANZA_LIRA_checkpoints":False,
        "data": "CRACKS v2; one previously inspected F3 cube",
        "external_confirmatory_test": False,
        "physical_fault_instance_labels_available": False,
        "expert_masks_used_in_prediction_or_threshold_selection": False,
        "training_only_crowd_annotations":True,
        "total_neural_fits":len(SEEDS)*len(METHODS),
        "seed_values":list(SEEDS),
        "full_train_steps_per_fit":TRAIN_STEPS,
        "thresholds_from_crowd_cal_only":calibration,
        "checkpoint_sha256":checkpoints,
        "image_zip_sha256":sha_file(images_path),
        "annotation_zip_sha256":sha_file(annotations_path),
        "per_section_csv_sha256":sha_file(output/"per_section_v28.csv"),
        "results":by_seed,
        "paired_section_bootstraps_descriptive_only":bootstraps,
        "interpretation":"seed sensitivity on the same already-opened evaluation images; no claims of geological physical identity or external generalization."
    }
    (output/"report_v28.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"results":by_seed,"paired":bootstraps},
                     ensure_ascii=False,indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",required=True,type=Path)
    p.add_argument("--annotations",required=True,type=Path)
    p.add_argument("--out",type=Path,default=Path("results/cracks_multiseed_v28"))
    a=p.parse_args()
    experiment(a.images,a.annotations,a.out)

if __name__=="__main__":
    main()
