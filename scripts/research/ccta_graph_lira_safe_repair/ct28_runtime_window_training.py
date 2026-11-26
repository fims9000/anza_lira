"""Train-only domain adaptation: use the actual five-tip full-mask windows.

All 84 features are scored by patient-held-out local models. Reference masks
are used only for controlled gap generation and conservative labels. No dev
or held-out CT is opened here; the existing frozen inference policy is intact.
"""
from __future__ import annotations

from collections import Counter
import numpy as np
import pandas as pd

import run_ct28_graph_lira as graph
import extract_ct28_scene_candidate_features as ct_features
from ccta_3d_an_patch_cache import load_verified_training_or_validation_ct, SEGMENTATIONS
from ccta_3d_structure_targets import load_aligned_train_reference
from ccta_3d_train_gap_simulator import TRAIN_CT28_IDS, synthesize_train_gaps
from ccta_3d_endpoints import extract_endpoints, build_candidate_scenes
from ccta_3d_k10 import K10Inference
from ct28_perturbations import perturb_scene_endpoints
from ct28_runtime_gap_supervision import gap_pair_labels, window_labels

PROTOCOL = "runtime_windows_train_oof_pair_none_v1"
DAMAGE_SEED = 20261102
PERTURBATION_SEED = 20260920
REPEATS = 15


def build_runtime_train_rows(*, geometry_folds, transform, archive_dir,
                             columns, historical_weight=1036.0):
    if set(geometry_folds) != TRAIN_CT28_IDS:
        raise ValueError("Runtime window training requires exactly the 17 train patients")
    if not np.isfinite(historical_weight) or historical_weight <= 0:
        raise ValueError("Invalid training domain weight")
    by_patient = {}
    audit = []
    arm_names = ct_features.arm_feature_names(ct_features.RADII_MM)
    for patient in sorted(TRAIN_CT28_IDS):
        ct, affine = load_verified_training_or_validation_ct(patient, archive_dir)
        reference, ref_sha = load_aligned_train_reference(
            patient, ct.shape, affine, SEGMENTATIONS, TRAIN_CT28_IDS)
        record = {"patient_id": patient, "reference_sha256": ref_sha}
        try:
            broken, gaps = synthesize_train_gaps(reference, affine, patient_id=patient,
                        gaps=3, radius_mm=1.5, seed=DAMAGE_SEED)
        except ValueError as exc:
            audit.append(dict(record, status="gap_generation_failed", detail=str(exc)))
            continue
        endpoints, _ = extract_endpoints(broken, affine, min_component_voxels=12,
                                        trace_mm=3., max_endpoints=256)
        scenes = build_candidate_scenes(endpoints, patient_label=f"TRAIN_ADAPT:{patient}",
                                        max_scenes=128)
        gap_records = gap_pair_labels(reference, broken, affine, gaps, endpoints, scenes)
        labels, skipped = window_labels(scenes, gap_records)
        record.update(status="labelled", gap_statuses=dict(Counter(g["status"] for g in gap_records)),
                      windows=len(scenes), labels=dict(Counter(labels.values())),
                      skipped=dict(Counter(skipped.values())), gaps=gap_records)
        audit.append(record)
        if not labels:
            continue
        frames = []
        for scene in scenes:
            if scene.scene_id not in labels:
                continue
            plan, tips, candidates = K10Inference._frames(scene, endpoints)
            plan["scan_id"] = patient
            plan["split"] = "train"
            plan["scene_family"] = "runtime_synthetic_pair_windows"
            plan["relation_class"] = labels[scene.scene_id]
            frames.append((plan,tips,candidates))
        plans = pd.concat([x[0] for x in frames], ignore_index=True)
        candidates = pd.concat([x[2] for x in frames], ignore_index=True)
        inverse = np.linalg.inv(affine)
        rows = []
        for repeat in range(-1, REPEATS):
            tip_parts, feature_parts = [], []
            for _, tips, scene_candidates in frames:
                noisy = tips if repeat < 0 else perturb_scene_endpoints(
                    tips, seed=PERTURBATION_SEED, repetition=repeat,
                    position_noise_mm=.8, tangent_noise_deg=20.)
                tip_parts.append(noisy)
                lookup = noisy.set_index("endpoint_id")
                feature_parts.append(pd.DataFrame([
                    graph.perturbed_candidate_features(c, lookup, ct, inverse, arm_names)
                    for _,c in scene_candidates.iterrows()]))
            all_tips = pd.concat(tip_parts, ignore_index=True)
            features = pd.concat(feature_parts, ignore_index=True)
            local = K10Inference._score_sparse_candidate_types(features, geometry_folds[patient])
            scores = transform(patient, max(repeat,0), all_tips, candidates, ct, inverse, local)
            feature_rows, actual_columns = graph.relation_feature_rows(
                plans, candidates, scores, ("geometry_score","ct_score"))
            if list(actual_columns) != list(columns) or len(feature_rows) != len(labels):
                raise ValueError("Runtime windows do not match the frozen 84-feature schema")
            for row in feature_rows:
                row["augmentation_index"] = repeat+1
                rows.append(row)
        if len(rows) != len(labels)*(REPEATS+1):
            raise ValueError("Incomplete clean/perturbed runtime scene set")
        by_patient[patient] = rows
        print(f"Train runtime windows {patient}: {dict(Counter(labels.values()))}", flush=True)
    positives = [r for r in audit if r.get("labels",{}).get("PAIR",0)]
    positive_count = sum(r.get("labels",{}).get("PAIR",0) for r in audit)
    negative_count = sum(r.get("labels",{}).get("NONE",0) for r in audit)
    if len(positives) < 8 or min(positive_count,negative_count) < 15:
        raise ValueError(f"Insufficient reliable runtime window supervision: {len(positives)} positive patients, {positive_count} PAIR / {negative_count} NONE; audit={audit}")
    # Equal total weight for the two train domains; equal patient weight
    # within the added domain, then equal window weight within each patient.
    output = []
    for patient, rows in sorted(by_patient.items()):
        n = len(rows)//(REPEATS+1)
        weight = historical_weight/(len(by_patient)*n)
        for row in rows:
            row["sample_weight"] = weight*(.5 if row["augmentation_index"]==0 else .5/REPEATS)
            output.append(row)
    if not np.isclose(sum(r["sample_weight"] for r in output), historical_weight):
        raise ValueError("Runtime domain sample weights do not match predeclared budget")
    return output, {"protocol":PROTOCOL, "train_patients":sorted(TRAIN_CT28_IDS),
                    "contributing_patients":sorted(by_patient),
                    "positive_patients":len(positives),
                    "positive_windows":positive_count,"negative_windows":negative_count,
                    "rows":len(output),"total_sample_weight":historical_weight,
                    "damage_seed":DAMAGE_SEED,"perturbation_seed":PERTURBATION_SEED,
                    "repeats":REPEATS,"patient_audit":audit,
                    "held_out_test_accessed":False}
