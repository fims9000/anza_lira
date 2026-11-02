"""Train-only 5-endpoint scene cardinality shift diagnostic for CCTA runtime.

Reads only *training* candidate counts from the committed controlled CT28
scene plan. Never reads relation labels, validation features, CT or test data.
This cannot prove in-distribution inference: it only detects a simple support
mismatch that is especially important for the historical count-sensitive HGB.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Sequence

from ccta_3d_endpoints import Scene


ROOT = Path(__file__).resolve().parents[3]
SCENE_PLAN = ROOT / "artifacts/varvara/ct28_graph_lira/scene_plan_train_val.csv"
PINNED_SCENE_SHA256 = "746e2d8e48c33ea281d6858a667189ae0ffb8106c60a7ef35935a6320e360871"


def training_count_support(path: Path = SCENE_PLAN, *, require_pinned: bool = True) -> Counter:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if require_pinned and digest != PINNED_SCENE_SHA256:
        raise ValueError("Controlled CT28 scene-plan hash changed; refusing shifted baseline")
    with path.open(newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        fields=set(reader.fieldnames or ())
        if not {"scene_id","split","scan_id","pair_candidate_count","junction_candidate_count"}<=fields:
            raise ValueError("Train count support table lacks frozen protocol columns")
        counts=Counter()
        patients=set()
        for row in reader:
            if row["split"] != "train":
                continue
            patients.add(int(row["scan_id"]))
            pair=int(row["pair_candidate_count"])
            junction=int(row["junction_candidate_count"])
            if min(pair,junction)<0:
                raise ValueError("Negative candidate counts")
            counts[(pair,junction)]+=1
    if not counts or len(patients)!=17:
        raise ValueError("Unexpected frozen CT28 train cohort for count support")
    return counts


def scene_counts(scene: Scene) -> tuple[int,int]:
    return (
        sum(c.kind=="PAIR" for c in scene.candidates),
        sum(c.kind=="JUNCTION" for c in scene.candidates),
    )


def describe_shift(scenes: Sequence[Scene], support: Counter) -> dict:
    seen=Counter()
    missing=[]
    for scene in scenes:
        pair,junction=scene_counts(scene)
        if support.get((pair,junction),0):
            seen[(pair,junction)]+=1
        else:
            missing.append({
                "scene_id":scene.scene_id,
                "pair_candidates":pair,
                "junction_candidates":junction,
            })
    return {
        "scope":"candidate cardinality support from CT28 train patients only",
        "train_scene_count":sum(support.values()),
        "distinct_train_count_patterns":len(support),
        "runtime_scenes":len(scenes),
        "supported_runtime_scenes":sum(seen.values()),
        "out_of_train_count_support":len(missing),
        "unsupported_scenes":missing,
        "warning":"Count support does not establish feature, anatomy, or label-distribution equivalence",
    }
