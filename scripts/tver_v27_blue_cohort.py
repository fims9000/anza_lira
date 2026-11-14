"""V27: fixed BLUE-positive source-availability cohort for real CRACKS.

V26 discovered that the last official expert sections have *zero* BLUE
confident fault pixels. V27 uses a label-availability-conditioned cohort:
only expert sections with at least one unbranched >=80-pixel BLUE trace
are eligible for link recovery. This is an *exploratory post-source-audit*
selection, never a fresh or external confirmatory benchmark.

The exact selection rule below is frozen before opening any V27 learned
model output; it never consults segmentation scores or LIRA predictions.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from scripts.cracks_real_oracle_gap_audit import (
    sections,COLORS,paths_from_mask,
)


def _rgb(z,p):
    with Image.open(io.BytesIO(z.read(p))) as im:
        return np.asarray(im.convert("RGB"),np.uint8)


def blue_available_sections(image_archive,annotation_archive):
    with zipfile.ZipFile(image_archive) as zi,zipfile.ZipFile(annotation_archive) as za:
        images=sections(zi)
        masks=sections(za,expert=True)
        ids=sorted(set(images)&set(masks))
        if len(ids)!=40:raise RuntimeError("V27 expected exact 40 original expert filenames")
        row=[]
        for sid in ids:
            rgb=_rgb(za,masks[sid])
            blue=np.all(rgb==COLORS["blue"],axis=-1)
            green=np.all(rgb==COLORS["green"],axis=-1)
            clean=blue&~ndi.binary_dilation(green,iterations=5)
            paths=sum(len(p)>=80 for p in paths_from_mask(clean))
            row.append({
                "section":sid,
                "blue_pixels":int(blue.sum()),
                "positive_trace_count_length_ge80":int(paths),
                "eligible":bool(paths>0),
            })
        return row


def frozen_positive_split(rows):
    ids=[int(r["section"]) for r in rows if r["eligible"]]
    if ids!=sorted(ids) or len(set(ids))!=len(ids):
        raise RuntimeError("Noncanonical BLUE candidate order")
    # Minimum exact disjoint annotated source support: five train,
    # two CAL, one buffered eligible section, two TEST. Do not weaken
    # this after observing masks or model outcomes. Return explicit STOP.
    if len(ids)<10:
        return {},{
            "status":"STOP_INSUFFICIENT_BLUE_SECTIONS_FOR_TRAIN_CAL_SPATIAL_TEST",
            "n_eligible":len(ids),"eligible_section_ids":ids,
            "minimum_eligible":10,
            "no_new_model_outcome_evaluated":True}
    n_train=len(ids)-5
    groups=("TRAIN",)*n_train+("CAL",)*2+("BUFFER",)+("TEST",)*2
    split=dict(zip(ids,groups))
    cal=[sid for sid in ids if split[sid]=="CAL"]
    test=[sid for sid in ids if split[sid]=="TEST"]
    receipt={
        "status":"V27_FROZEN_BLUE_AVAILABILITY_SOURCE_SPLIT",
        "source_conditioned_selection_after_V26_zero_BLUE_TEST":True,
        "not_unseen_confirmatory_test":True,
        "uses_only_expert_BLUE_path_length_for_section_eligibility":True,
        "no_segmentation_or_relation_score_used_to_select_sections":True,
        "n_eligible":len(ids),"eligible_section_ids":ids,
        "TRAIN_sections":[x for x in ids if split[x]=="TRAIN"],
        "CAL_sections":cal,
        "BUFFER_sections":[x for x in ids if split[x]=="BUFFER"],
        "TEST_sections":test,
        "TEST_min_minus_CAL_max_F3_sections":int(min(test)-max(cal)),
        "min_num_source_sections_by_split":{"TRAIN":5,"CAL":2,"BUFFER":1,"TEST":2},
    }
    if receipt["TEST_min_minus_CAL_max_F3_sections"]<3:
        receipt["status"]="STOP_INSUFFICIENT_SPATIAL_BUFFER"
        return {},receipt
    return split,receipt


def inspect(images,annotations,out):
    all_sections=blue_available_sections(images,annotations)
    mapping,receipt=frozen_positive_split(all_sections)
    report={
        "status":receipt["status"],
        "cohort":receipt,
        "all_original_expert_section_blue_availability":all_sections,
        "source_is_expert_annotation_for_AVAILABILITY_only":True,
        "no_learned_ANZA_or_LIRA_TEST_scores_seen_to_choose_split":True,
    }
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--images",type=Path,required=True)
    p.add_argument("--annotations",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    inspect(a.images,a.annotations,a.out)
