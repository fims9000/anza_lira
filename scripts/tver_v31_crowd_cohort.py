"""V31 source audit: crowd-supported continuous practitioner lines on new F3 ID bands.

This source-only preflight NEVER decodes TEST/B U F F E R practitioner pixels,
TEST image intensities or any expert annotation. It uses only TRAIN/CAL
practitioner BLUE paths and peer BLUE agreement. F3 numeric IDs are not
validated physical coordinates. All actual inferred SBPP ports later must
be segmentation-predicted, not these practitioner centerlines.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from scripts.cracks_real_oracle_gap_audit import COLORS,paths_from_mask,sections

TRAIN=(140,239)
CAL=(260,319)
TRAIN_CAL_GAP=(240,259)
BUFFER=(320,339)
TEST=(340,399)
SUPPORT_RADIUS_PX=3.
MIN_BLUE_PEERS=2
MIN_PATH_SUPPORTED_FRACTION=.60
MIN_PATH_LENGTH=80
MAX_SOURCE_PATHS_PER_SECTION=2
MIN_TRAIN_SOURCE_SECTIONS=15
MIN_CAL_SOURCE_SECTIONS=29
PROTOCOL="TV_V31_CROWD_PRACTITIONER_AGREEMENT_F3_NEW_NUMERIC_ID_BANDS"

def inside(section,bounds):
    return bounds[0]<=int(section)<=bounds[1]


def image_paths(zi):
    return sections(zi)


def practitioner_index(za):
    index=defaultdict(dict)
    for path in za.namelist():
        if path.endswith("/"):continue
        parts=path.split("/")
        if len(parts)<3:continue
        annotator=parts[-2].lower()
        match=re.fullmatch(r"section_(\d+)\.png",parts[-1],re.I)
        if match and re.fullmatch(r"practitioner\d+",annotator):
            sid=int(match.group(1))
            if annotator in index[sid]:
                raise RuntimeError("STOP duplicate practitioner section in original archive")
            index[sid][annotator]=path
    return {sid:dict(sorted(paths.items())) for sid,paths in sorted(index.items())}


def _rgb(za,path):
    with Image.open(io.BytesIO(za.read(path))) as im:
        rgb=np.asarray(im.convert("RGB"),dtype=np.uint8)
    if rgb.shape!=(255,701,3):
        raise RuntimeError("STOP unexpected real crowd source dimensions")
    known=np.zeros(rgb.shape[:2],dtype=bool)
    for color in COLORS.values():
        known|=np.all(rgb==color,axis=-1)
    if not np.all(known):
        raise RuntimeError("STOP invalid official practitioner annotation palette")
    return rgb


def crowd_practitioner_gaps(za,sid,index):
    """TRUE path only for supervised intervention/quality audit, never predictor.

    The strongest path comes from a single practitioner's unchanged BLUE mask,
    verified against two other practitioners within 3px tolerance. No merging
    or interpolation of independent annotators' disconnected lines.
    """
    paths=index.get(int(sid),{})
    if len(paths)<MIN_BLUE_PEERS+1:
        return [],{"section":int(sid),"practitioners":len(paths),
                   "accepted_paths":0,"reason":"INSUFFICIENT_PRACTITIONERS"}
    masks={annotator:_rgb(za,path) for annotator,path in paths.items()}
    # Support may only be another practitioner's certain BLUE pixels.
    blues={annotator:np.all(rgb==COLORS["blue"],axis=-1)
           for annotator,rgb in masks.items()}
    close_blue={name:ndi.distance_transform_edt(~blue)<=SUPPORT_RADIUS_PX
                for name,blue in blues.items()}
    greens={name:np.all(rgb==COLORS["green"],axis=-1)
            for name,rgb in masks.items()}
    raw_candidates=[]
    raw_unbranched_long_paths=0
    any_original_BLUE_pixels=sum(int(b.sum()) for b in blues.values())
    max_peer_supported_fraction_across_raw_paths=0.
    for name in sorted(blues):
        clean=blues[name]&~ndi.binary_dilation(greens[name],iterations=3)
        for trace_idx,original_path in enumerate(paths_from_mask(clean)):
            if len(original_path)<MIN_PATH_LENGTH:continue
            raw_unbranched_long_paths+=1
            trace=np.asarray(original_path,dtype=np.int32)
            peers=sum(close_blue[other][trace[:,0],trace[:,1]].astype(np.uint8)
                      for other in blues if other!=name)
            supported_fraction=float(np.mean(peers>=MIN_BLUE_PEERS))
            max_peer_supported_fraction_across_raw_paths=max(
                max_peer_supported_fraction_across_raw_paths,supported_fraction)
            if supported_fraction<MIN_PATH_SUPPORTED_FRACTION:continue
            raw_candidates.append({
                "annotator":name,"path_index":trace_idx,
                "supported_fraction":supported_fraction,
                "n_peers_confirmed_pixels":int(np.sum(peers>=MIN_BLUE_PEERS)),
                "path":trace,
            })
    raw_candidates.sort(key=lambda v:(-v["supported_fraction"],
                                       -len(v["path"]),v["annotator"],v["path_index"]))
    chosen=[];gaps=[]
    for item in raw_candidates:
        path=item["path"]
        if any(np.mean(ndi.distance_transform_edt(
            ~_rasterize(other["path"],blues[item["annotator"]].shape)
            )[path[:,0],path[:,1]]<=3.)>=.30 for other in chosen):
            continue
        gap=_path_gap(path,sid)
        if gap is None:continue
        # Reject a second candidate whose intervention rectangle would
        # interfere with an already selected gap in this source section.
        if any(_rect_overlap(gap["rectangle"],x["rectangle"]) for x in gaps):
            continue
        gap.update({
            "source_annotation_kind":"individually_drawn_practitioner_BLUE",
            "source_annotator":item["annotator"],
            "peer_agreement_radius_px":SUPPORT_RADIUS_PX,
            "at_least_two_other_practitioners_support_fraction":
                item["supported_fraction"],
            "nonexpert_identity_only_within_original_undivided_trace":True,
            "natural_geological_fault_identity_certified":False,
        })
        chosen.append(item);gaps.append(gap)
        if len(gaps)>=MAX_SOURCE_PATHS_PER_SECTION:break
    return gaps,{
        "section":int(sid),"practitioners":len(paths),
        "raw_traces_meeting_fixed_peer_rule":len(raw_candidates),
        "accepted_paths":len(gaps),
        "raw_unbranched_long_blue_paths":raw_unbranched_long_paths,
        "total_certain_BLUE_practitioner_pixels":any_original_BLUE_pixels,
        "max_peer_fraction_pre_admission":round(max_peer_supported_fraction_across_raw_paths,5),
        "accepted_lengths":[len(item["path"]) for item in chosen],
        "peer_support_fractions":[
            round(item["supported_fraction"],5) for item in chosen],
        "reason":("OK" if gaps else
                  "NO_INDIVIDUAL_UNBRANCHED_BLUE_PATH_LENGTH_GE80"
                  if not raw_unbranched_long_paths else
                  "NO_LONG_PATH_WITH_TWO_PEER_SUPPORT_OVER_60PCT"),
    }


def _rasterize(path,shape):
    mask=np.zeros(shape,dtype=bool)
    mask[path[:,0],path[:,1]]=True
    return mask


def _rect_overlap(a,b):
    return min(a[1],b[1])>max(a[0],b[0]) and min(a[3],b[3])>max(a[2],b[2])


def _path_gap(path,sid):
    mid=len(path)//2
    hidden=path[mid-6:mid+6]
    if len(hidden)!=12:return None
    y0=int(hidden[:,0].min()-3)
    y1=int(hidden[:,0].max()+4)
    x0=int(hidden[:,1].min()-3)
    x1=int(hidden[:,1].max()+4)
    if y0<9 or x0<9 or y1>=246 or x1>=692:
        return None
    inside=lambda p:y0<=p[0]<y1 and x0<=p[1]<x1
    first=next((p for p in reversed(path[:mid]) if not inside(p)),None)
    last=next((p for p in path[mid:] if not inside(p)),None)
    if first is None or last is None or np.linalg.norm(first-last)<6:
        return None
    return {
        "gap_id":f"crowd31:{sid}:{int(mid)}:{int(hidden[0,0])}:{int(hidden[0,1])}",
        "section":int(sid),"rectangle":(y0,y1,x0,x1),
        "true_ends":(tuple(map(float,first)),tuple(map(float,last))),
        "hidden_centerline":[tuple(map(int,p)) for p in hidden],
    }


def filenames_only_manifest(zi,za):
    ims=image_paths(zi)
    ann=practitioner_index(za)
    receipt={
        "protocol":PROTOCOL,
        "precommitted_numeric_bands":{
            "TRAIN":list(TRAIN),"CAL":list(CAL),
            "TRAIN_CAL_EXCLUSION":list(TRAIN_CAL_GAP),
            "BUFFER":list(BUFFER),"TEST":list(TEST)},
        "section_numeric_ids_not_proven_spatial_coordinates":True,
        "expert_contents_forbidden":True,
        "test_image_pixels_and_mask_pixels_not_decoded":True,
        "filename_counts":{
            name:{
                "image_section_files":sum(inside(sid,bounds) for sid in ims),
                "practitioner_section_filenames":sum(inside(sid,bounds) for sid in ann),
                "complete_image_and_three_practitioner_ids":[
                    sid for sid in sorted(set(ims)&set(ann))
                    if inside(sid,bounds) and len(ann[sid])>=3],
            }
            for name,bounds in (("TRAIN",TRAIN),("CAL",CAL),("BUFFER",BUFFER),("TEST",TEST))
        },
    }
    return ims,ann,receipt


def execute(images,annotations,out):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(images) as zi,zipfile.ZipFile(annotations) as za:
        ims,ann,receipt=filenames_only_manifest(zi,za)
        result={"protocol":PROTOCOL,"filename_inventory":receipt,
                "split_metrics":{},"no_TEST_or_BUFFER_decoding":True,
                "real_expert_annotations_not_decoded":True}
        for name,bounds in (("TRAIN",TRAIN),("CAL",CAL)):
            rows=[];eligible=0;gaps_total=0
            ids=[sid for sid in sorted(set(ims)&set(ann))
                 if inside(sid,bounds)]
            for sid in ids:
                gaps,row=crowd_practitioner_gaps(za,sid,ann)
                eligible+=bool(gaps);gaps_total+=len(gaps)
                rows.append(row)
            result["split_metrics"][name]={
                "n_image_and_practitioner_section_ids":len(ids),
                "n_eligible_distinct_source_sections":eligible,
                "n_peer_supported_artificial_gap_schedules":gaps_total,
                "sections":rows,
            }
        min_train=MIN_TRAIN_SOURCE_SECTIONS
        min_cal=MIN_CAL_SOURCE_SECTIONS
        ok=(result["split_metrics"]["TRAIN"]["n_eligible_distinct_source_sections"]>=min_train
            and result["split_metrics"]["CAL"]["n_eligible_distinct_source_sections"]>=min_cal)
        result["required_min_eligible_sections"]={"TRAIN":min_train,"CAL":min_cal}
        result["status"]=(
            "V31_NEW_CROWD_TRAIN_CAL_SOURCE_FEASIBLE_TEST_UNOPENED"
            if ok else
            "V31_STOP_INSUFFICIENT_CROWD_AGREEMENT_TRAIN_OR_CAL")
        result["next_action"]=(
            "Run separately frozen V31 predicted relation fit on this cohort then open TEST once"
            if ok else
            "No test use; collect independent region masks or expert source identity")
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "status":result["status"],
        "metrics":{k:{z:v for z,v in row.items() if z!="sections"}
                   for k,row in result["split_metrics"].items()},
        "reserved_test_section_filename_count":
            len(receipt["filename_counts"]["TEST"]["complete_image_and_three_practitioner_ids"]),
        "real_TEST_or_BUFFER_content_decoded":False,
        "real_expert_content_decoded":False,
    },indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--images",type=Path,required=True)
    parser.add_argument("--annotations",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    execute(args.images,args.annotations,args.out)
