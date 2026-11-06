"""Compact, source-verified REAL H50Z original/photo and machine overlays.

Local source-only evidence export from SAVED genuine S86/S92/S95/S96.
No synthetic generated photo is ever passed off as real. Source pixels are
copied from SHA-checked original 8-bit grayscale BMP via read-only memmap.

Two separate zip deliverables:
  - ORIGINAL_ONLY reviewer zip: neutral target locus and original crops with
    S95 model class NOT revealed (but ROI *selection* is S95-class-stratified).
  - MACHINE_AFTER_REVIEW zip: overlay of source-observed S86 centerlines and
    S95 classes, plus full provenance receipt; should only be opened after
    first zip review. No made-up physical particle identities or F1.

These examples diagnose machine-selected marks, NOT unbiased population
accuracy. For independent scene-level recall use S99 source-only spatial ROI.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import csv
import html
import io
import json
import re

import cv2
import numpy as np

from .bitmap_tiles import GrayBMP8,sha256_file
from .real_bmp_run_receipts import verify_stage_receipts
from .s87_seam_continuity import _read_source
from .s92_curve_review import collapse_observed_components,_write_private,_json
from .s95_all_structure_types import PALETTE_BGR

SCHEMA="H50Z_REAL_MACHINE_OBSERVED_PHOTO_EVIDENCE_V1"
CATEGORIES=tuple(PALETTE_BGR)
HUMAN_CLASSES=(
    "TRACK_LIKE","COILING_TRACK_LIKE","FIDUCIAL_CROSS",
    "ARTIFACT","CROSSING_OR_BUNDLE","UNKNOWN"
)
EXPECTED={
    "TRACK_LIKE_CANDIDATE":"TRACK_LIKE",
    "COILING_TRACE_CANDIDATE":"COILING_TRACK_LIKE",
    "CALIBRATION_CROSS_CONTEXT":"FIDUCIAL_CROSS",
    "ARTIFACT_LIKE_CANDIDATE":"ARTIFACT",
    "BUNDLE_OR_INTERSECTION_REVIEW":"CROSSING_OR_BUNDLE",
    "UNKNOWN":"UNKNOWN",
}
# A machine "calibration cross context" is not an independently detected cross,
# so agreement on that label must be considered contextual, not object accuracy.
NO_FIDUCIAL_AUTODETECTION_CLAIM=True


def _validate_paths(out_dir,raw_paths):
    dest=Path(out_dir).resolve()
    repo=Path(__file__).resolve().parents[1]
    if dest.exists() or dest==repo or repo in dest.parents:
        raise ValueError("New PRIVATE H50Z evidence directory required outside Git")
    for raw in raw_paths:
        p=Path(raw).resolve()
        if dest==p or dest in p.parents or p in dest.parents:
            raise ValueError("H50Z evidence must not overwrite or contain inputs")
    return dest


def _sample_components(components,typemap,*,source_sha,max_per_class):
    if type(max_per_class) is not int or not 1<=max_per_class<=8:
        raise ValueError("Bounded 1..8 samples per machine S95 category required")
    if not isinstance(source_sha,str) or len(source_sha)!=64:
        raise ValueError("Original H50Z SHA required for reproducible sampling")
    by_class=defaultdict(list)
    for c in components:
        cid=c["component_id"]
        cls=typemap[cid]
        # Random-looking, deterministic ordering within machine class.
        # Does NOT claim an unbiased random sample of true physical tracks.
        order=sha256(f"{SCHEMA}|{source_sha}|{cls}|{cid}".encode()).hexdigest()
        by_class[cls].append((order,cid))
    sampled=[]
    for cls in CATEGORIES:
        for _,cid in sorted(by_class.get(cls,[]))[:max_per_class]:
            sampled.append({"component_id":cid,"machine_S95_category":cls})
    return sampled


def _sample_crop(bmp,component,crop_size):
    if (type(crop_size) is not int or crop_size%16
        or not 128<=crop_size<=1024):
        raise ValueError("H50Z native crop dimension must be 128..1024, 16-aligned")
    points=np.asarray(component["points_xy"],float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<2:
        raise ValueError("Cannot show non-source 2D centerline")
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite S86 observed source trace")
    # The midpoint locates a source-observed pixel, not an invented track join.
    center=points[len(points)//2]
    gray,loc=bmp.crop(tuple(center),size=crop_size)
    return gray,loc,center


def _overlaid_right(gray,location,selected,components,typemap):
    pic=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    x0,y0,x1,y1=location.original_xyxy
    origin=np.array([x0,y0],float)
    for c in components:
        xy=np.asarray(c["points_xy"],float)
        if (xy[:,0].max()<x0 or xy[:,0].min()>=x1
            or xy[:,1].max()<y0 or xy[:,1].min()>=y1):
            continue
        cls=typemap[c["component_id"]]
        coords=np.rint(xy-origin).astype(np.int32)
        cv2.polylines(pic,[coords.reshape(-1,1,2)],False,
                      (135,135,135),1,lineType=cv2.LINE_8)
    chosen=np.asarray(selected["points_xy"],float)
    pts=np.rint(chosen-origin).astype(np.int32)
    cv2.polylines(pic,[pts.reshape(-1,1,2)],False,
                  tuple(int(x) for x in PALETTE_BGR[typemap[selected["component_id"]]]),
                  2,lineType=cv2.LINE_AA)
    return pic


def _png(path,array):
    success,encoded=cv2.imencode(".png",array,[cv2.IMWRITE_PNG_COMPRESSION,3])
    if not success:raise OSError("Cannot encode source-exact H50Z PNG")
    _write_private(path,encoded.tobytes())
    return sha256_file(path)


def _csv(path,rows,fields):
    buff=io.StringIO(newline="")
    writer=csv.DictWriter(buff,fieldnames=fields,extrasaction="ignore",
                          lineterminator="\n")
    writer.writeheader()
    for row in rows:writer.writerow(row)
    _write_private(path,buff.getvalue().encode("utf-8"))


def _zip(out,files,subdir,*,limit_bytes=32*1024*1024):
    with ZipFile(out,"x",compression=ZIP_DEFLATED,compresslevel=6) as zf:
        for file in sorted(files):
            p=Path(file).resolve()
            rel=p.relative_to(subdir)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe native photo package member")
            zf.write(p,arcname=str(rel))
    out.chmod(0o600)
    if out.stat().st_size>limit_bytes:
        raise ValueError("Photo evidence package above 32 MB; select fewer examples")
    return {"file":out.name,"sha256":sha256_file(out),
            "compressed_bytes":out.stat().st_size}


def package_verified_real_h50z(
    *,bmp_path:Path,s86_dir:Path,s92_dir:Path,s95_dir:Path,
    s96_dir:Path|None,private_out_dir:Path,
    max_per_class:int=2,crop_size:int=384,
    case_id:str="H50Z_REAL",max_zip_bytes:int=32*1024*1024,
):
    """Create local private minimal evidence from an already completed run."""
    bmp_path=Path(bmp_path).resolve()
    s86_dir=Path(s86_dir).resolve()
    s92_dir=Path(s92_dir).resolve()
    s95_dir=Path(s95_dir).resolve()
    s96_dir=Path(s96_dir).resolve() if s96_dir else None
    out=_validate_paths(private_out_dir,[bmp_path,s86_dir,s92_dir,s95_dir]+(
        [s96_dir] if s96_dir else []
    ))
    # Read all saved source S86 mask receipts and actual photo bytes first.
    verified=verify_stage_receipts(
        case_name=case_id,s86_dir=s86_dir,s92_dir=s92_dir,s95_dir=s95_dir,
        s96_dir=s96_dir,original_bmp=bmp_path
    )
    original_sha=sha256_file(bmp_path)
    assert verified["original_bmp_sha256"]==original_sha
    manifest,records,_=_read_source(s86_dir)
    comps,_=collapse_observed_components(records)
    by_id={c["component_id"]:c for c in comps}
    typemap={}
    original_types=s95_dir/"all_source_fragment_types.ndjson"
    original_types_sha=sha256_file(original_types)
    with original_types.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line)
            ident=row["component_id"]
            typ=row["primary_review_category"]
            if ident in typemap or ident not in by_id or typ not in CATEGORIES:
                raise ValueError("S95 stored machine category inventory changed")
            typemap[ident]=typ
    if set(typemap)!=set(by_id):
        raise ValueError("Incomplete source S95 category inventory")
    samples=_sample_components(
        comps,typemap,source_sha=original_sha,max_per_class=max_per_class
    )
    bmp=GrayBMP8(bmp_path)
    if min(bmp.height,bmp.width)<crop_size:
        raise ValueError("Requested image crop exceeds original photo size")
    out.mkdir(parents=True,mode=0o700)
    out.chmod(0o700)
    blind=out/"01_blind_original_only"
    machine=out/"02_machine_after_human_review"
    blind.mkdir(mode=0o700)
    machine.mkdir(mode=0o700)
    (blind/"original").mkdir(mode=0o700)
    (machine/"overlays").mkdir(mode=0o700)
    records_out=[]
    for i,sample in enumerate(samples):
        cid=sample["component_id"]
        c=by_id[cid]
        gray,loc,mid=_sample_crop(bmp,c,crop_size)
        local_id=f"case_{i+1:03d}"
        # Entire source photo crop is original gray intensity values.
        original_file=blind/"original"/f"{local_id}.png"
        orig_sha=_png(original_file,gray)
        # Neutral coordinate locator, NOT the model's type or predicted path.
        # Experts should view the original source image *and* this red dot
        # without seeing machine color-coded detection.
        mark=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        x=int(round(mid[0]-loc.original_xyxy[0]))
        y=int(round(mid[1]-loc.original_xyxy[1]))
        cv2.circle(mark,(x,y),7,(190,190,190),1,lineType=cv2.LINE_8)
        neutral=blind/"original"/f"{local_id}_neutral_target.png"
        neutral_sha=_png(neutral,mark)

        # Machine interpretation is SEALED IN THE SEPARATE zip.
        colored=_overlaid_right(gray,loc,c,comps,typemap)
        side=np.concatenate((cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),colored),axis=1)
        overlay_file=machine/"overlays"/f"{local_id}_S95_source.png"
        overlay_sha=_png(overlay_file,side)
        records_out.append({
            "sample_id":local_id,
            "source_component_id":cid,
            "source_xyxy":list(loc.original_xyxy),
            "observed_source_midpoint_xy":np.round(mid,4).tolist(),
            "source_original_png_sha256":orig_sha,
            "neutral_location_png_sha256":neutral_sha,
            "posthoc_machine_overlay_png_sha256":overlay_sha,
            "machine_S95_category":typemap[cid],
            "original_photo_pixels_are_real_and_unmodified":True,
            "sampled_from_machine_class_not_population_random":True,
        })
    # Two visibly distinct packages: no machine category in blind review.
    _csv(blind/"review_template.csv",(
        {"sample_id":x["sample_id"],"human_label":"","certainty":"","note":""}
        for x in records_out
    ),["sample_id","human_label","certainty","note"])
    _csv(machine/"source_machine_classes.csv",(
        {"sample_id":x["sample_id"],
         "original_component_id":x["source_component_id"],
         "machine_category":x["machine_S95_category"],
         "original_photo_crop_xyxy":json.dumps(x["source_xyxy"])}
        for x in records_out
    ),["sample_id","original_component_id","machine_category",
       "original_photo_crop_xyxy"])
    html_parts=[
        "<!doctype html><html lang='ru'><meta charset='utf-8'>"
        "<title>H50Z независимая проверка выбранных машинных объектов</title>"
        "<style>body{font:14px system-ui,sans-serif;margin:20px;color:#1c2228}"
        "img{max-width:480px;border:1px solid #999}"
        "section{padding:12px 0;border-top:1px solid #ddd}</style>"
        "<h1>Слепая оценка исходных участков H50Z</h1>"
        "<p>Слева исходная фотография без изменений. Справа тот же исходный "
        "фрагмент с нейтральной окружностью, указывающей на объект для оценки. "
        "Машинный класс и центрлиния здесь намеренно НЕ показаны. "
        "Проверьте объект в области окружности и заполните review_template.csv: "
        "TRACK_LIKE, COILING_TRACK_LIKE, FIDUCIAL_CROSS, ARTIFACT, "
        "CROSSING_OR_BUNDLE, UNKNOWN; certainty CLEAR/UNCERTAIN. "
        "Нельзя считать этот выбор независимым от модели: сами участки "
        "отобраны по машинным классам для качественного аудита ошибок.</p>"
    ]
    for item in records_out:
        sid=item["sample_id"]
        html_parts.extend([
            "<section><h3>"+html.escape(sid)+"</h3>",
            f"<img src='original/{sid}.png' alt='exact source H50Z grayscale crop'/>",
            f"<img src='original/{sid}_neutral_target.png' alt='same source with neutral target marker'/>",
            "</section>"
        ])
    html_parts.append("</html>")
    _write_private(blind/"index.html","".join(html_parts).encode("utf-8"))
    machine_parts=[
        "<!doctype html><html lang='ru'><meta charset='utf-8'>"
        "<title>H50Z машинные гипотезы после слепой оценки</title>"
        "<style>body{font:14px system-ui;margin:20px}"
        "img{max-width:100%;height:auto;border:1px solid #ccc}"
        "section{margin:20px 0}</style>"
        "<h1>Машинные категории S95: открыть только ПОСЛЕ разметки</h1>"
        "<p>Слева исходные пиксели BMP. Справа наблюдаемая центрлиния "
        "выбранного S86 компонента, окрашенная согласно машинному классу. "
        "Это НЕ подтверждённые физические частицы, не экспертные метки "
        "и не оценка полноты всего кадра.</p>"
    ]
    for item in records_out:
        sid=item["sample_id"]
        machine_parts.extend([
            "<section><h3>"+html.escape(sid)+" — "+
            html.escape(item["machine_S95_category"])+"</h3>",
            f"<img src='overlays/{sid}_S95_source.png' alt='real source and machine hypotheses'/>",
            "</section>"
        ])
    machine_parts.append("</html>")
    _write_private(machine/"index.html","".join(machine_parts).encode("utf-8"))
    # NO original full BMP, model weights, or full candidate mask set in ZIP.
    report={
        "schema":SCHEMA,
        "status":"REAL_SOURCE_SHA_VERIFIED_MACHINE_SELECTED_DIAGNOSTIC_CROPS_NOT_GT",
        "case":case_id,"source_bmp_name":bmp_path.name,
        "source_bmp_sha256":original_sha,
        "source_shape_yx":[bmp.height,bmp.width],
        "verified_full_S86_S92_S95_S96_saved_stage_receipts":verified,
        "source_s95_types_sha256":original_types_sha,
        "crop_side_px":crop_size,"max_per_machine_category":max_per_class,
        "sampled_actual_machine_source_components":len(records_out),
        "per_machine_S95_category_count":dict(Counter(
            x["machine_S95_category"] for x in records_out
        )),
        "sampled_original_photo_component_cases":records_out,
        "blind_original_only_review":True,
        "machine_colorized_centerlines_after_human_review_only":True,
        "sampling_conditioned_on_S95_machine_class_not_unbiased_GT":True,
        "independent_exhaustive_negative_pixel_mask_not_available":True,
        "human_visual_labels_not_yet_present":True,
        "no_real_accuracy_precision_recall_F1_claim":True,
        "filmmaker_particle_type_or_collision_causality_not_verified":True,
        "full_original_BMP_not_shared_in_compact_evidence":True,
    }
    _write_private(machine/"verified_machine_only_report.json",_json(report))
    source_list=list(blind.rglob("*"))
    blind_files=[p for p in source_list if p.is_file()]
    machine_files=[p for p in machine.rglob("*") if p.is_file()]
    blindzip=out/"H50Z_BLIND_ORIGINAL_CROPS_PRIVATE.zip"
    afterzip=out/"H50Z_MACHINE_AFTER_REVIEW_PRIVATE.zip"
    archives={
        "blind":_zip(blindzip,blind_files,blind,limit_bytes=max_zip_bytes),
        "posthoc_machine":_zip(afterzip,machine_files,machine,limit_bytes=max_zip_bytes),
    }
    if (sha256_file(bmp_path)!=original_sha
        or sha256_file(original_types)!=original_types_sha):
        raise ValueError("Source original BMP/S95 class inventory changed during compact export")
    summary={
        "schema":SCHEMA,"status":"TWO_SEPARATE_VERIFIED_PRIVATE_H50Z_DIAGNOSTIC_ARCHIVES",
        "source_bmp_name":bmp_path.name,"source_bmp_sha256":original_sha,
        "original_verified_S86_S95_count":len(comps),
        "blind_source_examples":len(records_out),
        "archives":archives,
        "blind_review_csv":str(blind/"review_template.csv"),
        "do_not_open_posthoc_until_human_review_complete":True,
        "not_exhaustive_true_physics_GT":True,
    }
    _write_private(out/"compact_evidence_receipt.json",_json(summary))
    return summary


def evaluate_blind_component_cases(
    *,machine_report_path:Path,blind_review_csv:Path,private_out_json:Path,
):
    report= json.loads(Path(machine_report_path).read_text(encoding="utf-8"))
    if report.get("schema")!=SCHEMA or report.get("status")!=(
        "REAL_SOURCE_SHA_VERIFIED_MACHINE_SELECTED_DIAGNOSTIC_CROPS_NOT_GT"
    ):
        raise ValueError("Incorrect H50Z machine candidate reference package")
    items=report["sampled_original_photo_component_cases"]
    targets={x["sample_id"]:x for x in items}
    if len(targets)!=len(items) or not targets:
        raise ValueError("Empty or duplicated sample records")
    reviews={}
    with Path(blind_review_csv).open(encoding="utf-8-sig",newline="") as fd:
        for row in csv.DictReader(fd):
            sid=row.get("sample_id")
            if sid not in targets or sid in reviews:
                raise ValueError("Blind source review contains unknown/duplicate sample")
            human=row.get("human_label","").strip()
            certainty=row.get("certainty","").strip()
            note=row.get("note","")
            if human not in HUMAN_CLASSES or certainty not in ("CLEAR","UNCERTAIN"):
                raise ValueError("Every H50Z selected source object requires explicit independent visual label/certainty")
            if not isinstance(note,str) or len(note)>600:
                raise ValueError("Invalid review note")
            reviews[sid]={"human_visual_class":human,"certainty":certainty,
                          "reviewer_note":note}
    if set(reviews)!=set(targets):
        raise ValueError("Incomplete blind source object review; must cover every diagnostic case")
    confusion=defaultdict(Counter)
    accepted=abstained=contradictory=0
    rows=[]
    for sid,record in targets.items():
        h=reviews[sid]
        cls=record["machine_S95_category"]
        expected=EXPECTED[cls]
        if h["certainty"]=="UNCERTAIN" or h["human_visual_class"]=="UNKNOWN":
            decision="ABSTAIN_INDEPENDENT_HUMAN_UNCLEAR"
            abstained+=1
        elif h["human_visual_class"]==expected:
            decision="CONSISTENT_MACHINE_SELECTED_2D_APPEARANCE"
            accepted+=1
        else:
            decision="CONTRADICTS_EXPERT_MACHINE_SELECTED_2D_APPEARANCE"
            contradictory+=1
        confusion[cls][h["human_visual_class"]]+=1
        rows.append({
            "sample_id":sid,
            "machine_s95_category":cls,
            "independent_human_visible_class":h["human_visual_class"],
            "human_certainty":h["certainty"],
            "diagnostic_review_decision":decision,
            "human_source_note":h["reviewer_note"],
            "sample_conditioned_on_machine_candidate_class":True,
            "not_truth_of_particle_or_background":True,
        })
    r={
        "schema":SCHEMA,
        "status":"H50Z_MACHINE_SELECTED_PARTIAL_VISUAL_DIAGNOSTIC_REVIEW_ONLY",
        "original_source_bmp_sha256":report["source_bmp_sha256"],
        "sampled_machine_components":len(items),
        "independently_reviewed_machine_candidates":len(rows),
        "visual_consistent_cases":accepted,
        "visual_contradictory_cases":contradictory,
        "human_uncertain_cases_abstained":abstained,
        "descriptive_consistency_among_human_clear_selected_candidates":(
            accepted/(accepted+contradictory) if accepted+contradictory else None
        ),
        "selected_machine_category_to_expert_visual_class":{
            k:dict(v) for k,v in confusion.items()
        },
        "cases":rows,
        "not_unbiased_population_precision_F1_recall":True,
        "requires_independent_original_S99_ROIs_for_full_detection_assessment":True,
        "physical_track_identity_not_evaluated":True,
    }
    destination=Path(private_out_json).resolve()
    if destination.exists() or destination==Path(machine_report_path).resolve():
        raise ValueError("H50Z human review result must be new immutable JSON")
    repo=Path(__file__).resolve().parents[1]
    if destination==repo or repo in destination.parents:
        raise ValueError("Private reviewer response cannot be committed in Git")
    destination.parent.mkdir(parents=True,mode=0o700,exist_ok=True)
    with destination.open("x",encoding="utf-8") as fd:
        fd.write(_json(r).decode("utf-8"))
    destination.chmod(0o600)
    return r
