"""Public-source safe audit of BMP crosses and separate sparse visualMetro tracks.

A start mark or fiducial cross is NOT a pixel segmentation or particle identity.
This module reads image headers and small ROIs without opening whole 200MB scans.
Sensitive filenames, original coordinates and source pixels remain PRIVATE.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import cv2

from .bitmap_tiles import GrayBMP8, BitmapFormatError
from .large_archive import audited_large_manifest

SCHEMA = "ANZA_LIRA_PHYSICISTS_TWO_ANNOTATION_INTAKE_V1"
MARKS = {"start", *(f"f{i}" for i in range(1, 9))}


def _validate_labels(path: Path):
    content = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(content, dict):
        raise ValueError("H50Z labels.json must map BMP names to metadata")
    for name, item in content.items():
        if (not isinstance(name, str) or Path(name).name != name or
                Path(name).suffix.lower() != ".bmp" or not isinstance(item, dict)):
            raise ValueError("Unsafe H50Z labels.json image name or record")
        points = item.get("keypoints")
        if not isinstance(points, list) or item.get("keypoints_count") != len(points):
            raise ValueError("H50Z keypoints_count mismatch")
        w, h = item.get("original_width"), item.get("original_height")
        if type(w) is not int or type(h) is not int or w <= 0 or h <= 0:
            raise ValueError("Invalid H50Z source dimensions")
        for point in points:
            if (not isinstance(point, dict) or point.get("label") not in MARKS or
                    not isinstance(point.get("x_px"), (int, float)) or
                    not isinstance(point.get("y_px"), (int, float)) or
                    not np.isfinite([point["x_px"], point["y_px"]]).all() or
                    not 0 <= point["x_px"] < w or not 0 <= point["y_px"] < h):
                raise ValueError("Invalid H50Z start/fiducial coordinate")
    return content


def inventory_bmps(bmp_root: Path, *, labels_json: Path | None = None, hash_bmps: bool = False):
    """Exact-name matching only. Ambiguous duplicated names cannot get labels."""
    if not bmp_root.is_dir():
        raise FileNotFoundError("BMP root is not an existing folder")
    labels = _validate_labels(labels_json) if labels_json is not None else {}
    files = defaultdict(list)
    for path in sorted(bmp_root.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".bmp":
            files[path.name].append(path)
    duplicates = {name: v for name, v in files.items() if len(v) > 1}
    accepted = []
    rejected = []
    status = Counter()
    for name in sorted(files):
        paths = files[name]
        if len(paths) != 1:
            status["ambiguous_bmp_names"] += len(paths)
            continue
        path = paths[0]
        try:
            image = GrayBMP8(path)
        except (BitmapFormatError, OSError, ValueError) as error:
            status["rejected_bmps"] += 1
            rejected.append({"relative_path": str(path.relative_to(bmp_root)),
                             "reason": type(error).__name__})
            continue
        label = labels.get(name)
        matched = label is not None
        if matched and (image.width, image.height) != (label["original_width"], label["original_height"]):
            raise ValueError("H50Z filename matches but BMP shape differs from labels.json: " + name)
        status["valid_bmps"] += 1
        status["matched_bmp_with_crosses"] += int(matched)
        status["valid_bmps_without_crosses"] += int(not matched)
        points = label["keypoints"] if matched else []
        for point in points:
            status["start_marks" if point["label"] == "start" else "fiducial_marks"] += 1
        record = {"relative_path": str(path.relative_to(bmp_root)), "basename": name,
                  "size_wh": [image.width, image.height], "matched_crosses": matched,
                  "marks": [{"label":x["label"],"xy_source_px":[x["x_px"],x["y_px"]]} for x in points]}
        if hash_bmps:
            hasher = sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    hasher.update(block)
            record["bmp_sha256"] = hasher.hexdigest()
        accepted.append(record)
    public = {
        "unique_names_found": len(files),
        "bmp_files_found": sum(map(len, files.values())),
        "ambiguous_names_count": len(duplicates),
        "cross_label_frames_described": len(labels),
        "cross_label_frames_missing_unambiguous_bmp": sum(len(files.get(k, [])) != 1 for k in labels),
        "valid_bmps": status["valid_bmps"],
        "invalid_bmps": status["rejected_bmps"],
        "bmp_files_with_duplicate_names": status["ambiguous_bmp_names"],
        "matched_bmp_with_crosses": status["matched_bmp_with_crosses"],
        "valid_bmps_without_crosses": status["valid_bmps_without_crosses"],
        "start_marks_on_matching_valid_bmps": status["start_marks"],
        "fiducial_marks_on_matching_valid_bmps": status["fiducial_marks"],
    }
    private = {"schema": SCHEMA+"_PRIVATE", "bmp_records": accepted,
               "ambiguous_filenames": sorted(duplicates), "rejected": rejected,
               "labels_json_sha256": sha256(labels_json.read_bytes()).hexdigest() if labels_json else None,
               "zero_certified_track_masks": True, "zero_certified_same_particle_links": True}
    return public, private


def check_manual_stereo_map(path: Path, bmp_root: Path, measured: dict):
    """Validate operator-declared film/view/BMP binding, not physical identity."""
    doc = json.loads(path.read_text(encoding="utf8"))
    if doc.get("schema") != "ANZA_LIRA_OPERATOR_STEREO_IMAGE_BINDINGS_V1":
        raise ValueError("Expected explicit operator stereo-map schema")
    entries = doc.get("bindings")
    if not isinstance(entries, list) or len(entries) > 10000:
        raise ValueError("Stereo bindings must be a bounded array")
    frames = {f["frame_id"]: f for f in measured["frames"]}
    base = bmp_root.resolve()
    seen = set()
    private = []
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("Invalid stereo map binding")
        frame = item.get("frame_id")
        if not isinstance(frame,str) or frame in seen or frame not in frames:
            raise ValueError("Duplicate, missing or unknown visualMetro frame identity")
        seen.add(frame)
        if item.get("operator_declares_same_physical_frame_and_view") is not True:
            raise ValueError("Image-to-measurement binding requires explicit operator declaration")
        if not isinstance(item.get("verified_by"),str) or not item["verified_by"].strip() or \
           not isinstance(item.get("physical_run_id"),str) or not item["physical_run_id"].strip():
            raise ValueError("Explicit operator and physical run identity required")
        views = item.get("views")
        if not isinstance(views,dict) or set(views) != {"1","2","3"}:
            raise ValueError("Need exactly 3 physical image views numbered 1,2,3")
        images = []
        view_records = {}
        for v in ("1","2","3"):
            rel = views[v]
            if not isinstance(rel,str) or Path(rel).is_absolute() or ".." in Path(rel).parts or \
               Path(rel).suffix.lower() != ".bmp":
                raise ValueError("Stereo image path must be a relative BMP within --bmp-root")
            full = (base/rel).resolve()
            if not full.is_relative_to(base) or not full.is_file():
                raise ValueError("Missing or escaping stereo BMP path")
            images.append(full)
            bmp = GrayBMP8(full)
            for track in frames[frame]["tracks"]:
                coords = np.asarray(track["measured_pixel_xy"][v], dtype=float)
                if coords.ndim != 2 or coords.shape[1] != 2 or np.any(coords < 0) or \
                   np.any(coords[:,0] >= bmp.width) or np.any(coords[:,1] >= bmp.height):
                    raise ValueError("Measured track clicks exceed manually paired image boundaries")
            for xy in frames[frame]["fiducials"]["view_pixel_xy"][v].values():
                if not (0 <= xy[0] < bmp.width and 0 <= xy[1] < bmp.height):
                    raise ValueError("Measured fiducial exceeds paired BMP boundaries")
            view_records[v] = {"relative_path":rel,"dimensions_wh":[bmp.width,bmp.height]}
        if len(set(images)) != 3:
            raise ValueError("Three stereo views cannot refer to the same BMP")
        private.append({"frame_id":frame,"physical_run_id":item["physical_run_id"],
                        "verified_by":item["verified_by"],"views":view_records,
                        "identity_proof":"OPERATOR_DECLARATION_ONLY_NOT_INDEPENDENT_PHYSICS_GROUND_TRUTH"})
    public = {"operator_declared_paired_stereo_frames":len(private),
              "operator_declared_distinct_run_names":len({x["physical_run_id"] for x in private}),
              "automatically_inferred_stereo_bindings":0,
              "pixel_bounds_and_fiducial_bounds_checked":True,
              "verified_physical_track_pixel_masks":0,
              "calibration_and_camera_projection_validated":False}
    return public,private


def inventory_sources(*,bmp_root:Path|None=None,crosses:Path|None=None,visualmetro:Path|None=None,
                      stereo_map:Path|None=None,hash_bmps:bool=False):
    if not (bmp_root or visualmetro):
        raise ValueError("Specify at least --bmp-root or --visualmetro-root")
    if crosses is not None and bmp_root is None:
        raise ValueError("H50Z cross labels require an explicitly named BMP root")
    if stereo_map is not None and (visualmetro is None or bmp_root is None):
        raise ValueError("Explicit stereo map needs both BMP and visualMetro roots")
    report = {"schema":SCHEMA,"privacy":"AGGREGATE_ONLY",
              "H50Z_crosses_are_particle_masks":False,
              "visualmetro_point_clicks_are_dense_pixel_masks":False,
              "auto_paired_070_with_H50Z":False,"native_physics_GT_count":0,
              "independent_physics_runs_verified":0,
              "physical_auto_joins_authorized":False,
              "allowed":"machine-only local review; expert confirmation needed"}
    private = {"schema":SCHEMA+"_PRIVATE","privacy":"PRIVATE_DO_NOT_UPLOAD",
               "H50Z":None,"visualmetro":None}
    if bmp_root is not None:
        public,secret = inventory_bmps(bmp_root,labels_json=crosses,hash_bmps=hash_bmps)
        report["bmp_crosses"] = public
        private["H50Z"] = secret
    if visualmetro is not None:
        data = audited_large_manifest(visualmetro)
        summary = data["summary"]
        report["visualmetro_sparse"] = {k:summary[k] for k in (
            "frame_folders_found","frames_complete_used","frames_quarantined",
            "tracks_used","spatial_points_used","stereo_pixel_measurements_used",
            "legacy_nonstandard_track_ids","matched_bmp_count")}
        report["visualmetro_sparse"]["film_count_in_valid_frames"] = len(summary["distinct_film_ids"])
        report["visualmetro_sparse"]["verified_BMP_view_correspondences"] = 0
        private["visualmetro"] = {
            "schema":data["schema"],"provenance_sha256":data["provenance_sha256"],
            "frame_ids_used":[f["frame_id"] for f in data["frames"]],
            "frame_ids_quarantined":[f["frame_id"] for f in data["excluded_frames"]],
            "source_070_is_not_corresponding_H50Z_image_without_manual_evidence":True
        }
        if stereo_map is not None:
            public_map,secret_map = check_manual_stereo_map(stereo_map,bmp_root,data)
            report["visualmetro_sparse"].update(public_map)
            private["visualmetro"]["manual_stereo_mapping_PRIVATE"] = secret_map
    return report,private


def make_mark_preview(gray:np.ndarray,marks:list[dict],*,origin_xy:tuple[int,int],
                      title:str="SOURCE ANNOTATIONS ONLY, NOT MODEL") ->np.ndarray:
    if not isinstance(gray,np.ndarray) or gray.ndim!=2 or gray.dtype!=np.uint8:
        raise ValueError("Expect uint8 2D crop")
    image = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    x0,y0 = origin_xy
    for mark in marks:
        x,y = mark["xy_source_px"]
        xx,yy = int(round(x-x0)),int(round(y-y0))
        if not (0<=xx<gray.shape[1] and 0<=yy<gray.shape[0]):
            continue
        color = (0,180,255) if mark["label"] == "start" else (255,180,70)
        cv2.drawMarker(image,(xx,yy),color,markerType=cv2.MARKER_CROSS,
                       markerSize=18,thickness=2)
        cv2.putText(image,mark["label"],(min(xx+11,gray.shape[1]-25),max(yy-9,12)),
                    cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
    cv2.putText(image,title[:55],(5,min(17,gray.shape[0]-1)),
                cv2.FONT_HERSHEY_SIMPLEX,.37,(40,200,255),1,cv2.LINE_AA)
    return image


def preview_crosses(*,bmp_root:Path,private:dict,output:Path,tile_size:int=512,limit:int=30):
    """Preview source annotation marks only, in a PRIVATE location."""
    root = Path(__file__).resolve().parents[1]
    destination = output.resolve()
    if destination == root or root in destination.parents:
        raise ValueError("PRIVATE annotated previews must be saved outside Git repository")
    if not 64<=tile_size<=2048 or tile_size%16 or not 1<=limit<=1000:
        raise ValueError("Invalid bounded preview size/count")
    output.mkdir(parents=True,exist_ok=True)
    count=0
    for record in private["bmp_records"]:
        if count>=limit: break
        if not record["matched_crosses"]: continue
        bmp=GrayBMP8(bmp_root/record["relative_path"])
        for mark in record["marks"]:
            if count>=limit: break
            if mark["label"]!="start":continue
            pixels,location=bmp.crop(tuple(mark["xy_source_px"]),tile_size)
            result=make_mark_preview(pixels,record["marks"],
                                     origin_xy=location.original_xyxy[:2])
            path=output/f"crosses_private_{count+1:04}.png"
            if path.exists():raise FileExistsError("Refuse to overwrite private preview")
            if not cv2.imwrite(str(path),result):
                raise OSError("Failed writing PRIVATE source-keypoint preview")
            count+=1
    return count
