"""S113 offline H50Z annotation view (NO physical ground truth).

Existing S111 track groups are presented as selectable geometric hypotheses.
Observed photo-backed parts and inferred connectors remain separately stored.
Operator fiducial exclusions are displayed as protected zones; their identifiers
are not fabricated from cross-mask blobs. No model training happens here.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

FRAME_SIZES = {"005": (9020, 26151), "006": (9268, 25324)}
PALETTE = ["#20b7e5", "#ee8739", "#6eb551", "#9979d2", "#e35e91", "#1dbda1", "#b7a24c", "#358bd0"]


def _xy(points, width: int, height: int) -> np.ndarray:
    data = np.asarray(points, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < 2 or not np.all(np.isfinite(data)):
        raise ValueError("Track geometry must contain finite XY coordinates")
    if np.any(data[:, 0] < -1) or np.any(data[:, 1] < -1) or np.any(data[:, 0] > width + 1) or np.any(data[:, 1] > height + 1):
        raise ValueError("Curve points outside local ROI")
    return data


def _simplify(points: np.ndarray, max_points: int = 600) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.round(np.linspace(0, len(points) - 1, max_points)).astype(int)
    return points[np.unique(indices)]


def _path(points: np.ndarray) -> str:
    points = _simplify(points)
    return "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _group_runs(points: np.ndarray, support: np.ndarray) -> list[dict[str, Any]]:
    """Keep photometric sample provenance, including inferred spans inside curves."""
    if len(points) != len(support):
        raise ValueError("Support flags must line up with sampled curve points")
    if len(points) < 2:
        return []
    # A segment is observed only when BOTH sampled end-points are photo-backed.
    edges = support[:-1] & support[1:]
    result = []
    begin = 0
    for k in range(1, len(edges) + 1):
        if k == len(edges) or edges[k] != edges[begin]:
            run = points[begin:k + 1]
            if len(run) >= 2:
                result.append({"kind": "observed_candidate" if edges[begin] else "inferred_inside_curve", "path": _path(run), "n_edges": int(k - begin)})
            begin = k
    return result


def _length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _cross_components(mask: np.ndarray) -> list[dict[str, Any]]:
    """Protected blobs, NOT verified f1..f8 labels or calibrated crosses."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    regions = []
    for i in range(1, n):
        x, y, w, h, area = (int(z) for z in stats[i])
        if area < 8:
            continue
        regions.append({"id": f"protected_{len(regions) + 1}", "bbox_xywh": [x, y, w, h],
                        "centroid_xy": [round(float(centroids[i, 0]), 2), round(float(centroids[i, 1]), 2)],
                        "mask_pixels": area, "operator_fiducial_id": None,
                        "status": ("LARGE_PROTECTED_SCAN_ARTIFACT_OR_VIGNETTE"
                                   if max(w,h) >= 160 or area >= 14000
                                   else "SMALL_FIDUCIAL_CANDIDATE_UNVERIFIED")})
    return regions


def build_annotation(root: Path, roi: str, labels_json: Path | None = None) -> dict[str, Any]:
    s111_path = root / "s111_global_results" / roi / "global_tracks.json"
    imgpath = root / "s108_results_all" / roi / "06_SOURCE_ORIGINAL_GRAY.png"
    protectedpath = root / "s108_results_all" / roi / "07_BLOCKED_SOURCE.png"
    v109path = root / "s109_vector_results" / roi / "vector_tracks.json"
    payload = json.loads(s111_path.read_text(encoding="utf-8"))
    v109 = json.loads(v109path.read_text(encoding="utf-8"))
    gray = cv2.imread(str(imgpath), cv2.IMREAD_GRAYSCALE)
    blocked = cv2.imread(str(protectedpath), cv2.IMREAD_GRAYSCALE)
    if gray is None or blocked is None or gray.shape != blocked.shape:
        raise ValueError("Missing or misaligned source and protected-region mask: " + roi)
    height, width = gray.shape
    origin = v109.get("original_roi_source_coordinates")
    if len(origin) != 4 or int(origin[2] - origin[0]) != width or int(origin[3] - origin[1]) != height:
        raise ValueError("Wrong original-source coordinate transform in " + roi)
    fiducials = _cross_components(blocked)
    operator_marks = []
    operator_labels_sha256 = None
    if labels_json is not None:
        labels_bytes = Path(labels_json).read_bytes()
        operator_labels_sha256 = hashlib.sha256(labels_bytes).hexdigest()
        keypoints = json.loads(labels_bytes.decode('utf-8-sig'))
        frame_id_for_marks = roi.split('_',1)[0]
        film_name = f'H50Z_030_01_{frame_id_for_marks}.bmp'
        if film_name not in keypoints:
            raise ValueError('No verified operator labels for '+film_name)
        rec = keypoints[film_name]
        if [int(rec['original_width']),int(rec['original_height'])]!=list(FRAME_SIZES[frame_id_for_marks]):
            raise ValueError('Crosses do not match this original source image size')
        for mark in rec['keypoints']:
            label = mark['label']
            if label!='start' and label not in {f'f{i}' for i in range(1,9)}:
                raise ValueError('Unexpected mark type')
            x=float(mark['x_px']);y=float(mark['y_px'])
            if origin[0] <= x < origin[2] and origin[1] <= y < origin[3]:
                operator_marks.append({'label':label,'full_frame_xy':[x,y],
                    'local_xy':[round(x-origin[0],3),round(y-origin[1],3)],
                    'source':'OPERATOR_LABELS_JSON_CONFIRMED_POSITION',
                    'not_a_complete_track_segmentation':True})
    s112_path = root / "s112_bubble_results" / roi / "bubble_graph_not_gt.json"
    s112 = json.loads(s112_path.read_text(encoding="utf-8"))
    if s112.get("source_roi") != roi or s112.get("s111_global_tracks_sha256") != hashlib.sha256(s111_path.read_bytes()).hexdigest():
        raise ValueError("S112 observations do not match this S111 model version")
    bead_proposals = []
    for family, key in (("anchored", "anchored_bead_chain_proposals"),
                        ("free", "unanchored_review_only_tracklets")):
        for item in s112[key]:
            pts = _xy(item["preview_spline_xy"], width, height)
            bead_proposals.append({"id": item["path_id"], "family":family, "status":item["status"],
                "geometry_path":_path(pts), "geometry_start_xy":list(map(float,pts[0])),
                "geometry_end_xy":list(map(float,pts[-1])),
                "photo_bubble_positions_xy":item["bubble_xy"],
                "photometric_candidate_count":len(item["bubble_ids"]),
                "length_px":round(_length(pts),2),
                "photometric_support_estimate":item.get("mean_photo_mask_support"),
                "not_expert_verified":True})
    bridge_proposals = []
    for item in s112['intergroup_review_only_bridge_proposals']:
        bridge_proposals.append({"id":item['bridge_id'],"family":"bridge",
            "status":item['status'],"geometry_path":item['candidate_inferred_spline']['svg_path'],
            "source_groups":item['source_groups'],
            "photo_bubble_positions_xy":item['source_dot_xy'],
            "photometric_candidate_count":len(item['common_bubble_ids']),
            "length_px":None,"not_expert_verified":True})
    by_id = {curve["curve_id"]: curve for curve in payload["curves"]}
    tracks = []
    used_curves = []
    for index, group in enumerate(payload["association"]["tracks"]):
        pieces = []
        occupied = []
        endpoints = []
        num_observed = num_inferred = 0
        total_length = 0.0
        members = group["ordered_members"]
        for member in members:
            cid = member["curve_id"]
            if cid not in by_id:
                raise ValueError("Association refers to missing curve " + cid)
            curve = by_id[cid]
            pts = _xy(curve["sampled_xy"], width, height)
            support = np.asarray(curve["sampled_source_supported"], dtype=bool)
            if member.get("reversed", False):
                pts = pts[::-1]
                support = support[::-1]
            if not endpoints:
                endpoints.append(pts[0].tolist())
            endpoint_end = pts[-1].tolist()
            used_curves.append(cid)
            for segment in _group_runs(pts, support):
                pieces.append({"source_curve_id": cid, **segment})
            num_observed += int(np.count_nonzero(support))
            num_inferred += int(np.count_nonzero(~support))
            total_length += _length(pts)
            occupied.extend((int(round(x)), int(round(y))) for x,y in pts[::max(1,len(pts)//150)])
        endpoints.append(endpoint_end)
        for connector in group.get("inferred_connectors", []):
            q = _xy(connector["sampled_xy"], width, height)
            pieces.append({"kind": "inferred_between_curves", "path": _path(q),
                           "source_curve_ids": [connector.get("from_curve_id"), connector.get("to_curve_id")],
                           "connector_id": connector["candidate_id"]})
        first, last = endpoints
        blocked_touch = 0
        for x, y in occupied[-min(len(occupied), 1200):]:
            if 0 <= x < width and 0 <= y < height and blocked[y, x]:
                blocked_touch += 1
        ratio = num_observed / max(1, num_observed + num_inferred)
        tracks.append({"id": group["global_track_id"], "index": index + 1,
                       "member_curve_ids": group["member_curve_ids"],
                       "geometry_start_xy": [round(float(z), 2) for z in first],
                       "geometry_end_xy": [round(float(z), 2) for z in last],
                       "full_frame_start_xy": [round(float(first[0] + origin[0]), 2), round(float(first[1] + origin[1]), 2)],
                       "full_frame_end_xy": [round(float(last[0] + origin[0]), 2), round(float(last[1] + origin[1]), 2)],
                       "arclength_photo_curve_px": round(total_length, 2),
                       "photo_candidate_sample_fraction": round(ratio, 4),
                       "inferred_connector_count": len(group.get("inferred_connectors", [])),
                       "protected_collision_sample_count": blocked_touch,
                       "geometry_orientation_is_particle_direction": False,
                       "physical_track_validated": False,
                       "review_status": "UNREVIEWED_MODEL_GROUP",
                       "pieces": pieces})
    if len(set(used_curves)) != len(used_curves) or len(used_curves) != len(by_id):
        raise AssertionError("Each curve must belong to exactly one global group")
    frame_id = roi.split("_", 1)[0]
    if frame_id not in FRAME_SIZES:
        raise ValueError("Unknown film/frame size — avoid guessing")
    frame_width, frame_height = FRAME_SIZES[frame_id]
    if origin[0] < 0 or origin[1] < 0 or origin[2] > frame_width or origin[3] > frame_height:
        raise ValueError("ROI lies outside declared original frame")
    report = {"schema": "S113_TRACK_CROSS_ANNOTATION_REVIEW_V1", "roi": roi,
              "frame_id": frame_id, "roi_size_wh": [width, height],
              "original_frame_size_wh": [frame_width, frame_height],
              "original_roi_source_coordinates": origin,
              "source_png_sha256": hashlib.sha256(imgpath.read_bytes()).hexdigest(),
              "operator_labels_sha256":operator_labels_sha256,
              "operator_keypoints_in_roi":len(operator_marks),
              "protected_mask_png_sha256": hashlib.sha256(protectedpath.read_bytes()).hexdigest(),
              "s111_global_tracks_sha256": hashlib.sha256(s111_path.read_bytes()).hexdigest(),
              "s112_bubble_proposals_sha256": hashlib.sha256(s112_path.read_bytes()).hexdigest(),
              "S111_global_group_count": len(tracks),
              "S111_curve_count": len(by_id),
              "protected_zone_candidates": len(fiducials),
              "small_protected_fiducial_candidates":sum(x["status"]=="SMALL_FIDUCIAL_CANDIDATE_UNVERIFIED" for x in fiducials),
              "large_scan_artifact_zones":sum(x["status"].startswith("LARGE_") for x in fiducials),
              "tracks_with_protected_region_collision": sum(t["protected_collision_sample_count"] > 0 for t in tracks),
              "geometry_endpoints_explicitly_available": all(t["geometry_start_xy"] and t["geometry_end_xy"] for t in tracks),
              "source_photograph_was_modified": False,
              "all_physical_particle_ids_unverified": True,
              "f1_to_f8_labels_not_inferred_from_blob_centroids": True,
              "frame_crosses_are_not_tracks": True,
              "tracks": tracks, "protected_regions": fiducials,
              "operator_marks":operator_marks,
              "bubble_chain_review_only":bead_proposals,"bridge_review_only":bridge_proposals}
    return report


def _bg_data_url(gray: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError("Image encode failure")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _svg_elements(report: dict[str, Any], marker_labels: bool = True) -> str:
    paths = []
    for t in report["tracks"]:
        key = html.escape(t["id"], quote=True)
        color = PALETTE[(t["index"] - 1) % len(PALETTE)]
        for p in t["pieces"]:
            inferred = p["kind"] != "observed_candidate"
            paths.append(f'<path class="trace {"hypothesis" if inferred else "observed"}" data-track="{key}" '
                         f'd="{html.escape(p["path"], quote=True)}" fill="none" stroke="{color}" '
                         f'stroke-width="{2.0 if not inferred else 1.65}" '
                         f'stroke-dasharray="{ "6 5" if inferred else "none" }"/>')
        (ax, ay), (bx, by) = t["geometry_start_xy"], t["geometry_end_xy"]
        paths.append(f'<g data-track="{key}" class="endpoints"><circle cx="{ax}" cy="{ay}" r="7" '
                     f'fill="#16a34a" stroke="#ffffff" stroke-width="1.6"/>'
                     f'<rect x="{bx-6:.2f}" y="{by-6:.2f}" width="12" height="12" '
                     f'fill="#dc2626" stroke="#ffffff" stroke-width="1.6"/>')
        if marker_labels:
            paths.append(f'<text x="{ax+9}" y="{ay-9}" font-size="12" stroke="#101827" stroke-width="2" paint-order="stroke" fill="#ffffff">A{t["index"]}</text>'
                         f'<text x="{bx+9}" y="{by-9}" font-size="12" stroke="#101827" stroke-width="2" paint-order="stroke" fill="#ffffff">B{t["index"]}</text>')
        paths.append('</g>')
    for prop in report["bubble_chain_review_only"]+report["bridge_review_only"]:
        key=html.escape(prop['id'],quote=True)
        color='#facc15' if prop['family']=='bridge' else '#f59e0b' if prop['family']=='anchored' else '#a78bfa'
        paths.append(f'<g class="bead-proposal hidden" data-bead="{key}">'
                     f'<path d="{html.escape(prop["geometry_path"],quote=True)}" fill="none" stroke="{color}" stroke-width="1.5" stroke-dasharray="5 6"/>')
        for x,y in prop['photo_bubble_positions_xy']:
            paths.append(f'<circle cx="{float(x):.2f}" cy="{float(y):.2f}" r="2.2" stroke="{color}" fill="none" stroke-width="1.2"/>')
        paths.append('</g>')
    for mark in report["operator_marks"]:
        x,y=mark['local_xy'];label=html.escape(mark['label'])
        if label=='start':
            paths.append(f'<g class="operator-start"><circle cx="{x}" cy="{y}" r="10" fill="none" stroke="#fbbf24" stroke-width="2"/>'
                         f'<text x="{x+13}" y="{y-10}" fill="#fbbf24" font-size="14">operator start</text></g>')
        else:
            paths.append(f'<g class="operator-cross"><path d="M {x-11},{y-11} L {x+11},{y+11} M {x-11},{y+11} L {x+11},{y-11}" stroke="#e879f9" stroke-width="2"/>'
                         f'<text x="{x+15}" y="{y-12}" fill="#e879f9" font-size="14">{label}</text></g>')
    for region in report["protected_regions"]:
        x, y, w, h = region["bbox_xywh"]
        cx, cy = region["centroid_xy"]
        is_artifact = region["status"].startswith("LARGE_")
        shade = "#9ca3af" if is_artifact else "#d946ef"
        paths.append(f'<g class="protected {"scan-artifact" if is_artifact else "fiducial-candidate"}" data-protected="{region["id"]}">'
                     f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                     f'fill="{shade}" fill-opacity=".04" stroke="{shade}" stroke-width="2"/>'
                     f'<circle cx="{cx}" cy="{cy}" r="8" fill="none" stroke="{shade}" stroke-width="1.3"/>'
                     f'<path d="M {cx-10},{cy} L {cx+10},{cy} M {cx},{cy-10} L {cx},{cy+10}" stroke="{shade}" stroke-width="1.5"/>'
                     f'<text x="{cx+12}" y="{cy-10}" fill="{shade}" font-size="12" stroke="#080b15" paint-order="stroke" stroke-width="2">{region["id"]} ?</text></g>')
    return "\n".join(paths)


def render_png(report: dict[str, Any], gray: np.ndarray, end_marker_limit: int | None = 12) -> np.ndarray:
    bg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    w, h = report["roi_size_wh"]
    if bg.shape[:2] != (h, w):
        raise ValueError("Source image dimensions changed")
    longest = sorted(report["tracks"],key=lambda t: t["arclength_photo_curve_px"],reverse=True)
    show_ends = {t["id"] for t in longest[:end_marker_limit]} if end_marker_limit is not None else {t["id"] for t in longest}
    for t in report["tracks"]:
        rgb = PALETTE[(t["index"] - 1) % len(PALETTE)].lstrip("#")
        color = (int(rgb[4:6],16),int(rgb[2:4],16),int(rgb[0:2],16))
        for p in t["pieces"]:
            # Store actual polyline coordinates in JSON/HTML; png is a preview.
            path = p["path"].replace("M", "").replace("L", "")
            pts = np.array([[float(a),float(b)] for a,b in (z.strip().split(",") for z in path.strip().split())],np.float32)
            if len(pts) < 2:
                continue
            pts = np.rint(pts).astype(np.int32)
            if p["kind"] != "observed_candidate":
                for i in range(0, len(pts)-1, 12):
                    short = pts[i:min(i+7,len(pts))]
                    if len(short) > 1:
                        cv2.polylines(bg,[short.reshape(-1,1,2)],False,color,1,cv2.LINE_AA)
            else:
                cv2.polylines(bg,[pts.reshape(-1,1,2)],False,color,2,cv2.LINE_AA)
        if t["id"] not in show_ends:
            continue
        a = tuple(map(lambda x:int(round(x)),t["geometry_start_xy"]))
        b = tuple(map(lambda x:int(round(x)),t["geometry_end_xy"]))
        cv2.circle(bg,a,6,(45,210,45),-1,cv2.LINE_AA)
        cv2.rectangle(bg,(b[0]-5,b[1]-5),(b[0]+5,b[1]+5),(15,20,220),-1)
        cv2.putText(bg,f'A{t["index"]}',(a[0]+6,a[1]-7),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(bg,f'B{t["index"]}',(b[0]+6,b[1]-7),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
    for m in report["operator_marks"]:
        x,y=(int(round(t)) for t in m['local_xy'])
        if m['label']=='start':
            cv2.circle(bg,(x,y),11,(10,210,245),2,cv2.LINE_AA)
            cv2.putText(bg,'operator START',(x+12,y-10),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,225,245),1,cv2.LINE_AA)
        else:
            cv2.line(bg,(x-11,y-11),(x+11,y+11),(225,30,225),2,cv2.LINE_AA)
            cv2.line(bg,(x-11,y+11),(x+11,y-11),(225,30,225),2,cv2.LINE_AA)
            cv2.putText(bg,m['label'],(x+14,y-12),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,40,230),1,cv2.LINE_AA)
    for z in report["protected_regions"]:
        x,y,w,h = z["bbox_xywh"]
        is_artifact = z["status"].startswith("LARGE_")
        cv2.rectangle(bg,(x,y),(x+w,y+h),(150,150,150) if is_artifact else (220,35,220),2,cv2.LINE_AA)
        cx,cy = (int(round(q)) for q in z["centroid_xy"])
        text = 'SCAN ARTIFACT' if is_artifact else (z["id"]+'?')
        cv2.putText(bg,text,(max(2,x+4),max(12,y+16)),cv2.FONT_HERSHEY_SIMPLEX,.45,(190,190,190) if is_artifact else (255,60,255),1,cv2.LINE_AA)
    return bg


def render_html(report: dict[str, Any], gray: np.ndarray) -> str:
    w, h = report["roi_size_wh"]
    tiny = {k:v for k,v in report.items() if k not in ("tracks","protected_regions","operator_marks","bubble_chain_review_only","bridge_review_only")}
    track_payload = [{k:v for k,v in t.items() if k!="pieces"} for t in report["tracks"]]
    data = json.dumps({"meta":tiny,"tracks":track_payload,"protected":report["protected_regions"],"operator_marks":report["operator_marks"],"beads":[{k:v for k,v in p.items() if k!="geometry_path"} for p in report["bubble_chain_review_only"]+report["bridge_review_only"]]},ensure_ascii=False,separators=(",",":"))
    data = data.replace("</", "<\\/").replace("<", "\\u003c")
    svg = _svg_elements(report)
    image = _bg_data_url(gray)
    # No fetch/network and no CDN: works when opened directly from a local ZIP.
    template = r'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>ANZA S113 — review ROI</title>
<style>
:root{color-scheme:dark}body{margin:0;background:#0d1017;color:#e8edf7;font:14px system-ui,Arial}
header{padding:12px 18px;border-bottom:1px solid #303847;display:flex;justify-content:space-between;align-items:center}
main{display:grid;grid-template-columns:minmax(0,1fr) 350px;height:calc(100vh - 55px)}
.viewer{position:relative;overflow:auto;background:#171a21;padding:12px}
svg{background:#2a2a2a;display:block;margin:auto;width:min(100%,920px);height:auto;box-shadow:0 0 0 1px #3c4350}
.trace{cursor:pointer;stroke-linecap:round;stroke-linejoin:round;pointer-events:stroke}
.trace:hover{stroke-width:4.6}.trace.active{stroke:#fff!important;stroke-width:3.6}
.endpoints{pointer-events:none}.protected{cursor:pointer}.hidden{display:none!important}
aside{overflow:auto;background:#131924;border-left:1px solid #303847;padding:15px}
button,select,input{color:#f7fafc;background:#222b3b;border:1px solid #4c596e;border-radius:6px;padding:7px;margin:2px}
button{cursor:pointer}button:hover{background:#364458}label{display:block;padding:4px 0}
mark{color:#fbf5da;background:#50401b}code{font-size:12px;overflow-wrap:anywhere}.key{color:#9cacc1}
.status{border-radius:6px;border:1px solid #4d596c;padding:9px;background:#192131;margin:9px 0}
.tiny{font-size:12px;color:#a3b0c0}
</style></head><body><header><strong>ANZA–LIRA S113 — __ROI__</strong><span>Синий / разноцветный: кандидаты треков · <span style="color:#d946ef">розовый: защищённые зоны</span> · зелёный A / красный B — геометрические концы</span></header>
<main><div class="viewer"><svg id="scene" viewBox="0 0 __W__ __H__" xmlns="http://www.w3.org/2000/svg"><image href="__IMG__" x="0" y="0" width="__W__" height="__H__"/>__SVG__</svg></div>
<aside><strong>Слои</strong>
<label><input type="checkbox" id="showObserved" checked> Фото-поддержанные фрагменты</label>
<label><input type="checkbox" id="showInferred" checked> Интерполяции (пунктир, НЕ фото)</label>
<label><input type="checkbox" id="showEnds" checked> Геометрическое начало A и конец B</label>
<label><input type="checkbox" id="showCrosses" checked> Защищённые области (X?)</label>
<label><input type="checkbox" id="showOperator" checked> Точные f1–f8 и операторские start из labels.json</label>
<label><input type="checkbox" id="showBeads"> S112: цепочки пузырьков и предлагаемые стыки (пунктир)</label>
<label>Минимальная длина траектории: <input id="minLength" type="number" min="0" max="3000" value="100" step="20" style="width:70px"> px</label>
<div class="status"><b>Не экспертная разметка</b><p class="tiny">S111 — геометрические гипотезы, S112 — точки-кандидаты. A/B определяют параметризацию сплайна, не направление полёта частицы. Protected ? — кандидат репера, не подтверждённый f1–f8.</p></div>
<div id="overall"></div><hr><strong>Выбранный объект</strong><div id="selected" class="status">Нажмите на кривую или розовую область.</div>
<label>Решение эксперта <select id="decision"><option value="uncertain">Неясно</option><option value="track">Трек</option><option value="artifact">Артефакт</option><option value="fiducial">Репер / крест</option></select></label>
<label>Комментарий <input id="note" placeholder="Например: конец у правого края" style="width:94%"></label>
<button id="saveOne">Сохранить решение</button><button id="export">Экспорт JSON</button>
<label>Импорт ранее сохранённых решений <input type="file" id="import" accept=".json,application/json"></label>
<p class="tiny">Экспорт привязан к SHA-256 фотографии и S111. Все решения хранятся только локально в JSON. Перед использованием для обучения требуется проверить координаты, направление и физический смысл меток.</p>
</aside></main><script>
const DATA = __DATA__;
let chosen=null,judgements={};const q=id=>document.getElementById(id),all=s=>[...document.querySelectorAll(s)];
function refresh(){const min=Number(q('minLength').value)||0;
 for(const p of all('.observed'))p.classList.toggle('hidden',!q('showObserved').checked);
 for(const p of all('.hypothesis'))p.classList.toggle('hidden',!q('showInferred').checked);
 for(const p of all('.endpoints'))p.classList.toggle('hidden',!q('showEnds').checked);
 for(const p of all('.protected'))p.classList.toggle('hidden',!q('showCrosses').checked);
 for(const p of all('.operator-cross,.operator-start'))p.classList.toggle('hidden',!q('showOperator').checked);
 for(const p of all('.bead-proposal'))p.classList.toggle('hidden',!q('showBeads').checked);
 for(const t of DATA.tracks){const visible=t.arclength_photo_curve_px>=min;
  for(const p of all('[data-track="'+t.id+'"]'))p.style.opacity=visible?'1':'0';}
}
for(const id of ['showObserved','showInferred','showEnds','showCrosses','showOperator','showBeads','minLength'])q(id).addEventListener('change',refresh);
function choose(kind,id){chosen={kind,id};all('.trace').forEach(p=>p.classList.toggle('active',p.dataset.track===id));
 const found=kind==='track'?DATA.tracks.find(x=>x.id===id):kind==='protected'?DATA.protected.find(x=>x.id===id):DATA.beads.find(x=>x.id===id);
 if(!found)return;const previous=judgements[kind+':'+id]||{};
 q('decision').value=previous.decision||'uncertain';q('note').value=previous.note||'';
 if(kind==='track'){q('selected').innerHTML='<b>Траектория '+found.index+'</b><p><code>'+found.id+'</code></p><div>A (локальные): '+found.geometry_start_xy.join(', ')+
 '</div><div>B (локальные): '+found.geometry_end_xy.join(', ')+'</div><div>A/B (полный кадр): '+found.full_frame_start_xy.join(', ')+' / '+found.full_frame_end_xy.join(', ')+
 '</div><div>Длина: '+found.arclength_photo_curve_px+' px</div><div>Фото-поддержка (сэмплы): '+(100*found.photo_candidate_sample_fraction).toFixed(1)+'%</div><div>Предположительных стыков: '+found.inferred_connector_count+'</div><div>Точки в защищённой зоне: '+found.protected_collision_sample_count+'</div><small>Физический ID не подтверждён.</small>';
 }else if(kind==='protected'){q('selected').textContent=found.id+' | центр '+found.centroid_xy.join(', ')+' | площадь '+found.mask_pixels+' px | '+found.status+' | настоящий f1–f8 не установлен';}
 else {q('selected').textContent=found.id+' | S112 '+found.family+' | '+found.status+' | подтверждающих тёмных точек: '+found.photometric_candidate_count+' | НЕ наблюдаемая кривая и НЕ физический ID';}}
all('.trace').forEach(p=>p.addEventListener('click',()=>choose('track',p.dataset.track)));
all('.protected').forEach(p=>p.addEventListener('click',()=>choose('protected',p.dataset.protected)));
all('.bead-proposal').forEach(p=>p.addEventListener('click',()=>choose('bead_path',p.dataset.bead)));
q('saveOne').onclick=()=>{if(!chosen)return alert('Выберите кривую или защищённую область');judgements[chosen.kind+':'+chosen.id]={...chosen,decision:q('decision').value,note:q('note').value};q('saveOne').textContent='Записано в текущий JSON';};
function payload(){return {schema:'S113_EXPERT_REVIEW_V1',source_sha256:DATA.meta.source_png_sha256,s111_sha256:DATA.meta.s111_global_tracks_sha256,s112_sha256:DATA.meta.s112_bubble_proposals_sha256,operator_labels_sha256:DATA.meta.operator_labels_sha256,
  roi:DATA.meta.roi,geometry_A_B_is_not_particle_direction:true,decisions:Object.values(judgements)};}
q('export').onclick=()=>{const body=JSON.stringify(payload(),null,2),blob=new Blob([body],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=DATA.meta.roi+'_S113_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
q('import').onchange=async event=>{const file=event.target.files[0];if(!file)return;
 let data;try{data=JSON.parse(await file.text());}catch(e){alert('Неправильный JSON');return;}
 if(data.schema!=='S113_EXPERT_REVIEW_V1'||data.roi!==DATA.meta.roi||data.source_sha256!==DATA.meta.source_png_sha256||data.s111_sha256!==DATA.meta.s111_global_tracks_sha256||data.s112_sha256!==DATA.meta.s112_bubble_proposals_sha256||data.operator_labels_sha256!==DATA.meta.operator_labels_sha256){alert('SHA/ROI не совпадают: импорт запрещён');return;}
 judgements={};for(const x of data.decisions||[]){if((x.kind==='track'&&DATA.tracks.some(y=>y.id===x.id))||(x.kind==='protected'&&DATA.protected.some(y=>y.id===x.id))||(x.kind==='bead_path'&&DATA.beads.some(y=>y.id===x.id))){judgements[x.kind+':'+x.id]=x;}}q('overall').textContent='Импортировано решений: '+Object.keys(judgements).length;};
q('overall').textContent=DATA.meta.S111_global_group_count+' групп кривых / '+DATA.meta.S111_curve_count+' фрагментов, защищённых областей: '+DATA.meta.protected_zone_candidates;
const list=document.createElement('div');list.className='status';list.style.maxHeight='165px';list.style.overflow='auto';
for (const t of [...DATA.tracks].sort((a,b)=>b.arclength_photo_curve_px-a.arclength_photo_curve_px)){
 let el=document.createElement('button');el.textContent='#'+t.index+' · '+Math.round(t.arclength_photo_curve_px)+'px';el.onclick=()=>{q('minLength').value=0;refresh();choose('track',t.id);};list.appendChild(el);
}
q('overall').appendChild(list);
refresh();
</script></body></html>'''
    for key, val in {"__ROI__": html.escape(report["roi"]), "__W__": str(w), "__H__": str(h), "__IMG__": image, "__SVG__": svg, "__DATA__": data}.items():
        template = template.replace(key, val)
    return template


def validate_review_decisions(report: dict[str, Any], decisions: dict[str, Any]) -> dict[str, Any]:
    if decisions.get("schema") != "S113_EXPERT_REVIEW_V1" or decisions.get("roi") != report["roi"]:
        raise ValueError("Foreign schema or ROI")
    if decisions.get("source_sha256") != report["source_png_sha256"] or decisions.get("s111_sha256") != report["s111_global_tracks_sha256"] or decisions.get("s112_sha256") != report["s112_bubble_proposals_sha256"] or decisions.get("operator_labels_sha256") != report["operator_labels_sha256"]:
        raise ValueError("Source/model version mismatch; refuse importing expert decisions")
    allowed = ({("track", z["id"]) for z in report["tracks"]}
               | {("protected", z["id"]) for z in report["protected_regions"]}
               | {("bead_path", z["id"]) for z in report["bubble_chain_review_only"]+report["bridge_review_only"]})
    output = []
    seen = set()
    for d in decisions.get("decisions", []):
        key = (d.get("kind"), d.get("id"))
        if key not in allowed or key in seen:
            raise ValueError("Invalid or duplicate reviewed identity")
        if d.get("decision") not in {"uncertain", "track", "artifact", "fiducial"}:
            raise ValueError("Invalid expert decision type")
        if len(d.get("note", "")) > 4000:
            raise ValueError("Review note too long")
        seen.add(key)
        output.append({"kind": key[0], "id": key[1], "decision": d["decision"], "note": d.get("note", "")})
    return {"accepted_review_count": len(output), "reviewed": output,
            "machine_to_expert_gt_promoted_automatically": False}