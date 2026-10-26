#!/usr/bin/env python3
"""Run S48 native and synthetic fixed-registration stress tests.

Private photos, coordinates, overlays, and raw evidence stay in --private-out.
Only pixel-free aggregate metrics belong in the public docs tree.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import html
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bubble_chamber.s40_fiducial_safety import fiducial_guard
from bubble_chamber.s44_crossframe_persistence import (
    ExactBMP8, warp_corresponding_window, compare_registered_ridges)
from bubble_chamber.s48_registration_jackknife import (
    SCHEMA, leave_one_fiducial_out, photo_transforms,
    ridge_binary, assess_candidate, annotate_graph)
from scripts.research.bubble_chamber.s46_registered_repeat_counterfactuals import (
    generate_path, perpendicular_shift, inject)

SEED = 20261048
SCENARIOS = ("one_photo_only", "same_position_two_photos",
             "normal_shift_four_px", "normal_shift_eight_px")


def _load_checked(private_bmp: Path, s39: Path, s44: Path,
                  s40: Path, s39_graph: Path, s40_graph: Path):
    src = json.loads(s39.read_text())
    previous = json.loads(s44.read_text())
    fid = json.loads(s40.read_text())
    graph = json.loads(s39_graph.read_text())
    guarded = json.loads(s40_graph.read_text())
    if (src.get("native_expert_gt_created") != 0
            or len(src.get("tiles", [])) != 39
            or len(previous.get("registered_crops_PRIVATE", [])) != 39
            or len(graph.get("graphs", [])) != 39
            or len(guarded.get("graphs", [])) != 39
            or len(fid.get("fiducials_PRIVATE", [])) != 15):
        raise ValueError("Private dataset provenance differs from S39/S44/S40")
    for i, tile in enumerate(src["tiles"]):
        if (tile["crop_id"] != previous["registered_crops_PRIVATE"][i]["crop_id"]
                or tile["crop_id"] != graph["graphs"][i]["crop_id"]
                or tile["crop_id"] != guarded["graphs"][i]["crop_id"]):
            raise ValueError("Cannot pair different private real crop identities")
    images = {p: ExactBMP8(private_bmp / f"H50Z_030_01_{p}.bmp")
              for p in ("005", "006")}
    calib = leave_one_fiducial_out(json.loads((private_bmp / "labels.json").read_text()))
    return src, previous, fid, graph, guarded, images, calib


def _read_pair(tile: dict, images: dict, calib: dict):
    photo = tile["photo_id"]
    counterpart = "006" if photo == "005" else "005"
    full, alternatives = photo_transforms(calib, photo)
    origin = tile["native_origin_xy_PRIVATE"]
    source = images[photo].crop(*origin, 512, 512)
    if sha256(source.tobytes()).hexdigest() != tile["source_gray_sha256"]:
        raise ValueError("S39 native photo SHA verification FAILED")
    other = warp_corresponding_window(images[counterpart], origin, full)
    return source, other, full, alternatives


def _machine_native(src: dict, old: dict, zones: dict, graphs: dict, guarded: dict,
                    images: dict, calib: dict, private_out: Path):
    counters, graph_total = Counter(), Counter()
    by_photo = {k: Counter() for k in ("005", "006")}
    displacements = []
    detail = []
    examples = []
    for idx, tile in enumerate(src["tiles"]):
        source, other, full, variants = _read_pair(tile, images, calib)
        old_tile = old["registered_crops_PRIVATE"][idx]
        if old_tile.get("overlap_status") != "BOTH_IMAGES_VALID":
            raise ValueError("S44 cross-frame image coverage unexpectedly missing")
        evidence = old_tile["private_candidate_evidence"]
        if len(evidence) != len(tile["proposals"]):
            raise ValueError("S39 candidate and S44 score cardinality differs")
        support = ridge_binary(other)
        desc = []
        evidence_by_id = {}
        for candidate, s44 in zip(tile["proposals"], evidence):
            # The S44 flag is recomputed independently from known S40 controls
            # to prevent accidental transductive source provenance changes.
            separate_fid = fiducial_guard(
                candidate["points_xy"], tile["native_origin_xy_PRIVATE"],
                tile["photo_id"], zones["fiducials_PRIVATE"])["fiducial_uncertainty_any"]
            if s44.get("known_fiducial_uncertainty_NOT_NATIVE_TRUTH") != separate_fid:
                raise ValueError("S44 and S40 fiducial provenance mismatch")
            d = assess_candidate(source, other, candidate, s44,
                                 tile["native_origin_xy_PRIVATE"], full, variants,
                                 precomputed_destination_binary=support)
            desc.append(d)
            if d["draft_id"] in evidence_by_id:
                raise ValueError("Duplicated native candidate ID")
            evidence_by_id[d["draft_id"]] = d
            counters["all_original_machine_candidates_preserved"] += 1
            counters[d["registration_status"]] += 1
            counters["original_repeat_risk"] += int(d["S45_original_repeat_risk"])
            counters["pose_only_extra_risk"] += int(d["new_pose_only_warning"])
            by_photo[tile["photo_id"]][d["registration_status"]] += 1
            displacements.append(d["jackknife_max_path_displacement_px_PRIVATE"])
        g = annotate_graph(graphs["graphs"][idx],
                           guarded["graphs"][idx]["machine_graph_risk_annotation_PRIVATE"],
                           evidence_by_id)
        for k, v in g["counters"].items():
            graph_total[k] += v
        detail.append({"crop_id": tile["crop_id"], "photo": tile["photo_id"],
                       "source_sha256_PRIVATE": tile["source_gray_sha256"],
                       "candidate_evidence_PRIVATE": desc,
                       "graph_evidence_PRIVATE": g["per_edge_PRIVATE"]})
        panel = []
        for base in (source, other):
            canvas = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            for c, d in zip(tile["proposals"], desc):
                points = np.rint(np.asarray(c["points_xy"])).astype("int32")
                status = d["registration_status"]
                color = ((90, 145, 255) if status.startswith("BASELINE_REPEAT") else
                         (250, 70, 220) if "POSE_ONLY" in status else (85, 215, 100))
                cv2.polylines(canvas, [points], False, color, 1, lineType=cv2.LINE_AA)
            panel.append(canvas)
        p = f"S48_MACHINE_{idx:02d}.jpg"
        cv2.imwrite(str(private_out / "PRIVATE_IMAGE_PANELS" / p),
                    cv2.hconcat(panel), [cv2.IMWRITE_JPEG_QUALITY, 80])
        examples.append(p)
        print(f"S48 original real crop {idx + 1}/39: {len(desc)} candidates", flush=True)
    if counters["all_original_machine_candidates_preserved"] != 567 or graph_total["edge_count"] != 182:
        raise ValueError("Native proposal count changed; stopping")
    report = {"schema": SCHEMA, "source_real_photos": 2, "independent_physics_runs": 1,
              "six_fiducial_leave_one_out": True,
              "fiducial_heldout_mean_px": round(calib["heldout_mean_px"], 4),
              "fiducial_heldout_max_px": round(calib["heldout_max_px"], 4),
              "fiducial_in_sample_mean_px": round(calib["in_sample_mean_px"], 4),
              "fiducial_in_sample_max_px": round(calib["in_sample_max_px"], 4),
              "position_variant_displacement_px_on_original_candidates": {
                  "median": round(float(np.median(displacements)), 4),
                  "p90": round(float(np.quantile(displacements, .90)), 4),
                  "max": round(float(max(displacements)), 4)},
              "machine_candidate_counts": dict(counters),
              "source_photo_counts": {k: dict(v) for k, v in by_photo.items()},
              "graph_counts": dict(graph_total),
              "all_original_candidates_retained": True,
              "native_expert_gt": 0, "native_physical_auto_joins": 0,
              "cross_photo_transductive_advisory_only": True,
              "heldout_photo_training_from_this_output_allowed": False,
              "repeated_image_structure_not_particle_scratch_identity": True}
    (private_out / "S48_PRIVATE_MACHINE_DETAILS.json").write_text(
        json.dumps({"schema": SCHEMA,
                    "fiducial_loo_transforms_PRIVATE": [m.tolist() for m in calib["loo_forward_PRIVATE"]],
                    "fiducial_heldout_errors_by_identifier_PRIVATE": dict(zip(
                        calib["heldout_identifiers"], calib["heldout_errors_px"])),
                    "machine_crops_PRIVATE": detail}, indent=2))
    (private_out / "S48_PRIVATE_MACHINE_DEMO" / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>ANZA-LIRA S48</title>'
        '<style>body{font:16px system-ui;background:#0d1724;color:#f5f7fb;margin:26px}'
        'article{padding:14px;margin:20px 0;background:#172637;border-radius:12px}'
        'img{width:100%;max-width:1024px}p{color:#bfcfdf}</style><h1>ANZA-LIRA S48: machine-only</h1>'
        '<p>Two native photo views, 39 crops. Teal: no high repeat evidence; orange: original repeat-risk; '
        'purple: pose-sensitive possible repeat. These are NOT particle/scratch GT. '
        'Two photos were accessed for inference; never use these annotations for photo-heldout training.</p>'
        + ''.join('<article><h3>' + html.escape(src["tiles"][i]["crop_id"])
                  + '</h3><img loading="lazy" src="PRIVATE_IMAGE_PANELS/' + html.escape(name)
                  + '"></article>' for i, name in enumerate(examples)))
    return report


def _counterfactuals(src: dict, zones: dict, images: dict, calib: dict,
                     private_out: Path):
    rng = np.random.default_rng(SEED)
    counts = defaultdict(Counter)
    details = []
    for idx, tile in enumerate(src["tiles"]):
        gray, other, full, variants = _read_pair(tile, images, calib)
        path = generate_path(rng, zones["fiducials_PRIVATE"],
                             tile["native_origin_xy_PRIVATE"], tile["photo_id"])
        rows = []
        for scen in SCENARIOS:
            new_src = inject(gray, path, color=32)
            new_other = other.copy()
            if scen == "same_position_two_photos":
                new_other = inject(other, path, color=32)
            elif scen == "normal_shift_four_px":
                new_other = inject(other, perpendicular_shift(path, 4.), color=32)
            elif scen == "normal_shift_eight_px":
                new_other = inject(other, perpendicular_shift(path, 8.), color=32)
            proposal = {"draft_id": "S48_SYNTHETIC_NOT_GT", "points_xy": path.tolist()}
            evidence = compare_registered_ridges(new_src, new_other, [proposal])[0]
            evidence["known_fiducial_uncertainty_NOT_NATIVE_TRUTH"] = False
            d = assess_candidate(new_src, new_other, proposal, evidence,
                                 tile["native_origin_xy_PRIVATE"], full, variants)
            counts[scen]["n"] += 1
            counts[scen]["S45_repeated"] += int(d["S45_original_repeat_risk"])
            counts[scen]["S48_any_variant_repeated"] += int(
                d["S45_original_repeat_risk"] or d["jackknife_variant_repeat_hits"] > 0)
            counts[scen]["S48_4of6_variant_repeated"] += int(
                d["jackknife_variant_repeat_hits"] >= 4)
            rows.append({"scenario": scen, "candidate_advisory_PRIVATE": d})
        details.append({"crop_id": tile["crop_id"], "geometry_PRIVATE": path.round(2).tolist(),
                        "scenario_rows": rows})
        print(f"S48 synthetic controlled background {idx+1}/39", flush=True)
    (private_out / "S48_PRIVATE_SYNTHETIC_CONTROLS.json").write_text(
        json.dumps({"seed": SEED, "crop_details_PRIVATE": details}, indent=2))
    return {"fixed_new_seed": SEED, "native_background_crops": 39,
            "scenario_total": 39 * len(SCENARIOS),
            "different_from_S46_S47_injected_geometry": True,
            "independent_physical_events": 0,
            "scenario_counts": {k: dict(v) for k, v in counts.items()},
            "impossibility_note": "A separate synthetic line with the same pixels as a shifted repeat is observationally identical; offset recovery cannot determine particle versus scratch or match identity."}


def run(args):
    args.private_out.mkdir(parents=True, exist_ok=True)
    (args.private_out / "PRIVATE_IMAGE_PANELS").mkdir(exist_ok=True)
    (args.private_out / "S48_PRIVATE_MACHINE_DEMO" / "PRIVATE_IMAGE_PANELS").mkdir(parents=True, exist_ok=True)
    args.public_out.mkdir(parents=True, exist_ok=True)
    src, old, zones, graphs, guarded, images, estimate = _load_checked(
        args.bmp_directory, args.s39_private, args.s44_private, args.s40_private,
        args.s39_graph, args.s40_graph)
    native = _machine_native(src, old, zones, graphs, guarded,
                             images, estimate, args.private_out)
    # Duplicate only compressed JPG in private demo subfolder; no copy of raw BMP.
    for jpg in sorted((args.private_out / "PRIVATE_IMAGE_PANELS").glob("*.jpg")):
        dest = args.private_out / "S48_PRIVATE_MACHINE_DEMO" / "PRIVATE_IMAGE_PANELS" / jpg.name
        dest.write_bytes(jpg.read_bytes())
    controls = _counterfactuals(src, zones, images, estimate, args.private_out)
    report = {"S48_native_machine": native, "S48_new_synthetic_controls": controls}
    (args.public_out / "S48_PUBLIC_AGGREGATE_RESULTS.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    for name in ("bmp-directory", "s39-private", "s44-private", "s40-private",
                 "s39-graph", "s40-graph", "private-out", "public-out"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
