#!/usr/bin/env python3
"""Standalone, fail-closed reconstruction from saved CT28 routes and a broken mask.

Input mask must be a predicted damaged vessel mask in the path voxel grid.
The reference full segmentation must never be used as the input to a repair
improvement experiment. Held-out CT28 subjects are blocked in this tool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ct28_mask_reconstruction import RepairProposal, apply_approved_paths


HELD_OUT = {954, 958, 972, 973, 980, 984}


def load_route_record(path: Path, scene_id: str) -> dict:
    matches = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("scene_id") == scene_id:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError("Expected exactly one selected route record for scene")
    record = matches[0]
    patient = int(record["scan_id"])
    if patient in HELD_OUT:
        raise ValueError("Held-out CT28 patient is blocked before protocol freeze")
    if scene_id.startswith("CT28S:"):
        tokens = scene_id.split(":")
        if len(tokens) < 5 or tokens[1] not in {"train", "val"} or int(tokens[3]) != patient:
            raise ValueError("Scene ID does not match frozen train/validation patient")
    if len(record.get("routes", [])) not in (1, 2):
        raise ValueError("Expected one relation or a BOTH pair of relations")
    if "selected_candidate_ids" in record and len(record["selected_candidate_ids"]) != len(record["routes"]):
        raise ValueError("Selected candidate IDs and route count disagree")
    for route in record["routes"]:
        if route.get("status") != "ok" or route.get("extra_endpoint_hit", False):
            raise ValueError("Unsuccessful or unsafe route cannot be rasterized")
    return record


def proposals_for_record(record: dict) -> list[RepairProposal]:
    return [
        RepairProposal(
            route["kind"],
            tuple(tuple(tuple(voxel) for voxel in arm)
                  for arm in route["paths_voxel"]),
        )
        for route in record["routes"]
    ]


def run_from_files(
    *,
    routes: Path,
    scene_id: str,
    mask_file: Path,
    affine_file: Path,
    output_mask: Path,
    output_report: Path,
    radius_mm: float,
    max_new_voxels: int,
    allowed_file: Path | None = None,
) -> dict:
    protected = {mask_file.resolve(), affine_file.resolve(), routes.resolve()}
    if allowed_file is not None:
        protected.add(allowed_file.resolve())
    if output_mask.resolve() in protected or output_report.resolve() in protected or output_mask.resolve() == output_report.resolve():
        raise ValueError("Cannot replace original input files")
    if output_mask.suffix != ".npy" or output_report.suffix != ".json":
        raise ValueError("Expected output mask .npy and output report .json")
    if output_mask.exists() or output_report.exists():
        raise FileExistsError("Output exists: use a new path for immutable evidence")
    record = load_route_record(routes, scene_id)
    mask = np.load(mask_file, allow_pickle=False)
    affine = np.load(affine_file, allow_pickle=False)
    allowed = np.load(allowed_file, allow_pickle=False) if allowed_file else None
    result = apply_approved_paths(
        mask, affine, proposals_for_record(record),
        radius_mm=radius_mm,
        max_new_voxels=max_new_voxels,
        allowed_new_voxels=allowed,
    )
    summary = {
        "phase": "experimental_mask_reconstruction_train_val_only",
        "scene_id": scene_id,
        "scan_id": int(record["scan_id"]),
        "accepted": result.accepted,
        "reason": result.reason,
        "input_mask": str(mask_file),
        "routes_file": str(routes),
        "radius_mm": radius_mm,
        "max_new_voxels": max_new_voxels,
        "added_voxels": result.added_voxels,
        "original_components": result.original_components,
        "final_components": result.final_components,
        "component_labels_connected": result.connected_components,
        "output_mask": str(output_mask) if result.accepted else None,
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    if result.accepted:
        output_mask.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_mask, result.mask, allow_pickle=False)
    output_report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True,
                        help="Existing path v2 JSONL with routes[].paths_voxel")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--broken-mask", type=Path, required=True,
                        help="Boolean 3D .npy predicted broken mask, NOT reference segmentation")
    parser.add_argument("--affine", type=Path, required=True,
                        help="4x4 .npy NIfTI voxel-to-RAS affine")
    parser.add_argument("--radius-mm", type=float, required=True,
                        help="Radius preselected on training data")
    parser.add_argument("--max-new-voxels", type=int, required=True,
                        help="Volume safety budget preselected on training data")
    parser.add_argument("--allowed-new-voxels", type=Path,
                        help="Optional train-defined CT support boolean .npy")
    parser.add_argument("--output-mask", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    report = run_from_files(
        routes=args.routes, scene_id=args.scene_id, mask_file=args.broken_mask,
        affine_file=args.affine, output_mask=args.output_mask,
        output_report=args.output_report, radius_mm=args.radius_mm,
        max_new_voxels=args.max_new_voxels, allowed_file=args.allowed_new_voxels,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
