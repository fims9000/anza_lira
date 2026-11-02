#!/usr/bin/env python3
"""Run experimental CT28 Graph-LIRA on one *predicted* 3D CCTA vessel mask.

Examples:
 python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_cli.py \\
   --ct scan.nii.gz --mask predicted_vessels.nii.gz --patient-id external_001 \\
   --mode candidates --output-mask unchanged.nii.gz --report candidates.json

For the K10/15 mode, supply the frozen local relation_models.joblib pack
along with a preselected radius and added-voxel budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from ccta_3d_k10 import K10Inference
from ccta_3d_runtime import RuntimeConfig, frozen_path_protocol, run_volume


DEFAULT_PATH = Path(__file__).resolve().parents[3] / (
    "artifacts/varvara/ct28_graph_lira/path/selected_path_protocol.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pair(ct_path: Path, mask_path: Path, threshold: float):
    """Require shared voxel grid; do not resample silently."""
    if not np.isfinite(threshold):
        raise ValueError("Mask threshold is not finite")
    ct_nii = nib.load(str(ct_path))
    mask_nii = nib.load(str(mask_path))
    if len(ct_nii.shape) != 3 or len(mask_nii.shape) != 3 or ct_nii.shape != mask_nii.shape:
        raise ValueError("CT and predicted mask must have identical 3D shape")
    if not np.allclose(ct_nii.affine, mask_nii.affine, rtol=0, atol=1e-4):
        raise ValueError("CT and predicted mask affine mismatch; register explicitly")
    ct = np.asarray(ct_nii.get_fdata(dtype=np.float32))
    raw_mask = np.asanyarray(mask_nii.dataobj)
    if not np.isfinite(raw_mask).all():
        raise ValueError("Predicted mask has non-finite voxels")
    mask = np.asarray(raw_mask > threshold, dtype=bool)
    if not mask.any():
        raise ValueError("Predicted mask is entirely empty")
    return ct, mask, mask_nii


def run_cli(
    *,
    ct_path: Path,
    mask_path: Path,
    patient_id: str,
    mode: str,
    model_pack: Path | None,
    output_mask: Path,
    report_path: Path,
    radius_mm: float,
    max_new_voxels: int,
    min_component_voxels: int,
    trace_mm: float,
    max_endpoints: int,
    max_scenes: int,
    threshold: float,
    path_protocol: Path = DEFAULT_PATH,
    output_additions: Path | None = None,
    strict_count_support: bool = False,
    engine_override: object | None = None,
) -> dict:
    if mode not in {"candidates", "k10"}:
        raise ValueError("Mode must be candidates or k10")
    if mode == "k10" and model_pack is None:
        raise ValueError("K10 mode requires the exact frozen relation_models.joblib")
    protected = {ct_path.resolve(), mask_path.resolve(), path_protocol.resolve()}
    if model_pack:
        protected.add(model_pack.resolve())
    if (output_mask.resolve() in protected or report_path.resolve() in protected or
            report_path.resolve() == output_mask.resolve()):
        raise ValueError("Never overwrite CT, predicted mask, model pack or path protocol")
    if output_additions is not None and (
        output_additions.resolve() in protected or
        output_additions.resolve() in {output_mask.resolve(), report_path.resolve()}
    ):
        raise ValueError("Overlay path conflicts with inputs or other outputs")
    if output_mask.exists() or report_path.exists() or (
        output_additions is not None and output_additions.exists()
    ):
        raise FileExistsError("Immutable outputs already exist; choose a new directory")
    if not output_mask.name.endswith(".nii.gz") and output_mask.suffix != ".nii":
        raise ValueError("Output mask must be .nii or .nii.gz")
    if output_additions is not None and not output_additions.name.endswith(".nii.gz"):
        raise ValueError("Overlay must be a .nii.gz NIfTI")
    if report_path.suffix != ".json":
        raise ValueError("Report must end in .json")
    # Block test patients before any CCTA bytes are opened.
    from ccta_3d_runtime import HELD_OUT_CT28_IDS
    if str(patient_id).strip() in HELD_OUT_CT28_IDS:
        raise ValueError("Held-out CT28 patient prohibited before pre-registered test")
    tic = time.perf_counter()
    ct, mask, mask_nii = load_pair(ct_path, mask_path, threshold)
    path_config, slack = frozen_path_protocol(path_protocol)
    if engine_override is not None and mode != "k10":
        raise ValueError("Experimental engine override requires k10-style inference mode")
    engine = (
        engine_override if engine_override is not None else
        K10Inference.from_file(model_pack) if mode == "k10" else None
    )
    result, report = run_volume(
        ct, mask, mask_nii.affine,
        patient_label=patient_id, engine=engine,
        params=RuntimeConfig(
            radius_mm=radius_mm, max_new_voxels=max_new_voxels,
            min_component_voxels=min_component_voxels,
            trace_mm=trace_mm, max_endpoints=max_endpoints,
            max_scenes=max_scenes,
        ),
        path_config=path_config, slack_fraction=slack,
        strict_count_support=strict_count_support,
    )
    report["input_provenance"] = {
        "ct_path": str(ct_path.resolve()),
        "ct_sha256": sha256(ct_path),
        "mask_path": str(mask_path.resolve()),
        "mask_sha256": sha256(mask_path),
        "path_protocol_sha256": sha256(path_protocol),
        "model_pack_path": str(model_pack.resolve()) if model_pack else None,
        "mask_threshold": threshold,
        "mask_affine": mask_nii.affine.tolist(),
    }
    report["elapsed_seconds"] = round(time.perf_counter() - tic, 3)
    if engine_override is not None:
        report["experimental_encoder_variant"] = getattr(
            engine_override, "variant", "unknown_experimental_encoder"
        )
    out_nii = nib.Nifti1Image(
        result.astype(np.uint8), mask_nii.affine,
        header=mask_nii.header.copy(),
    )
    out_nii.header.set_data_dtype(np.uint8)
    # Write into temporary sibling files first, then promote only verified NIfTI.
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_nii = output_mask.with_name(".tmp_" + output_mask.name)
    temp_json = report_path.with_name(".tmp_" + report_path.name)
    temp_additions = (
        output_additions.with_name(".tmp_" + output_additions.name)
        if output_additions is not None else None
    )
    if temp_nii.exists() or temp_json.exists() or (
        temp_additions is not None and temp_additions.exists()
    ):
        raise FileExistsError("Temporary evidence file already exists")
    try:
        nib.save(out_nii, str(temp_nii))
        verify = nib.load(str(temp_nii))
        if verify.shape != mask_nii.shape or not np.allclose(
            verify.affine, mask_nii.affine, atol=1e-4, rtol=0
        ):
            raise RuntimeError("Written NIfTI geometry does not reproduce input")
        if not np.array_equal(np.asanyarray(verify.dataobj) > 0, result):
            raise RuntimeError("Written NIfTI mask content does not reproduce result")
        report["output_mask"] = str(output_mask.resolve())
        report["output_added_voxels"] = str(output_additions.resolve()) if output_additions else None
        if temp_additions is not None:
            output_additions.parent.mkdir(parents=True, exist_ok=True)
            added = result & ~mask
            overlay = nib.Nifti1Image(added.astype(np.uint8), mask_nii.affine,
                                      header=mask_nii.header.copy())
            overlay.header.set_data_dtype(np.uint8)
            nib.save(overlay, str(temp_additions))
            if not np.array_equal(np.asanyarray(nib.load(str(temp_additions)).dataobj) > 0, added):
                raise RuntimeError("NIfTI added-voxel overlay did not verify")
        temp_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        if output_mask.exists() or report_path.exists():
            raise FileExistsError("Output appeared during inference; refusing overwrite")
        os.rename(temp_nii, output_mask)
        if temp_additions is not None:
            os.rename(temp_additions, output_additions)
        os.rename(temp_json, report_path)
    finally:
        if temp_nii.exists():
            temp_nii.unlink()
        if temp_json.exists():
            temp_json.unlink()
        if temp_additions is not None and temp_additions.exists():
            temp_additions.unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True,
                        help="Predicted broken vessel mask, NOT reference segmentation")
    parser.add_argument("--patient-id", required=True,
                        help="External/train/validation ID; known held-out CT28 patients blocked")
    parser.add_argument("--mode", choices=("candidates", "k10"), default="candidates")
    parser.add_argument("--model-pack", type=Path,
                        help="Trusted local K10/15 SHA-pinned relation_models.joblib")
    parser.add_argument("--path-protocol", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--radius-mm", type=float, default=0.0,
                        help="Train-selected 3D tube radius; default zero for centerline pilot")
    parser.add_argument("--max-new-voxels", type=int, default=250,
                        help="Train-selected maximum new voxels per accepted local repair")
    parser.add_argument("--min-component-voxels", type=int, default=12)
    parser.add_argument("--trace-mm", type=float, default=3.0)
    parser.add_argument("--max-endpoints", type=int, default=256)
    parser.add_argument("--max-scenes", type=int, default=128)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--output-mask", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-additions", type=Path,
                        help="Optional NIfTI overlay containing newly added voxels only")
    parser.add_argument("--strict-count-support", action="store_true",
                        help="Fail closed for a candidate-count pattern unseen in 17 CT28 train subjects")
    args = parser.parse_args()
    report = run_cli(
        ct_path=args.ct, mask_path=args.mask, patient_id=args.patient_id,
        mode=args.mode, model_pack=args.model_pack,
        output_mask=args.output_mask, report_path=args.report,
        radius_mm=args.radius_mm, max_new_voxels=args.max_new_voxels,
        min_component_voxels=args.min_component_voxels,
        trace_mm=args.trace_mm, max_endpoints=args.max_endpoints,
        max_scenes=args.max_scenes, threshold=args.mask_threshold,
        path_protocol=args.path_protocol, output_additions=args.output_additions,
        strict_count_support=args.strict_count_support,
    )
    print(json.dumps({
        "status": report["status"],
        "mode": args.mode,
        "patient_label": report["patient_label"],
        "scenes": report["scene_count"],
        "accepted": report["accepted_decisions"],
        "added_voxels": report["added_voxels"],
        "output_mask": str(args.output_mask),
        "report": str(args.report),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
