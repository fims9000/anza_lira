#!/usr/bin/env python3
"""One-command reproducible CT28 train-patient CCTA repair pilot.

End-to-end: original ImageCAS CCTA archive -> verified CT -> damaged training
reference mask -> synthetic controlled gap -> unlabelled candidate inspection
-> frozen K10/15 + path v2 + mask repair -> posthoc 3D metrics -> manifest.

The original ground truth is NEVER passed to K10/path/mask inference.
Only CT28 training patients are authorized. A run is create-only and retains
failed execution evidence; medical image data stays in the ignored local
workspace, NEVER in Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np

from ccta_3d_cli import run_cli, DEFAULT_PATH
from ccta_3d_evaluate import evaluate_files
from ccta_3d_export_training_ct import export_training_ct, ARCHIVE
from ccta_3d_k10 import K10_15_SHA256
from ccta_3d_train_gap_simulator import (
    TRAIN_CT28_IDS,
    synthesize_train_gaps,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_DIR = REPO_ROOT / (
    "varvara/anza_lira_local_data/Varvara_CT28_junction_source_pack/segmentations"
)
SELECTED_PACK = REPO_ROOT / (
    "varvara/anza_lira_local_results/ct28_copy_count_ablations/"
    "joint_check/fusion_10_relation_15/relation_models.joblib"
)
DEFAULT_ROOT = REPO_ROOT / "varvara/anza_lira_local_results/ccta_3d_runtime_pilot"


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _reference_mask(
    reference_path: Path, ct_path: Path,
) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    ct = nib.load(str(ct_path))
    ref = nib.load(str(reference_path))
    if ct.ndim != 3 or ref.ndim != 3 or ct.shape != ref.shape:
        raise ValueError("Reference training mask and CCTA grid differ")
    if not np.allclose(ct.affine, ref.affine, atol=1e-4, rtol=0):
        raise ValueError("Reference training mask and CCTA affine differ")
    volume = np.asanyarray(ref.dataobj)
    if not np.isfinite(volume).all():
        raise ValueError("Nonfinite reference mask")
    return volume > .5, ref.affine, ref


def execute_train_pilot(
    *,
    scan_id: int,
    output_dir: Path,
    reference_path: Path | None = None,
    archive_dir: Path = ARCHIVE,
    model_pack: Path | None = SELECTED_PACK,
    mode: str = "k10",
    gaps: int = 3,
    gap_radius_mm: float = 1.5,
    repair_radius_mm: float = 0.0,
    max_new_voxels: int = 250,
    min_component_voxels: int = 12,
    trace_mm: float = 3.0,
    max_endpoints: int = 256,
    max_scenes: int = 128,
    gap_seed: int = 20261102,
    path_protocol: Path = DEFAULT_PATH,
) -> dict:
    # Check the firewall and immutable output before opening any local scan.
    if scan_id not in TRAIN_CT28_IDS:
        raise ValueError("CT28 pilot is train-only: validation/test access forbidden")
    if mode not in {"candidates", "k10"}:
        raise ValueError("Pilot mode must be candidates or k10")
    if output_dir.exists():
        raise FileExistsError("Pilot directory already exists; experiments are immutable")
    reference_path = reference_path or REFERENCE_DIR / f"{scan_id}.coronary.nii.gz"
    if mode == "k10":
        if model_pack is None or not model_pack.is_file():
            raise FileNotFoundError("Selected K10/15 model pack not found")
        if digest(model_pack) != K10_15_SHA256:
            raise ValueError("Selected K10/15 model SHA mismatch")
    if not reference_path.is_file():
        raise FileNotFoundError(f"Training coronary reference mask missing: {reference_path}")
    if not path_protocol.is_file():
        raise FileNotFoundError("Frozen CT path-v2 protocol file not found")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(
        prefix=f".pilot_{scan_id}_staging_", dir=output_dir.parent
    ))
    summary: dict = {}
    try:
        ct_file = work / "ct.nii.gz"
        ct_meta = export_training_ct(
            scan_id, archive_dir, ct_file, work / "ct_export.json"
        )
        original, affine, reference_nifti = _reference_mask(
            reference_path, ct_file
        )
        damaged, gaps_meta = synthesize_train_gaps(
            original, affine, patient_id=scan_id, gaps=gaps,
            radius_mm=gap_radius_mm, seed=gap_seed
        )
        broken_path = work / "broken_mask.nii.gz"
        broken = nib.Nifti1Image(
            damaged.astype(np.uint8), affine,
            header=reference_nifti.header.copy()
        )
        broken.header.set_data_dtype(np.uint8)
        nib.save(broken, str(broken_path))
        (work / "gaps.json").write_text(
            json.dumps(gaps_meta, indent=2) + "\n", encoding="utf-8"
        )
        inspector = run_cli(
            ct_path=ct_file, mask_path=broken_path,
            patient_id=str(scan_id), mode="candidates", model_pack=None,
            output_mask=work / "inspection_unchanged.nii.gz",
            report_path=work / "candidates.json",
            radius_mm=repair_radius_mm, max_new_voxels=max_new_voxels,
            min_component_voxels=min_component_voxels, trace_mm=trace_mm,
            max_endpoints=max_endpoints, max_scenes=max_scenes,
            threshold=.5, path_protocol=path_protocol,
        )
        if mode == "k10":
            decision = run_cli(
                ct_path=ct_file, mask_path=broken_path,
                patient_id=str(scan_id), mode="k10", model_pack=model_pack,
                output_mask=work / "repaired.nii.gz",
                output_additions=work / "added_voxels.nii.gz",
                report_path=work / "decisions.json",
                radius_mm=repair_radius_mm, max_new_voxels=max_new_voxels,
                min_component_voxels=min_component_voxels, trace_mm=trace_mm,
                max_endpoints=max_endpoints, max_scenes=max_scenes,
                threshold=.5, path_protocol=path_protocol,
            )
            evaluation = evaluate_files(
                patient_id=str(scan_id),
                before=broken_path, after=work / "repaired.nii.gz",
                reference=reference_path,
                output=work / "posthoc_metrics.json",
            )
        else:
            decision = None
            evaluation = None
        summary = {
            "status": "completed_experimental_train_pilot",
            "patient_id": scan_id,
            "mode": mode,
            "source_kind": "controlled_gap_from_reference_not_natural_segmentation_failure",
            "ground_truth_used_in_inference": False,
            "ct_export_sha256": ct_meta["raw_ct_sha256"],
            "reference_sha256": digest(reference_path),
            "model_pack_sha256": digest(model_pack) if mode == "k10" else None,
            "path_protocol_sha256": digest(path_protocol),
            "gap_generation": gaps_meta,
            "candidate_scene_count": inspector["scene_count"],
            "candidate_endpoint_count": inspector["endpoint_count"],
            "accepted_repairs": decision["accepted_decisions"] if decision else None,
            "added_voxels": decision["added_voxels"] if decision else None,
            "dice_delta": evaluation["dice_delta"] if evaluation else None,
            "cldice_delta": evaluation["cldice_delta"] if evaluation else None,
            "files": sorted(p.name for p in work.iterdir() if p.is_file()),
            "limitations": [
                "Only one training patient and artificially damaged reference segmentation.",
                "A successful executable pilot does not prove out-of-patient generalization.",
                "3D ANZA encoder is not present in frozen K10/15.",
            ],
        }
        (work / "pilot_manifest.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as error:
        failure = {
            "status": "failed_experimental_train_pilot",
            "patient_id": scan_id,
            "mode": mode,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=8),
            "completed_files": sorted(p.name for p in work.iterdir() if p.is_file()),
        }
        (work / "pilot_failure.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(work, output_dir)
        raise

    os.rename(work, output_dir)
    # CLI sub-reports were produced under staging. Rewrite only relocated
    # local output paths; source data hashes remain unchanged.
    for json_file in output_dir.glob("*.json"):
        source = json_file.read_text(encoding="utf-8")
        moved = source.replace(str(work), str(output_dir))
        if moved != source:
            json_file.write_text(moved, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id", type=int, default=953)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-mask", type=Path)
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE)
    parser.add_argument("--model-pack", type=Path, default=SELECTED_PACK)
    parser.add_argument("--mode", choices=("candidates", "k10"), default="k10")
    parser.add_argument("--gaps", type=int, default=3)
    parser.add_argument("--gap-radius-mm", type=float, default=1.5)
    parser.add_argument("--repair-radius-mm", type=float, default=0)
    parser.add_argument("--max-new-voxels", type=int, default=250)
    parser.add_argument("--min-component-voxels", type=int, default=12)
    parser.add_argument("--trace-mm", type=float, default=3.)
    parser.add_argument("--max-endpoints", type=int, default=256)
    parser.add_argument("--max-scenes", type=int, default=128)
    parser.add_argument("--gap-seed", type=int, default=20261102)
    parser.add_argument("--path-protocol", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        DEFAULT_ROOT / f"{args.patient_id}_{args.mode}_g{args.gaps}_seed{args.gap_seed}"
    )
    summary = execute_train_pilot(
        scan_id=args.patient_id, output_dir=output_dir,
        reference_path=args.reference_mask,
        archive_dir=args.archive_dir, model_pack=args.model_pack,
        mode=args.mode, gaps=args.gaps, gap_radius_mm=args.gap_radius_mm,
        repair_radius_mm=args.repair_radius_mm,
        max_new_voxels=args.max_new_voxels,
        min_component_voxels=args.min_component_voxels,
        trace_mm=args.trace_mm,
        max_endpoints=args.max_endpoints, max_scenes=args.max_scenes,
        gap_seed=args.gap_seed, path_protocol=args.path_protocol,
    )
    print(json.dumps({
        "status": summary["status"],
        "patient": summary["patient_id"],
        "mode": summary["mode"],
        "scenes": summary["candidate_scene_count"],
        "accepted_repairs": summary["accepted_repairs"],
        "dice_delta": summary["dice_delta"],
        "cldice_delta": summary["cldice_delta"],
        "directory": str(output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
