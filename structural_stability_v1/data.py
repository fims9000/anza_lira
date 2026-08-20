"""CRACKS location, provenance, rank split, and historical T1 audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from PIL import Image

from cracks_experiment.partial_label_evaluation import T1_ROOT
from cracks_experiment.partial_label_training import T1_PROTOCOL, T1RunSpec, _model, t1_protocol_hash
from cracks_experiment.partial_labels import audit_nonexpert_annotations
from cracks_experiment.training import NORMALIZATION
from lira_final.protocol import HELDOUT_ANNOTATORS, TRAIN_ANNOTATORS
from structural_stability_v1.protocol import PROTOCOL, ROOT, canonical_hash, protocol_hash


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_cracks_root() -> tuple[Path, str]:
    candidates: list[tuple[Path, str]] = []
    candidates.append((ROOT / "data/cracks", "project_config"))
    if os.environ.get("CRACKS_ROOT"):
        candidates.append((Path(os.environ["CRACKS_ROOT"]), "CRACKS_ROOT"))
    inventory = ROOT / "results/cracks_study/archive_inventory.json"
    if inventory.is_file():
        candidates.append((ROOT / "data/cracks", "existing_inventory"))
    for candidate, source in candidates:
        if (candidate / "images").is_dir() and (candidate / "annotations").is_dir():
            return candidate.resolve(), source
    raise FileNotFoundError("STOP_SS_CRACKS_DATA_NOT_FOUND")


def _section_ids(paths: list[Path]) -> set[int]:
    return {int(path.stem.split("_")[-1]) for path in paths}


def _tree_hash(root: Path, files: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        row = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        rows.append(row)
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest(), rows


def build_split(common_ids: list[int]) -> dict[str, list[int]]:
    if len(common_ids) < 341:
        raise ValueError("fewer than 341 common valid sections cannot satisfy the frozen rank split")
    ordered = list(sorted(map(int, common_ids)))
    return {
        "SS_TRAIN": ordered[0:220], "BUFFER_1": ordered[220:230],
        "SS_CALIBRATION": ordered[230:270], "BUFFER_2": ordered[270:280],
        "SS_DEVELOPMENT": ordered[280:330], "BUFFER_3": ordered[330:340],
        "SS_CONFIRM": ordered[340:],
    }


def git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE_DIRTY_WORKTREE"


def audit_dataset(output: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cracks_root, location_source = locate_cracks_root()
    image_files = sorted((cracks_root / "images").glob("section_*.png"))
    annotation_dirs = sorted(path for path in (cracks_root / "annotations").iterdir() if path.is_dir())
    nonexpert_dirs = [path for path in annotation_dirs if path.name != "expert"]
    expert_dir = cracks_root / "annotations/expert"
    annotation_files = sorted(path for directory in annotation_dirs for path in directory.glob("section_*.png"))
    nonexpert_files = sorted(path for directory in nonexpert_dirs for path in directory.glob("section_*.png"))
    expert_files = sorted(expert_dir.glob("section_*.png")) if expert_dir.is_dir() else []
    image_ids = _section_ids(image_files)
    nonexpert_ids = _section_ids(nonexpert_files)
    common_ids = sorted(image_ids & nonexpert_ids)
    shapes = {}
    invalid_images = []
    for path in image_files:
        try:
            with Image.open(path) as image:
                key = f"{image.height}x{image.width}x{len(image.getbands())}"
                shapes[key] = shapes.get(key, 0) + 1
        except Exception as error:  # pragma: no cover - corruption evidence path
            invalid_images.append({"path": str(path), "error": str(error)})
    dataset_sha, file_rows = _tree_hash(cracks_root, image_files + annotation_files)
    nonexpert_audit = audit_nonexpert_annotations(cracks_root / "annotations", tuple(path.name for path in nonexpert_dirs), common_ids)
    split = build_split(common_ids)
    all_members = [section for values in split.values() for section in values]
    if len(all_members) != len(set(all_members)) or sorted(all_members) != common_ids:
        raise AssertionError("rank split is overlapping or incomplete")
    manifest = {
        "protocol_sha256": protocol_hash(), "dataset_root": str(cracks_root), "dataset_root_sha256": dataset_sha,
        "dataset_file_count": len(file_rows), "ordered_common_valid_section_ids": common_ids,
        "missing_integer_section_ids_1_400": [section for section in range(1, 401) if section not in common_ids],
        "split_basis": "rank in ordered common valid section IDs because count != 400",
        "splits": split, "buffers_used_for_training_or_selection": False,
        "git_sha": git_sha(), "frozen_before_ss1_metrics": True,
    }
    manifest["sha256"] = canonical_hash(manifest)
    previous_expert_complete = ROOT / "results/anza_v2_study/cracks/setting_a_expert/complete.json"
    expert_hash, expert_rows = _tree_hash(cracks_root, expert_files)
    expert_provenance = {
        "expert_files_present": bool(expert_files), "expert_file_count": len(expert_files),
        "expert_files_hash": expert_hash, "expert_file_manifest": expert_rows,
        "expert_label_pixels_loaded_by_ss0_ss1": False,
        "expert_previously_accessed": previous_expert_complete.is_file(),
        "evidence": [
            "results/anza_v2_study/cracks/setting_a_expert/complete.json",
            "results/anza_v2_study/cracks/archive_audit/expert_availability_audit.json",
            "results/cracks_study/expert_color_audit.json",
        ],
        "expert_final_confirm_allowed": "HISTORICALLY_EXPOSED_FINAL_EVALUATION_ONLY",
        "untouched_expert_claim_allowed": False,
    }
    audit = {
        "status": "SS_S0_DATA_AUDIT_PASS", "location_source": location_source, "dataset_root": str(cracks_root),
        "dataset_root_sha256": dataset_sha, "images": len(image_files), "image_shapes": shapes,
        "invalid_images": invalid_images, "annotation_files": len(annotation_files),
        "nonexpert_annotation_files": len(nonexpert_files), "expert_annotation_files": len(expert_files),
        "annotator_directories": {path.name: len(list(path.glob("section_*.png"))) for path in annotation_dirs},
        "common_valid_sections": len(common_ids), "common_valid_section_ids": common_ids,
        "nonexpert_palette_audit": nonexpert_audit,
        "official_semantics": {"blue": "certain fault", "green": "uncertain fault", "orange": "certain no-fault", "white": "unknown/zero weight"},
        "old_manifests": [
            "results/cracks_study/archive_inventory.json", "results/anza_v2_study/protocol.json",
            "results/final_practical_cycle/cracks_t1/audit.json",
        ],
        "expert_pixels_used": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_file_manifest.json").write_text(json.dumps(file_rows, indent=2, sort_keys=True) + "\n")
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "split_manifest.sha256").write_text(manifest["sha256"] + "\n")
    (output / "EXPERT_PROVENANCE.json").write_text(json.dumps(expert_provenance, indent=2, sort_keys=True) + "\n")
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (output / "CRACKS_SS_DATA_AUDIT.md").write_text(
        "# CRACKS Structural Stability V1 data audit\n\n"
        "The existing local CRACKS extraction was used; no network download occurred.\n\n"
        f"- Dataset root: `{cracks_root}`\n- Dataset SHA-256: `{dataset_sha}`\n"
        f"- Images: `{len(image_files)}`; annotations: `{len(annotation_files)}`; common valid sections: `{len(common_ids)}`\n"
        f"- Missing integer IDs 1-400: `{manifest['missing_integer_section_ids_1_400']}`\n"
        f"- Image shapes: `{shapes}`; corrupt images: `{len(invalid_images)}`\n"
        "- Frozen semantics: blue certain fault, green uncertain fault, orange certain no-fault, white unknown with zero supervision weight.\n"
        f"- Expert files: `{len(expert_files)}`; historically accessed: `{expert_provenance['expert_previously_accessed']}`. They are not untouched and were not decoded by SS0/SS1.\n"
        f"- Rank split SHA-256: `{manifest['sha256']}`. Buffer sections are excluded.\n\nSS_S0_PASS\n"
    )
    return audit, manifest, expert_provenance


def backbone_provenance(output: Path) -> dict[str, Any]:
    checkpoints = []
    for seed in (41, 42, 43):
        spec = T1RunSpec(f"t1_unet_s{seed}", "unet", seed)
        run_dir = T1_ROOT / f"{spec.run_id}-{spec.run_hash}"
        checkpoint = run_dir / "checkpoint-last.pt"
        status = json.loads((run_dir / "status.json").read_text())
        if status.get("status") != "COMPLETE" or status.get("expert_data_accessed") is not False:
            raise ValueError(f"historical T1 provenance invalid for seed {seed}")
        checkpoints.append({"seed": seed, "run_id": spec.run_id, "run_hash": spec.run_hash, "path": str(checkpoint.relative_to(ROOT)), "sha256": sha256_file(checkpoint)})
    model = _model(T1RunSpec("t1_unet_s41", "unet", 41))
    evaluation = json.loads((T1_ROOT / "evaluation/t1_unet_s41/evaluation.json").read_text())
    payload = {
        "status": "BACKBONE_PROVENANCE_FROZEN", "selected_h0": "t1_unet_s41",
        "selection_rule": "canonical earliest-seed frozen final T1 U-Net; no SS development result used",
        "source_module": "cracks_experiment.partial_label_training._model -> cracks_experiment.training.build_real_model",
        "git_sha": git_sha(), "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "historical_checkpoints": checkpoints, "protocol_sha256": t1_protocol_hash(),
        "optimizer": T1_PROTOCOL["optimizer"], "learning_rate": T1_PROTOCOL["learning_rate"], "epochs": T1_PROTOCOL["epochs"],
        "effective_batch_size": T1_PROTOCOL["effective_batch_size"], "crop_size": T1_PROTOCOL["crop_size"],
        "foreground_crop_probability": T1_PROTOCOL["foreground_crop_probability"],
        "normalization": NORMALIZATION, "augmentations": "no image augmentation in historical T1; deterministic foreground-aware crop selection only",
        "historical_selected_threshold": evaluation["selected_threshold"],
        "historical_evaluation_sha256": sha256_file(T1_ROOT / "evaluation/t1_unet_s41/evaluation.json"),
        "expert_scores_used_for_recipe_selection": False,
    }
    (output / "BACKBONE_PROVENANCE.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload

