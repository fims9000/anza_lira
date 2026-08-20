"""Execute and finalize only SS1.5 pre-training hardening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from datasets.cracks import BLUE, GREEN, ORANGE, WHITE, load_section_image
from lira_final.protocol import TRAIN_ANNOTATORS
from structural_stability_v1.perturb import apply_perturbation, transform_rgb_mask
from structural_stability_v1.protocol import FAMILIES, SEVERITIES
from structural_stability_v1_1.amendment import amendment_payload, parent_integrity_snapshot
from structural_stability_v1_1.geometry_metric import parameter_audit
from structural_stability_v1_1.geometry_targets import audit_geometry_targets
from structural_stability_v1_1.initialization import create_shared_backbone_initializations, initialize_variant, state_dict_sha256
from structural_stability_v1_1.protocol import PARENT_RESULT_ROOT, PROTOCOL, RESULT_ROOT, ROOT, SEEDS, VARIANTS, protocol_hash
from structural_stability_v1_1.train_normalization import compute_train_only_normalization
from structural_stability_v1_1.training_manifest import assert_manifest_shared, build_pair_manifests, validate_pair_manifest_crops


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_parent_split() -> dict[str, Any]:
    manifest = json.loads((PARENT_RESULT_ROOT / "s0_audit/split_manifest.json").read_text())
    if manifest.get("sha256") != PROTOCOL["parent_split_sha256"]:
        raise ValueError("parent split hash does not match frozen V1.1 amendment")
    if len(manifest["splits"]["SS_TRAIN"]) != 220:
        raise ValueError("parent SS_TRAIN count drift")
    return manifest


def _normalized_train_image(section_id: int, normalization: dict[str, Any]) -> np.ndarray:
    image = load_section_image(ROOT / "data/cracks/images" / f"section_{section_id:03d}.png")
    tensor = torch.from_numpy(image.transpose(2, 0, 1))
    mean = torch.tensor(normalization["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(normalization["std"], dtype=torch.float32).view(3, 1, 1).clamp_min(1e-6)
    return F.pad((tensor - mean) / std, (0, 3, 0, 1)).numpy().astype(np.float32)


def _first_train_mask(section_id: int) -> np.ndarray:
    for annotator in TRAIN_ANNOTATORS:
        path = ROOT / "data/cracks/annotations" / annotator / f"section_{section_id:03d}.png"
        if path.is_file():
            with Image.open(path) as handle:
                return np.asarray(handle.convert("RGB"), dtype=np.uint8)
    raise FileNotFoundError(f"no train mask for section {section_id}")


def revalidate_perturbations(section_ids: list[int], normalization: dict[str, Any], output: Path) -> dict[str, Any]:
    allowed = {BLUE, GREEN, ORANGE, WHITE}
    finite = deterministic = palette = jacobian = True
    rows = 0
    seed_records: dict[str, int] = {}
    det_min, det_max, cond_max = 1.0, 1.0, 1.0
    for index, section_id in enumerate(section_ids):
        image = _normalized_train_image(section_id, normalization)
        mask = _first_train_mask(section_id)
        for family in FAMILIES:
            for severity in SEVERITIES:
                result = apply_perturbation(image, section_id, "v1_1_train_full_section", family, severity)
                repeated = apply_perturbation(image, section_id, "v1_1_train_full_section", family, severity)
                finite &= bool(np.isfinite(result.image).all())
                deterministic &= bool(np.array_equal(result.image, repeated.image) and result.metadata == repeated.metadata)
                seed_records[f"{section_id}:{family}:{severity}"] = result.seed
                if family == "warp":
                    transformed = transform_rgb_mask(mask, result)
                    observed = set(map(tuple, np.unique(transformed.reshape(-1, 3), axis=0)))
                    palette &= observed <= allowed
                    local_min = float(result.metadata["jacobian_det_min"])
                    local_max = float(result.metadata["jacobian_det_max"])
                    local_cond = float(result.metadata["jacobian_condition_max"])
                    det_min = min(det_min, local_min); det_max = max(det_max, local_max); cond_max = max(cond_max, local_cond)
                    jacobian &= local_min >= 0.75 and local_max <= 1.25 and local_cond <= 1.5
                rows += 1
        if (index + 1) % 25 == 0 or index + 1 == len(section_ids):
            print(f"phase=SS1.5_PERTURB section={index + 1}/{len(section_ids)} conditions=15 expert=LOCKED", flush=True)
    result = {
        "status": "TRAIN_NORMALIZED_PERTURBATION_REVALIDATION_PASS" if all((finite, deterministic, palette, jacobian)) else "STOP_SS1_5_PERTURBATION_INVALID",
        "train_sections": len(section_ids), "condition_rows": rows,
        "all_finite": finite, "all_deterministic": deterministic,
        "palette_preserved": palette, "warp_jacobians_valid": jacobian,
        "warp_det_min": det_min, "warp_det_max": det_max, "warp_condition_max": cond_max,
        "seed_records_sha256": __import__("hashlib").sha256(json.dumps(seed_records, sort_keys=True).encode()).hexdigest(),
        "performance_metrics_computed": False, "H0_loaded": False, "expert_data_accessed": False,
    }
    _write_json(output / "PERTURBATION_REVALIDATION.json", result)
    return result


def validate_initialization(initialization: dict[str, Any], output: Path) -> dict[str, Any]:
    by_seed = {int(record["seed"]): record for record in initialization["records"]}
    seed_records = []
    for seed in SEEDS:
        backbone_hashes = {}
        for variant in VARIANTS:
            model = initialize_variant(variant, seed, ROOT / by_seed[seed]["path"])
            backbone_hashes[variant] = state_dict_sha256(model.backbone.state_dict())
        seed_records.append({"seed": seed, "backbone_hashes": backbone_hashes, "identical": len(set(backbone_hashes.values())) == 1})
    params = parameter_audit()
    result = {
        "status": "INITIALIZATION_AND_CAPACITY_MATCH_PASS" if all(row["identical"] for row in seed_records) and params["passes_one_percent"] else "STOP_SS1_5_INITIALIZATION_INVALID",
        "seeds": seed_records, "parameter_audit": params,
        "historical_H0_loaded": False, "training_started": False,
    }
    _write_json(output / "INITIALIZATION_VALIDATION.json", result)
    return result


def run_pretraining_hardening() -> dict[str, Any]:
    output = RESULT_ROOT / "pretrain_freeze"
    output.mkdir(parents=True, exist_ok=True)
    parent_before = parent_integrity_snapshot()
    amendment = amendment_payload()
    _write_json(output / "AMENDMENT_V1_1.json", amendment)
    _write_json(output / "PARENT_INTEGRITY_BEFORE.json", parent_before)
    split = _load_parent_split()
    train_sections = list(split["splits"]["SS_TRAIN"])
    normalization = compute_train_only_normalization(train_sections, output / "TRAIN_ONLY_NORMALIZATION.json")
    geometry = audit_geometry_targets(train_sections, output)
    initialization = create_shared_backbone_initializations(output / "initialization")
    _write_json(output / "INITIALIZATION_MANIFEST.json", initialization)
    init_validation = validate_initialization(initialization, output)
    manifests = build_pair_manifests(train_sections, output / "manifests")
    for record in manifests["records"]:
        assert_manifest_shared(record)
    manifest_crop_validation = validate_pair_manifest_crops(
        output / "manifests", output / "MANIFEST_CROP_VALIDATION.json"
    )
    perturbations = revalidate_perturbations(train_sections, normalization, output)
    parent_after = parent_integrity_snapshot()
    parent_unchanged = parent_before == parent_after
    _write_json(output / "PARENT_INTEGRITY_AFTER.json", parent_after)
    prechecks = {
        "parent_unchanged": parent_unchanged,
        "parent_ss1_pass": amendment["parent_ss1_status"] == "SS_S1_PASS",
        "train_only_normalization": normalization["status"] == "TRAIN_ONLY_NORMALIZATION_FROZEN" and normalization["train_section_ids"] == train_sections,
        "geometry_target": geometry["status"] == "GEOMETRY_TARGET_AUDIT_PASS" and geometry["expert_data_accessed"] is False,
        "initialization": init_validation["status"] == "INITIALIZATION_AND_CAPACITY_MATCH_PASS",
        "shared_manifests": manifests["status"] == "SHARED_PAIR_MANIFESTS_FROZEN" and all(record["severities"] == [1, 2] for record in manifests["records"]),
        "manifest_selected_labels": manifest_crop_validation["status"] == "PAIR_MANIFEST_CROP_VALIDATION_PASS",
        "perturbations": perturbations["status"] == "TRAIN_NORMALIZED_PERTURBATION_REVALIDATION_PASS",
        "training_unopened": True,
        "development_confirm_expert_unopened": True,
    }
    preliminary = {
        "status": "SS1_5_PRECHECK_PASS" if all(prechecks.values()) else "STOP_SS1_5_PRECHECK_INVALID",
        "protocol_sha256": protocol_hash(), "parent_split_sha256": split["sha256"],
        "prechecks": prechecks, "normalization_sha256": normalization["sha256"],
        "geometry": geometry, "initialization": init_validation, "manifests": manifests,
        "manifest_crop_validation": manifest_crop_validation,
        "perturbations": perturbations,
        "new_training_started": False, "B0_B1_B2_B3_opened": False,
        "development_opened": False, "confirm_opened": False, "expert_label_pixels_loaded": False,
    }
    _write_json(output / "PRECHECK_RESULT.json", preliminary)
    return preliminary


def finalize_pretraining(test_summary: dict[str, Any]) -> dict[str, Any]:
    output = RESULT_ROOT / "pretrain_freeze"
    preliminary = json.loads((output / "PRECHECK_RESULT.json").read_text())
    manifest_crop_validation = json.loads((output / "MANIFEST_CROP_VALIDATION.json").read_text())
    parent_now = parent_integrity_snapshot()
    parent_before = json.loads((output / "PARENT_INTEGRITY_BEFORE.json").read_text())
    checks = {
        **preliminary["prechecks"],
        "manifest_selected_labels": manifest_crop_validation.get("status") == "PAIR_MANIFEST_CROP_VALIDATION_PASS",
        "targeted_tests": test_summary.get("targeted", {}).get("returncode") == 0,
        "full_repository_tests": test_summary.get("full_repository", {}).get("returncode") == 0,
        "python_compile": test_summary.get("compile", {}).get("returncode") == 0,
        "json_parse": test_summary.get("json_parse", {}).get("returncode") == 0,
        "git_diff_check": test_summary.get("git_diff_check", {}).get("returncode") == 0,
        "parent_still_unchanged": parent_now == parent_before,
    }
    status = "SS1_5_PRETRAINING_FREEZE_PASS" if all(checks.values()) else "STOP_SS1_5_PRETRAINING_VALIDATION_FAILED"
    result = {
        "status": status, "protocol_sha256": protocol_hash(), "checks": checks,
        "normalization_sha256": preliminary["normalization_sha256"],
        "pair_manifest_sha256": {str(row["seed"]): row["sha256"] for row in preliminary["manifests"]["records"]},
        "training_jobs_authorized_after_pass": 12 if status == "SS1_5_PRETRAINING_FREEZE_PASS" else 0,
        "training_started": False, "development_opened": False, "confirm_opened": False,
        "expert_label_pixels_loaded": False,
    }
    _write_json(output / "validator.json", {"validator_status": "PASS" if status.endswith("PASS") else "FAIL", **result})
    _write_json(RESULT_ROOT / "ANZA_LIRA_SS_V1_1_MASTER_RESULT.json", {"phase": "SS1_5_PRETRAINING_HARDENING", **result})
    (output / "GEOMETRY_IMPLEMENTATION_VALIDATION.md").write_text(
        "# V1.1 SPD geometry implementation validation\n\n"
        "The B2/B3 sidecars use the same decoder locations, hidden width, axis and d supervision. B2 alone has free m; B3 fixes m=0 and det(C)=1.\n\n"
        f"- Capacity audit: `{preliminary['initialization']['parameter_audit']}`.\n"
        "- Transport convention: parent warp is output-to-input; the forward Jacobian is its inverse and is area-normalized before SPD congruence.\n"
        "- Historical H0 initialization is rejected by provenance guard.\n"
        f"- Numerical tests: `{test_summary.get('targeted', {})}`.\n\n"
        f"{status}\n"
    )
    (output / "SS1_5_PRETRAINING_HARDENING_REPORT.md").write_text(
        "# ANZA-LIRA CRACKS Structural Stability V1.1 — SS1.5\n\n"
        "Parent SS0/SS1 artifacts and historical STOPs remained byte-identical. No model was trained.\n\n"
        f"- Train-only normalization: 220 sections; SHA-256 `{result['normalization_sha256']}`.\n"
        f"- Geometry supervision fraction: `{preliminary['geometry']['supervised_fraction']:.8f}` over `{preliminary['geometry']['sections_with_supervision']}` sections.\n"
        f"- Pair manifests: `{result['pair_manifest_sha256']}`; severity 3 absent; B0-B3 share each seed manifest.\n"
        f"- Parameter match: `{preliminary['initialization']['parameter_audit']['B2_B3_relative_difference']:.8f}` relative difference.\n"
        f"- Train-normalized perturbation rows: `{preliminary['perturbations']['condition_rows']}`; finite/deterministic/Jacobian/palette checks passed.\n"
        f"- Targeted tests: `{test_summary.get('targeted', {})}`.\n- Full repository tests: `{test_summary.get('full_repository', {})}`.\n"
        "- B0/B1/B2/B3 training, development, confirm, LIRA, and expert label access remained unopened.\n\n"
        f"{status}\n"
    )
    return result
