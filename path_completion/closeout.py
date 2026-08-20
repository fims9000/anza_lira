"""Claim-safe closeout for the max-min/path and CleanANZA R0 cycle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from cracks_experiment.clean_anza_evaluation import _heldout_dataset, _r0_runs
from cracks_experiment.training import NORMALIZATION, build_real_model, load_real_checkpoint
from cracks_experiment.validation import tiled_probability


FINAL_STATUS = "MAXMIN_PATH_ORACLE_PASS"
TERMINAL_REASON = "STOP_LEARNED_CONFIRM_AND_CLEAN_ANZA_GATES_FAILED"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_inputs(project_root: Path) -> dict[str, Any]:
    root = Path(project_root)
    oracle = json.loads((root / "results/path_completion/oracle/oracle_summary.json").read_text())
    pair = json.loads((root / "results/path_completion/pair_classifier/result.json").read_text())
    confirm = json.loads((root / "results/path_completion/learned_confirm/result.json").read_text())
    r0 = json.loads((root / "results/maxmin_path_study/cracks/r0_clean_anza/analysis/r0_result.json").read_text())
    checks = {
        "oracle_pass": oracle.get("status") == "MAXMIN_PATH_ORACLE_PASS",
        "pair_classifier_pass": pair.get("status") == "ENDPOINT_PAIR_CLASSIFIER_PASS",
        "learned_confirm_frozen_fail": confirm.get("status") == "LEARNED_PATH_SYNTHETIC_CONFIRM_FAIL",
        "clean_anza_r0_frozen_fail": r0.get("status") == "CLEAN_ANZA_REAL_GATE_FAIL",
        "confirm_threshold_unchanged": confirm.get("frozen_pair_classifier", {}).get("threshold") == pair.get("threshold_frozen_from_train"),
        "confirm_no_latent_inference": confirm.get("inference_uses_latent_connectivity") is False and confirm.get("inference_uses_gap_or_instance_truth") is False,
        "v5_test_unopened": oracle.get("test_v5_samples_opened") == pair.get("test_v5_samples_opened") == confirm.get("test_v5_samples_opened") == 0,
        "expert_unopened": oracle.get("expert_data_accessed") is pair.get("expert_data_accessed") is confirm.get("expert_data_accessed") is r0.get("expert_data_accessed") is False,
        "cracks_completion_not_run": confirm.get("cracks_samples_opened") == 0,
    }
    if not all(checks.values()):
        raise ValueError(f"closeout input validation failed: {checks}")
    return {"status": "PASS", "checks": checks, "oracle": oracle, "pair": pair, "confirm": confirm, "r0": r0}


def _selected_section(dataset: Any) -> tuple[int, int, dict[str, Any]]:
    rule = "minimum sha256('maxmin-r0-qualitative-v1:' + section_id); metrics not consulted"
    candidates = list(enumerate(dataset.section_ids))
    index, section_id = min(
        candidates,
        key=lambda item: hashlib.sha256(f"maxmin-r0-qualitative-v1:{item[1]}".encode()).hexdigest(),
    )
    receipt = {
        "selection_rule": rule,
        "selected_section_id": int(section_id),
        "candidate_count": len(candidates),
        "expert_data_accessed": False,
    }
    return int(index), int(section_id), receipt


def _seed42_models(device: torch.device) -> dict[str, tuple[torch.nn.Module, float]]:
    selected: dict[str, tuple[torch.nn.Module, float]] = {}
    for model_name, spec, training_root, threshold, _freeze in _r0_runs():
        if int(spec.seed) != 42:
            continue
        model = build_real_model(spec).to(device)
        checkpoint = Path(training_root) / f"{spec.run_id}-{spec.run_hash}" / "checkpoint-last.pt"
        load_real_checkpoint(checkpoint, spec.run_hash, model)
        selected[model_name] = (model.eval(), float(threshold))
    if set(selected) != {"unet", "anza_v1", "clean_anza"}:
        raise ValueError("seed42 qualitative model set incomplete")
    return selected


def generate_required_figures(project_root: Path, output_root: Path, *, device: str) -> dict[str, Any]:
    root, output = Path(project_root), Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    dataset = _heldout_dataset()
    index, section_id, selection = _selected_section(dataset)
    batch = dataset[index]
    height, width = batch["original_hw"]
    normalized = batch["image"]
    mean = torch.tensor(NORMALIZATION["mean"]).view(3, 1, 1)
    std = torch.tensor(NORMALIZATION["std"]).view(3, 1, 1)
    rgb = (normalized[:, :height, :width] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    target = batch["target"][0, :height, :width].numpy() >= 0.5
    valid = batch["valid"][0, :height, :width].numpy().astype(bool)
    torch_device = torch.device(device)
    models = _seed42_models(torch_device)
    predictions: dict[str, np.ndarray] = {}
    for name, (model, threshold) in models.items():
        probability = tiled_probability(model, normalized).numpy()[:height, :width]
        predictions[name] = (probability >= threshold) & valid

    fig, axes = plt.subplots(2, 4, figsize=(15, 6), constrained_layout=True)
    plain_panels = (
        (axes[0, 0], rgb, "input"),
        (axes[0, 1], target, "crowd reference"),
        (axes[0, 2], predictions["unet"], "U-Net"),
        (axes[1, 0], predictions["anza_v1"], "legacy ANZA"),
        (axes[1, 2], predictions["clean_anza"], "CleanANZA"),
    )
    for axis, image, title in plain_panels:
        axis.imshow(image, cmap=None if image.ndim == 3 else "gray")
        axis.set_title(title)
        axis.axis("off")
    error_cmap = ListedColormap(["black", "#4daf4a", "#e41a1c", "#377eb8"])
    for axis, name, title in (
        (axes[0, 3], "unet", "U-Net errors"),
        (axes[1, 1], "anza_v1", "legacy errors"),
        (axes[1, 3], "clean_anza", "CleanANZA errors"),
    ):
        pred = predictions[name]
        error = np.zeros(target.shape, dtype=np.uint8)
        error[pred & target & valid] = 1
        error[pred & ~target & valid] = 2
        error[~pred & target & valid] = 3
        axis.imshow(error, cmap=error_cmap, vmin=0, vmax=3)
        axis.set_title(title + "\nTP green / FP red / FN blue")
        axis.axis("off")
    fig.suptitle(f"Deterministic crowd-heldout section {section_id}; seed 42; no expert data")
    fig.savefig(output / "fig_cracks_clean_anza.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "fig_cracks_clean_anza.svg", bbox_inches="tight")
    plt.close(fig)

    legacy = models["anza_v1"][0]
    clean = models["clean_anza"][0]
    tile = normalized[:, :256, :256].unsqueeze(0).to(torch_device)
    with torch.inference_mode():
        legacy_logits = legacy.enc1.spatial.gate_conv(tile)
        clean_logits = clean.enc1.spatial.gate_conv(tile)
        legacy_mu = torch.softmax(legacy_logits, dim=1)[0].cpu().numpy()
        clean_mu = torch.sigmoid(clean_logits)[0].cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    for row, (memberships, label) in enumerate(((legacy_mu, "legacy softmax"), (clean_mu, "independent sigmoid"))):
        for mode in range(4):
            axes[row, mode].imshow(memberships[mode], cmap="viridis", vmin=0, vmax=1)
            axes[row, mode].set_title(f"{label}: mode {mode + 1}")
            axes[row, mode].axis("off")
        axes[row, 4].imshow(memberships.sum(axis=0), cmap="magma", vmin=0, vmax=4)
        axes[row, 4].set_title(f"sum memberships\nmean={memberships.sum(axis=0).mean():.3f}")
        axes[row, 4].axis("off")
    fig.suptitle(f"Membership semantics on deterministic crowd section {section_id}, seed 42")
    fig.savefig(output / "fig_clean_anza_memberships.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "fig_clean_anza_memberships.svg", bbox_inches="tight")
    plt.close(fig)
    selection["legacy_membership_sum_mean"] = float(legacy_mu.sum(axis=0).mean())
    selection["clean_membership_sum_mean"] = float(clean_mu.sum(axis=0).mean())
    selection["figures"] = ["fig_cracks_clean_anza", "fig_clean_anza_memberships"]
    (output / "FIGURE_SELECTION_RECEIPT.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    return selection


def _comparison(result: dict[str, Any], metric: str) -> dict[str, Any]:
    return next(row for row in result["comparisons"] if row["comparison"] == "clean_anza_minus_anza_v1" and row["metric"] == metric)


def _write_closeout_documents(output: Path, frozen: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    oracle, pair, confirm, r0 = frozen["oracle"], frozen["pair"], frozen["confirm"], frozen["r0"]
    numbers = {
        "status": FINAL_STATUS,
        "terminal_reason": TERMINAL_REASON,
        "claim_eligibility": {
            "maxmin_oracle_math": True,
            "balanced_pair_discrimination": True,
            "learned_completion_success": False,
            "clean_anza_real_improvement": False,
            "cracks_completion_improvement": False,
        },
        "oracle": {
            "maxmin_gap_recovery": next(row["positive_gap_recovery"] for row in oracle["comparison"] if row["method"] == "maxmin_closure"),
            "widest_path_gap_recovery": next(row["positive_gap_recovery"] for row in oracle["comparison"] if row["method"] == "widest_path"),
            "widest_path_false_bridge": next(row["false_bridge_rate"] for row in oracle["comparison"] if row["method"] == "widest_path"),
        },
        "pair_classifier_validation": pair["validation_metrics"],
        "learned_confirm": {"pair_metrics": confirm["pair_metrics"], "summary": confirm["summary"], "checks": confirm["checks"]},
        "clean_anza_minus_legacy_v1": {metric: _comparison(r0, metric) for metric in ("dice", "auprc", "cldice", "skeleton_f1_at_2px", "precision", "recall")},
        "provenance": {
            "expert_data_accessed": False,
            "synthetic_v5_test_opened": False,
            "confirm_retuning": "FORBIDDEN_NOT_PERFORMED",
            "cracks_completion_run": False,
            "qualitative_selection": selection,
        },
    }
    (output / "THESIS_NUMBERS.json").write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    evidence = f"""# Thesis evidence — max-min path cycle

## What is supported

- Exact max-min/widest-path oracle: PASS (recovery 1.0, false bridge 0.0).
- Balanced endpoint-pair discrimination: PASS (validation AUROC {pair['validation_metrics']['auroc']:.4f}; independent confirm AUROC {confirm['pair_metrics']['auroc']:.4f}).

## What is not supported

- Learned completion did not pass its immutable confirm gate: recovery {confirm['summary']['positive_gap_recovery']:.4f} < 0.70, despite false bridge {confirm['summary']['false_bridge_rate']:.4f}.
- CleanANZA did not improve crowd-heldout CRACKS versus legacy v1: Dice delta {_comparison(r0, 'dice')['mean_delta']:.4f}, clDice delta {_comparison(r0, 'cldice')['mean_delta']:.4f}, skeleton-F1 delta {_comparison(r0, 'skeleton_f1_at_2px')['mean_delta']:.4f}.
- No claim of CRACKS or expert improvement is permitted.

## Locks

- No expert data was accessed.
- CrossingTraceBench-v5 test remained unopened.
- The train-frozen pair threshold was not changed after confirm.
"""
    (output / "THESIS_EVIDENCE.md").write_text(evidence)
    report = f"""# ANZA max-min path completion — final cycle report

Status: `{FINAL_STATUS}`  
Terminal reason: `{TERMINAL_REASON}`

The algebraic repair succeeded: with perfect connectivity, max-min closure and widest-path completion recovered every positive gap and created no negative bridge. A frozen balanced endpoint-pair classifier also generalized well (validation AUROC {pair['validation_metrics']['auroc']:.4f}; confirm AUROC {confirm['pair_metrics']['auroc']:.4f}).

The learned completion claim nevertheless failed its predeclared gate. At the train-frozen threshold {confirm['frozen_pair_classifier']['threshold']:.6f}, confirm gap recovery was {confirm['summary']['positive_gap_recovery']:.4f}, below 0.70. False bridge was {confirm['summary']['false_bridge_rate']:.4f}, latent clDice rose from {confirm['summary']['base_latent_cldice']:.4f} to {confirm['summary']['completion_latent_cldice']:.4f}, and endpoint F1 rose from {confirm['summary']['base_endpoint_f1']:.4f} to {confirm['summary']['completion_endpoint_f1']:.4f}. These supportive diagnostics do not override the failed gate. The confirm threshold was not retuned.

The independent real R0 was also negative. Across 392 non-expert crowd-heldout sections and seeds 41/42/43, CleanANZA versus legacy v1 changed Dice by {_comparison(r0, 'dice')['mean_delta']:.4f} (95% CI {_comparison(r0, 'dice')['ci95_low']:.4f} to {_comparison(r0, 'dice')['ci95_high']:.4f}), AUPRC by {_comparison(r0, 'auprc')['mean_delta']:.4f}, clDice by {_comparison(r0, 'cldice')['mean_delta']:.4f}, and skeleton F1 by {_comparison(r0, 'skeleton_f1_at_2px')['mean_delta']:.4f}. Recall increased {_comparison(r0, 'recall')['mean_delta']:.4f}, but precision decreased {_comparison(r0, 'precision')['mean_delta']:.4f}.

Therefore the defensible conclusion is narrow: max-min bottleneck algebra is capable in the oracle, and endpoint-pair context is identifiable, but this frozen learned operating point and CleanANZA R0 do not establish an improved CRACKS method. Completion was not applied to CRACKS, expert masks remained locked, and v5 test remained unopened.
"""
    (output / "FINAL_REPORT.md").write_text(report)
    return numbers


def _package_files(project_root: Path, final_root: Path) -> list[Path]:
    root = Path(project_root)
    files: set[Path] = set()
    for directory in (
        root / "results/path_completion/oracle",
        root / "results/path_completion/pair_classifier",
        root / "results/path_completion/learned_confirm",
        final_root,
    ):
        files.update(path for path in directory.rglob("*") if path.is_file() and path.suffix != ".zip")
    files.update((root / "results/maxmin_path_study/cracks/r0_clean_anza/analysis").glob("*"))
    files.update({
        root / "results/maxmin_path_study/cracks/r0_clean_anza/clean_threshold_freeze.json",
        root / "results/maxmin_path_study/cracks/r0_clean_anza/reuse_contract.json",
        root / "path_completion/maxmin.py",
        root / "path_completion/widest_path.py",
        root / "path_completion/oracle.py",
        root / "path_completion/pair_classifier.py",
        root / "path_completion/learned_confirm.py",
        root / "path_completion/closeout.py",
        root / "scripts/run_path_pair_classifier.py",
        root / "scripts/run_learned_path_confirm.py",
        root / "scripts/close_maxmin_path_study.py",
        root / "tests/test_maxmin_path_completion.py",
        root / "tests/test_path_pair_classifier.py",
        root / "tests/test_learned_path_confirm.py",
        root / "tests/test_maxmin_path_closeout.py",
    })
    return sorted(path for path in files if path.is_file() and path.name not in {"SHA256SUMS.txt", "PACKAGE_RECEIPT.json"})


def build_closeout(project_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    root = Path(project_root)
    final_root = root / "results/maxmin_path_study/final"
    final_root.mkdir(parents=True, exist_ok=True)
    frozen = validate_frozen_inputs(root)
    selection = generate_required_figures(root, final_root / "figures", device=device)
    numbers = _write_closeout_documents(final_root, frozen, selection)
    validation = {
        "status": "PASS",
        "final_status": FINAL_STATUS,
        "terminal_reason": TERMINAL_REASON,
        "checks": frozen["checks"],
        "expert_data_accessed": False,
        "synthetic_v5_test_opened": False,
        "post_confirm_retuning": False,
        "cracks_completion_run": False,
    }
    (final_root / "VALIDATION_RECEIPT.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    package_files = _package_files(root, final_root)
    manifest = {
        "status": FINAL_STATUS,
        "terminal_reason": TERMINAL_REASON,
        "file_count_excluding_checksums": len(package_files),
        "files": [str(path.relative_to(root)) for path in package_files],
        "checkpoints": [str(path.relative_to(root)) for path in package_files if path.suffix == ".pt"],
        "r0_training_checkpoints_included": False,
    }
    manifest_path = final_root / "PACKAGE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    package_files = _package_files(root, final_root)
    sums_path = final_root / "SHA256SUMS.txt"
    sums_path.write_text("".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in package_files))
    package_files = sorted({*package_files, sums_path})
    zip_path = root / "results/maxmin_path_study/ANZA_MAXMIN_PATH_CLOSEOUT_20260818.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in package_files:
            archive.write(path, path.relative_to(root))
    with zipfile.ZipFile(zip_path) as archive:
        bad_member = archive.testzip()
        member_count = len(archive.namelist())
    if bad_member is not None:
        raise ValueError(f"ZIP CRC failed at {bad_member}")
    receipt = {
        "status": "PASS",
        "zip": str(zip_path),
        "zip_sha256": _sha256(zip_path),
        "zip_member_count": member_count,
        "zip_crc": "PASS",
        "internal_checksum_entries": len(package_files) - 1,
        "thesis_numbers_sha256": _sha256(final_root / "THESIS_NUMBERS.json"),
        "claim_eligibility": numbers["claim_eligibility"],
    }
    (final_root / "PACKAGE_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
