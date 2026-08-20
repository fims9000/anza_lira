"""Machine-linked reporting for the frozen context-repair cycle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from method_repair.context_matrix import context_matrix, context_protocol_hash
from method_repair.context_training import build_context_candidate, load_context_checkpoint
from synthetic.context_repair_losses import effective_mode_count
from synthetic.crossing_trace_bench_v3 import generate_sample_v3


def _load_summaries(root: Path) -> dict[str, dict[str, Any]]:
    return {
        spec.candidate_id: json.loads(
            (root / "validation" / f"{spec.candidate_id}-{spec.run_hash}.json").read_text()
        )
        for spec in context_matrix()
    }


def _load_model(root: Path, candidate_id: str, device: str) -> torch.nn.Module:
    spec = next(spec for spec in context_matrix() if spec.candidate_id == candidate_id)
    status = json.loads(
        (root / "development" / f"{spec.candidate_id}-{spec.run_hash}" / "status.json").read_text()
    )
    model = build_context_candidate(spec, widths=tuple(status["widths"])).to(device)
    load_context_checkpoint(
        root / "development" / f"{spec.candidate_id}-{spec.run_hash}" / "checkpoint-last.pt",
        expected_hash=spec.run_hash,
        model=model,
    )
    return model.eval()


def _inference(model: torch.nn.Module, sample: dict[str, Any], device: str) -> dict[str, np.ndarray]:
    image = torch.as_tensor(sample["image"], device=device).unsqueeze(0)
    with torch.inference_mode():
        output = model(image, return_diagnostics=True)
        diagnostics = output["transport_diagnostics"][0]
        return {
            "probability": torch.sigmoid(output["visible_logits"])[0, 0].cpu().numpy(),
            "gate": diagnostics["ambiguity_gate"][0].cpu().numpy(),
            "neff": effective_mode_count(diagnostics["membership"])[0].cpu().numpy(),
            "correction": diagnostics["correction"][0].abs().mean(dim=0).cpu().numpy(),
            "membership": diagnostics["membership"][0].cpu().numpy(),
        }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def generate_context_figures(root: Path, *, device: str = "cuda") -> list[Path]:
    root = Path(root)
    figure_root = root / "final" / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    b0 = _load_model(root, "B0", device)
    b3 = _load_model(root, "B3", device)

    x_sample = generate_sample_v3("validation", 256, image_size=128)
    x_output = _inference(b3, x_sample, device)
    fig, axes = plt.subplots(1, 5, figsize=(14, 3), constrained_layout=True)
    panels = (
        (x_sample["image"][0], "Synthetic input", "gray"),
        (x_sample["gate_target"], "GT gate target", "magma"),
        (x_output["gate"], "B3 predicted gate", "magma"),
        (x_output["neff"], "B3 effective modes", "viridis"),
        (
            x_output["correction"],
            f"Residual |correction|\nmean={x_output['correction'].mean():.2e}",
            "inferno",
        ),
    )
    for axis, (array, title, cmap) in zip(axes, panels):
        axis.imshow(array, cmap=cmap)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    path_context = figure_root / "fig_context_gate"
    _save_figure(fig, path_context)

    positive = generate_sample_v3("validation", 11, image_size=128)
    negative = generate_sample_v3("validation", 139, image_size=128)
    b0_pos, b0_neg = _inference(b0, positive, device), _inference(b0, negative, device)
    b3_pos, b3_neg = _inference(b3, positive, device), _inference(b3, negative, device)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), constrained_layout=True)
    for row, (sample, old, repaired, label) in enumerate(
        ((positive, b0_pos, b3_pos, "Positive gap"), (negative, b0_neg, b3_neg, "Matched negative"))
    ):
        corridor = sample["positive_gap_mask"] | sample["negative_gap_mask"]
        panels = (
            (sample["image"][0], f"{label}: input", "gray", 0.0, 1.0),
            (sample["visible_fault_mask"], "Visible target", "gray", 0.0, 1.0),
            (corridor, "Evaluated corridor", "gray", 0.0, 1.0),
            (old["probability"], "B0 probability", "viridis", 0.0, 1.0),
            (repaired["probability"], "B3 probability", "viridis", 0.0, 1.0),
        )
        for axis, (array, title, cmap, low, high) in zip(axes[row], panels):
            axis.imshow(array, cmap=cmap, vmin=low, vmax=high)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
    path_gap = figure_root / "fig_positive_negative_gap"
    _save_figure(fig, path_gap)

    fig, axes = plt.subplots(3, 4, figsize=(11, 8), constrained_layout=True)
    for row, (index, label) in enumerate(((256, "X"), (257, "T"), (258, "Y"))):
        sample = generate_sample_v3("validation", index, image_size=128)
        output = _inference(b3, sample, device)
        panels = (
            (sample["image"][0], f"{label}: input", "gray", None, None),
            (sample["gt_mode_count"], "GT mode count", "viridis", 0, 4),
            (output["neff"], "Predicted N_eff", "viridis", 1, 4),
            (output["membership"].max(axis=0), "Max membership", "magma", 0, 1),
        )
        for axis, (array, title, cmap, low, high) in zip(axes[row], panels):
            axis.imshow(array, cmap=cmap, vmin=low, vmax=high)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
    path_modes = figure_root / "fig_modes"
    _save_figure(fig, path_modes)
    return [path_context, path_gap, path_modes]


def _gate_target_audit() -> dict[str, Any]:
    target_mass = 0.0
    valid_pixels = 0
    junction_samples = 0
    for index in range(512):
        sample = generate_sample_v3("validation", index, image_size=128)
        target = np.asarray(sample["gate_target"], dtype=np.float64)
        valid = np.asarray(sample["gate_valid_mask"], dtype=bool)
        target_mass += float(target[valid].sum())
        valid_pixels += int(valid.sum())
        junction_samples += int(target.any())
    raw_fraction = target_mass / max(valid_pixels, 1)
    positive_weight = 4.0
    weighted_fraction = positive_weight * target_mass / max(
        positive_weight * target_mass + valid_pixels - target_mass, 1.0
    )
    return {
        "junction_samples": junction_samples,
        "soft_positive_target_mass": target_mass,
        "valid_pixels": valid_pixels,
        "raw_soft_positive_mass_fraction": raw_fraction,
        "effective_weighted_positive_mass_fraction": weighted_fraction,
    }


def _root_cause(gate: dict[str, Any], summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failures = {
        candidate: [name for name, passed in result["checks"].items() if not passed]
        for candidate, result in gate["decisions"].items()
    }
    b0 = summaries["B0"]["metrics"]
    b3 = summaries["B3"]["metrics"]
    causes: list[str] = []
    terminal_failures = failures.get("B3", [])
    if "gate_auroc" in terminal_failures or "gate_delta_ci" in terminal_failures:
        causes.append("CONTEXT_GATE_LOCALIZATION_NOT_ESTABLISHED")
    if any(name.startswith("neff") or name == "membership_kl" for name in terminal_failures):
        causes.append("MODE_CARDINALITY_SPECIALIZATION_NOT_ESTABLISHED")
    if "false_bridge" in terminal_failures or "false_bridge_reduction" in terminal_failures:
        causes.append("NEGATIVE_CONTINUATION_CONTROL_INSUFFICIENT")
    if "gap_recovery" in terminal_failures:
        causes.append("POSITIVE_COMPLETION_RECOVERY_INSUFFICIENT")
    if "visible_dice_safe" in terminal_failures or "visible_cldice_safe" in terminal_failures:
        causes.append("RESIDUAL_BRANCH_VIOLATED_SEGMENTATION_SAFETY")
    if "route_ap" in terminal_failures or "route_entropy" in terminal_failures:
        causes.append("ROUTING_SHARPNESS_GATE_NOT_MET")
    gate_audit = _gate_target_audit()
    if (
        summaries["B3"]["metrics"].get("gate_auroc", 1.0) < 0.85
        and gate_audit["effective_weighted_positive_mass_fraction"] < 0.05
    ):
        causes.append("GATE_SUPERVISION_EFFECTIVE_POSITIVE_MASS_TOO_LOW")
    if (
        summaries["B3"]["metrics"].get("mode_count_accuracy", 1.0) < 0.01
        and summaries["B3"]["metrics"].get("membership_set_kl", 0.0) > 0.70
    ):
        causes.append("MODE_CARDINALITY_COLLAPSE_NEAR_UNIFORM")
    if (
        summaries["B3"]["metrics"].get("correction_to_base_abs_mean_ratio", 1.0) < 1e-3
        and summaries["B3"]["metrics"].get("false_bridge_rate", 1.0)
        < summaries["B0"]["metrics"].get("false_bridge_rate", 1.0)
    ):
        causes.append("GAP_LOSS_BYPASSED_RESIDUAL_BRANCH_THROUGH_BASE_PATH")
    if (
        summaries["B2"]["metrics"].get("route_entropy_normalized", 0.0)
        >= summaries["B1"]["metrics"].get("route_entropy_normalized", 1.0)
    ):
        causes.append("CONTRASTIVE_ROUTE_DID_NOT_SHARPEN_VALIDATION_ROUTING")
    return {
        "status": "ROOT_CAUSE_RECORDED" if causes else "NO_FAILURE_ROOT_CAUSE_REQUIRED",
        "failed_checks": failures,
        "causes": causes,
        "b0": b0,
        "b3": b3,
        "gate_target_audit": gate_audit,
        "interpretation_rule": "terminal causes are mapped from B3 predeclared failures; mechanistic explanations additionally require direct stored diagnostics and are hypotheses where causality is not identifiable",
    }


def build_context_report(root: Path, *, device: str = "cuda") -> dict[str, Any]:
    root = Path(root)
    final = root / "final"
    final.mkdir(parents=True, exist_ok=True)
    summaries = _load_summaries(root)
    gate = json.loads((root / "mechanism_gate.json").read_text())
    terminal = gate["status"] == "CONTEXT_MECHANISM_FAIL"
    status = "CONTEXT_REPAIR_NEGATIVE_WITH_ROOT_CAUSE" if terminal else "CONTEXT_REPAIR_VALIDATION_PASS_CONFIRM_PENDING"
    root_cause = _root_cause(gate, summaries)
    (final / "ROOT_CAUSE.json").write_text(json.dumps(root_cause, indent=2, sort_keys=True) + "\n")

    metric_names = (
        "visible_dice", "visible_cldice", "route_average_precision",
        "route_entropy_normalized", "orientation_error_model_modes_median_deg",
        "membership_set_kl", "neff_junction_minus_straight",
        "neff_junction_minus_straight_median", "gate_auroc", "gate_auprc",
        "gate_junction_minus_straight", "false_bridge_rate", "gap_recovery_rate",
        "topology_constrained_pairing_score", "endpoint_f1",
    )
    with (final / "synthetic_context_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("candidate_id", *metric_names))
        writer.writeheader()
        for candidate in ("B0", "B1", "B2", "B3"):
            metrics = summaries[candidate]["metrics"]
            writer.writerow({"candidate_id": candidate, **{name: metrics.get(name) for name in metric_names}})

    generate_context_figures(root, device=device)
    numbers = {
        "status": status,
        "protocol_hash": context_protocol_hash(),
        "selected_candidate": gate.get("selected_candidate"),
        "cracks_authorized": False,
        "test_v3_status": "LOCKED_UNOPENED",
        "expert_data_accessed": False,
        "metrics": {candidate: summary["metrics"] for candidate, summary in summaries.items()},
        "gate": gate,
    }
    (final / "THESIS_NUMBERS.json").write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    b0, b1, b2, b3 = (summaries[name]["metrics"] for name in ("B0", "B1", "B2", "B3"))
    report = f"""# ANZA context-repair report

Status: `{status}`

This report is generated only from the frozen B0-B3 validation JSON/CSV artifacts. The legacy synthetic test, v3 test, CRACKS, and expert masks were not opened in this cycle.

## Required scientific questions

1. Contextual gate localization: B1/B2/B3 gate AUROC = {b1['gate_auroc']:.4f} / {b2['gate_auroc']:.4f} / {b3['gate_auroc']:.4f}; B3 junction-minus-straight = {b3['gate_junction_minus_straight']:.4f}.
2. Mode separation: B3 mean N_eff junction-minus-straight = {b3['neff_junction_minus_straight']:.4f}, median separation = {b3['neff_junction_minus_straight_median']:.4f}, KL = {b3['membership_set_kl']:.4f}.
3. Gap tradeoff: false bridge B0 -> B3 = {b0['false_bridge_rate']:.4f} -> {b3['false_bridge_rate']:.4f}; B3 positive gap recovery = {b3['gap_recovery_rate']:.4f}.
4. Routing: B3 route AP = {b3['route_average_precision']:.4f}; normalized entropy = {b3['route_entropy_normalized']:.4f}.
5. Segmentation safety: Visible Dice B0 -> B3 = {b0['visible_dice']:.4f} -> {b3['visible_dice']:.4f}; visible clDice = {b0['visible_cldice']:.4f} -> {b3['visible_cldice']:.4f}.
6. B1 (context heads/direct gate) changed Dice by {b1['visible_dice'] - b0['visible_dice']:+.4f}, Route AP by {b1['route_average_precision'] - b0['route_average_precision']:+.4f}, and gate AUROC remained {b1['gate_auroc']:.4f}; it did not localize junctions.
7. B2 (additional contrastive route) changed Route AP by {b2['route_average_precision'] - b1['route_average_precision']:+.4f} and entropy by {b2['route_entropy_normalized'] - b1['route_entropy_normalized']:+.4f}; the entropy change is in the wrong direction.
8. B3 (additional paired corridor loss) reduced false bridge by {b2['false_bridge_rate'] - b3['false_bridge_rate']:+.4f} and raised gap recovery by {b3['gap_recovery_rate'] - b2['gap_recovery_rate']:+.4f}, but residual/base magnitude ratio collapsed to {b3['correction_to_base_abs_mean_ratio']:.6f}; this improvement is not evidence that the multimode residual mechanism caused the change.
9. CRACKS authorized: **NO** at this artifact stage. A validation PASS still requires 3-seed confirm_v3.
10. Thesis claim boundary: the data may support only the individual gates that passed. It does not support a real-seismic improvement claim, because CRACKS was not run.

## Gate result

`{gate['status']}`; selected candidate: `{gate.get('selected_candidate')}`.

## Root-cause mapping

{', '.join(root_cause['causes']) if root_cause['causes'] else 'No validation failure root cause; confirmation is pending.'}

No B4/B5 was introduced after observing results. Thesis documents were not edited.
"""
    (final / "FINAL_REPORT.md").write_text(report)
    evidence = {
        "status": status,
        "source_files": {
            candidate: summaries[candidate]["rows_csv"] for candidate in summaries
        },
        "protocol": str(root / "protocol.json"),
        "benchmark": str(root / "benchmark_v3_config.json"),
        "locked_streams": {"legacy_test": 0, "v3_test": 0, "cracks": 0, "expert": 0},
    }
    (final / "EVIDENCE.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return numbers


def package_context_report(root: Path) -> tuple[Path, str]:
    root = Path(root)
    final = root / "final"
    files = sorted(path for path in final.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    checksum_lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(final)}" for path in files
    ]
    (final / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n")
    archive = root / "ANZA_CONTEXT_REPAIR_20260818.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(final.rglob("*")):
            if path.is_file():
                handle.write(path, Path("context_repair") / path.relative_to(final))
        for path in (root / "protocol.json", root / "benchmark_v3_config.json", root / "mechanism_gate.json"):
            handle.write(path, Path("context_repair") / path.name)
        for path in sorted((root / "validation").glob("*")):
            if not path.is_file() or path.suffix not in {".json", ".csv"}:
                continue
            handle.write(path, Path("context_repair/validation") / path.name)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (root / f"{archive.name}.sha256").write_text(f"{checksum}  {archive.name}\n")
    return archive, checksum
