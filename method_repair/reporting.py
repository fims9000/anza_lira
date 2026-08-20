"""Build a claim-safe final package for the bounded negative repair result."""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from method_repair.audit import FROZEN_FILES, sha256_file
from method_repair.matrix import COMMON_PROTOCOL, protocol_hash, synthetic_matrix
from method_repair.training import build_candidate_model, load_candidate_checkpoint
from models.azconv_repaired import ambiguity_components
from synthetic.crossing_trace_bench_v2 import generate_sample_v2


REQUIRED_FINAL_FILES = (
    "FINAL_REPORT.md",
    "METHOD_REPAIR_AUDIT.md",
    "PROTOCOL.json",
    "DATA_SEMANTICS_AUDIT.md",
    "MODEL_FORMULAS.md",
    "raw_per_section.csv",
    "main_metrics.csv",
    "paired_comparisons.csv",
    "ablations.csv",
    "synthetic_mechanism.csv",
    "calibration.csv",
    "THESIS_NUMBERS.json",
    "THESIS_EVIDENCE.md",
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_validation(synthetic_root: Path) -> dict[str, dict[str, Any]]:
    return {
        spec.candidate_id: json.loads(
            (synthetic_root / "validation" / f"{spec.candidate_id}-{spec.run_hash}.json").read_text()
        )
        for spec in synthetic_matrix()
    }


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(base.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_matrix(summaries: dict[str, dict[str, Any]], figure_root: Path) -> None:
    names = list(summaries)
    visible = [summaries[name]["metrics"]["visible_dice"] for name in names]
    route_ap = [summaries[name]["metrics"]["route_average_precision"] or 0.0 for name in names]
    entropy = [summaries[name]["metrics"]["route_entropy_normalized"] or 0.0 for name in names]
    false_bridge = [summaries[name]["metrics"]["false_bridge_rate"] for name in names]
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.8), constrained_layout=True)
    for axis, values, title in zip(
        axes,
        (visible, route_ap, entropy, false_bridge),
        ("Visible Dice", "Route AP", "Route entropy", "False bridge"),
    ):
        axis.bar(names, values, color="#4c78a8")
        axis.set_title(title)
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, figure_root / "fig_synthetic_matrix")
    (figure_root / "fig_synthetic_matrix.json").write_text(json.dumps({
        "split": "CrossingTraceBench-v2 validation[0:256]",
        "candidate_ids": names,
        "metrics": {
            "visible_dice": visible,
            "route_average_precision": route_ap,
            "route_entropy_normalized": entropy,
            "false_bridge_rate": false_bridge,
        },
        "interpretation": "Route readout improved, but the complete predeclared gate failed.",
    }, indent=2, sort_keys=True) + "\n")


def _load_model(candidate_id: str, synthetic_root: Path, device: torch.device) -> tuple[Any, Any]:
    spec = next(item for item in synthetic_matrix() if item.candidate_id == candidate_id)
    run_dir = synthetic_root / "development" / f"{candidate_id}-{spec.run_hash}"
    status = json.loads((run_dir / "status.json").read_text())
    model = build_candidate_model(spec, widths=tuple(status["widths"])).to(device).eval()
    load_candidate_checkpoint(run_dir / "checkpoint-last.pt", expected_hash=spec.run_hash, model=model)
    return spec, model


def _plot_gate_failure(synthetic_root: Path, figure_root: Path, device: str) -> None:
    torch_device = torch.device(device)
    _spec, model = _load_model("A3", synthetic_root, torch_device)
    sample = generate_sample_v2("validation", 0, image_size=128, case="x_junction")
    image = torch.as_tensor(sample["image"], device=torch_device).unsqueeze(0)
    with torch.inference_mode():
        output = model(image, return_diagnostics=True)
        probability = torch.sigmoid(output["visible_logits"])[0, 0].cpu().numpy()
        diagnostic = output["transport_diagnostics"][0]
        membership = diagnostic["membership"]
        neff = torch.exp(-(membership * membership.clamp_min(1e-8).log()).sum(dim=1))[0].cpu().numpy()
        gate = diagnostic["ambiguity_gate"][0].cpu().numpy()
        correction = torch.linalg.vector_norm(diagnostic["correction"][0], dim=0).cpu().numpy()
    input_panel = np.moveaxis(np.asarray(sample["image"]), 0, -1)
    panels = (
        (input_panel, "Synthetic input", "gray"),
        (sample["gt_mode_count"], "GT mode count", "viridis"),
        (neff, "Predicted effective modes", "viridis"),
        (gate, "Ambiguity gate", "magma"),
        (correction, "Residual magnitude", "magma"),
        (probability, "Visible probability", "gray"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.8), constrained_layout=True)
    for axis, (array, title, cmap) in zip(axes.flat, panels):
        axis.imshow(array, cmap=None if array.ndim == 3 else cmap)
        axis.set_title(title)
        axis.set_axis_off()
    _save_figure(fig, figure_root / "fig_gate_failure")
    (figure_root / "fig_gate_failure.json").write_text(json.dumps({
        "candidate_id": "A3",
        "selection_reason": "predeclared direct-mode 3x3 mechanistic candidate; not selected by expert or test performance",
        "split": "validation",
        "index": 0,
        "forced_case_for_diagnostic": "x_junction",
        "expert_data_used": False,
        "test_data_used": False,
    }, indent=2, sort_keys=True) + "\n")


def _plot_failure_cases(
    synthetic_root: Path,
    summaries: dict[str, dict[str, Any]],
    figure_root: Path,
    device: str,
) -> None:
    by_candidate: dict[str, dict[int, dict[str, str]]] = {}
    for candidate_id in ("A0", "A3"):
        with Path(summaries[candidate_id]["rows_csv"]).open(newline="") as handle:
            by_candidate[candidate_id] = {int(row["index"]): row for row in csv.DictReader(handle)}
    indices = sorted(
        by_candidate["A0"],
        key=lambda index: (
            float(by_candidate["A3"][index]["family_a_visible_dice"])
            - float(by_candidate["A0"][index]["family_a_visible_dice"]),
            index,
        ),
    )[:2]
    torch_device = torch.device(device)
    models = {name: _load_model(name, synthetic_root, torch_device)[1] for name in ("A0", "A3")}
    fig, axes = plt.subplots(2, 5, figsize=(11.2, 4.8), constrained_layout=True)
    receipt = []
    for row_index, index in enumerate(indices):
        sample = generate_sample_v2("validation", index, image_size=128)
        image = torch.as_tensor(sample["image"], device=torch_device).unsqueeze(0)
        predictions = {}
        for name, model in models.items():
            threshold = float(summaries[name]["selected_visible_threshold"])
            with torch.inference_mode():
                predictions[name] = torch.sigmoid(model(image))[0, 0].cpu().numpy() >= threshold
        truth = np.asarray(sample["visible_fault_mask"], dtype=bool)
        input_panel = np.moveaxis(np.asarray(sample["image"]), 0, -1)
        panels = (
            (input_panel, "Input"),
            (truth, "Visible GT"),
            (predictions["A0"], "A0 v1"),
            (predictions["A3"], "A3 repaired"),
            (predictions["A3"] != truth, "A3 error"),
        )
        for axis, (array, title) in zip(axes[row_index], panels):
            axis.imshow(array, cmap=None if array.ndim == 3 else "gray")
            axis.set_title(title)
            axis.set_axis_off()
        receipt.append({
            "index": index,
            "case": sample["case"],
            "selection": "two lowest A3-minus-A0 visible Dice deltas on frozen validation",
            "a0_visible_dice": float(by_candidate["A0"][index]["family_a_visible_dice"]),
            "a3_visible_dice": float(by_candidate["A3"][index]["family_a_visible_dice"]),
        })
    _save_figure(fig, figure_root / "fig_failure_cases")
    (figure_root / "fig_failure_cases.json").write_text(json.dumps({
        "split": "validation",
        "expert_data_used": False,
        "test_data_used": False,
        "cases": receipt,
    }, indent=2, sort_keys=True) + "\n")


def build_negative_package(
    project_root: Path,
    final_root: Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    project_root = Path(project_root)
    final_root = Path(final_root)
    final_root.mkdir(parents=True, exist_ok=True)
    synthetic_root = project_root / "results" / "method_repair" / "synthetic_v2"
    root_cause = json.loads((project_root / "results" / "method_repair" / "root_cause.json").read_text())
    gate = json.loads((synthetic_root / "mechanism_gate.json").read_text())
    if root_cause["status"] != "METHOD_REPAIR_NEGATIVE_WITH_ROOT_CAUSE" or gate["cracks_authorized"] is not False:
        raise ValueError("negative package requires the frozen failed mechanism gate")
    summaries = _load_validation(synthetic_root)

    protocol_payload = {
        "status": "FROZEN_BOUNDED_CYCLE_COMPLETE",
        "protocol_hash": protocol_hash(),
        "common_protocol": COMMON_PROTOCOL,
        "matrix": [{**asdict(spec), "run_hash": spec.run_hash} for spec in synthetic_matrix()],
        "result": "SYNTHETIC_MECHANISM_FAIL",
        "cracks_authorized": False,
    }
    (final_root / "PROTOCOL.json").write_text(json.dumps(protocol_payload, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(project_root / "docs" / "research" / "METHOD_REPAIR_AUDIT.md", final_root / "METHOD_REPAIR_AUDIT.md")

    crowd = json.loads((project_root / "results" / "method_repair" / "audit" / "crowd_target.json").read_text())
    pair = crowd["pair_summary"]
    data_audit = f"""# CRACKS data-semantics audit

Status: **CROWD_ONLY_COMPLETE; WHITE_NOT_ESTABLISHED**.

Official material defines orange certain-no-fault, blue certain-fault, and green uncertain-fault, but does not establish white as an explicit semantic class. The historical `paper_like` policy is retained only as an inferred baseline. No expert mask was accessed in this audit.

On the deterministic crowd-only sample of {crowd['selection']['sample_count']} sections, mean annotator pixel Dice was {pair['pixel_dice_mean']:.6f}; mean nearest-trace coverage within 5 px was {pair['within_5px_fraction_mean']:.6f}, including {pair['displaced_2_to_5px_fraction_mean']:.6f} at a nonzero displacement up to 5 px. This supports a target-fusion sensitivity study, but no alternative target was selected because the synthetic architecture gate failed before CRACKS.
"""
    (final_root / "DATA_SEMANTICS_AUDIT.md").write_text(data_audit)
    (final_root / "MODEL_FORMULAS.md").write_text("""# Repaired model formulas

The unchanged base path is `B(x) = AZConv2d_v1(x)`. Initial mode state is `z_r^0 = mu_r V`; source membership is not repeated in transport. Destination acceptance, axial compatibility, sign-invariant displacement compatibility, and source-row normalization define `T`. Fusion is `R = sum_r z_r` without another membership multiplier.

The residual output is `Y = B + lambda g Delta`, with `lambda = tanh(a)` and `a=0` initially. `Delta` is initialized nonzero so `lambda` has a wake-up gradient; setting both `lambda` and `W_delta` to zero would deadlock the branch.

The gate uses `D = (1-sum(mu_r^2))/(1-1/R)`, angular diversity `A`, and `g = sigmoid(s(J-tau))`, `J=DA`. The bounded result shows this pointwise predictor did not localize junction ambiguity.
""")

    mechanism_rows = []
    for spec in synthetic_matrix():
        metrics = summaries[spec.candidate_id]["metrics"]
        decision = gate["decisions"].get(spec.candidate_id, {})
        mechanism_rows.append({
            "candidate_id": spec.candidate_id,
            "run_hash": spec.run_hash,
            "visible_dice": metrics["visible_dice"],
            "visible_cldice": metrics["visible_cldice"],
            "route_average_precision": metrics["route_average_precision"],
            "route_mrr": metrics["route_mrr"],
            "route_entropy_normalized": metrics["route_entropy_normalized"],
            "route_excess_over_chance": metrics["route_excess_over_chance"],
            "neff_junction_minus_straight": metrics["neff_junction_minus_straight"],
            "neff_ci95_low": metrics["neff_junction_minus_straight_ci95"][0],
            "ambiguity_junction_minus_straight": metrics["ambiguity_junction_minus_straight"],
            "ambiguity_ci95_low": metrics["ambiguity_junction_minus_straight_ci95"][0],
            "false_bridge_rate": metrics["false_bridge_rate"],
            "all_gates_pass": decision.get("all_gates_pass", False),
        })
    _write_csv(final_root / "synthetic_mechanism.csv", list(mechanism_rows[0]), mechanism_rows)
    _write_csv(final_root / "ablations.csv", list(mechanism_rows[0]), mechanism_rows)
    _write_csv(final_root / "main_metrics.csv", ["scope", "status", "reason"], [{
        "scope": "CRACKS_R0_R3",
        "status": "NOT_RUN",
        "reason": "SYNTHETIC_MECHANISM_GATE_FAILED",
    }])
    for name, fields in (
        ("raw_per_section.csv", ["section_id", "model", "metric", "value", "status"]),
        ("paired_comparisons.csv", ["comparison", "metric", "delta", "ci95_low", "ci95_high", "status"]),
        ("calibration.csv", ["model", "brier", "ece", "auprc", "status"]),
    ):
        _write_csv(final_root / name, fields, [{fields[0]: "", "status": "NOT_RUN_SYNTHETIC_GATE_FAILED"}])

    numbers = {
        "status": "METHOD_REPAIR_NEGATIVE_WITH_ROOT_CAUSE",
        "frozen_deadline_sha256": {name: sha256_file(path) for name, path in FROZEN_FILES.items()},
        "protocol_hash": protocol_hash(),
        "synthetic_gate": gate["status"],
        "cracks": {"training": "NOT_RUN", "expert_evaluation": "NOT_RUN", "expert_data_accessed": False},
        "test_access": {"old_synthetic_samples": 0, "new_synthetic_samples": 0},
        "synthetic_candidates": {name: summary["metrics"] for name, summary in summaries.items()},
        "root_causes": root_cause["root_causes"],
        "next_experiment_not_executed": root_cause["next_experiment_not_executed"],
    }
    (final_root / "THESIS_NUMBERS.json").write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    a0 = numbers["synthetic_candidates"]["A0"]
    a3 = numbers["synthetic_candidates"]["A3"]
    report = f"""# ANZA method-repair final report

Status: **{numbers['status']}**.

## Answers required by the protocol

1. **Why did the frozen mode-resolved model lose?** It repeatedly attenuated fuzzy states, marginalized mode identity in route supervision, imposed persistent half-mode polarity, and routed at all three encoder stages. These are frozen forensic findings, not rewritten results.
2. **What was fixed mathematically?** The repair uses one membership source gate, axial sign-invariant source-row transport, sum fusion, and a zero-scaled residual on the unchanged v1 base.
3. **What was fixed in supervision?** CrossingTraceBench-v2 provides exact tangent sets; matching is permutation-invariant; branch routing retains matched mode identity.
4. **What was fixed in the data target?** White semantics were left `NOT_ESTABLISHED`; crowd displacement was audited. No target was silently changed.
5. **Did the real result improve?** Not evaluated. CRACKS R0-R3 was forbidden because the synthetic mechanism gate failed.
6. **What effect was observed?** A3 validation visible Dice was {a3['visible_dice']:.6f} versus A0 {a0['visible_dice']:.6f}; route AP was {a3['route_average_precision']:.6f} and normalized entropy {a3['route_entropy_normalized']:.6f}. These partial effects do not satisfy the complete gate.
7. **Where does the method still lose?** A3 junction-minus-straight N_eff CI lower bound was {a3['neff_junction_minus_straight_ci95'][0]:.6f}, ambiguity CI lower bound {a3['ambiguity_junction_minus_straight_ci95'][0]:.6f}, and false-bridge rate {a3['false_bridge_rate']:.6f}. A4 also violated visible non-inferiority.
8. **Claims allowed for theses:** the repair removed the identified attenuation/identity bugs; mode-specific routing is above chance on controlled validation; residual initialization preserves the v1 path.
9. **Claims forbidden:** real CRACKS improvement, expert improvement, successful junction-local ambiguity, false-bridge control, and structural superiority.

## Root cause

The pointwise `1x1` mode/gate heads do not receive enough spatial context to distinguish junctions from straight traces in this implementation. Direct supervision learned axes but not the required membership cardinality/localized gate. Route/mode loss also supplied no matched-negative gap pressure, so every candidate remained saturated at false bridge 1.0.

The next proposed experiment is recorded but was not executed because adding A5 after seeing A0-A4 would violate the frozen search budget.
"""
    (final_root / "FINAL_REPORT.md").write_text(report)
    evidence = f"""# Thesis evidence

The bounded method-repair cycle is a negative result. All reported numbers come from `THESIS_NUMBERS.json`, which was built from A0-A4 validation JSON/CSV and the frozen mechanism gate.

- Protocol hash: `{protocol_hash()}`
- Synthetic gate: `{gate['status']}`
- CRACKS training: `NOT_RUN`
- Expert evaluation: `NOT_RUN`
- Old/new synthetic test samples opened: `0/0`
- Frozen deadline ZIP remains `{numbers['frozen_deadline_sha256']['deadline_zip']}`.

No thesis claim of real-data superiority is authorized by this package.
"""
    (final_root / "THESIS_EVIDENCE.md").write_text(evidence)

    figure_root = final_root / "figures"
    _plot_matrix(summaries, figure_root)
    _plot_gate_failure(synthetic_root, figure_root, device)
    _plot_failure_cases(synthetic_root, summaries, figure_root, device)
    return {
        "status": numbers["status"],
        "final_root": str(final_root),
        "required_files": list(REQUIRED_FINAL_FILES),
        "figure_bases": ["fig_synthetic_matrix", "fig_gate_failure", "fig_failure_cases"],
    }
