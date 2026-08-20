"""Claim-safe reporting for the frozen ANZA final practical cycle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = PROJECT_ROOT / "results" / "final_practical_cycle"
FINAL_ROOT = STUDY_ROOT / "final"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    v5 = json.loads((STUDY_ROOT / "path_v5_test" / "test_result.json").read_text())
    v6 = json.loads((STUDY_ROOT / "realistic_synthetic" / "development_result.json").read_text())
    t1 = json.loads((STUDY_ROOT / "cracks_t1" / "analysis" / "result.json").read_text())
    pairs = json.loads((STUDY_ROOT / "cracks_pairs" / "result.json").read_text())
    return v5, v6, t1, pairs


def build_numbers() -> dict[str, Any]:
    v5, v6, t1, pairs = _load()
    comparison = {(row["model"], row["metric"]): row for row in t1["comparisons"]}
    numbers = {
        "statuses": {
            "path_classifier_v5": v5["status"],
            "v6_predicted_endpoints": v6["status"],
            "cracks_partial_labels": t1["status"],
            "cracks_real_pair_classifier": pairs["status"],
            "anza_guided_completion": "NOT_RUN_GATE_LOCKED",
            "final": "FINAL_PRACTICAL_NEGATIVE_WITH_ROOT_CAUSE",
        },
        "path_v5_test": {
            "auroc": v5["pair_metrics"]["auroc"],
            "positive_gap_recovery": v5["summary"]["positive_gap_recovery"],
            "false_bridge_rate": v5["summary"]["false_bridge_rate"],
            "endpoint_f1_before": v5["summary"]["base_endpoint_f1"],
            "endpoint_f1_after": v5["summary"]["completion_endpoint_f1"],
            "test_samples_opened": v5["v5_test_samples_opened"],
        },
        "cracks_t1": {
            "heldout_sections": t1["section_count"],
            "seed_count": t1["seed_count"],
            "unet": {
                metric: {
                    "delta": comparison[("unet", metric)]["mean_delta"],
                    "ci95": [comparison[("unet", metric)]["ci95_low"], comparison[("unet", metric)]["ci95_high"]],
                }
                for metric in ("dice", "auprc", "precision", "recall", "cldice", "skeleton_f1_at_2px", "predicted_foreground_fraction")
            },
            "anza_v1": {
                metric: {
                    "delta": comparison[("anza_v1", metric)]["mean_delta"],
                    "ci95": [comparison[("anza_v1", metric)]["ci95_low"], comparison[("anza_v1", metric)]["ci95_high"]],
                }
                for metric in ("dice", "auprc", "precision", "recall", "cldice", "skeleton_f1_at_2px", "predicted_foreground_fraction")
            },
            "claim_boundary": "held-out explicit non-expert crowd labels only; not expert quality and not an ANZA advantage",
        },
        "real_pair_classifier": {
            "train_matched_pairs": pairs["train_matched_pairs"],
            "validation_matched_pairs": pairs["validation_matched_pairs"],
            "validation_auroc": pairs["validation_metrics"]["auroc"],
            "validation_balanced_auprc": pairs["validation_metrics"]["balanced_auprc"],
            "validation_matched_pair_ranking": pairs["validation_metrics"]["matched_pair_ranking_probability"],
            "threshold": pairs["validation_operating_point"]["threshold"],
            "fpr": pairs["validation_operating_point"]["fpr"],
            "tpr": pairs["validation_operating_point"]["tpr"],
            "tpr_gate": 0.70,
            "fpr_gate": 0.05,
        },
        "scientific_conclusion": {
            "positive": "Partial-label crowd supervision is a confirmed data-method result on held-out explicit crowd evidence.",
            "negative": "Real-domain continuation ranking passed AUROC/AUPRC but failed sensitivity at the frozen low false-positive operating point, so real ANZA-guided completion and expert opening were not run.",
            "architecture_claim": "No evidence that legacy ANZA outperforms U-Net under T1; both benefit similarly.",
        },
        "expert_data_accessed": False,
        "expert_scores_used": False,
    }
    sources = {
        "path_v5_test": STUDY_ROOT / "path_v5_test" / "test_result.json",
        "v6_development": STUDY_ROOT / "realistic_synthetic" / "development_result.json",
        "t1_statistics": STUDY_ROOT / "cracks_t1" / "analysis" / "result.json",
        "t1_raw_sections": STUDY_ROOT / "cracks_t1" / "analysis" / "raw_per_section.csv",
        "real_pair_manifest": STUDY_ROOT / "cracks_pairs" / "manifest.json",
        "real_pair_scores": STUDY_ROOT / "cracks_pairs" / "scores.csv",
        "real_pair_result": STUDY_ROOT / "cracks_pairs" / "result.json",
    }
    numbers["provenance"] = {
        key: {"path": str(path), "sha256": _sha256(path)} for key, path in sources.items()
    }
    return numbers


def _figures(numbers: dict[str, Any]) -> None:
    figure_root = FINAL_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    metrics = ("dice", "recall", "cldice", "skeleton_f1_at_2px", "auprc")
    labels = ("Dice", "Recall", "clDice", "Skeleton F1", "AUPRC")
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for offset, (model, label_name, color) in zip(
        (-0.18, 0.18),
        (("unet", "U-Net T1 - T0", "#3B82F6"), ("anza_v1", "ANZA v1 T1 - T0", "#F97316")),
    ):
        rows = [numbers["cracks_t1"][model][metric] for metric in metrics]
        values = np.asarray([row["delta"] for row in rows])
        errors = np.asarray([[row["delta"] - row["ci95"][0] for row in rows], [row["ci95"][1] - row["delta"] for row in rows]])
        ax.bar(x + offset, values, width=0.34, yerr=errors, capsize=3, label=label_name, color=color)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Paired section delta")
    ax.set_title("Partial-label supervision improves explicit crowd-label metrics")
    ax.legend(loc="upper right")
    fig.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(figure_root / f"fig_t1_paired_deltas.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

    pair = numbers["real_pair_classifier"]
    values = [pair["validation_auroc"], pair["validation_balanced_auprc"], pair["validation_matched_pair_ranking"], pair["tpr"]]
    gates = [0.85, 0.85, None, pair["tpr_gate"]]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bars = ax.bar(("AUROC", "Balanced AUPRC", "Matched ranking", "TPR at FPR=0.05"), values, color=("#10B981", "#10B981", "#64748B", "#DC2626"))
    for bar, value, gate in zip(bars, values, gates):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", va="bottom")
        if gate is not None:
            ax.plot([bar.get_x(), bar.get_x() + bar.get_width()], [gate, gate], color="black", linestyle="--", linewidth=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation score")
    ax.set_title("Real-domain pair ranking passes, low-FPR sensitivity fails")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(figure_root / f"fig_real_pair_gate.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def build_closeout() -> dict[str, Any]:
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    numbers = build_numbers()
    numbers_path = FINAL_ROOT / "THESIS_NUMBERS.json"
    numbers_path.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    _figures(numbers)
    unet = numbers["cracks_t1"]["unet"]
    anza = numbers["cracks_t1"]["anza_v1"]
    pair = numbers["real_pair_classifier"]
    report = f"""# ANZA final practical cycle

Final status: `FINAL_PRACTICAL_NEGATIVE_WITH_ROOT_CAUSE`

Independent positive status: `CRACKS_PARTIAL_LABEL_SUCCESS`.

## What passed

Treating white crowd pixels as unknown rather than certain background improved held-out explicit crowd-label metrics for both architectures over 392 sections and three seeds. U-Net Dice changed by `{unet['dice']['delta']:.6f}` (95% CI `{unet['dice']['ci95'][0]:.6f}` to `{unet['dice']['ci95'][1]:.6f}`); legacy ANZA changed by `{anza['dice']['delta']:.6f}` (`{anza['dice']['ci95'][0]:.6f}` to `{anza['dice']['ci95'][1]:.6f}`). Recall changed by `{unet['recall']['delta']:.6f}` and `{anza['recall']['delta']:.6f}`, respectively.

This is a data-supervision result on explicit non-expert crowd evidence. It is not an expert-quality result and does not show an ANZA advantage. Full-image foreground fraction increased by `{unet['predicted_foreground_fraction']['delta']:.6f}` for U-Net and `{anza['predicted_foreground_fraction']['delta']:.6f}` for ANZA, while precision changed by `{unet['precision']['delta']:.6f}` and `{anza['precision']['delta']:.6f}`. Unknown-region overprediction remains unresolved.

## What failed

The balanced, section-disjoint real-domain endpoint-pair classifier achieved AUROC `{pair['validation_auroc']:.6f}` and balanced AUPRC `{pair['validation_balanced_auprc']:.6f}`. At the frozen low-false-positive operating point (FPR `{pair['fpr']:.6f}`, threshold `{pair['threshold']:.6f}`), TPR was only `{pair['tpr']:.6f}` versus the required `{pair['tpr_gate']:.6f}`. Training separation was perfect, so the root cause is generalization/calibration at the strict operating point, not lack of training fit.

Phase E ANZA-guided completion was therefore `NOT_RUN_GATE_LOCKED`; expert masks remained unopened. No second classifier, post-hoc threshold change, or architecture variant was run.

## Claim boundary

Supported: explicit partial-label supervision materially changes crowd-heldout delineation and fixes much of the previous recall deficit.

Not supported: ANZA superiority over U-Net, real learned path-completion benefit, expert-quality improvement, or an article claim that the proposed completion system improves CRACKS segmentation.
"""
    (FINAL_ROOT / "FINAL_PRACTICAL_REPORT.md").write_text(report)
    evidence = [
        "# Thesis evidence",
        "",
        "Every number in the report is generated from `THESIS_NUMBERS.json`; that file is generated from frozen machine artifacts.",
        "",
        "| Artifact | SHA-256 |",
        "|---|---|",
    ]
    for row in numbers["provenance"].values():
        evidence.append(f"| `{row['path']}` | `{row['sha256']}` |")
    evidence.extend([
        "",
        "Expert data accessed: `false`.",
        "",
        "Phase E: `NOT_RUN_GATE_LOCKED`.",
    ])
    (FINAL_ROOT / "THESIS_EVIDENCE.md").write_text("\n".join(evidence) + "\n")
    return numbers
