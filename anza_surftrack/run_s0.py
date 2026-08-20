"""Complete bounded SurfTrack S0 execution and frozen causal decision."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .eval.observability import evaluate_observability
from .eval.statistics import calibrated_confidence, fit_margin_calibration, paired_bootstrap, risk_coverage
from .eval.tracking import evaluate_method_batches, summarize, summarize_strata
from .figures import generate_figures
from .protocol import METHODS, PROTOCOL, ROOT, canonical_hash, freeze_protocol, load_frozen, write_json
from .synthetic3d.families import generate_batch
from .transport.fit import fit_all


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else (0.0 if numerator == 0 else math.inf)


def _gate(
    iid: dict[str, dict], ood: dict[str, dict], strata: list[dict], bootstrap: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    g1, g2, g3, g4 = (ood[key] for key in METHODS[1:])
    top_g1 = g4["top1"] - g1["top1"]; switch_g1 = _ratio(g4["switch"], g1["switch"])
    top_g2 = g4["top1"] - g2["top1"]; switch_g2 = _ratio(g4["switch"], g2["switch"])
    top_g3 = g4["top1"] - g3["top1"]; switch_g3 = _ratio(g4["switch"], g3["switch"])
    iid_delta_g3 = iid["G4_anza_cocycle"]["top1"] - iid["G3_free_compose"]["top1"]
    composition = ((top_g1 >= .08 and bootstrap["G4_minus_G1_top1"]["ci95_low"] > 0) or
                   (switch_g1 <= .70 and bootstrap["G1_minus_G4_switch"]["ci95_low"] > 0))
    hyperbolic = ((top_g2 >= .04 and bootstrap["G4_minus_G2_top1"]["ci95_low"] > 0) or
                  (switch_g2 <= .80 and bootstrap["G2_minus_G4_switch"]["ci95_low"] > 0))
    flexible = iid_delta_g3 >= -.01 and (
        (top_g3 >= .03 and bootstrap["G4_minus_G3_top1"]["ci95_low"] > 0) or
        (switch_g3 <= .85 and bootstrap["G3_minus_G4_switch"]["ci95_low"] > 0)
    )
    target_families = PROTOCOL["gates"]["per_stratum"]["families"]; wins = {}
    for family in target_families:
        local = {row["method"]: row for row in strata if row["split"] == "geom_dev_ood" and row["family"] == family}
        delta = local["G4_anza_cocycle"]["top1"] - local["G1_local_reset"]["top1"]
        reduction = 1 - _ratio(local["G4_anza_cocycle"]["switch"], local["G1_local_reset"]["switch"])
        wins[family] = {"top1_delta": delta, "switch_reduction": reduction, "pass": delta >= .05 or reduction >= .20}
    stratum_pass = sum(int(row["pass"]) for row in wins.values()) >= 3
    full = composition and hyperbolic and flexible and stratum_pass
    status = "ANZA_SURFTRACK_COCYCLE_PASS" if full else (
        "SURFTRACK_COMPOSITION_PASS_ANOSOV_NOT_SPECIFIC" if composition else "STOP_ANOSOV_SURFTRACK_NO_CAUSAL_VALUE"
    )
    return status, {
        "composition_G4_vs_G1": composition, "hyperbolic_G4_vs_G2": hyperbolic,
        "flexible_control_G4_vs_G3": flexible, "per_stratum_3_of_5": stratum_pass,
        "top1_delta_G4_G1_ood": top_g1, "switch_ratio_G4_G1_ood": switch_g1,
        "top1_delta_G4_G2_ood": top_g2, "switch_ratio_G4_G2_ood": switch_g2,
        "top1_delta_G4_G3_iid": iid_delta_g3, "top1_delta_G4_G3_ood": top_g3,
        "switch_ratio_G4_G3_ood": switch_g3, "per_stratum": wins, "full_pass": full,
    }


def _report(status: str, observability: dict, iid: dict, ood: dict, checks: dict) -> str:
    lines = ["# ANZA-LIRA SurfTrack V1 — S0 causal geometry", "", "## Status", "", f"`{status}`", "",
             "Zero-training geometry only. No seismic rendering, CNN, Thebe, CRACKS, or confirm data were opened.", "",
             "## Observability", "", f"- Center-only AUROC: `{observability['center_auroc']:.6f}` (required 0.45–0.55).",
             f"- Adjacent-history oracle Top1: `{observability['context_oracle_top1']:.6f}` (required >=0.85).", "",
             "| Method | IID Top1 | IID switch | OOD Top1 | OOD switch | Survival@7 OOD |", "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        lines.append(f"| {method} | {iid[method]['top1']:.4f} | {iid[method]['switch']:.4f} | {ood[method]['top1']:.4f} | {ood[method]['switch']:.4f} | {ood[method]['survival_7']:.4f} |")
    if (ROOT / "fitted_params.json").exists():
        fitted = json.loads((ROOT / "fitted_params.json").read_text()).get("methods", {})
        if fitted:
            lines.extend(["", "## Train-only fitted transport", ""])
            for method in METHODS[1:]:
                lines.append(f"- `{method}`: `{fitted[method]['params']}`; bound hits `{fitted[method]['bound_hits']}`.")
    if (ROOT / "bootstrap.json").exists():
        bootstrap = json.loads((ROOT / "bootstrap.json").read_text())
        lines.extend(["", "## Paired OOD bootstrap", ""])
        for key, row in bootstrap.items():
            lines.append(f"- `{key}`: `{row['mean_delta']:+.6f}`, 95% CI `[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}]`.")
    lines.extend(["", "## Frozen gates", ""])
    for key, value in checks.items():
        if key != "per_stratum": lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Claim boundary", "", "Only a full causal PASS could open learned SurfTrack. Every other status keeps S1, confirm, real data, and all Anosov-specific repairs locked.", ""])
    return "\n".join(lines)


def run_s0() -> dict[str, Any]:
    frozen = freeze_protocol(); protocol, split = load_frozen()
    observability = evaluate_observability(); write_json(ROOT / "observability.json", observability)
    if not observability["center_gate_pass"] or not observability["context_gate_pass"]:
        status = "STOP_SURFTRACK_BENCH_NOT_CAUSAL" if not observability["center_gate_pass"] else "STOP_SURFTRACK_LINEAGE_NOT_OBSERVABLE"
        result = {"status": status, "observability": observability, "confirm_accessed": False, **{f"{key}_opened": False for key in ("s1", "rendering", "cnn", "thebe", "cracks")}}
        write_json(ROOT / "fitted_params.json", {"status": "NOT_RUN_OBSERVABILITY_STOP"}); write_json(ROOT / "metrics.json", result)
        (ROOT / "ANZA_SURFTRACK_S0_REPORT.md").write_text(_report(status, observability, {m:{} for m in METHODS}, {m:{} for m in METHODS}, {}))
        from .validator import validate_s0
        return {"freeze": frozen["action"], "metrics": result, "validation": validate_s0()}

    fitted = fit_all(); fitted_receipt = {"status": "FROZEN", "fit_split": "geom_train", "dev_accessed": False, "methods": fitted}
    fitted_receipt["freeze_sha256"] = canonical_hash(fitted_receipt); write_json(ROOT / "fitted_params.json", fitted_receipt)
    all_rows = []; all_strata = []; metrics_by_split = {}; results_by_split = {}; selective_rows = []; calibrations = {}
    for split_name in ("geom_calibration", "geom_dev_iid", "geom_dev_ood"):
        batches = [generate_batch(split_name, start, min(1000, 10_000 - start)) for start in range(0, 10_000, 1000)]
        results_by_split[split_name] = {}; metrics_by_split[split_name] = {}
        for method in METHODS:
            result = evaluate_method_batches(method, fitted[method]["params"], split_name, batches); results_by_split[split_name][method] = result
            metrics_by_split[split_name][method] = summarize(result.rows)
            if split_name != "geom_calibration":
                all_rows.extend(result.rows); all_strata.extend(summarize_strata(result.rows))
            if split_name == "geom_calibration":
                calibrations[method] = fit_margin_calibration(result.margin, result.switch)
        del batches
    write_json(ROOT / "margin_calibration.json", calibrations)
    for split_name in ("geom_dev_iid", "geom_dev_ood"):
        for method in METHODS:
            result = results_by_split[split_name][method]; confidence = calibrated_confidence(result.margin, calibrations[method])
            for row in risk_coverage(confidence, result.switch): selective_rows.append({"split": split_name, "method": method, **row})
    iid = metrics_by_split["geom_dev_iid"]; ood = metrics_by_split["geom_dev_ood"]
    write_json(ROOT / "iid_metrics.json", iid); write_json(ROOT / "ood_metrics.json", ood)
    _write_csv(ROOT / "per_case.csv", all_rows); _write_csv(ROOT / "per_stratum.csv", all_strata); _write_csv(ROOT / "selective_curve.csv", selective_rows)
    ood_results = results_by_split["geom_dev_ood"]
    bootstrap = {
        "G4_minus_G1_top1": paired_bootstrap(ood_results["G4_anza_cocycle"].transition_correct, ood_results["G1_local_reset"].transition_correct),
        "G1_minus_G4_switch": paired_bootstrap(ood_results["G1_local_reset"].switch, ood_results["G4_anza_cocycle"].switch, seed=7302),
        "G4_minus_G2_top1": paired_bootstrap(ood_results["G4_anza_cocycle"].transition_correct, ood_results["G2_shear_compose"].transition_correct, seed=7303),
        "G2_minus_G4_switch": paired_bootstrap(ood_results["G2_shear_compose"].switch, ood_results["G4_anza_cocycle"].switch, seed=7304),
        "G4_minus_G3_top1": paired_bootstrap(ood_results["G4_anza_cocycle"].transition_correct, ood_results["G3_free_compose"].transition_correct, seed=7305),
        "G3_minus_G4_switch": paired_bootstrap(ood_results["G3_free_compose"].switch, ood_results["G4_anza_cocycle"].switch, seed=7306),
    }
    write_json(ROOT / "bootstrap.json", bootstrap)
    status, checks = _gate(iid, ood, all_strata, bootstrap)
    result = {"status": status, "observability": observability, "checks": checks, "iid": iid, "ood": ood,
              "fitted_params_freeze_sha256": fitted_receipt["freeze_sha256"], "confirm_accessed": False,
              **{f"{key}_opened": False for key in ("s1", "rendering", "cnn", "thebe", "cracks")}}
    write_json(ROOT / "metrics.json", result); generate_figures(ROOT / "figures", fitted, all_strata, iid, ood, selective_rows)
    (ROOT / "ANZA_SURFTRACK_S0_REPORT.md").write_text(_report(status, observability, iid, ood, checks))
    from .validator import validate_s0
    validation = validate_s0()
    if validation["status"] != "PASS": raise RuntimeError("SurfTrack S0 validation failed")
    return {"freeze": frozen["action"], "metrics": result, "validation": validation}
