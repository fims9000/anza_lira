"""Independent SurfTrack S0 artifact and frozen-decision validator."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .eval.tracking import summarize
from .protocol import METHODS, ROOT, canonical_hash, load_frozen, write_json
from .run_s0 import _gate


VALID = {
    "ANZA_SURFTRACK_COCYCLE_PASS", "SURFTRACK_COMPOSITION_PASS_ANOSOV_NOT_SPECIFIC",
    "STOP_ANOSOV_SURFTRACK_NO_CAUSAL_VALUE", "STOP_SURFTRACK_BENCH_NOT_CAUSAL",
    "STOP_SURFTRACK_LINEAGE_NOT_OBSERVABLE",
}


def validate_s0() -> dict:
    protocol, split = load_frozen(); required = [
        ROOT / "protocol.json", ROOT / "protocol_hash.txt", ROOT / "split_manifest.json", ROOT / "generator_manifest.json",
        ROOT / "code_hashes.json", ROOT / "fitted_params.json", ROOT / "observability.json", ROOT / "metrics.json", ROOT / "ANZA_SURFTRACK_S0_REPORT.md",
    ]
    checks = {f"exists:{path.name}": path.is_file() for path in required}
    if not all(checks.values()):
        result = {"status": "FAIL", "checks": checks}; write_json(ROOT / "validator.json", result); return result
    metrics = json.loads((ROOT / "metrics.json").read_text()); observability = json.loads((ROOT / "observability.json").read_text())
    project = Path(__file__).resolve().parents[1]; code_hashes = json.loads((ROOT / "code_hashes.json").read_text())
    checks.update({
        "valid_research_status": metrics.get("status") in VALID,
        "observability_recomputed": observability["center_gate_pass"] == (0.45 <= observability["center_auroc"] <= 0.55)
        and observability["context_gate_pass"] == (observability["context_oracle_top1"] >= 0.85),
        "split_seeds_disjoint": split["seeds_disjoint"] is True,
        "confirm_hash_only": split["confirm_access"] == "HASH_ONLY_NOT_GENERATED" and metrics.get("confirm_accessed") is False,
        "downstream_locked": all(metrics.get(f"{key}_opened") is False for key in ("s1", "rendering", "cnn", "thebe", "cracks")),
        "code_hashes": all(hashlib.sha256((project / relative).read_bytes()).hexdigest() == expected for relative, expected in code_hashes.items()),
    })
    if observability["center_gate_pass"] and observability["context_gate_pass"]:
        extra = [ROOT / "per_case.csv", ROOT / "per_stratum.csv", ROOT / "iid_metrics.json", ROOT / "ood_metrics.json",
                 ROOT / "selective_curve.csv", ROOT / "bootstrap.json", ROOT / "margin_calibration.json"]
        checks.update({f"exists:{path.name}": path.is_file() for path in extra})
        params = json.loads((ROOT / "fitted_params.json").read_text()); params_sha = params.pop("freeze_sha256", None)
        checks["params_freeze"] = params_sha == canonical_hash(params) and params["fit_split"] == "geom_train" and params["dev_accessed"] is False
        if all(path.is_file() for path in extra):
            with (ROOT / "per_case.csv").open(newline="") as handle: rows = list(csv.DictReader(handle))
            converted = [{**row, **{key: float(row[key]) for key in ("top1", "switch", "survival_3", "survival_7", "survival_15", "mean_position_error_correct_lineage", "margin")}} for row in rows]
            by = {(split_name, method): [row for row in converted if row["split"] == split_name and row["method"] == method]
                  for split_name in ("geom_dev_iid", "geom_dev_ood") for method in METHODS}
            recomputed_iid = {method: summarize(by[("geom_dev_iid", method)]) for method in METHODS}
            recomputed_ood = {method: summarize(by[("geom_dev_ood", method)]) for method in METHODS}
            stored_iid = json.loads((ROOT / "iid_metrics.json").read_text()); stored_ood = json.loads((ROOT / "ood_metrics.json").read_text())
            checks["metric_rows_complete"] = all(len(local) == 10_000 for local in by.values())
            checks["metrics_recomputed"] = all(np.isclose(recomputed_iid[m][k], stored_iid[m][k], equal_nan=True) and np.isclose(recomputed_ood[m][k], stored_ood[m][k], equal_nan=True)
                                                       for m in METHODS for k in ("top1", "switch", "survival_3", "survival_7", "survival_15"))
            with (ROOT / "per_stratum.csv").open(newline="") as handle: strata = list(csv.DictReader(handle))
            strata = [{**row, **{key: float(row[key]) for key in ("top1", "switch", "survival_3", "survival_7", "survival_15")}} for row in strata]
            bootstrap = json.loads((ROOT / "bootstrap.json").read_text()); status, gate = _gate(recomputed_iid, recomputed_ood, strata, bootstrap)
            checks["gate_recomputed"] = status == metrics["status"] and gate == metrics["checks"]
        figures = list((ROOT / "figures").glob("*.png")); checks["eight_code_figures"] = len(figures) == 8
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "research_status": metrics.get("status"), "checks": checks,
              "confirm_accessed": False, "real_data_accessed": False}
    write_json(ROOT / "validator.json", result); return result
