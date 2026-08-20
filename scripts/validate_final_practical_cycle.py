#!/usr/bin/env python3
"""Fail-closed validator for the ANZA final practical cycle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cracks_experiment.partial_label_training import t1_matrix
from path_completion.final_practical_reporting import FINAL_ROOT, STUDY_ROOT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate() -> dict[str, object]:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    required = (
        FINAL_ROOT / "THESIS_NUMBERS.json",
        FINAL_ROOT / "THESIS_EVIDENCE.md",
        FINAL_ROOT / "FINAL_PRACTICAL_REPORT.md",
        FINAL_ROOT / "figures" / "fig_t1_paired_deltas.png",
        FINAL_ROOT / "figures" / "fig_real_pair_gate.png",
    )
    for path in required:
        check(path.is_file() and path.stat().st_size > 0, f"missing or empty {path}")
    if failures:
        return {"status": "FAIL", "failures": failures}
    numbers = json.loads((FINAL_ROOT / "THESIS_NUMBERS.json").read_text())
    statuses = numbers["statuses"]
    check(statuses["path_classifier_v5"] == "PATH_CLASSIFIER_TEST_FAIL", "v5 status drift")
    check(statuses["v6_predicted_endpoints"] == "V6_PREDICTED_ENDPOINT_NEGATIVE", "v6 status drift")
    check(statuses["cracks_partial_labels"] == "CRACKS_PARTIAL_LABEL_SUCCESS", "T1 positive status missing")
    check(statuses["cracks_real_pair_classifier"] == "CRACKS_REAL_PAIR_CLASSIFIER_GATE_FAIL", "real-pair failure missing")
    check(statuses["anza_guided_completion"] == "NOT_RUN_GATE_LOCKED", "Phase E was not kept locked")
    check(statuses["final"] == "FINAL_PRACTICAL_NEGATIVE_WITH_ROOT_CAUSE", "final status is not claim-safe")
    check(numbers.get("expert_data_accessed") is False, "numbers do not preserve expert lock")
    check(numbers.get("expert_scores_used") is False, "numbers used expert scores")
    for row in numbers["provenance"].values():
        path = Path(row["path"])
        check(path.is_file() and _sha256(path) == row["sha256"], f"provenance hash drift: {path}")
    complete = 0
    for spec in t1_matrix():
        run = STUDY_ROOT / "cracks_t1" / f"{spec.run_id}-{spec.run_hash}" / "status.json"
        if not run.is_file():
            failures.append(f"missing T1 run {spec.run_id}")
            continue
        payload = json.loads(run.read_text())
        check(payload.get("status") == "COMPLETE" and payload.get("epoch") == 20, f"incomplete T1 run {spec.run_id}")
        check(payload.get("expert_data_accessed") is False, f"expert lock failed in {spec.run_id}")
        complete += payload.get("status") == "COMPLETE"
    check(complete == 6, "T1 matrix is not 6/6")
    t1 = json.loads((STUDY_ROOT / "cracks_t1" / "analysis" / "result.json").read_text())
    check(t1.get("section_count") == 392 and t1.get("seed_count") == 3, "T1 aggregation size drift")
    check(t1.get("expert_data_accessed") is False, "T1 analysis accessed expert")
    pair = json.loads((STUDY_ROOT / "cracks_pairs" / "result.json").read_text())
    check(pair.get("section_disjoint") is True and pair.get("balanced_50_50") is True, "pair dataset contract failed")
    check(pair.get("checks", {}).get("auroc") is True, "pair AUROC did not pass")
    check(pair.get("checks", {}).get("balanced_auprc") is True, "pair AUPRC did not pass")
    check(pair.get("checks", {}).get("tpr") is False, "pair TPR failure disappeared")
    check(pair.get("validation_operating_point", {}).get("fpr") <= 0.05, "pair FPR drift")
    check(pair.get("validation_operating_point", {}).get("tpr") < 0.70, "pair TPR no longer explains gate")
    check(pair.get("expert_data_accessed") is False, "pair stage accessed expert")
    v6 = json.loads((STUDY_ROOT / "realistic_synthetic" / "development_result.json").read_text())
    check(v6.get("v6_test_samples_opened") == 0, "v6 test was opened despite failed development gate")
    check(not (STUDY_ROOT / "real_completion").exists(), "Phase E artifacts exist despite gate lock")
    report = (FINAL_ROOT / "FINAL_PRACTICAL_REPORT.md").read_text()
    for value in (
        numbers["cracks_t1"]["unet"]["dice"]["delta"],
        numbers["cracks_t1"]["anza_v1"]["dice"]["delta"],
        numbers["real_pair_classifier"]["validation_auroc"],
        numbers["real_pair_classifier"]["tpr"],
    ):
        check(f"{value:.6f}" in report, f"report number not generated from THESIS_NUMBERS: {value}")
    check("ANZA_GUIDED_COMPLETION_SUCCESS" not in report, "report makes a false completion claim")
    status = "PASS" if not failures else "FAIL"
    return {
        "status": status,
        "final_status": statuses["final"],
        "independent_positive_status": statuses["cracks_partial_labels"],
        "t1_runs_complete": complete,
        "expert_data_accessed": False,
        "failures": failures,
    }


def main() -> int:
    result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
