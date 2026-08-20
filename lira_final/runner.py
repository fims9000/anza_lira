"""Gated orchestrator for the authorized F0--F3 final endgame slice."""

from __future__ import annotations

from typing import Any

from lira_final.f0 import run_f0
from lira_final.f1 import run_f1
from lira_final.protocol import RESULT_ROOT


def _write_locked_downstream(reason: str) -> None:
    f2 = RESULT_ROOT / "f2_candidate"
    f3 = RESULT_ROOT / "f3_relation_s41"
    f2.mkdir(parents=True, exist_ok=True)
    f3.mkdir(parents=True, exist_ok=True)
    (f2 / "LIRA_REAL_CANDIDATE_REPORT.md").write_text(
        "# LIRA Real Candidate Report\n\nStatus: `LOCKED_NOT_RUN`\n\n"
        f"F2 was not opened because F1 ended with `{reason}`. No SBPP metrics were computed.\n"
    )
    (f3 / "LIRA_REAL_RELATION_S41_REPORT.md").write_text(
        "# LIRA Real Relation Seed 41 Report\n\nStatus: `LOCKED_NOT_RUN`\n\n"
        f"F3 was not opened because F1 ended with `{reason}`. P0 was not fine-tuned, calibrated, or scored.\n"
    )


def run(*, device: str = "cuda", stop_after: str = "F3") -> dict[str, Any]:
    if stop_after not in {"F1", "F2", "F3"}:
        raise ValueError("stop_after must be F1, F2, or F3")
    f0 = run_f0()
    f1, context = run_f1(device=device)
    summary: dict[str, Any] = {"f0": f0["status"], "f1": f1["status"], "f2": "LOCKED", "f3": "LOCKED"}
    if stop_after == "F1" or f1["status"] not in {"F1_PASS", "F1_LIMITED_SAMPLE"}:
        if f1["status"] not in {"F1_PASS", "F1_LIMITED_SAMPLE"}:
            _write_locked_downstream(str(f1["status"]))
        return summary
    from lira_final.f2 import run_f2
    f2, context = run_f2(context)
    summary["f2"] = f2["status"]
    if stop_after == "F2" or f2["status"] != "F2_PASS":
        return summary
    from lira_final.f3 import run_f3
    f3 = run_f3(context, device=device)
    summary["f3"] = f3["status"]
    return summary
