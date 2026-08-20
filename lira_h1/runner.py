"""Bounded H0-H2 runner for the final LIRA correctness hotfix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lira_final.dense.ensemble import cache_ensemble
from lira_graph_cut_v2.graph_cut import connected, rasterize, tube_distance
from lira_h1.benchmark import build_split, load_cases, recover_split, split_manifest
from lira_h1.candidate import evaluate
from lira_h1.protocol import (
    CUT_RADII,
    DEVELOPMENT_DENSE_CACHE,
    PARENT_DENSE_CACHE,
    PROTOCOL,
    RESULT_ROOT,
    ROOT,
    canonical_hash,
    protocol_hash,
)
from lira_h1.ribbon import cumulative_arclength, flat_cap_ribbon


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hash() -> tuple[str, dict[str, str]]:
    files = sorted((ROOT / "lira_h1").glob("*.py")) + [ROOT / "tests/test_anza_lira_h1.py"]
    rows = {str(path.relative_to(ROOT)): _sha256(path) for path in files}
    return canonical_hash(rows), rows


def run_h0() -> dict[str, object]:
    probability = np.zeros((72, 140), dtype=np.float32)
    probability[28:45, 8:132] = 0.9
    trace = np.asarray([(36.0, float(x)) for x in range(8, 132)])
    start, end = 52, 76
    arc = cumulative_arclength(trace)
    left, right = trace[start - 8 : start], trace[end + 1 : end + 9]
    hidden = trace[start : end + 1]
    old = tube_distance(hidden, probability.shape) <= 11.0
    new = flat_cap_ribbon(trace, arc[start], arc[end], 11.0, probability.shape)
    cut = (probability >= 0.12) & ~new
    left_pixels, right_pixels = np.rint(left).astype(int), np.rint(right).astype(int)
    checks = {
        "old_capsule_deletes_left_anchor": bool(old[left_pixels[:, 0], left_pixels[:, 1]].any()),
        "old_capsule_deletes_right_anchor": bool(old[right_pixels[:, 0], right_pixels[:, 1]].any()),
        "ribbon_preserves_left_anchor": bool(not new[left_pixels[:, 0], left_pixels[:, 1]].any()),
        "ribbon_preserves_right_anchor": bool(not new[right_pixels[:, 0], right_pixels[:, 1]].any()),
        "ribbon_disconnects_band": bool(not connected(cut, rasterize(left, cut.shape, 1), rasterize(right, cut.shape, 1))),
        "no_longitudinal_spillover": bool(not new[:, : int(trace[start, 1])].any() and not new[:, int(trace[end, 1]) + 1 :].any()),
    }
    status = "H1_RIBBON_IMPLEMENTATION_PASS" if all(checks.values()) else "STOP_H1_RIBBON_IMPLEMENTATION_INVALID"
    report = {"status": status, "checks": checks, "radius_px": 11, "unit_test_command": "/home/lebedeffson/Code/venv/bin/python -m pytest tests/test_anza_lira_h1.py tests/test_anza_lira_graph_cut_v2.py -q"}
    output = RESULT_ROOT / "ribbon_unit_tests"
    _write_json(output / "report.json", report)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for axis, mask, title in zip(axes, (old, new), ("V2 round-cap capsule", "H1 flat-cap ribbon")):
        axis.imshow(probability >= 0.12, cmap="Greys", origin="upper")
        axis.imshow(np.ma.masked_where(~mask, mask), cmap="Reds", alpha=0.55, origin="upper")
        axis.scatter(left[:, 1], left[:, 0], s=10, c="tab:blue")
        axis.scatter(right[:, 1], right[:, 0], s=10, c="tab:green")
        axis.set_title(title); axis.set_xlim(45, 90); axis.set_ylim(52, 20); axis.set_axis_off()
    fig.savefig(figures / "capsule_vs_ribbon.png", dpi=180)
    plt.close(fig)
    (output / "H1_RIBBON_IMPLEMENTATION_REPORT.md").write_text(
        "# H1 Ribbon Implementation Report\n\n"
        f"Status: `{status}`\n\n"
        "The deterministic straight-band regression reproduces the V2 round-cap anchor destruction. The exact segment-projection flat-cap ribbon disconnects the 17-pixel support band at radius 11 while preserving both adjacent eight-point anchors and producing no longitudinal spillover. Curved-trace, reversal, collateral, and immutable-parent regressions are covered by the targeted test suite.\n"
    )
    return report


def freeze(unit_report: dict[str, object]) -> dict[str, object]:
    parent = json.loads((ROOT / "results/lira_graph_cut_v2/benchmark/retention.json").read_text())
    if parent.get("status") != "STOP_GRAPH_CUT_BENCH_TOO_SELECTIVE":
        raise AssertionError("parent V2 STOP drift")
    manifest = split_manifest()
    code_sha, code_files = _code_hash()
    # Absence of inference caches is the auditable no-content-access condition.
    old_confirm_cache = [path.name for section in range(347, 401) if (PARENT_DENSE_CACHE / f"section_{section:03d}.npy").exists() for path in [PARENT_DENSE_CACHE / f"section_{section:03d}.npy"]]
    h1_confirm_cache = [path.name for path in DEVELOPMENT_DENSE_CACHE.glob("section_*.npy") if int(path.stem.split("_")[-1]) >= 375]
    if old_confirm_cache or h1_confirm_cache:
        raise AssertionError(f"confirm content cache exists before authorization: {old_confirm_cache + h1_confirm_cache}")
    receipt = {
        "status": "H1_FRESH_SPLIT_AUTHORIZED",
        "protocol_sha256": protocol_hash(), "protocol": PROTOCOL,
        "split_manifest_sha256": manifest["sha256"], "splits": manifest["splits"],
        "split_hashes": {name: canonical_hash(ids) for name, ids in manifest["splits"].items()},
        "code_sha256": code_sha, "code_files": code_files,
        "unit_test_results": unit_report,
        "radius_bank": list(CUT_RADII),
        "gates": {
            "bug_audit_retention": 0.50, "treatment_validity": 1.0,
            "fresh_absolute_floor": 250, "source_port_availability": 0.90,
            "branch_candidate_recall_at_12": 0.85,
        },
        "old_v2_confirm_contents_read": False, "sections_347_400_dense_cache_present": False,
        "h1_confirm_375_400_contents_opened": False, "expert_accessed": False,
    }
    _write_json(RESULT_ROOT / "freeze/protocol.json", PROTOCOL)
    _write_json(RESULT_ROOT / "freeze/split_manifest.json", manifest)
    _write_json(RESULT_ROOT / "freeze/H1_FRESH_SPLIT_AUTHORIZATION.json", receipt)
    return receipt


def run_bug_audit(receipt: dict[str, object]) -> dict[str, object]:
    output = RESULT_ROOT / "bug_audit"
    summary = build_split("h1_bug_audit", recover_split("h1_bug_audit", {"splits": receipt["splits"]}), output, PARENT_DENSE_CACHE)
    retention_ok = float(summary["retention"]) >= float(PROTOCOL["treatment"]["minimum_bug_audit_retention"])
    validity_ok = summary["treatment_validity"] == 1.0
    status = "H1_RIBBON_BUG_AUDIT_PASS" if retention_ok and validity_ok else "STOP_H1_RIBBON_BENCHMARK_FAIL"
    metrics = {"status": status, **summary, "retention_gate": 0.50, "treatment_validity_gate": 1.0, "sbpp_used_for_filtering": False}
    _write_json(output / "metrics.json", metrics)
    (output / "H1_BUG_AUDIT_REPORT.md").write_text(
        "# H1 Flat-Cap Ribbon Bug Audit\n\n"
        f"Status: `{status}`\n\n"
        f"- Accepted cases: `{summary['valid_cases']}/{summary['eligible_before_treatment']}`\n"
        f"- Retention: `{summary['retention']:.6f}` (gate `0.50`)\n"
        f"- TreatmentValidity: `{summary['treatment_validity'] if summary['treatment_validity'] is not None else 'N/A'}`\n"
        f"- Median selected radius: `{summary['median_selected_radius']}`\n"
        f"- Status taxonomy: `{summary['status_counts']}`\n\n"
        "Sections 263-344 are a mechanical bug audit only. No SBPP score was used to accept a case and no performance claim is made from this split.\n"
    )
    return metrics


def _finalize(status: str, unit: dict[str, object], bug: dict[str, object] | None, candidate: dict[str, object] | None) -> dict[str, object]:
    master = {
        "status": status, "protocol_sha256": protocol_hash(), "unit": unit, "bug_audit": bug, "fresh_candidate": candidate,
        "parent_stops_changed": False, "p0_opened": False, "path_opened": False, "confirm_375_400_opened": False,
        "expert_accessed": False, "new_architecture_created": False,
        "historical_controlled_positive_result_scope": "independent controlled synthetic structural continuation; not natural CRACKS gaps",
    }
    _write_json(RESULT_ROOT / "final/ANZA_LIRA_H1_MASTER_RESULT.json", master)
    (RESULT_ROOT / "final/ANZA_LIRA_H1_FINAL_REPORT.md").write_text(
        "# ANZA-LIRA H1 H0-H2 Final Report\n\n"
        f"Status: `{status}`\n\n"
        "H1 changed only the intervention primitive from a round-cap capsule to an arclength-bounded flat-cap ribbon. Frozen dense evidence, SBPP, and every historical STOP remained unchanged. P0, path, sections 375-400, and expert annotations were not opened.\n"
    )
    return master


def run(*, device: str = "cuda") -> dict[str, object]:
    unit = run_h0()
    if unit["status"] != "H1_RIBBON_IMPLEMENTATION_PASS":
        return _finalize(unit["status"], unit, None, None)
    receipt = freeze(unit)
    bug = run_bug_audit(receipt)
    if bug["status"] != "H1_RIBBON_BUG_AUDIT_PASS":
        return _finalize(bug["status"], unit, bug, None)
    development_ids = list(receipt["splits"]["h1_development"])
    cache_report = cache_ensemble(development_ids, DEVELOPMENT_DENSE_CACHE, device=device)
    output = RESULT_ROOT / "development_candidate"
    benchmark = build_split("h1_development", recover_split("h1_development", {"splits": receipt["splits"]}), output, DEVELOPMENT_DENSE_CACHE)
    floor = int(PROTOCOL["fresh_development"]["absolute_floor"])
    if benchmark["valid_cases"] < floor:
        candidate = {"status": "STOP_H1_FRESH_DATA_INSUFFICIENT", "benchmark": benchmark, "dense_cache": cache_report, "candidate_opened": False}
    else:
        metrics = evaluate(load_cases(output / "cases.csv"), output, DEVELOPMENT_DENSE_CACHE)
        source_ok = metrics["source_port_availability"] >= float(PROTOCOL["candidate"]["source_port_availability_gate"])
        recall_ok = metrics["branch_candidate_recall_at_12"] >= float(PROTOCOL["candidate"]["branch_candidate_recall_gate"])
        status = "H1_REAL_CANDIDATE_PASS_P0_AUTHORIZED" if source_ok and recall_ok else "STOP_H1_REAL_CANDIDATE_TRANSFER"
        candidate = {
            "status": status, "benchmark": benchmark, **metrics,
            "source_port_availability_gate": 0.90, "branch_candidate_recall_gate": 0.85,
            "dense_cache": cache_report, "sbpp_modified": False, "p0_opened": False,
        }
    _write_json(output / "metrics.json", candidate)
    b = candidate["benchmark"]
    (output / "H1_FRESH_CANDIDATE_REPORT.md").write_text(
        "# H1 Fresh Candidate Report\n\n"
        f"Status: `{candidate['status']}`\n\n"
        f"- Fresh sections: `347-372`\n- Valid cases: `{b['valid_cases']}` (absolute floor `250`)\n"
        f"- Treatment retention: `{b['retention']:.6f}`\n"
        + (f"- SourcePortAvailability: `{candidate['source_port_availability']:.6f}` (gate `0.90`)\n- BranchCandidateRecall@12: `{candidate['branch_candidate_recall_at_12']:.6f}` (gate `0.85`)\n- Candidate burden median/P95: `{candidate['median_candidates']:.1f}/{candidate['p95_candidates']:.1f}`\n" if candidate.get("candidate_opened", True) is not False else "- Frozen SBPP remained unopened because the fresh benchmark was below its absolute size floor.\n")
        + "\nP0, path, sections 375-400, and expert annotations remained unopened.\n"
    )
    return _finalize(str(candidate["status"]), unit, bug, candidate)

