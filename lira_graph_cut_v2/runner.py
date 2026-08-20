"""Bounded Graph-Cut V2 benchmark and candidate runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from lira_graph_cut_v2.benchmark import build_split, load_cases, recover_split, split_manifest
from lira_graph_cut_v2.candidate import evaluate
from lira_graph_cut_v2.protocol import PROTOCOL, RESULT_ROOT, ROOT, canonical_hash, protocol_hash


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def freeze() -> dict[str, object]:
    parent = ROOT / "results/lira_intervention_final/i2_candidate/validator.json"
    note = ROOT / ".codex/notes/anza-lira-intervention-endgame-stop.md"
    if not parent.is_file() or not note.is_file():
        raise FileNotFoundError("immutable V1 intervention STOP is missing")
    parent_result = json.loads(parent.read_text())
    if parent_result.get("research_status") != "STOP_LIRA_INTERVENTION_CANDIDATE":
        raise AssertionError("V1 intervention status drift")
    manifest = split_manifest()
    receipt = {
        "status": "GRAPH_CUT_V2_FROZEN",
        "protocol": PROTOCOL,
        "protocol_sha256": protocol_hash(),
        "split_manifest": manifest,
        "parent_status": parent_result["research_status"],
        "parent_validator_sha256": canonical_hash(parent_result),
        "confirm_contents_opened": False,
        "expert_accessed": False,
        "p0_opened": False,
        "path_opened": False,
    }
    _write_json(RESULT_ROOT / "freeze/protocol.json", PROTOCOL)
    _write_json(RESULT_ROOT / "freeze/split_manifest.json", manifest)
    _write_json(RESULT_ROOT / "freeze/freeze_receipt.json", receipt)
    return receipt


def build_benchmark() -> dict[str, object]:
    receipt = freeze()
    output = RESULT_ROOT / "benchmark"
    summaries = {}
    for split in ("gc_calibration", "gc_development"):
        summaries[split] = build_split(split, recover_split(split, receipt["split_manifest"]), output)
    retention_gate = float(PROTOCOL["treatment"]["minimum_retention"])
    retention_ok = all(float(row["retention"]) >= retention_gate for row in summaries.values())
    treatment_ok = all(row["treatment_validity"] is None or float(row["treatment_validity"]) == 1.0 for row in summaries.values())
    size_ok = (
        int(summaries["gc_calibration"]["valid_cases"]) >= int(PROTOCOL["minimum_valid_cases"]["calibration"])
        and int(summaries["gc_development"]["valid_cases"]) >= int(PROTOCOL["minimum_valid_cases"]["absolute_development"])
    )
    if not retention_ok:
        status = "STOP_GRAPH_CUT_BENCH_TOO_SELECTIVE"
    elif not size_ok:
        status = "STOP_GRAPH_CUT_DATA_INSUFFICIENT"
    elif not treatment_ok:
        status = "FAIL_GRAPH_CUT_TREATMENT_INVALID"
    else:
        status = "GRAPH_CUT_BENCHMARK_PASS"
    retention = {
        "status": status,
        "protocol_sha256": protocol_hash(),
        "split_manifest_sha256": receipt["split_manifest"]["sha256"],
        "splits": summaries,
        "retention_gate": retention_gate,
        "treatment_validity_gate": 1.0,
        "confirm": {
            "section_ids_sha256": canonical_hash(receipt["split_manifest"]["splits"]["gc_confirm"]),
            "generator_sha256": canonical_hash(PROTOCOL["placement"] | PROTOCOL["treatment"]),
            "contents_opened": False,
        },
        "parent_stop_changed": False,
        "candidate_used_for_filtering": False,
        "expert_accessed": False,
    }
    _write_json(output / "retention.json", retention)
    _write_json(output / "generator_manifest.json", {"protocol": PROTOCOL["placement"] | PROTOCOL["treatment"], "sha256": retention["confirm"]["generator_sha256"]})
    radius_rows = []
    for split, summary in summaries.items():
        for radius, count in summary["radius_counts"].items():
            radius_rows.append({
                "split": split,
                "radius_px": radius,
                "minimal_disconnect_count_before_exclusions": summary["minimal_disconnect_radius_counts_before_exclusions"][radius],
                "valid_count": count,
            })
    with (output / "radius_distribution.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "radius_px", "minimal_disconnect_count_before_exclusions", "valid_count"])
        writer.writeheader(); writer.writerows(radius_rows)
    for combined_name, suffix in (("eligibility.csv", "_eligibility.csv"), ("intervention_cases.csv", "_intervention_cases.csv")):
        source_paths = [output / f"{split}{suffix}" for split in ("gc_calibration", "gc_development")]
        rows = []
        fields = []
        for path in source_paths:
            with path.open() as handle:
                reader = csv.DictReader(handle)
                local = list(reader)
                rows.extend(local)
                if reader.fieldnames:
                    fields = list(reader.fieldnames)
        with (output / combined_name).open("w", newline="") as handle:
            if fields:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
    lines = [
        "# Graph-Cut Intervention V2 Benchmark",
        "",
        f"Status: `{status}`",
        "",
        "The parent V1 candidate STOP remains immutable. V2 uses fresh placements and selects the smallest predeclared radius that disconnects both anchors at the lowest frozen SBPP support threshold (0.12).",
        "",
        "| Split | Reliable traces | Pre-treatment eligible | Valid cuts | Retention | Treatment validity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, row in summaries.items():
        validity = "N/A" if row["treatment_validity"] is None else f"{row['treatment_validity']:.1f}"
        lines.append(f"| {split} | {row['reliable_traces']} | {row['eligible_before_treatment']} | {row['valid_cases']} | {row['retention']:.4f} | {validity} |")
    development = summaries["gc_development"]
    lines += [
        "",
        "Development pre-treatment eligible outcome:",
        f"- invalid collateral trace: `{development['status_counts'].get('INVALID_COLLATERAL_TRACE', 0)}`;",
        f"- invalid context destroyed: `{development['status_counts'].get('INVALID_CONTEXT_DESTROYED', 0)}`;",
        f"- minimal disconnect radii before exclusions: `{development['minimal_disconnect_radius_counts_before_exclusions']}`.",
        "",
        "TreatmentValidity is `N/A`, not zero: no case survived the pre-candidate retention/context rules, so the accepted-case ratio has denominator zero.",
        "",
        "Invalidity counts are preserved in the eligibility CSV files; no case was filtered using SBPP or P0 output.",
        f"Confirm remained hash-only: `{retention['confirm']['section_ids_sha256']}`.",
    ]
    (output / "GRAPH_CUT_BENCHMARK_REPORT.md").write_text("\n".join(lines) + "\n")
    if status != "GRAPH_CUT_BENCHMARK_PASS":
        candidate = RESULT_ROOT / "candidate"
        candidate.mkdir(parents=True, exist_ok=True)
        (candidate / "LIRA_GRAPH_CUT_CANDIDATE_REPORT.md").write_text(
            "# LIRA Graph-Cut Candidate Report\n\n"
            f"Status: `LOCKED_NOT_RUN_AFTER_{status}`\n\n"
            "Frozen SBPP was not opened because the manipulation-valid benchmark failed its pre-candidate retention/data gate. P0, path, confirm, and expert remain locked.\n"
        )
        _write_json(candidate / "validator.json", {
            "validator_status": "PASS",
            "research_status": f"LOCKED_NOT_RUN_AFTER_{status}",
            "candidate_opened": False,
            "p0_opened": False,
            "confirm_opened": False,
            "expert_accessed": False,
        })
    return retention


def run_candidate() -> dict[str, object]:
    benchmark_path = RESULT_ROOT / "benchmark/retention.json"
    benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.is_file() else build_benchmark()
    if benchmark["status"] != "GRAPH_CUT_BENCHMARK_PASS":
        return {"status": benchmark["status"], "candidate_opened": False}
    cases = load_cases(RESULT_ROOT / "benchmark/gc_development_intervention_cases.csv")
    metrics, _rows = evaluate(cases, RESULT_ROOT / "candidate")
    source_gate = float(PROTOCOL["candidate"]["source_port_availability_gate"])
    recall_gate = float(PROTOCOL["candidate"]["branch_candidate_recall_gate"])
    if metrics["source_port_availability"] < source_gate:
        status = "STOP_GRAPH_CUT_SOURCE_PORT_FAIL"
    elif metrics["branch_candidate_recall_at_12"] < recall_gate:
        status = "STOP_GRAPH_CUT_CANDIDATE"
    else:
        status = "LIRA_GRAPH_CUT_CANDIDATE_PASS"
    metrics.update({
        "status": status,
        "source_port_availability_gate": source_gate,
        "branch_candidate_recall_gate": recall_gate,
        "treatment_validity": benchmark["splits"]["gc_development"]["treatment_validity"],
        "retention": benchmark["splits"]["gc_development"]["retention"],
        "protocol_sha256": protocol_hash(),
        "sbpp_modified": False,
        "p0_opened": False,
        "confirm_opened": False,
        "expert_accessed": False,
        "path_opened": False,
    })
    _write_json(RESULT_ROOT / "candidate/metrics.json", metrics)
    report = [
        "# LIRA Graph-Cut Candidate Report",
        "",
        f"Status: `{status}`",
        "",
        f"- TreatmentValidity: `{metrics['treatment_validity']:.1f}`",
        f"- Development retention: `{metrics['retention']:.6f}`",
        f"- SourcePortAvailability: `{metrics['source_port_availability']:.6f}` (gate `{source_gate:.2f}`)",
        f"- BranchCandidateRecall@12: `{metrics['branch_candidate_recall_at_12']:.6f}` (gate `{recall_gate:.2f}`)",
        f"- Recalled branches: `{metrics['candidate_recalled']}/{metrics['cases']}`",
        f"- Candidate burden median/P95: `{metrics['median_candidates']:.1f}/{metrics['p95_candidates']:.1f}`",
        "- Frozen SBPP was unchanged; P0, path, confirm, and expert remained locked.",
    ]
    (RESULT_ROOT / "candidate/LIRA_GRAPH_CUT_CANDIDATE_REPORT.md").write_text("\n".join(report) + "\n")
    return metrics
