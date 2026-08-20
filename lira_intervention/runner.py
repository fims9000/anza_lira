"""Execute only the authorized I0--I3 intervention phases."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from lira_intervention.candidate import evaluate_cases
from lira_intervention.data import build_interventions, recover_split_traces, save_jsonl, split_manifest
from lira_intervention.protocol import PROTOCOL, RESULT_ROOT, ROOT, canonical_hash, protocol_hash


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def freeze_i0() -> dict[str, object]:
    output = RESULT_ROOT / "i0_freeze"
    output.mkdir(parents=True, exist_ok=True)
    parent_validator = ROOT / "results/lira_final/f1_gap_audit/validator.json"
    parent_note = ROOT / ".codex/notes/anza-lira-final-f1-data-stop.md"
    if not parent_validator.is_file() or not parent_note.is_file():
        raise FileNotFoundError("immutable natural-gap STOP evidence is missing")
    manifest = split_manifest()
    receipt = {
        "status": "INTERVENTION_I0_FROZEN",
        "protocol": PROTOCOL,
        "protocol_sha256": protocol_hash(),
        "split_manifest": manifest,
        "parent_stop": "STOP_LIRA_REAL_GAP_DATA_INSUFFICIENT",
        "parent_validator_sha256": canonical_hash(json.loads(parent_validator.read_text())),
        "parent_note_present": True,
        "confirm_contents_opened": False,
        "expert_accessed": False,
        "path_opened": False,
    }
    _write_json(output / "protocol.json", PROTOCOL)
    _write_json(output / "split_manifest.json", manifest)
    _write_json(output / "freeze_receipt.json", receipt)
    return receipt


def build_i1() -> dict[str, object]:
    freeze = freeze_i0()
    output = RESULT_ROOT / "i1_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    manifest = freeze["split_manifest"]
    summaries = {}
    for split in ("ig_calibration", "ig_development"):
        trace_report, traces = recover_split_traces(split, manifest)
        cases = build_interventions(split, traces)
        save_jsonl(output / f"{split}.jsonl", cases)
        gap_counts = Counter(case.gap_length_px for case in cases)
        summaries[split] = {
            "sections": len(manifest["splits"][split]),
            "reliable_traces": len(traces),
            "interventions": len(cases),
            "gap_length_counts": {str(key): int(gap_counts[key]) for key in sorted(gap_counts)},
            "one_intervention_per_trace": len({case.trace_id for case in cases}) == len(cases),
            "image_changed": False,
            "expert_accessed": False,
            "trace_audit_status": trace_report["status"],
        }
        print(f"phase=I1_BENCHMARK split={split} traces={len(traces)} cases={len(cases)}", flush=True)
    calibration_ok = summaries["ig_calibration"]["interventions"] >= int(PROTOCOL["benchmark_minimum"]["calibration"])
    development_ok = summaries["ig_development"]["interventions"] >= int(PROTOCOL["benchmark_minimum"]["development"])
    report = {
        "status": "INTERVENTION_BENCHMARK_PASS" if calibration_ok and development_ok else "STOP_INTERVENTION_DATA_INSUFFICIENT",
        "protocol_sha256": protocol_hash(),
        "split_manifest_sha256": manifest["sha256"],
        "summaries": summaries,
        "confirm": {
            "section_ids_sha256": canonical_hash(manifest["splits"]["ig_confirm"]),
            "generator_sha256": canonical_hash(PROTOCOL["intervention"]),
            "contents_opened": False,
            "case_count": None,
        },
        "claim_boundary": PROTOCOL["claim_boundary"],
        "old_natural_gap_stop_changed": False,
        "expert_accessed": False,
    }
    _write_json(output / "benchmark_summary.json", report)
    lines = [
        "# CRACKS Intervention Benchmark V1",
        "",
        f"Status: `{report['status']}`",
        "",
        "This is a separate controlled-intervention endpoint. The immutable natural-gap result remains `STOP_LIRA_REAL_GAP_DATA_INSUFFICIENT`.",
        "",
        "The seismic image is unchanged. Only the frozen dense-evidence channel is erased in a 3-pixel tube over an internal segment of a real crowd trace.",
        "",
        "| Split | Sections | Reliable traces | Interventions |",
        "|---|---:|---:|---:|",
    ]
    for split in ("ig_calibration", "ig_development"):
        row = summaries[split]
        lines.append(f"| {split} | {row['sections']} | {row['reliable_traces']} | {row['interventions']} |")
    lines += [
        "",
        f"Confirm is hash-only and unopened: `{report['confirm']['section_ids_sha256']}`.",
        "",
        f"Claim boundary: {PROTOCOL['claim_boundary']}.",
    ]
    (output / "INTERVENTION_BENCHMARK_REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def run_i2() -> dict[str, object]:
    benchmark_path = RESULT_ROOT / "i1_benchmark/benchmark_summary.json"
    benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.is_file() else build_i1()
    if benchmark["status"] != "INTERVENTION_BENCHMARK_PASS":
        return {"status": benchmark["status"]}
    from lira_intervention.data import load_jsonl

    cases = load_jsonl(RESULT_ROOT / "i1_benchmark/ig_development.jsonl")
    summary, _rows = evaluate_cases(cases, RESULT_ROOT / "i2_candidate/development_candidates.jsonl")
    gate = float(PROTOCOL["i2_gate"]["candidate_recall_at_12"])
    summary.update({
        "status": "LIRA_INTERVENTION_CANDIDATE_PASS" if summary["candidate_recall_at_12"] >= gate else "STOP_LIRA_INTERVENTION_CANDIDATE",
        "gate": gate,
        "protocol_sha256": protocol_hash(),
        "confirm_opened": False,
        "expert_accessed": False,
    })
    _write_json(RESULT_ROOT / "i2_candidate/summary.json", summary)
    report = [
        "# LIRA Intervention Candidate Report",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"- BranchCandidateRecall@12: `{summary['candidate_recall_at_12']:.6f}` (gate `{gate:.2f}`)",
        f"- Recalled: `{summary['candidate_recalled']}/{summary['cases']}`",
        f"- Source available: `{summary['source_available']}/{summary['cases']}`",
        f"- Candidate burden median/P95: `{summary['median_candidates']:.1f}/{summary['p95_candidates']:.1f}`",
        f"- Image unchanged for every intervention: `{summary['image_unchanged_all']}`",
        "- Confirm, expert, P0, and path remained unopened in I2.",
    ]
    (RESULT_ROOT / "i2_candidate/LIRA_INTERVENTION_CANDIDATE_REPORT.md").write_text("\n".join(report) + "\n")
    return summary


def diagnose_i2() -> dict[str, object]:
    from lira_intervention.data import load_jsonl
    from lira_intervention.diagnostics import diagnose

    result = diagnose(
        load_jsonl(RESULT_ROOT / "i1_benchmark/ig_development.jsonl"),
        RESULT_ROOT / "i2_candidate/development_candidates.jsonl",
        RESULT_ROOT / "i2_candidate/development_diagnostics.csv",
    )
    _write_json(RESULT_ROOT / "i2_candidate/diagnostics.json", result)
    return result
