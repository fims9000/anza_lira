"""Real CRACKS trace identity, dense calibration, and natural-gap audit."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from lira_final.data.cracks_trace_audit import CrowdTrace, audit_trace_identity
from lira_final.data.natural_gaps import NaturalGap, gaps_for_split
from lira_final.data.splits import build_split_manifest
from lira_final.dense.calibration import calibrate_dense_threshold
from lira_final.dense.ensemble import cache_ensemble
from lira_final.io import write_csv, write_json
from lira_final.protocol import HELDOUT_ANNOTATORS, PROTOCOL, RESULT_ROOT, ROOT, protocol_hash


def _negative_source_count(traces: Iterable[CrowdTrace]) -> int:
    local = list(traces)
    count = 0
    for source in local:
        if len(source.points_yx) < 8:
            continue
        for endpoint, inner in ((source.points_yx[0], source.points_yx[min(5, len(source.points_yx) - 1)]), (source.points_yx[-1], source.points_yx[max(0, len(source.points_yx) - 6)])):
            tangent = endpoint - inner
            tangent /= max(float(np.linalg.norm(tangent)), 1e-8)
            eligible = False
            for other in local:
                if other.trace_id == source.trace_id:
                    continue
                delta = other.points_yx - endpoint
                distance = np.linalg.norm(delta, axis=1)
                valid = (distance >= 6.0) & (distance <= 68.0)
                if not np.any(valid):
                    continue
                directions = delta[valid] / np.maximum(distance[valid, None], 1e-8)
                if np.any(directions @ tangent > np.cos(np.deg2rad(78.0))):
                    eligible = True
                    break
            count += int(eligible)
    return count


def _gap_rows(gaps: tuple[NaturalGap, ...], split: str) -> list[dict[str, object]]:
    rows = []
    for gap in gaps:
        row = gap.row()
        row["split"] = split
        row["source_y"] = gap.source_yx[0]
        row["source_x"] = gap.source_yx[1]
        row["destination_y"] = gap.destination_yx[0]
        row["destination_x"] = gap.destination_yx[1]
        for key in ("source_yx", "destination_yx", "source_tangent_yx", "destination_tangent_yx", "gap_points_yx", "destination_context_yx"):
            row.pop(key, None)
        rows.append(row)
    return rows


def run_f1(*, device: str) -> tuple[dict[str, object], dict[str, object]]:
    output = RESULT_ROOT / "f1_gap_audit"
    output.mkdir(parents=True, exist_ok=True)
    split_manifest = build_split_manifest()
    write_json(output / "split_manifest.json", split_manifest)
    opened_sections = sorted(set(sum((split_manifest["splits"][name] for name in ("relation_train", "dense_calibration", "lira_calibration", "lira_development")), [])))
    dense_receipt = cache_ensemble(opened_sections, output / "dense_cache", device=device)
    write_json(output / "dense_ensemble_receipt.json", dense_receipt)
    identity, traces = audit_trace_identity(ROOT / "data/cracks/annotations", opened_sections, HELDOUT_ANNOTATORS)
    write_json(output / "trace_identity_audit.json", identity)
    lines = [
        "# CRACKS Trace Identity Audit", "", f"Status: `{identity['status']}`", "",
        "The release provides RGB raster masks, not polyline/instance IDs. Final LIRA therefore uses only ordered skeleton segments without junctions, loops, or border truncation as local, defensible trace identities. Different local `trace_id` values may be used as negatives; proximity alone never defines identity.", "",
        f"- reliable local traces: `{identity['reliable_trace_count']}`", f"- missing nonexpert annotation files: `{identity['missing_annotation_files']}`", "- expert accessed: `false`", "",
        "This supports local continuation evaluation only. It does not identify geological fault instances through crossings or across disconnected raster components.",
    ]
    (output / "CRACKS_TRACE_IDENTITY_AUDIT.md").write_text("\n".join(lines) + "\n")
    if identity["status"] != "PASS":
        result = {"status": "STOP_REAL_LINEAGE_LABELS_UNAVAILABLE", "phase": "F1_REAL_GAP_AUDIT", "expert_accessed": False}
        write_json(output / "metrics.json", result)
        return result, {"traces": traces, "gaps": {}}
    calibration = calibrate_dense_threshold(output / "dense_cache", ROOT / "data/cracks/annotations", split_manifest["splits"]["dense_calibration"])
    write_json(output / "dense_threshold_freeze.json", calibration)
    threshold = float(calibration["selected_threshold"])
    gaps = {}
    all_gap_rows = []
    split_summaries = {}
    for split in ("lira_calibration", "lira_development"):
        local = gaps_for_split(output / "dense_cache", split_manifest["splits"][split], traces, threshold)
        gaps[split] = local
        rows = _gap_rows(local, split)
        all_gap_rows.extend(rows)
        section_counts = Counter(gap.section_id for gap in local)
        trace_groups = [trace for (section, _), values in traces.items() if section in split_manifest["splits"][split] for trace in values]
        split_summaries[split] = {
            "positive_gaps": len(local),
            "negative_sources_available": _negative_source_count(trace_groups),
            "sections_with_gaps": len(section_counts),
            "per_section": dict(sorted(section_counts.items())),
            "gap_length_quantiles": {str(q): float(np.quantile([gap.length_px for gap in local], q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)} if local else {},
        }
    diagnostic_sections = sorted(set(opened_sections) - set(split_manifest["splits"]["lira_confirm"]))
    diagnostic_gaps = gaps_for_split(output / "dense_cache", diagnostic_sections, traces, threshold)
    write_csv(output / "natural_gaps.csv", all_gap_rows)
    write_json(output / "natural_gap_manifest.json", {split: [gap.row() for gap in values] for split, values in gaps.items()})
    dev = split_summaries["lira_development"]
    enough = dev["positive_gaps"] >= 150 and dev["negative_sources_available"] >= 150
    absolute = dev["positive_gaps"] >= 75 and dev["negative_sources_available"] >= 75
    status = "F1_PASS" if enough else ("F1_LIMITED_SAMPLE" if absolute else "STOP_LIRA_REAL_GAP_DATA_INSUFFICIENT")
    result = {
        "status": status,
        "phase": "F1_REAL_GAP_AUDIT",
        "protocol_sha256": protocol_hash(),
        "dense_threshold": threshold,
        "dense_calibration": calibration["selected"],
        "splits": split_summaries,
        "all_opened_nonconfirm_diagnostic": {
            "section_count": len(diagnostic_sections),
            "positive_gaps": len(diagnostic_gaps),
            "purpose": "pre-gate data sufficiency diagnostic only; not model or threshold selection",
            "interpretation": "marginally above the pooled absolute floor, but cannot supply independent calibration/development/confirm sets and cannot justify changing the frozen split after counts were observed",
        },
        "confirm": {"sections": len(split_manifest["splits"]["lira_confirm"]), "hash_only": True, "inference_opened": False, "metrics_opened": False, "minimum_count_gate_deferred_until_authorized_one_shot" : True},
        "expert_accessed": False,
        "claim_boundary": PROTOCOL["claim_boundary"],
    }
    write_json(output / "metrics.json", result)
    report = [
        "# LIRA Real Gap Audit", "", f"Status: `{status}`", "", f"Frozen dense threshold: `{threshold:.4f}`.", "",
        "## Opened nonexpert splits", "",
    ]
    for split, summary in split_summaries.items():
        report += [f"- {split}: `{summary['positive_gaps']}` natural positive gaps; `{summary['negative_sources_available']}` eligible different-trace negative sources."]
    report += ["", "The confirm section IDs and hash are frozen, but confirm inference and natural-gap counts remain unopened. Expert annotations were not accessed.", "", f"Claim boundary: {PROTOCOL['claim_boundary']}"]
    report += ["", f"Across all `{len(diagnostic_sections)}` already-opened non-confirm sections, the same frozen definition found `{len(diagnostic_gaps)}` gaps. This is only marginally above the pooled absolute floor of 75, cannot supply independent calibration/development/confirm cohorts, and does not authorize reallocating the frozen split after counts were observed."]
    (output / "LIRA_REAL_GAP_AUDIT.md").write_text("\n".join(report) + "\n")
    return result, {"traces": traces, "gaps": gaps, "split_manifest": split_manifest, "threshold": threshold}
