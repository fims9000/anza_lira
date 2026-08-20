"""Post-STOP diagnostics that do not alter intervention cases or gates."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from lira_final.dense.ensemble import load_probability
from lira_intervention.candidate import masked_probability
from lira_intervention.data import Intervention
from lira_intervention.protocol import DENSE_CACHE, PROTOCOL


def _pixels(points: tuple[tuple[float, float], ...], shape: tuple[int, int]) -> np.ndarray:
    value = np.rint(np.asarray(points)).astype(int)
    value[:, 0] = np.clip(value[:, 0], 0, shape[0] - 1)
    value[:, 1] = np.clip(value[:, 1], 0, shape[1] - 1)
    return value


def diagnose(cases: tuple[Intervention, ...], candidate_jsonl: Path, output_csv: Path) -> dict[str, object]:
    candidate_rows = {row["case_id"]: row for row in map(json.loads, candidate_jsonl.open())}
    loaded: dict[int, np.ndarray] = {}
    rows = []
    threshold = float(PROTOCOL["dense"]["tau_h"])
    for index, case in enumerate(cases):
        probability = loaded.setdefault(case.section_id, load_probability(DENSE_CACHE, case.section_id))
        intervened, _tube = masked_probability(probability, case)
        source = _pixels(case.source_context_yx, probability.shape)
        destination = _pixels(case.destination_context_yx, probability.shape)
        source_fraction = float(np.mean(probability[source[:, 0], source[:, 1]] >= threshold))
        destination_fraction = float(np.mean(probability[destination[:, 0], destination[:, 1]] >= threshold))
        components, _count = label(intervened >= threshold, structure=np.ones((3, 3), dtype=np.uint8))
        source_labels = set(components[source[:, 0], source[:, 1]].tolist()) - {0}
        destination_labels = set(components[destination[:, 0], destination[:, 1]].tolist()) - {0}
        proposal = candidate_rows[case.case_id]
        if proposal["candidate_recalled"]:
            taxonomy = "C_RECALLED"
        elif proposal["source_available"]:
            taxonomy = "B_SOURCE_NO_CORRECT_CANDIDATE"
        else:
            taxonomy = "A_NO_SOURCE_PORT"
        rows.append({
            "case_id": case.case_id,
            "section_id": case.section_id,
            "annotator": case.annotator,
            "trace_id": case.trace_id,
            "gap_length_px": case.gap_length_px,
            "taxonomy": taxonomy,
            "source_context_hard_fraction": source_fraction,
            "destination_context_hard_fraction": destination_fraction,
            "both_contexts_majority_supported": int(source_fraction >= 0.5 and destination_fraction >= 0.5),
            "contexts_still_hard_connected_after_intervention": int(bool(source_labels & destination_labels)),
        })
        if (index + 1) % 500 == 0 or index + 1 == len(cases):
            print(f"phase=I2_DIAGNOSTIC case={index + 1}/{len(cases)}", flush=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {name: sum(row["taxonomy"] == name for row in rows) for name in ("A_NO_SOURCE_PORT", "B_SOURCE_NO_CORRECT_CANDIDATE", "C_RECALLED")}
    return {
        "cases": len(rows),
        "taxonomy": counts,
        "both_contexts_majority_supported": int(sum(row["both_contexts_majority_supported"] for row in rows)),
        "contexts_still_hard_connected_after_intervention": int(sum(row["contexts_still_hard_connected_after_intervention"] for row in rows)),
        "still_connected_among_no_source": int(sum(row["contexts_still_hard_connected_after_intervention"] for row in rows if row["taxonomy"] == "A_NO_SOURCE_PORT")),
        "diagnostic_only": True,
        "gate_changed": False,
    }

