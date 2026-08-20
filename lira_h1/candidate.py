"""Frozen SBPP evaluation on accepted H1 ribbon interventions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from datasets.cracks import load_section_image
from lira_final.branches.real_sbpp import propose_for_gap
from lira_final.data.natural_gaps import NaturalGap
from lira_final.dense.ensemble import load_probability
from lira_h1.benchmark import RibbonCase
from lira_h1.protocol import PROTOCOL, ROOT
from lira_h1.ribbon import flat_cap_ribbon


def intervene(probability: np.ndarray, case: RibbonCase) -> tuple[np.ndarray, np.ndarray]:
    ribbon = flat_cap_ribbon(np.asarray(case.trace_yx), case.s_a, case.s_b, case.radius_px, probability.shape)
    output = np.asarray(probability, dtype=np.float32).copy()
    output[ribbon] = 0.0
    return output, ribbon


def _as_gap(case: RibbonCase) -> NaturalGap:
    destination = np.asarray(case.destination_context_yx, dtype=np.float64)
    vector = destination[0] - destination[min(3, len(destination) - 1)]
    vector /= max(float(np.linalg.norm(vector)), 1e-8)
    return NaturalGap(
        gap_id=case.case_id, section_id=case.section_id, annotator=case.annotator, trace_id=case.trace_id,
        start_index=case.start_index, end_index=case.end_index, length_px=float(case.gap_length_px),
        source_yx=case.source_yx, destination_yx=tuple(map(float, destination[0])), source_tangent_yx=case.source_tangent_yx,
        destination_tangent_yx=tuple(map(float, vector)), gap_points_yx=case.hidden_yx,
        destination_context_yx=case.destination_context_yx,
    )


def propose(case: RibbonCase, probability: np.ndarray, image: np.ndarray) -> dict[str, object]:
    intervened, ribbon = intervene(probability, case)
    before = np.asarray(image).copy()
    result = propose_for_gap(
        _as_gap(case), intervened, image, float(PROTOCOL["dense"]["hard_threshold"]),
        landing_band=float(PROTOCOL["candidate"]["landing_band_px"]), k=int(PROTOCOL["candidate"]["k"]),
    )
    if not np.array_equal(before, image):
        raise AssertionError("frozen SBPP modified the real image")
    return {
        "case_id": case.case_id, "section_id": case.section_id, "annotator": case.annotator, "trace_id": case.trace_id,
        "gap_length_px": case.gap_length_px, "radius_px": case.radius_px, "ribbon_pixels": int(ribbon.sum()),
        "image_unchanged": True, **result,
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    counts = np.asarray([int(row["candidate_count"]) for row in rows], dtype=float)
    return {
        "cases": len(rows),
        "source_available": int(sum(bool(row["source_available"]) for row in rows)),
        "source_port_availability": float(np.mean([bool(row["source_available"]) for row in rows])) if rows else 0.0,
        "candidate_recalled": int(sum(bool(row["candidate_recalled"]) for row in rows)),
        "branch_candidate_recall_at_12": float(np.mean([bool(row["candidate_recalled"]) for row in rows])) if rows else 0.0,
        "median_candidates": float(np.median(counts)) if len(counts) else 0.0,
        "p95_candidates": float(np.quantile(counts, 0.95)) if len(counts) else 0.0,
    }


def evaluate(cases: tuple[RibbonCase, ...], output: Path, dense_cache: Path) -> dict[str, object]:
    rows = []
    loaded: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, case in enumerate(cases):
        if case.section_id not in loaded:
            loaded[case.section_id] = (
                load_probability(dense_cache, case.section_id),
                load_section_image(ROOT / "data/cracks/images" / f"section_{case.section_id:03d}.png"),
            )
        rows.append(propose(case, *loaded[case.section_id]))
        if (index + 1) % 100 == 0 or index + 1 == len(cases):
            print(f"phase=H1_SBPP case={index + 1}/{len(cases)}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    flat = []
    for row in rows:
        candidates = row.pop("candidates")
        flat.append({**row, "candidates_json": json.dumps(candidates, separators=(",", ":"))})
        row["candidates"] = candidates
    if flat:
        with (output / "per_case.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader(); writer.writerows(flat)
    return _aggregate(rows)

