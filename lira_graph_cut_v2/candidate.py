"""Evaluate the unchanged real SBPP on treatment-valid graph cuts."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from datasets.cracks import load_section_image
from lira_final.branches.real_sbpp import propose_for_gap
from lira_final.data.natural_gaps import NaturalGap
from lira_final.dense.ensemble import load_probability
from lira_graph_cut_v2.benchmark import GraphCutCase
from lira_graph_cut_v2.graph_cut import rasterize
from lira_graph_cut_v2.protocol import DENSE_CACHE, PROTOCOL, ROOT


def cut_probability(probability: np.ndarray, case: GraphCutCase) -> tuple[np.ndarray, np.ndarray]:
    seed = rasterize(np.asarray(case.hidden_yx), probability.shape, 0)
    tube = distance_transform_edt(~seed) <= float(case.radius_px)
    output = np.asarray(probability, dtype=np.float32).copy()
    output[tube] = 0.0
    return output, tube


def _as_gap(case: GraphCutCase) -> NaturalGap:
    destination = np.asarray(case.destination_context_yx, dtype=np.float64)
    vector = destination[0] - destination[min(3, len(destination) - 1)]
    vector /= max(float(np.linalg.norm(vector)), 1e-8)
    return NaturalGap(
        gap_id=case.case_id,
        section_id=case.section_id,
        annotator=case.annotator,
        trace_id=case.trace_id,
        start_index=case.start_index,
        end_index=case.end_index,
        length_px=float(case.gap_length_px),
        source_yx=case.source_yx,
        destination_yx=tuple(map(float, destination[0])),
        source_tangent_yx=case.source_tangent_yx,
        destination_tangent_yx=tuple(map(float, vector)),
        gap_points_yx=case.hidden_yx,
        destination_context_yx=case.destination_context_yx,
    )


def propose(case: GraphCutCase, probability: np.ndarray, image: np.ndarray) -> dict[str, object]:
    intervened, tube = cut_probability(probability, case)
    before = np.asarray(image).copy()
    proposal = propose_for_gap(
        _as_gap(case),
        intervened,
        image,
        float(PROTOCOL["dense"]["hard_threshold"]),
        landing_band=float(PROTOCOL["candidate"]["landing_band_px"]),
        k=int(PROTOCOL["candidate"]["k"]),
    )
    if not np.array_equal(before, image):
        raise AssertionError("SBPP modified the real seismic image")
    return {
        "case_id": case.case_id,
        "split": case.split,
        "section_id": case.section_id,
        "annotator": case.annotator,
        "trace_id": case.trace_id,
        "gap_length_px": case.gap_length_px,
        "radius_px": case.radius_px,
        "tube_pixels": int(tube.sum()),
        "image_unchanged": True,
        **proposal,
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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def evaluate(cases: tuple[GraphCutCase, ...], output: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    loaded: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, case in enumerate(cases):
        if case.section_id not in loaded:
            loaded[case.section_id] = (
                load_probability(DENSE_CACHE, case.section_id),
                load_section_image(ROOT / "data/cracks/images" / f"section_{case.section_id:03d}.png"),
            )
        rows.append(propose(case, *loaded[case.section_id]))
        if (index + 1) % 100 == 0 or index + 1 == len(cases):
            print(f"phase=GRAPH_CUT_SBPP case={index + 1}/{len(cases)}", flush=True)
    flat_rows = []
    for row in rows:
        candidates = row.pop("candidates")
        flat_rows.append({**row, "candidates_json": json.dumps(candidates, separators=(",", ":"))})
        row["candidates"] = candidates
    _write_csv(output / "per_case.csv", flat_rows)
    by_radius = []
    for radius in PROTOCOL["treatment"]["candidate_radii_px"]:
        local = [row for row in rows if int(row["radius_px"]) == int(radius)]
        by_radius.append({"radius_px": radius, **_aggregate(local)})
    by_length = []
    for length in PROTOCOL["placement"]["gap_lengths_px"]:
        local = [row for row in rows if int(row["gap_length_px"]) == int(length)]
        by_length.append({"gap_length_px": length, **_aggregate(local)})
    _write_csv(output / "per_radius.csv", by_radius)
    _write_csv(output / "per_gap_length.csv", by_length)
    return _aggregate(rows), rows

