"""Frozen SBPP evaluation after a dense-evidence-only intervention."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from datasets.cracks import load_section_image
from lira_final.branches.real_sbpp import propose_for_gap
from lira_final.data.natural_gaps import NaturalGap
from lira_final.dense.ensemble import load_probability
from lira_intervention.data import Intervention
from lira_intervention.protocol import DENSE_CACHE, PROTOCOL, ROOT


def masked_probability(probability: np.ndarray, case: Intervention) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.rint(np.asarray(case.hidden_yx)).astype(int)
    seed = np.zeros(probability.shape, dtype=bool)
    hidden[:, 0] = np.clip(hidden[:, 0], 0, seed.shape[0] - 1)
    hidden[:, 1] = np.clip(hidden[:, 1], 0, seed.shape[1] - 1)
    seed[hidden[:, 0], hidden[:, 1]] = True
    radius = int(PROTOCOL["intervention"]["dense_evidence_tube_radius_px"])
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    tube = binary_dilation(seed, structure=(yy * yy + xx * xx <= radius * radius))
    output = np.asarray(probability, dtype=np.float32).copy()
    output[tube] = 0.0
    return output, tube


def as_gap(case: Intervention) -> NaturalGap:
    destination = np.asarray(case.destination_context_yx, dtype=np.float64)
    destination_tangent = destination[0] - destination[min(3, len(destination) - 1)]
    destination_tangent /= max(float(np.linalg.norm(destination_tangent)), 1e-8)
    return NaturalGap(
        gap_id=case.case_id,
        section_id=case.section_id,
        annotator=case.annotator,
        trace_id=case.trace_id,
        start_index=0,
        end_index=len(case.hidden_yx),
        length_px=case.gap_length_px,
        source_yx=case.source_yx,
        destination_yx=tuple(map(float, destination[0])),
        source_tangent_yx=case.source_tangent_yx,
        destination_tangent_yx=tuple(map(float, destination_tangent)),
        gap_points_yx=case.hidden_yx,
        destination_context_yx=case.destination_context_yx,
    )


def propose_case(case: Intervention, probability: np.ndarray, image: np.ndarray) -> dict[str, object]:
    intervened, tube = masked_probability(probability, case)
    image_before = np.asarray(image).copy()
    result = propose_for_gap(
        as_gap(case),
        intervened,
        image,
        float(PROTOCOL["dense"]["tau_h"]),
        landing_band=float(PROTOCOL["candidate"]["landing_band_px"]),
        k=int(PROTOCOL["candidate"]["k"]),
    )
    if not np.array_equal(image, image_before):
        raise AssertionError("intervention modified the seismic image")
    return {
        "case_id": case.case_id,
        "split": case.split,
        "section_id": case.section_id,
        "annotator": case.annotator,
        "trace_id": case.trace_id,
        "gap_length_px": case.gap_length_px,
        "tube_pixels": int(tube.sum()),
        "image_unchanged": True,
        **result,
    }


def evaluate_cases(cases: tuple[Intervention, ...], output_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    loaded: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, case in enumerate(cases):
        if case.section_id not in loaded:
            loaded[case.section_id] = (
                load_probability(DENSE_CACHE, case.section_id),
                load_section_image(ROOT / "data/cracks/images" / f"section_{case.section_id:03d}.png"),
            )
        probability, image = loaded[case.section_id]
        rows.append(propose_case(case, probability, image))
        if (index + 1) % 100 == 0 or index + 1 == len(cases):
            print(f"phase=I2_SBPP case={index + 1}/{len(cases)}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    counts = np.asarray([int(row["candidate_count"]) for row in rows], dtype=float)
    summary = {
        "cases": len(rows),
        "source_available": int(sum(bool(row["source_available"]) for row in rows)),
        "candidate_recalled": int(sum(bool(row["candidate_recalled"]) for row in rows)),
        "candidate_recall_at_12": float(np.mean([bool(row["candidate_recalled"]) for row in rows])) if rows else 0.0,
        "median_candidates": float(np.median(counts)) if len(counts) else 0.0,
        "p95_candidates": float(np.quantile(counts, 0.95)) if len(counts) else 0.0,
        "image_unchanged_all": bool(all(row["image_unchanged"] for row in rows)),
        "k": int(PROTOCOL["candidate"]["k"]),
    }
    return summary, rows
