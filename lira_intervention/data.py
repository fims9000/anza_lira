"""Section-disjoint controlled interventions on reliable CRACKS tracelets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from lira_final.data.cracks_trace_audit import CrowdTrace, audit_trace_identity
from lira_final.data.splits import available_sections
from lira_intervention.protocol import (
    GAP_LENGTHS,
    HELDOUT_ANNOTATORS,
    PROTOCOL,
    ROOT,
    SPLIT_RANGES,
    TRAIN_ANNOTATORS,
    canonical_hash,
)


@dataclass(frozen=True)
class Intervention:
    case_id: str
    split: str
    section_id: int
    annotator: str
    trace_id: str
    gap_length_px: int
    source_yx: tuple[float, float]
    source_tangent_yx: tuple[float, float]
    source_context_yx: tuple[tuple[float, float], ...]
    hidden_yx: tuple[tuple[float, float], ...]
    destination_context_yx: tuple[tuple[float, float], ...]


def split_manifest() -> dict[str, object]:
    available = set(available_sections())
    splits = {
        name: [section for section in range(low, high + 1) if section in available]
        for name, (low, high) in SPLIT_RANGES.items()
    }
    for left, left_ids in splits.items():
        for right, right_ids in splits.items():
            if left < right and set(left_ids) & set(right_ids):
                raise AssertionError(f"section overlap: {left}/{right}")
    payload = {
        "splits": splits,
        "ranges": {key: list(value) for key, value in SPLIT_RANGES.items()},
        "confirm_contents_opened": False,
        "expert_accessed": False,
        "frozen_before_intervention_counts_or_scores": True,
    }
    payload["sha256"] = canonical_hash(payload)
    return payload


def _stable_int(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def _arc(points: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))


def _unit(vector: np.ndarray) -> tuple[float, float]:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return (0.0, 1.0)
    return tuple((vector / norm).tolist())


def _slice_by_arc(points: np.ndarray, cumulative: np.ndarray, low: float, high: float) -> np.ndarray:
    selected = points[(cumulative >= low) & (cumulative <= high)]
    if len(selected) >= 2:
        return selected
    indices = np.unique(np.clip(np.searchsorted(cumulative, [low, high]), 0, len(points) - 1))
    return points[indices]


def make_intervention(trace: CrowdTrace, split: str, gap_length: int) -> Intervention | None:
    points = np.asarray(trace.points_yx, dtype=np.float64)
    if _stable_int(trace.trace_id, split, "direction") % 2:
        points = points[::-1].copy()
    cumulative = _arc(points)
    total = float(cumulative[-1])
    context = float(PROTOCOL["intervention"]["minimum_visible_context_px_each_side"])
    if total < float(gap_length) + 2.0 * context:
        return None
    span = total - float(gap_length) - 2.0 * context
    fraction = ((_stable_int(trace.trace_id, split, gap_length, "position") % 10001) / 10000.0)
    start = context + span * (0.25 + 0.5 * fraction)
    end = start + float(gap_length)
    hidden = _slice_by_arc(points, cumulative, start, end)
    source = _slice_by_arc(points, cumulative, max(0.0, start - context), start)
    destination = _slice_by_arc(points, cumulative, end, min(total, end + context))
    if min(len(source), len(hidden), len(destination)) < 2:
        return None
    source_yx = source[-1]
    tangent = _unit(source[-1] - source[max(0, len(source) - 4)])
    case_id = hashlib.sha256(f"{split}|{trace.trace_id}|{gap_length}|{start:.6f}".encode()).hexdigest()[:20]
    serialize = lambda value: tuple(tuple(map(float, row)) for row in np.asarray(value))
    return Intervention(
        case_id=case_id,
        split=split,
        section_id=trace.section_id,
        annotator=trace.annotator,
        trace_id=trace.trace_id,
        gap_length_px=int(gap_length),
        source_yx=tuple(map(float, source_yx)),
        source_tangent_yx=tangent,
        source_context_yx=serialize(source),
        hidden_yx=serialize(hidden),
        destination_context_yx=serialize(destination),
    )


def build_interventions(split: str, traces: Iterable[CrowdTrace]) -> tuple[Intervention, ...]:
    counts = {length: 0 for length in GAP_LENGTHS}
    output: list[Intervention] = []
    ordered = sorted(traces, key=lambda trace: (_stable_int(trace.trace_id, split, "order"), trace.trace_id))
    for trace in ordered:
        total = float(_arc(np.asarray(trace.points_yx, dtype=np.float64))[-1])
        feasible = [length for length in GAP_LENGTHS if total >= length + 2 * PROTOCOL["intervention"]["minimum_visible_context_px_each_side"]]
        for gap_length in sorted(feasible, key=lambda length: (counts[length], _stable_int(trace.trace_id, length))):
            case = make_intervention(trace, split, gap_length)
            if case is not None:
                output.append(case)
                counts[gap_length] += 1
                break
    return tuple(sorted(output, key=lambda case: (case.section_id, case.annotator, case.trace_id)))


def save_jsonl(path: Path, rows: Iterable[Intervention]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def load_jsonl(path: Path) -> tuple[Intervention, ...]:
    rows = []
    with path.open() as handle:
        for line in handle:
            value = json.loads(line)
            rows.append(Intervention(
                **{key: value[key] for key in ("case_id", "split", "section_id", "annotator", "trace_id", "gap_length_px")},
                source_yx=tuple(value["source_yx"]),
                source_tangent_yx=tuple(value["source_tangent_yx"]),
                source_context_yx=tuple(map(tuple, value["source_context_yx"])),
                hidden_yx=tuple(map(tuple, value["hidden_yx"])),
                destination_context_yx=tuple(map(tuple, value["destination_context_yx"])),
            ))
    return tuple(rows)


def recover_split_traces(split: str, manifest: dict[str, object]) -> tuple[dict[str, object], tuple[CrowdTrace, ...]]:
    if split == "ig_confirm":
        raise PermissionError("confirm contents are hash-only before I3 PASS")
    annotators = TRAIN_ANNOTATORS if split == "ig_train" else HELDOUT_ANNOTATORS
    report, recovered = audit_trace_identity(
        ROOT / "data/cracks/annotations",
        list(manifest["splits"][split]),
        tuple(annotators),
    )
    traces = tuple(trace for local in recovered.values() for trace in local)
    return report, traces

