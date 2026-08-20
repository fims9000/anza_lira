"""Build a fresh, treatment-valid CRACKS graph-cut benchmark."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass, fields as dataclass_fields
import hashlib
import json
from pathlib import Path

import numpy as np

from lira_final.data.cracks_trace_audit import CrowdTrace, audit_trace_identity
from lira_final.data.splits import available_sections
from lira_final.dense.ensemble import load_probability
from lira_graph_cut_v2.graph_cut import CutResult, minimal_valid_cut, rasterize
from lira_graph_cut_v2.protocol import (
    DENSE_CACHE,
    GAP_LENGTHS,
    HELDOUT_ANNOTATORS,
    PROTOCOL,
    ROOT,
    SPLIT_RANGES,
    TRAIN_ANNOTATORS,
    canonical_hash,
)


@dataclass(frozen=True)
class GraphCutCase:
    case_id: str
    split: str
    section_id: int
    annotator: str
    trace_id: str
    gap_length_px: int
    start_index: int
    end_index: int
    radius_px: int
    collateral_fraction: float
    source_yx: tuple[float, float]
    source_tangent_yx: tuple[float, float]
    hidden_yx: tuple[tuple[float, float], ...]
    left_anchor_yx: tuple[tuple[float, float], ...]
    right_anchor_yx: tuple[tuple[float, float], ...]
    destination_context_yx: tuple[tuple[float, float], ...]


def split_manifest() -> dict[str, object]:
    available = set(available_sections())
    splits = {
        name: [section for section in range(low, high + 1) if section in available]
        for name, (low, high) in SPLIT_RANGES.items()
    }
    values = list(splits.values())
    if any(set(values[i]) & set(values[j]) for i in range(len(values)) for j in range(i + 1, len(values))):
        raise AssertionError("graph-cut sections overlap")
    payload = {
        "splits": splits,
        "ranges": {key: list(value) for key, value in SPLIT_RANGES.items()},
        "frozen_before_validity_counts_or_candidate_scores": True,
        "confirm_contents_opened": False,
        "expert_accessed": False,
    }
    payload["sha256"] = canonical_hash(payload)
    return payload


def _stable_int(*parts: object) -> int:
    namespace = PROTOCOL["placement"]["seed_namespace"]
    return int(hashlib.sha256("|".join(map(str, (namespace, *parts))).encode()).hexdigest()[:16], 16)


def _arc(points: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))


def _unit(vector: np.ndarray) -> tuple[float, float]:
    vector = np.asarray(vector, dtype=np.float64)
    return tuple((vector / max(float(np.linalg.norm(vector)), 1e-8)).tolist())


def _interval(trace: CrowdTrace, split: str, gap_length: int) -> tuple[np.ndarray, int, int] | None:
    points = np.asarray(trace.points_yx, dtype=np.float64)
    if _stable_int(trace.trace_id, split, "direction") % 2:
        points = points[::-1].copy()
    cumulative = _arc(points)
    context = int(PROTOCOL["placement"]["minimum_context_px_each_side"])
    feasible_start = np.flatnonzero((cumulative >= context) & (cumulative <= cumulative[-1] - context - gap_length))
    if not len(feasible_start):
        return None
    fraction = (_stable_int(trace.trace_id, split, gap_length, "position") % 10001) / 10000.0
    start = int(feasible_start[min(len(feasible_start) - 1, int(fraction * len(feasible_start)))])
    end = int(np.searchsorted(cumulative, cumulative[start] + gap_length, side="left"))
    if end >= len(points) - context or start < context or end <= start:
        return None
    return points, start, end


def _serialize(points: np.ndarray) -> tuple[tuple[float, float], ...]:
    return tuple(tuple(map(float, point)) for point in np.asarray(points))


def _select_source(points: np.ndarray, start: int, cut_support: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]] | None:
    pixels = np.rint(points[:start]).astype(int)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, cut_support.shape[0] - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, cut_support.shape[1] - 1)
    supported = np.flatnonzero(cut_support[pixels[:, 0], pixels[:, 1]])
    if not len(supported):
        return None
    index = int(supported[-1])
    previous = int(supported[max(0, len(supported) - 4)])
    return tuple(map(float, points[index])), _unit(points[index] - points[previous])


def _placement_result(
    trace: CrowdTrace,
    split: str,
    gap_length: int,
    probability: np.ndarray,
    other_trace_mask: np.ndarray,
) -> tuple[dict[str, object], GraphCutCase | None]:
    interval = _interval(trace, split, gap_length)
    base = {"split": split, "section_id": trace.section_id, "annotator": trace.annotator, "trace_id": trace.trace_id, "gap_length_px": gap_length}
    if interval is None:
        return {**base, "status": "INELIGIBLE_INTERVAL", "eligible_before_treatment": 0}, None
    points, start, end = interval
    hidden = points[start : end + 1]
    margin = int(PROTOCOL["placement"]["image_border_margin_px"])
    if (
        hidden[:, 0].min() < margin
        or hidden[:, 1].min() < margin
        or hidden[:, 0].max() >= probability.shape[0] - margin
        or hidden[:, 1].max() >= probability.shape[1] - margin
    ):
        return {**base, "start_index": start, "end_index": end, "status": "INELIGIBLE_BORDER", "eligible_before_treatment": 0}, None
    left_immediate = points[max(0, start - 12) : start]
    right_immediate = points[end + 1 : min(len(points), end + 13)]
    left_context = points[:start]
    right_context = points[end + 1 :]
    threshold = float(PROTOCOL["treatment"]["validation_threshold"])
    def supported_count(local: np.ndarray) -> int:
        pixels = np.rint(local).astype(int)
        return int(np.sum(probability[pixels[:, 0], pixels[:, 1]] >= threshold))
    minimum = int(PROTOCOL["placement"]["minimum_supported_context_points_each_side"])
    if supported_count(left_immediate) < minimum or supported_count(right_immediate) < minimum:
        return {**base, "start_index": start, "end_index": end, "status": "INELIGIBLE_CONTEXT_SUPPORT", "eligible_before_treatment": 0}, None
    anchor_count = int(PROTOCOL["treatment"]["anchor_points_each_side"])
    left_anchor = points[start - anchor_count : start]
    right_anchor = points[end + 1 : end + 1 + anchor_count]
    cut, tube, cut_support = minimal_valid_cut(
        probability,
        hidden,
        left_anchor,
        right_anchor,
        left_context,
        right_context,
        other_trace_mask,
    )
    eligible = int(cut.status not in ("INELIGIBLE_ANCHOR_SUPPORT", "INELIGIBLE_PRE_DISCONNECTED"))
    row = {
        **base,
        "start_index": start,
        "end_index": end,
        "status": cut.status,
        "eligible_before_treatment": eligible,
        "radius_px": "" if cut.radius is None else cut.radius,
        "collateral_fraction": cut.collateral_fraction,
        "pre_connected": int(cut.pre_connected),
        "post_connected": "" if cut.post_connected is None else int(cut.post_connected),
        "left_context_supported": cut.left_context_supported,
        "right_context_supported": cut.right_context_supported,
        "tube_pixels": cut.tube_pixels,
    }
    if cut.status != "VALID" or tube is None or cut_support is None or cut.radius is None:
        return row, None
    source = _select_source(points, start, cut_support)
    if source is None:
        row["status"] = "INVALID_CONTEXT_DESTROYED"
        return row, None
    destination_pixels = np.rint(right_context).astype(int)
    keep = cut_support[destination_pixels[:, 0], destination_pixels[:, 1]]
    destination = right_context[keep][:24]
    if len(destination) < minimum:
        row["status"] = "INVALID_CONTEXT_DESTROYED"
        return row, None
    case_id = hashlib.sha256(f"{split}|{trace.trace_id}|{start}|{end}|{cut.radius}".encode()).hexdigest()[:20]
    case = GraphCutCase(
        case_id=case_id,
        split=split,
        section_id=trace.section_id,
        annotator=trace.annotator,
        trace_id=trace.trace_id,
        gap_length_px=gap_length,
        start_index=start,
        end_index=end,
        radius_px=cut.radius,
        collateral_fraction=cut.collateral_fraction,
        source_yx=source[0],
        source_tangent_yx=source[1],
        hidden_yx=_serialize(hidden),
        left_anchor_yx=_serialize(left_anchor),
        right_anchor_yx=_serialize(right_anchor),
        destination_context_yx=_serialize(destination),
    )
    row["case_id"] = case_id
    return row, case


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) + sorted(set().union(*(row.keys() for row in rows)) - set(rows[0])) if rows else []
    with path.open("w", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def save_cases(path: Path, cases: list[GraphCutCase]) -> None:
    rows = []
    for case in cases:
        row = asdict(case)
        for key in ("source_yx", "source_tangent_yx", "hidden_yx", "left_anchor_yx", "right_anchor_yx", "destination_context_yx"):
            row[key] = json.dumps(row[key], separators=(",", ":"))
        rows.append(row)
    if rows:
        _write_csv(path, rows)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[field.name for field in dataclass_fields(GraphCutCase)])
            writer.writeheader()


def load_cases(path: Path) -> tuple[GraphCutCase, ...]:
    rows = []
    with path.open() as handle:
        for value in csv.DictReader(handle):
            rows.append(GraphCutCase(
                case_id=value["case_id"], split=value["split"], section_id=int(value["section_id"]), annotator=value["annotator"], trace_id=value["trace_id"],
                gap_length_px=int(value["gap_length_px"]), start_index=int(value["start_index"]), end_index=int(value["end_index"]), radius_px=int(value["radius_px"]), collateral_fraction=float(value["collateral_fraction"]),
                source_yx=tuple(json.loads(value["source_yx"])), source_tangent_yx=tuple(json.loads(value["source_tangent_yx"])),
                hidden_yx=tuple(map(tuple, json.loads(value["hidden_yx"]))), left_anchor_yx=tuple(map(tuple, json.loads(value["left_anchor_yx"]))), right_anchor_yx=tuple(map(tuple, json.loads(value["right_anchor_yx"]))), destination_context_yx=tuple(map(tuple, json.loads(value["destination_context_yx"]))),
            ))
    return tuple(rows)


def recover_split(split: str, manifest: dict[str, object]) -> dict[tuple[int, str], tuple[CrowdTrace, ...]]:
    if split == "gc_confirm":
        raise PermissionError("gc_confirm contents are hash-only")
    annotators = TRAIN_ANNOTATORS if split == "gc_train" else HELDOUT_ANNOTATORS
    _report, recovered = audit_trace_identity(ROOT / "data/cracks/annotations", list(manifest["splits"][split]), tuple(annotators))
    return recovered


def build_split(split: str, recovered: dict[tuple[int, str], tuple[CrowdTrace, ...]], output: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    cases: list[GraphCutCase] = []
    length_counts = Counter({length: 0 for length in GAP_LENGTHS})
    groups = sorted(recovered.items())
    processed = 0
    for (section_id, annotator), local in groups:
        probability = load_probability(DENSE_CACHE, section_id)
        dilated = [rasterize(trace.points_yx, probability.shape, int(PROTOCOL["treatment"]["collateral_trace_radius_px"])) for trace in local]
        total = np.sum(np.stack(dilated), axis=0, dtype=np.uint16) if dilated else np.zeros(probability.shape, dtype=np.uint16)
        for trace, own in zip(local, dilated):
            arc_length = float(_arc(np.asarray(trace.points_yx, dtype=np.float64))[-1])
            feasible = [length for length in GAP_LENGTHS if arc_length >= length + 2 * int(PROTOCOL["placement"]["minimum_context_px_each_side"])]
            if not feasible:
                rows.append({"split": split, "section_id": section_id, "annotator": annotator, "trace_id": trace.trace_id, "gap_length_px": "", "status": "INELIGIBLE_INTERVAL", "eligible_before_treatment": 0})
                continue
            gap_length = min(feasible, key=lambda length: (length_counts[length], _stable_int(trace.trace_id, length)))
            length_counts[gap_length] += 1
            row, case = _placement_result(trace, split, gap_length, probability, (total - own.astype(np.uint16)) > 0)
            rows.append(row)
            if case is not None:
                cases.append(case)
        processed += len(local)
        if processed and (processed % 500 < len(local) or (section_id, annotator) == groups[-1][0]):
            print(f"phase=GRAPH_CUT split={split} traces={processed}", flush=True)
    _write_csv(output / f"{split}_eligibility.csv", rows)
    save_cases(output / f"{split}_intervention_cases.csv", cases)
    eligible = sum(int(row.get("eligible_before_treatment", 0)) for row in rows)
    retention = len(cases) / eligible if eligible else 0.0
    status_counts = Counter(str(row["status"]) for row in rows)
    radii = Counter(case.radius_px for case in cases)
    minimal_radii = Counter(int(row["radius_px"]) for row in rows if row.get("radius_px") not in (None, ""))
    summary = {
        "split": split,
        "reliable_traces": len(rows),
        "eligible_before_treatment": eligible,
        "valid_cases": len(cases),
        "retention": retention,
        "treatment_validity": 1.0 if cases else None,
        "status_counts": dict(sorted(status_counts.items())),
        "radius_counts": {str(radius): radii[radius] for radius in PROTOCOL["treatment"]["candidate_radii_px"]},
        "minimal_disconnect_radius_counts_before_exclusions": {str(radius): minimal_radii[radius] for radius in PROTOCOL["treatment"]["candidate_radii_px"]},
        "gap_length_counts": {str(length): sum(case.gap_length_px == length for case in cases) for length in GAP_LENGTHS},
        "image_changed": False,
        "candidate_or_p0_used_for_filtering": False,
    }
    return summary
