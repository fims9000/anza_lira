#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from build_junction_plan import candidate_span, junction_geometry


REPO_ROOT = Path(__file__).resolve().parents[3]
PAIR_PLAN = REPO_ROOT / "artifacts/varvara/ct28_pair/relation_pair_plan.csv"
JUNCTION_PLAN = (
    REPO_ROOT
    / "artifacts/varvara/ct28_junction/junction_relation_plan_train_val.csv"
)
JUNCTION_ENDPOINTS = (
    REPO_ROOT
    / "artifacts/varvara/ct28_junction/junction_candidate_endpoints_train_val.csv"
)
DEFAULT_OUT = REPO_ROOT / "artifacts/varvara/ct28_graph_lira"

PROTOCOL_VERSION = "ct28_scene_protocol_v1"
DEVELOPMENT_SPLITS = frozenset({"train", "val"})
EXPECTED_PATIENT_IDS = {
    "train": {953, 955, 956, 959, 960, 963, 964, 967, 969, 970, 971, 975, 976, 977, 979, 982, 983},
    "val": {957, 961, 965, 966, 974},
}
PAIR_MAX_DISTANCE_MM = 12.0
JUNCTION_MAX_SPAN_MM = 16.0
JUNCTION_DEGREE = 3
JUNCTION_DECOY_LIMIT = 2
SCENE_ENDPOINT_COUNT = 5

# Possible future stress modes for the same frozen scene protocol:
# clean geometry; 15 deg + 0.5 mm; 30 deg + 1.0 mm;
# 45 deg + 1.0 mm; 60 deg + 1.5 mm.
# This builder intentionally generates clean geometry only. Stress is applied
# later, outside candidate construction, by the feature/evaluation pipeline.

PAIR_GEOMETRY_COLUMNS = [f"pair_g{i}" for i in range(7)]
JUNCTION_GEOMETRY_COLUMNS = [
    "junction_span_mm",
    "junction_residual_mean_mm",
    "junction_residual_max_mm",
    "junction_forward_min_mm",
    "junction_forward_mean_mm",
    "junction_alignment_min",
    "junction_alignment_mean",
    "junction_center_distance_max_mm",
    "junction_center_distance_std_mm",
]


class SceneBuildError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic CT28 train/validation Graph-LIRA scenes."
    )
    parser.add_argument("--pair-plan", type=Path, default=PAIR_PLAN)
    parser.add_argument("--junction-plan", type=Path, default=JUNCTION_PLAN)
    parser.add_argument("--junction-endpoints", type=Path, default=JUNCTION_ENDPOINTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise SceneBuildError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray([float(row[f"{prefix}_{axis}"]) for axis in range(3)])


def endpoint(
    token: str,
    component: str,
    source_id: str,
    role: str,
    position: Iterable[float],
    tangent: Iterable[float],
) -> dict[str, object]:
    position_array = np.asarray(tuple(position), dtype=float)
    tangent_array = np.asarray(tuple(tangent), dtype=float)
    norm = float(np.linalg.norm(tangent_array))
    if norm <= 1e-12:
        raise SceneBuildError(f"Zero tangent for {token}")
    return {
        "token": token,
        "component": component,
        "source_id": source_id,
        "role": role,
        "position": position_array,
        "tangent": tangent_array / norm,
    }


def group_unique(rows: Sequence[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return dict(grouped)


def pair_components(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, object]]:
    components = {}
    for pair_id, group in sorted(group_unique(rows, "pair_id").items()):
        if len(group) != 2 or {int(row["y"]) for row in group} != {0, 1}:
            raise SceneBuildError(f"PAIR source is not 1:1 positive/negative: {pair_id}")
        positive = next(row for row in group if int(row["y"]) == 1)
        negative = next(row for row in group if int(row["y"]) == 0)
        if positive["split"] not in DEVELOPMENT_SPLITS:
            continue
        if negative["split"] != positive["split"] or negative["scan_id"] != positive["scan_id"]:
            raise SceneBuildError(f"PAIR source split/patient mismatch: {pair_id}")
        endpoints = []
        for index in range(2):
            endpoints.append(
                endpoint(
                    f"{pair_id}:true:{index}",
                    "pair",
                    pair_id,
                    "pair_true",
                    point(positive, f"p{index + 1}"),
                    point(positive, f"a{index + 1}"),
                )
            )
        for index in range(2):
            endpoints.append(
                endpoint(
                    f"{pair_id}:hard_negative:{index}",
                    "pair",
                    pair_id,
                    "pair_hard_negative",
                    point(negative, f"p{index + 1}"),
                    point(negative, f"a{index + 1}"),
                )
            )
        components[pair_id] = {
            "source_id": pair_id,
            "scan_id": int(positive["scan_id"]),
            "split": positive["split"],
            "endpoints": endpoints,
            "true_tokens": {endpoints[0]["token"], endpoints[1]["token"]},
            "center": np.mean([item["position"] for item in endpoints[:2]], axis=0),
        }
    return components


def endpoint_rows_by_candidate(
    rows: Sequence[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    grouped = group_unique(rows, "candidate_id")
    for candidate_id, group in grouped.items():
        grouped[candidate_id] = sorted(group, key=lambda row: int(row["endpoint_index"]))
    return grouped


def junction_endpoint(
    row: dict[str, str],
    *,
    token: str,
    source_id: str,
    role: str,
) -> dict[str, object]:
    return endpoint(
        token,
        "junction",
        source_id,
        role,
        [float(row[f"{axis}_lps"]) for axis in "xyz"],
        [float(row[f"tangent_to_candidate_{axis}"]) for axis in "xyz"],
    )


def decoy_physical_key(row: dict[str, str]) -> tuple[int, float]:
    return int(row["source_line_id"]), round(float(row["decoy_arc_mm"]), 9)


def junction_components(
    plan_rows: Sequence[dict[str, str]],
    endpoint_rows: Sequence[dict[str, str]],
) -> dict[str, dict[str, object]]:
    endpoints_by_candidate = endpoint_rows_by_candidate(endpoint_rows)
    components = {}
    for source_id, group in sorted(group_unique(plan_rows, "source_junction_id").items()):
        if group[0]["split"] not in DEVELOPMENT_SPLITS:
            raise SceneBuildError(f"Unexpected JUNCTION split: {group[0]['split']}")
        positives = [row for row in group if int(row["y"]) == 1]
        if len(positives) != 1 or int(positives[0]["degree"]) != JUNCTION_DEGREE:
            raise SceneBuildError(f"Expected one degree-3 positive: {source_id}")
        positive = positives[0]
        positive_id = positive["candidate_id"]
        raw_true = endpoints_by_candidate[positive_id]
        if len(raw_true) != JUNCTION_DEGREE:
            raise SceneBuildError(f"Wrong true endpoint count: {positive_id}")
        true_endpoints = [
            junction_endpoint(
                row,
                token=f"{source_id}:true:{index}",
                source_id=source_id,
                role="junction_true",
            )
            for index, row in enumerate(raw_true)
        ]
        negatives = sorted(
            (row for row in group if int(row["y"]) == 0),
            key=lambda row: (float(row["geometry_match_distance"]), row["candidate_id"]),
        )
        decoys = []
        seen_physical = set()
        for negative in negatives:
            for raw in endpoints_by_candidate[negative["candidate_id"]]:
                if raw["source_role"] != "decoy":
                    continue
                physical_key = decoy_physical_key(raw)
                if physical_key in seen_physical:
                    continue
                seen_physical.add(physical_key)
                decoy = junction_endpoint(
                    raw,
                    token=f"{source_id}:decoy:{len(decoys)}",
                    source_id=source_id,
                    role="junction_decoy",
                )
                decoy["physical_key"] = physical_key
                decoys.append(decoy)
        one_arm = [row for row in negatives if int(row["n_replaced_arms"]) == 1]
        incomplete = None
        if one_arm:
            selected = one_arm[0]
            replaced = [int(value) for value in selected["replaced_arm_indices"].split("|")]
            if len(replaced) != 1:
                raise SceneBuildError(f"Invalid one-arm replacement: {selected['candidate_id']}")
            selected_decoys = [
                raw
                for raw in endpoints_by_candidate[selected["candidate_id"]]
                if raw["source_role"] == "decoy"
            ]
            if len(selected_decoys) != 1:
                raise SceneBuildError(f"Invalid one-arm decoy count: {selected['candidate_id']}")
            required_key = decoy_physical_key(selected_decoys[0])
            required = next(item for item in decoys if item["physical_key"] == required_key)
            remaining = [item for item in decoys if item["token"] != required["token"]]
            incomplete = {
                "withheld_arm_index": replaced[0],
                "required_decoy": required,
                "extra_decoys": remaining,
                "source_negative_id": selected["candidate_id"],
            }
        components[source_id] = {
            "source_id": source_id,
            "scan_id": int(positive["scan_id"]),
            "split": positive["split"],
            "side": positive["side"],
            "benchmark_group": positive["benchmark_group"],
            "true_endpoints": true_endpoints,
            "decoys": decoys,
            "true_tokens": {item["token"] for item in true_endpoints},
            "center": np.mean([item["position"] for item in true_endpoints], axis=0),
            "incomplete": incomplete,
        }
    return components


def pair_geometry(first: dict[str, object], second: dict[str, object]) -> list[float]:
    vector = second["position"] - first["position"]
    distance = float(np.linalg.norm(vector))
    direction = vector / max(distance, 1e-12)
    first_tangent = first["tangent"]
    second_tangent = second["tangent"]
    return [
        distance,
        abs(float(first_tangent @ direction)),
        abs(float(second_tangent @ direction)),
        abs(float(first_tangent @ second_tangent)),
        abs(float(direction[2])),
        abs(float(direction[0])),
        abs(float(direction[1])),
    ]


def oriented_junction_geometry(items: Sequence[dict[str, object]]) -> list[float]:
    positions = np.stack([item["position"] for item in items])
    axes = np.stack([item["tangent"] for item in items])
    projection = np.eye(3)[None, :, :] - axes[:, :, None] * axes[:, None, :]
    center = np.linalg.pinv(projection.sum(axis=0)) @ np.einsum(
        "nij,nj->i", projection, positions
    )
    toward = center[None, :] - positions
    signs = np.where(np.sum(toward * axes, axis=1) >= 0.0, 1.0, -1.0)
    tangents = axes * signs[:, None]
    values = junction_geometry(positions.tolist(), tangents.tolist())
    return [
        values["jg_span_mm"],
        values["jg_residual_mean_mm"],
        values["jg_residual_max_mm"],
        values["jg_forward_min_mm"],
        values["jg_forward_mean_mm"],
        values["jg_alignment_min"],
        values["jg_alignment_mean"],
        values["jg_center_distance_max_mm"],
        values["jg_center_distance_std_mm"],
    ]


def mixed_pairing(
    pairs: dict[str, dict[str, object]], junctions: dict[str, dict[str, object]]
) -> dict[str, str]:
    pairs_by_patient: dict[int, list[dict[str, object]]] = defaultdict(list)
    for component in pairs.values():
        pairs_by_patient[int(component["scan_id"])].append(component)
    assignment = {}
    used = defaultdict(set)
    for source_id, junction in sorted(junctions.items()):
        scan_id = int(junction["scan_id"])
        options = []
        for pair in pairs_by_patient[scan_id]:
            if pair["source_id"] in used[scan_id]:
                continue
            center_distance = float(np.linalg.norm(pair["center"] - junction["center"]))
            options.append((center_distance, pair["source_id"], pair))
        if not options:
            raise SceneBuildError(f"No unused PAIR source for mixed scene {source_id}")
        _, pair_id, _ = min(options)
        assignment[source_id] = pair_id
        used[scan_id].add(pair_id)
    return assignment


def clone_endpoints(items: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            **item,
            "position": np.array(item["position"], dtype=float, copy=True),
            "tangent": np.array(item["tangent"], dtype=float, copy=True),
        }
        for item in items
    ]


def filler_pools(
    pairs: dict[str, dict[str, object]],
) -> dict[int, list[dict[str, object]]]:
    pools: dict[int, list[dict[str, object]]] = defaultdict(list)
    for pair in pairs.values():
        pools[int(pair["scan_id"])].extend(pair["endpoints"][2:])
    return pools


def fill_endpoints(
    base: Sequence[dict[str, object]],
    pool: Sequence[dict[str, object]],
    *,
    target_count: int = SCENE_ENDPOINT_COUNT,
) -> list[dict[str, object]]:
    selected = []
    for item in base:
        if any(
            np.array_equal(item["position"], current["position"])
            for current in selected
        ):
            continue
        selected.append(item)
    used_tokens = {str(item["token"]) for item in selected}
    while len(selected) < target_count:
        nearby_options = []
        fallback_options = []
        for item in pool:
            if str(item["token"]) in used_tokens:
                continue
            if any(
                np.array_equal(item["position"], current["position"])
                for current in selected
            ):
                continue
            distances = [
                float(np.linalg.norm(item["position"] - current["position"]))
                for current in selected
            ]
            minimum = min(distances)
            option = (minimum, str(item["token"]), item)
            fallback_options.append(option)
            if minimum <= JUNCTION_MAX_SPAN_MM:
                nearby_options.append(option)
        options = nearby_options or fallback_options
        if not options:
            raise SceneBuildError(
                f"Cannot fill endpoint pool from {len(selected)} to {target_count}"
            )
        _, token, chosen = min(options)
        selected.append(chosen)
        used_tokens.add(token)
    if len(selected) != target_count:
        raise SceneBuildError(f"Endpoint target exceeded: {len(selected)}")
    return selected


def make_scene(
    *,
    scene_id: str,
    family: str,
    relation_class: str,
    scan_id: int,
    split: str,
    endpoints: Sequence[dict[str, object]],
    true_pair_tokens: set[str] | None = None,
    true_junction_tokens: set[str] | None = None,
    pair_source_id: str = "",
    junction_source_id: str = "",
    withheld_tokens: Sequence[str] = (),
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    if split not in DEVELOPMENT_SPLITS:
        raise SceneBuildError(f"Forbidden split in scene {scene_id}: {split}")
    local_endpoints = clone_endpoints(endpoints)
    token_to_id = {}
    endpoint_rows = []
    for index, item in enumerate(local_endpoints):
        endpoint_id = f"E{index:02d}"
        if item["token"] in token_to_id:
            raise SceneBuildError(f"Duplicate endpoint token in {scene_id}: {item['token']}")
        token_to_id[item["token"]] = endpoint_id
        item["endpoint_id"] = endpoint_id
        endpoint_rows.append(
            {
                "scene_id": scene_id,
                "endpoint_id": endpoint_id,
                "component": item["component"],
                "source_id": item["source_id"],
                "role": item["role"],
                "x_lps": item["position"][0],
                "y_lps": item["position"][1],
                "z_lps": item["position"][2],
                "tangent_x": item["tangent"][0],
                "tangent_y": item["tangent"][1],
                "tangent_z": item["tangent"][2],
            }
        )

    true_pair = {token_to_id[token] for token in (true_pair_tokens or set())}
    true_junction = {
        token_to_id[token] for token in (true_junction_tokens or set())
    }
    candidate_rows = []
    pair_recalled = not true_pair
    junction_recalled = not true_junction
    for first_index, second_index in itertools.combinations(range(len(local_endpoints)), 2):
        first, second = local_endpoints[first_index], local_endpoints[second_index]
        geometry = pair_geometry(first, second)
        if geometry[0] > PAIR_MAX_DISTANCE_MM + 1e-12:
            continue
        endpoint_ids = sorted([first["endpoint_id"], second["endpoint_id"]])
        label = int(set(endpoint_ids) == true_pair)
        pair_recalled |= bool(label)
        row = {
            "scene_id": scene_id,
            "candidate_id": "PAIR:" + "+".join(endpoint_ids),
            "candidate_type": "PAIR",
            "endpoint_ids": "|".join(endpoint_ids),
            "degree": 2,
            "y": label,
        }
        row.update(dict(zip(PAIR_GEOMETRY_COLUMNS, geometry)))
        row.update({name: "" for name in JUNCTION_GEOMETRY_COLUMNS})
        candidate_rows.append(row)

    for indices in itertools.combinations(range(len(local_endpoints)), JUNCTION_DEGREE):
        items = [local_endpoints[index] for index in indices]
        if candidate_span([item["position"] for item in items]) > JUNCTION_MAX_SPAN_MM + 1e-12:
            continue
        endpoint_ids = sorted(item["endpoint_id"] for item in items)
        label = int(set(endpoint_ids) == true_junction)
        junction_recalled |= bool(label)
        geometry = oriented_junction_geometry(items)
        row = {
            "scene_id": scene_id,
            "candidate_id": "JUNCTION:" + "+".join(endpoint_ids),
            "candidate_type": "JUNCTION",
            "endpoint_ids": "|".join(endpoint_ids),
            "degree": JUNCTION_DEGREE,
            "y": label,
        }
        row.update({name: "" for name in PAIR_GEOMETRY_COLUMNS})
        row.update(dict(zip(JUNCTION_GEOMETRY_COLUMNS, geometry)))
        candidate_rows.append(row)

    if not candidate_rows:
        raise SceneBuildError(f"Scene has no candidates: {scene_id}")
    pair_count = sum(row["candidate_type"] == "PAIR" for row in candidate_rows)
    junction_count = sum(row["candidate_type"] == "JUNCTION" for row in candidate_rows)
    plan_row = {
        "scene_id": scene_id,
        "scan_id": scan_id,
        "split": split,
        "scene_family": family,
        "relation_class": relation_class,
        "pair_source_id": pair_source_id,
        "junction_source_id": junction_source_id,
        "true_pair_endpoint_ids": "|".join(sorted(true_pair)),
        "true_junction_endpoint_ids": "|".join(sorted(true_junction)),
        "withheld_source_tokens": "|".join(sorted(withheld_tokens)),
        "endpoint_count": len(local_endpoints),
        "pair_candidate_count": pair_count,
        "junction_candidate_count": junction_count,
        "pair_candidate_recall": int(pair_recalled),
        "junction_candidate_recall": int(junction_recalled),
        "candidate_recall": int(pair_recalled and junction_recalled),
    }
    return plan_row, endpoint_rows, candidate_rows


def build_scenes(
    pairs: dict[str, dict[str, object]],
    junctions: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    plans = []
    endpoints = []
    candidates = []
    fillers = filler_pools(pairs)

    def append(result):
        plan, endpoint_rows, candidate_rows = result
        plans.append(plan)
        endpoints.extend(endpoint_rows)
        candidates.extend(candidate_rows)

    for pair_id, pair in sorted(pairs.items()):
        base = fill_endpoints(pair["endpoints"], fillers[pair["scan_id"]])
        append(
            make_scene(
                scene_id=f"CT28S:{pair['split']}:pair_only:{pair_id}",
                family="pair_only",
                relation_class="PAIR",
                scan_id=pair["scan_id"],
                split=pair["split"],
                endpoints=base,
                true_pair_tokens=pair["true_tokens"],
                pair_source_id=pair_id,
            )
        )
        withheld = base[1]["token"]
        orphan_endpoints = fill_endpoints(
            [pair["endpoints"][0], pair["endpoints"][2], pair["endpoints"][3]],
            fillers[pair["scan_id"]],
        )
        append(
            make_scene(
                scene_id=f"CT28S:{pair['split']}:none_orphan_pair:{pair_id}",
                family="none_orphan_pair",
                relation_class="NONE",
                scan_id=pair["scan_id"],
                split=pair["split"],
                endpoints=orphan_endpoints,
                pair_source_id=pair_id,
                withheld_tokens=[withheld],
            )
        )

    mixed = mixed_pairing(pairs, junctions)
    for junction_id, junction in sorted(junctions.items()):
        junction_base = fill_endpoints(
            junction["true_endpoints"] + junction["decoys"][:JUNCTION_DECOY_LIMIT],
            fillers[junction["scan_id"]],
        )
        append(
            make_scene(
                scene_id=f"CT28S:{junction['split']}:junction_only:{junction_id}",
                family="junction_only",
                relation_class="JUNCTION",
                scan_id=junction["scan_id"],
                split=junction["split"],
                endpoints=junction_base,
                true_junction_tokens=junction["true_tokens"],
                junction_source_id=junction_id,
            )
        )

        pair = pairs[mixed[junction_id]]
        append(
            make_scene(
                scene_id=f"CT28S:{junction['split']}:mixed:{junction_id}",
                family="mixed",
                relation_class="BOTH",
                scan_id=junction["scan_id"],
                split=junction["split"],
                endpoints=pair["endpoints"][:2] + junction["true_endpoints"],
                true_pair_tokens=pair["true_tokens"],
                true_junction_tokens=junction["true_tokens"],
                pair_source_id=pair["source_id"],
                junction_source_id=junction_id,
            )
        )

        incomplete = junction["incomplete"]
        if incomplete is None:
            continue
        withheld_index = int(incomplete["withheld_arm_index"])
        retained_true = [
            item
            for index, item in enumerate(junction["true_endpoints"])
            if index != withheld_index
        ]
        incomplete_base = (
            retained_true
            + [incomplete["required_decoy"]]
            + incomplete["extra_decoys"][:2]
        )
        incomplete_endpoints = fill_endpoints(
            incomplete_base,
            fillers[junction["scan_id"]],
        )
        append(
            make_scene(
                scene_id=f"CT28S:{junction['split']}:none_incomplete_junction:{junction_id}",
                family="none_incomplete_junction",
                relation_class="NONE",
                scan_id=junction["scan_id"],
                split=junction["split"],
                endpoints=incomplete_endpoints,
                junction_source_id=junction_id,
                withheld_tokens=[junction["true_endpoints"][withheld_index]["token"]],
            )
        )
    return plans, endpoints, candidates


def validate_outputs(
    plans: Sequence[dict[str, object]],
    endpoints: Sequence[dict[str, object]],
    candidates: Sequence[dict[str, object]],
) -> None:
    scene_ids = [row["scene_id"] for row in plans]
    if len(scene_ids) != len(set(scene_ids)):
        raise SceneBuildError("Duplicate scene_id")
    if {row["split"] for row in plans} != DEVELOPMENT_SPLITS:
        raise SceneBuildError("Scene output is not exactly train/val")
    for split, expected_ids in EXPECTED_PATIENT_IDS.items():
        actual_ids = {int(row["scan_id"]) for row in plans if row["split"] == split}
        if actual_ids != expected_ids:
            raise SceneBuildError(
                f"Patient split mismatch for {split}: {sorted(actual_ids)}"
            )
    if {row["relation_class"] for row in plans} != {"NONE", "PAIR", "JUNCTION", "BOTH"}:
        raise SceneBuildError("Missing relation class")
    expected_families = {
        "pair_only",
        "junction_only",
        "mixed",
        "none_orphan_pair",
        "none_incomplete_junction",
    }
    if {row["scene_family"] for row in plans} != expected_families:
        raise SceneBuildError("Missing scene family")
    if not all(int(row["candidate_recall"]) == 1 for row in plans):
        raise SceneBuildError("Candidate recall failure")
    endpoint_counts = Counter(row["scene_id"] for row in endpoints)
    if set(endpoint_counts) != set(scene_ids) or set(endpoint_counts.values()) != {
        SCENE_ENDPOINT_COUNT
    }:
        raise SceneBuildError("Every scene must contain the frozen endpoint count")
    endpoint_keys = {(row["scene_id"], row["endpoint_id"]) for row in endpoints}
    if len(endpoint_keys) != len(endpoints):
        raise SceneBuildError("Duplicate endpoint key")
    positions_by_scene: dict[str, set[tuple[float, float, float]]] = defaultdict(set)
    for row in endpoints:
        position = tuple(float(row[f"{axis}_lps"]) for axis in "xyz")
        if position in positions_by_scene[str(row["scene_id"])]:
            raise SceneBuildError(
                f"Exact coordinate duplicate: {row['scene_id']} {row['endpoint_id']}"
            )
        positions_by_scene[str(row["scene_id"])].add(position)
    candidate_keys = {(row["scene_id"], row["candidate_id"]) for row in candidates}
    if len(candidate_keys) != len(candidates):
        raise SceneBuildError("Duplicate candidate key")
    for row in candidates:
        ids = row["endpoint_ids"].split("|")
        if any((row["scene_id"], endpoint_id) not in endpoint_keys for endpoint_id in ids):
            raise SceneBuildError(f"Candidate references missing endpoint: {row}")
        if row["candidate_type"] == "PAIR" and float(row["pair_g0"]) > PAIR_MAX_DISTANCE_MM + 1e-12:
            raise SceneBuildError("PAIR envelope violation")
        if row["candidate_type"] == "JUNCTION" and float(row["junction_span_mm"]) > JUNCTION_MAX_SPAN_MM + 1e-12:
            raise SceneBuildError("JUNCTION envelope violation")
        geometry_columns = (
            PAIR_GEOMETRY_COLUMNS
            if row["candidate_type"] == "PAIR"
            else JUNCTION_GEOMETRY_COLUMNS
        )
        if not all(math.isfinite(float(row[column])) for column in geometry_columns):
            raise SceneBuildError(f"Non-finite candidate geometry: {row['candidate_id']}")
    positive_by_scene = defaultdict(Counter)
    candidate_counts_by_scene = defaultdict(Counter)
    for row in candidates:
        candidate_counts_by_scene[row["scene_id"]][row["candidate_type"]] += 1
        if int(row["y"]):
            positive_by_scene[row["scene_id"]][row["candidate_type"]] += 1
    for row in plans:
        counts = positive_by_scene[row["scene_id"]]
        expected = {
            "NONE": (0, 0),
            "PAIR": (1, 0),
            "JUNCTION": (0, 1),
            "BOTH": (1, 1),
        }[row["relation_class"]]
        actual = (counts["PAIR"], counts["JUNCTION"])
        if actual != expected:
            raise SceneBuildError(
                f"Ground-truth candidate mismatch {row['scene_id']}: {actual} != {expected}"
            )
        if int(row["endpoint_count"]) != endpoint_counts[row["scene_id"]]:
            raise SceneBuildError(f"Endpoint count mismatch: {row['scene_id']}")
        pair_count = candidate_counts_by_scene[row["scene_id"]]["PAIR"]
        junction_count = candidate_counts_by_scene[row["scene_id"]]["JUNCTION"]
        if pair_count != int(row["pair_candidate_count"]) or junction_count != int(
            row["junction_candidate_count"]
        ):
            raise SceneBuildError(f"Candidate count mismatch: {row['scene_id']}")


def candidate_count_diagnostic(
    plans: Sequence[dict[str, object]],
) -> dict[str, object]:
    train_lookup: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for row in plans:
        if row["split"] != "train":
            continue
        key = (int(row["pair_candidate_count"]), int(row["junction_candidate_count"]))
        train_lookup[key][str(row["relation_class"])] += 1
    validation = [row for row in plans if row["split"] == "val"]
    covered = 0
    correct = 0
    for row in validation:
        key = (int(row["pair_candidate_count"]), int(row["junction_candidate_count"]))
        if key not in train_lookup:
            continue
        covered += 1
        counts = train_lookup[key]
        prediction = max(sorted(counts), key=lambda label: counts[label])
        correct += prediction == row["relation_class"]
    return {
        "method": "train exact lookup by (pair_candidate_count, junction_candidate_count)",
        "validation_coverage": covered / max(len(validation), 1),
        "validation_accuracy_on_covered": correct / max(covered, 1),
        "role": "protocol diagnostic; candidate counts are intentionally included in the historical 42-feature relation input",
    }


def mixed_compatibility_diagnostic(
    plans: Sequence[dict[str, object]],
    candidates: Sequence[dict[str, object]],
) -> dict[str, int]:
    candidates_by_scene: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        candidates_by_scene[str(row["scene_id"])].append(row)
    mixed_count = 0
    true_relations_disjoint = 0
    false_conflict_count = 0
    for plan in plans:
        if plan["scene_family"] != "mixed":
            continue
        mixed_count += 1
        rows = candidates_by_scene[str(plan["scene_id"])]
        positive_pair = next(
            row
            for row in rows
            if row["candidate_type"] == "PAIR" and int(row["y"]) == 1
        )
        positive_junction = next(
            row
            for row in rows
            if row["candidate_type"] == "JUNCTION" and int(row["y"]) == 1
        )
        pair_ids = set(str(positive_pair["endpoint_ids"]).split("|"))
        junction_ids = set(str(positive_junction["endpoint_ids"]).split("|"))
        true_relations_disjoint += pair_ids.isdisjoint(junction_ids)
        false_pairs = [
            set(str(row["endpoint_ids"]).split("|"))
            for row in rows
            if row["candidate_type"] == "PAIR" and int(row["y"]) == 0
        ]
        false_junctions = [
            set(str(row["endpoint_ids"]).split("|"))
            for row in rows
            if row["candidate_type"] == "JUNCTION" and int(row["y"]) == 0
        ]
        false_conflict_count += any(
            pair_ids & junction_ids
            for pair_ids in false_pairs
            for junction_ids in false_junctions
        )
    return {
        "mixed_scenes": mixed_count,
        "true_pair_junction_disjoint": true_relations_disjoint,
        "scenes_with_conflicting_false_pair_junction_candidates": false_conflict_count,
    }


def main() -> None:
    args = parse_args()
    args.pair_plan = args.pair_plan.resolve()
    args.junction_plan = args.junction_plan.resolve()
    args.junction_endpoints = args.junction_endpoints.resolve()
    args.out_dir = args.out_dir.resolve()
    pair_rows = read_csv(args.pair_plan)
    junction_rows = read_csv(args.junction_plan)
    junction_endpoint_rows = read_csv(args.junction_endpoints)
    pairs = pair_components(pair_rows)
    junctions = junction_components(junction_rows, junction_endpoint_rows)
    plans, endpoints, candidates = build_scenes(pairs, junctions)
    validate_outputs(plans, endpoints, candidates)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.out_dir / "scene_plan_train_val.csv"
    endpoint_path = args.out_dir / "scene_endpoints_train_val.csv"
    candidate_path = args.out_dir / "scene_candidates_train_val.csv"
    protocol_path = args.out_dir / "scene_protocol.json"
    write_csv(plan_path, plans)
    write_csv(endpoint_path, endpoints)
    write_csv(candidate_path, candidates)

    family_counts = Counter((row["split"], row["scene_family"]) for row in plans)
    class_counts = Counter((row["split"], row["relation_class"]) for row in plans)
    count_diagnostic = candidate_count_diagnostic(plans)
    mixed_diagnostic = mixed_compatibility_diagnostic(plans, candidates)
    incomplete_coverage = Counter()
    for junction in junctions.values():
        key = (
            str(junction["split"]),
            str(junction["benchmark_group"]),
            "included" if junction["incomplete"] is not None else "omitted",
        )
        incomplete_coverage[key] += 1
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "train_val_draft_pending_protocol_review",
        "historical_reproduction": False,
        "held_out_junction_test_accessed": False,
        "splits": ["train", "val"],
        "patient_ids": {
            split: sorted(patient_ids)
            for split, patient_ids in EXPECTED_PATIENT_IDS.items()
        },
        "source_files": {
            "pair_plan": str(args.pair_plan.relative_to(REPO_ROOT)),
            "junction_plan": str(args.junction_plan.relative_to(REPO_ROOT)),
            "junction_endpoints": str(args.junction_endpoints.relative_to(REPO_ROOT)),
        },
        "source_sha256": {
            "pair_plan": sha256(args.pair_plan),
            "junction_plan": sha256(args.junction_plan),
            "junction_endpoints": sha256(args.junction_endpoints),
        },
        "scene_families": {
            "pair_only": "positive PAIR endpoints plus its frozen matched hard-negative endpoints",
            "junction_only": "positive degree-3 JUNCTION endpoints plus up to two retained decoy endpoints",
            "mixed": "same-patient pair_only and junction_only components with disjoint true endpoints",
            "none_orphan_pair": "one true PAIR endpoint withheld; matched hard-negative endpoints retained",
            "none_incomplete_junction": "one true arm withheld and replaced by the nearest retained one-arm negative decoy",
        },
        "candidate_generation": {
            "pair": f"all endpoint pairs with distance <= {PAIR_MAX_DISTANCE_MM} mm",
            "junction": f"all degree-{JUNCTION_DEGREE} endpoint subsets with maximum span <= {JUNCTION_MAX_SPAN_MM} mm",
            "anatomical_labels_used": False,
            "junction_axis_orientation": "orient each undirected tangent toward the fitted candidate center",
        },
        "composition": {
            "scene_endpoint_count": SCENE_ENDPOINT_COUNT,
            "junction_decoy_limit": JUNCTION_DECOY_LIMIT,
            "mixed_pairing": "nearest unused PAIR component from the same patient",
            "minimum_component_separation_mm": None,
            "duplicate_coordinate_policy": "exclude exact coordinate matches; no distance tolerance",
            "source_reuse_across_scene_families": True,
        },
        "relation_input_policy": {
            "candidate_pool": "all span-valid candidates remain available to Graph-LIRA",
            "representation": "historical_42_score_landscape",
            "status": "canonical",
            "features_per_local_score_stream": 42,
            "geometry_only_feature_count": 42,
            "cardinality_handling": "PAIR and JUNCTION candidate counts are included as historical model features",
            "score_landscape": "18 PAIR features plus 18 JUNCTION features plus six cross-stream features",
            "missing_rank_policy": "top-1 through top-7 are zero-padded when fewer candidates exist",
            "direct_pair_candidate_count": "included",
            "direct_junction_candidate_count": "included",
            "score_sum_and_top_over_sum": "top-over-sum included per candidate type",
        },
        "counts": {
            "scenes": len(plans),
            "endpoints": len(endpoints),
            "candidates": len(candidates),
            "by_split_family": {
                f"{split}:{family}": count
                for (split, family), count in sorted(family_counts.items())
            },
            "by_split_class": {
                f"{split}:{relation_class}": count
                for (split, relation_class), count in sorted(class_counts.items())
            },
        },
        "candidate_recall": {
            "repair_scenes": float(
                np.mean(
                    [
                        int(row["candidate_recall"])
                        for row in plans
                        if row["relation_class"] != "NONE"
                    ]
                )
            ),
            "all_scenes_protocol_validity": float(
                np.mean([int(row["candidate_recall"]) for row in plans])
            ),
        },
        "candidate_count_only_diagnostic": count_diagnostic,
        "mixed_compatibility_diagnostic": mixed_diagnostic,
        "none_incomplete_source_coverage": {
            f"{split}:{group}:{status}": count
            for (split, group, status), count in sorted(incomplete_coverage.items())
        },
        "outputs": {
            "scene_plan": plan_path.name,
            "scene_endpoints": endpoint_path.name,
            "scene_candidates": candidate_path.name,
        },
        "output_sha256": {
            "scene_plan": sha256(plan_path),
            "scene_endpoints": sha256(endpoint_path),
            "scene_candidates": sha256(candidate_path),
        },
        "pending_review_choices": [
            "whether mixed scenes should use nearest, farthest, or anatomically constrained component pairing",
            "whether two decoy endpoints per JUNCTION component is the intended candidate burden",
            "whether none_incomplete_junction should require a retained one-arm negative and omit sources without one",
            "whether source components may be reused across different scene families",
            "whether degree-4 remains audit-only for the first relation experiment",
            "whether perturbations belong in the frozen scene pack or only in the later consistency evaluation",
        ],
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(protocol["counts"], indent=2))
    print("candidate recall", protocol["candidate_recall"])
    print("OUT", args.out_dir)


if __name__ == "__main__":
    main()
