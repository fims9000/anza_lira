"""Independent repair-calibration stream using frozen V2 geometry rules."""

from __future__ import annotations

import hashlib

import numpy as np

from anza_tracegraph.data_v2.generator import BUILDERS, RELATION_CORRIDOR_X, _render
from anza_tracegraph.data_v2.strata import POSITIVE_STRATA, STRATA


REPAIR_CALIBRATION_SIZE = 3840
REPAIR_CALIBRATION_SEED = 5_231_000_000


def generate_repair_scene(index: int) -> dict:
    if not 0 <= index < REPAIR_CALIBRATION_SIZE: raise ValueError("unknown repair-calibration index")
    rng = np.random.default_rng(REPAIR_CALIBRATION_SEED + int(index)); stratum = STRATA[index % len(STRATA)]; geometry = BUILDERS[stratum](rng)
    model_input, visible_evidence = _render(geometry, rng)
    source = np.asarray(geometry["source"], dtype=np.float64); target = None if geometry["target"] is None else np.asarray(geometry["target"], dtype=np.float64)
    tangent = source[-1] - source[max(0, len(source) - 6)]; tangent /= max(float(np.linalg.norm(tangent)), 1e-8)
    return {
        "input": {"model_input": model_input, "source_query_yx": tuple(map(float, source[-1])), "source_tangent_yx": tuple(map(float, tangent)), "relation_corridor_x": RELATION_CORRIDOR_X, "stratum": stratum, "split": "repair_calibration", "index": int(index)},
        "truth": {"has_valid_continuation": stratum in POSITIVE_STRATA, "source_branch": source, "destination_branch": target, "distractor_branches": tuple(np.asarray(item, dtype=np.float64) for item in geometry["distractors"]), "topology": geometry["topology"], "visible_evidence": visible_evidence},
    }


def repair_scene_digest(scene: dict) -> bytes:
    digest = hashlib.sha256(); public = scene["input"]; truth = scene["truth"]
    digest.update(public["model_input"].tobytes()); digest.update(public["stratum"].encode()); digest.update(np.asarray(public["source_query_yx"], dtype=np.float64).tobytes()); digest.update(str(bool(truth["has_valid_continuation"])).encode()); digest.update(truth["source_branch"].tobytes())
    if truth["destination_branch"] is not None: digest.update(truth["destination_branch"].tobytes())
    for branch in truth["distractor_branches"]: digest.update(branch.tobytes())
    return digest.digest()


def repair_calibration_hash() -> str:
    digest = hashlib.sha256()
    for index in range(REPAIR_CALIBRATION_SIZE): digest.update(index.to_bytes(4, "little")); digest.update(repair_scene_digest(generate_repair_scene(index)))
    return digest.hexdigest()
