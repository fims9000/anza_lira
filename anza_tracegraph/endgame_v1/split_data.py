"""Fresh, disjoint relation splits using the frozen V2 geometry builders."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from anza_tracegraph.data_v2 import generator as frozen_generator
from anza_tracegraph.data_v2.strata import POSITIVE_STRATA, STRATA

from .protocol import PROTOCOL


SPLIT_SETTINGS = dict(PROTOCOL["splits"])


def generate_scene(split: str, index: int) -> dict[str, Any]:
    """Generate one scene without modifying the frozen V2 generator module."""
    if split not in SPLIT_SETTINGS:
        raise ValueError(f"unknown endgame split: {split}")
    settings = SPLIT_SETTINGS[split]
    if not 0 <= index < int(settings["size"]):
        raise IndexError(index)
    rng = np.random.default_rng(int(settings["seed"]) + int(index))
    stratum = STRATA[index % len(STRATA)]
    geometry = frozen_generator.BUILDERS[stratum](rng)
    model_input, visible_evidence = frozen_generator._render(geometry, rng)
    source = np.asarray(geometry["source"], dtype=np.float64)
    target = None if geometry["target"] is None else np.asarray(geometry["target"], dtype=np.float64)
    tangent = source[-1] - source[max(0, len(source) - 6)]
    tangent /= max(float(np.linalg.norm(tangent)), 1e-8)
    return {
        "input": {
            "model_input": model_input,
            "source_query_yx": tuple(map(float, source[-1])),
            "source_tangent_yx": tuple(map(float, tangent)),
            "relation_corridor_x": frozen_generator.RELATION_CORRIDOR_X,
            "stratum": stratum,
            "split": split,
            "index": int(index),
        },
        "truth": {
            "has_valid_continuation": stratum in POSITIVE_STRATA,
            "source_branch": source,
            "destination_branch": target,
            "distractor_branches": tuple(np.asarray(item, dtype=np.float64) for item in geometry["distractors"]),
            "topology": geometry["topology"],
            "destination_signal": None if target is None else float(geometry["amplitudes"][1]),
            "competitor_signals": tuple(map(float, geometry["amplitudes"][2 if target is not None else 1 :])),
            "visible_evidence": visible_evidence,
        },
    }


def scene_digest(scene: dict[str, Any]) -> bytes:
    return frozen_generator.scene_digest(scene)


def split_hash(split: str) -> str:
    digest = hashlib.sha256()
    for index in range(int(SPLIT_SETTINGS[split]["size"])):
        digest.update(index.to_bytes(4, "little"))
        digest.update(scene_digest(generate_scene(split, index)))
    return digest.hexdigest()


def assert_seed_hygiene() -> None:
    old = {5_201_000_000, 5_211_000_000, 5_221_000_000, 5_231_000_000}
    current = {int(settings["seed"]) for settings in SPLIT_SETTINGS.values()}
    if len(current) != len(SPLIT_SETTINGS) or current & old:
        raise AssertionError("relation split seed overlap")
