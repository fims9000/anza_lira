"""Frozen S0 protocol and split manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np

from cracks_experiment.matrix import PROJECT_ROOT


ROOT = PROJECT_ROOT / "results" / "anza_surftrack" / "s0"
METHODS = ("G0_euclidean", "G1_local_reset", "G2_shear_compose", "G3_free_compose", "G4_anza_cocycle")
FAMILIES = (
    "plane_smooth", "curved_surface", "rotating_strike", "s_warp", "stepover",
    "close_parallel", "diverging_parallel", "converging_parallel", "projection_crossing",
    "center_collinear_ambiguous", "multi_slice_gap_1", "multi_slice_gap_3",
    "multi_slice_gap_7", "competing_branch", "terminating_surface",
    "two_surface_near_touch", "combined_rotate_gap_parallel",
)
SPLITS = {
    "geom_train": {"size": 50_000, "seed": 31_001, "opened": True, "role": "parameter_fit_only"},
    "geom_calibration": {"size": 10_000, "seed": 41_003, "opened": True, "role": "margin_calibration_only"},
    "geom_dev_iid": {"size": 10_000, "seed": 51_007, "opened": True, "role": "iid_evaluation"},
    "geom_dev_ood": {"size": 10_000, "seed": 61_009, "opened": True, "role": "ood_evaluation"},
    "geom_confirm": {"size": 20_000, "seed": 71_011, "opened": False, "role": "hash_only_locked"},
}


PROTOCOL: dict[str, Any] = {
    "name": "ANZA_LIRA_SURFTRACK_V1_S0",
    "phase": "S0_ZERO_TRAINING_CAUSAL_GEOMETRY_ONLY",
    "volume_geometry": [17, 96, 96],
    "methods": list(METHODS), "families": list(FAMILIES), "splits": SPLITS,
    "common_mean": "constant velocity from last two accepted positions; first transition holds position",
    "candidate_count": 5, "end_candidate_index": 4,
    "parameter_bounds": {
        "sigma0": [0.25, 8.0], "q": [1e-4, 8.0], "alpha": [-0.75, 0.75],
        "a": [-0.50, 0.50], "b": [-0.50, 0.50], "lambda": [0.0, 0.50],
        "sigma_u": [0.25, 8.0], "sigma_s": [0.25, 8.0],
    },
    "fit": {"source": "geom_train", "optimizer": "bounded L-BFGS-B", "deterministic_starts": 3, "dev_tuning": False},
    "observability": {"center_auroc_min": 0.45, "center_auroc_max": 0.55, "context_top1_min": 0.85},
    "bootstrap_resamples": 10_000, "bootstrap_unit": "synthetic_scene",
    "gates": {
        "g4_vs_g1": {"top1_delta": 0.08, "switch_ratio": 0.70},
        "g4_vs_g2": {"top1_delta": 0.04, "switch_ratio": 0.80},
        "g4_vs_g3": {"iid_top1_noninferiority": -0.01, "ood_top1_delta": 0.03, "ood_switch_ratio": 0.85},
        "per_stratum": {"required": 3, "of": 5, "top1_delta": 0.05, "switch_reduction": 0.20,
                         "families": ["rotating_strike", "close_parallel", "multi_slice_gap_3", "multi_slice_gap_7", "center_collinear_ambiguous"]},
    },
    "locks": {"confirm": False, "rendering": False, "cnn": False, "thebe": False, "cracks": False, "s1": False},
    "claim_boundary": "Anosov-inspired determinant-one hyperbolic covariance transport prior; no physical Anosov, ergodicity, real seismic, or segmentation claim.",
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def split_manifest() -> dict[str, Any]:
    entries = {}
    for name, spec in SPLITS.items():
        descriptor = {"name": name, **spec, "rng": "numpy.PCG64", "index_range": [0, int(spec["size"]) - 1]}
        descriptor["descriptor_sha256"] = canonical_hash(descriptor)
        entries[name] = descriptor
    manifest = {
        "version": "SurfTrackBench3D-V1", "splits": entries,
        "seeds_disjoint": len({spec["seed"] for spec in SPLITS.values()}) == len(SPLITS),
        "ood_definition": "unseen combined family plus shifted curvature/spacing/gap severity; excluded from fitting",
        "confirm_access": "HASH_ONLY_NOT_GENERATED", "test_data_opened": False,
    }
    manifest["sha256"] = canonical_hash(manifest)
    return manifest


def freeze_protocol() -> dict[str, Any]:
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol_hash = canonical_hash(PROTOCOL); split = split_manifest()
    if (ROOT / "protocol.json").exists():
        existing = json.loads((ROOT / "protocol.json").read_text())
        existing_split = json.loads((ROOT / "split_manifest.json").read_text())
        if canonical_hash(existing) != protocol_hash or existing_split != split:
            raise ValueError("SurfTrack S0 frozen input drift")
        return {"action": "SKIP", "protocol": existing, "split": existing_split}
    write_json(ROOT / "protocol.json", PROTOCOL); (ROOT / "protocol_hash.txt").write_text(protocol_hash + "\n")
    write_json(ROOT / "split_manifest.json", split)
    generator = {
        "families": list(FAMILIES), "dedicated_constructors": [f"make_{name}" for name in FAMILIES],
        "surface_id": "immutable int64 per scene and candidate surface", "truth_feature_access": False,
        "generated_on_demand": True, "large_voxel_corpus_stored": False,
        "confirm_descriptor_sha256": split["splits"]["geom_confirm"]["descriptor_sha256"],
    }
    write_json(ROOT / "generator_manifest.json", generator)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        head = "UNAVAILABLE"
    (ROOT / "git_head.txt").write_text(head + "\n")
    write_json(ROOT / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()})
    return {"action": "RUN", "protocol": PROTOCOL, "split": split}


def load_frozen() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads((ROOT / "protocol.json").read_text()); split = json.loads((ROOT / "split_manifest.json").read_text())
    if canonical_hash(protocol) != (ROOT / "protocol_hash.txt").read_text().strip():
        raise ValueError("SurfTrack protocol hash drift")
    if split["sha256"] != canonical_hash({key: value for key, value in split.items() if key != "sha256"}):
        raise ValueError("SurfTrack split hash drift")
    return protocol, split
