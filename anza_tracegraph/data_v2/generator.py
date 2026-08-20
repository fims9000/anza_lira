"""Deterministic seismic-like TRACEGRAPH_RELATION_V2 generator.

Latent branch lineage is returned under ``truth`` and is evaluation-only.  The
frozen dense model receives only ``model_input``.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable

import numpy as np
from scipy.ndimage import gaussian_filter

from .strata import NEGATIVE_STRATA, POSITIVE_STRATA, SPLIT_SEEDS, SPLIT_SIZES, STRATA


SIZE = 96
RELATION_CORRIDOR_X = (35, 50)


def _rng(split: str, index: int) -> np.random.Generator:
    if split not in SPLIT_SIZES or not 0 <= index < SPLIT_SIZES[split]:
        raise ValueError("unknown TRACEGRAPH_RELATION_V2 split/index")
    return np.random.default_rng(SPLIT_SEEDS[split] + int(index))


def _sample(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    raw = np.column_stack((ys, xs)).astype(np.float64)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
    count = max(2, int(math.ceil(float(distance[-1]))) + 1)
    target = np.linspace(0.0, float(distance[-1]), count)
    return np.column_stack((np.interp(target, distance, ys), np.interp(target, distance, xs)))


def _curve(xs: np.ndarray, *, center: float, slope: float, amplitude: float, phase: float, frequency: float) -> np.ndarray:
    return center + slope * (xs - 48.0) + amplitude * np.sin(frequency * math.pi * xs / SIZE + phase)


def _base(rng: np.random.Generator, *, kind: str = "curve", long: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    center = float(rng.uniform(30.0, 66.0)); slope = float(rng.uniform(-0.14, 0.14)); phase = float(rng.uniform(-math.pi, math.pi))
    amplitude = 0.15 if kind == "straight" else float(rng.uniform(2.0, 5.5)); frequency = 2.05 if kind == "s" else float(rng.uniform(0.7, 1.35))
    if kind == "s": amplitude = float(rng.uniform(4.5, 7.0))
    gap_end = float(rng.uniform(61.0, 66.0) if long else rng.uniform(50.0, 57.0))
    left_x = np.linspace(4.0, 34.0, 90); right_x = np.linspace(gap_end, 92.0, 100)
    args = {"center": center, "slope": slope, "amplitude": amplitude, "phase": phase, "frequency": frequency}
    return _sample(left_x, _curve(left_x, **args)), _sample(right_x, _curve(right_x, **args)), {**args, "gap_end": gap_end}


def _parallel(path: np.ndarray, offset: float) -> np.ndarray:
    output = path.copy(); output[:, 0] += offset; return output


def _line(start: tuple[float, float], end: tuple[float, float], count: int = 100) -> np.ndarray:
    y = np.linspace(start[0], end[0], count); x = np.linspace(start[1], end[1], count); return _sample(x, y)


def _positive(rng: np.random.Generator, *, kind: str = "curve", long: bool = False) -> dict[str, Any]:
    source, target, params = _base(rng, kind=kind, long=long)
    return {"source": source, "target": target, "distractors": [], "amplitudes": [0.90, 0.82], "params": params, "topology": "gap"}


def _straight_gap(rng: np.random.Generator) -> dict[str, Any]: return _positive(rng, kind="straight")
def _curved_gap(rng: np.random.Generator) -> dict[str, Any]: return _positive(rng)
def _s_curve_gap(rng: np.random.Generator) -> dict[str, Any]: return _positive(rng, kind="s")
def _long_gap(rng: np.random.Generator) -> dict[str, Any]: return _positive(rng, long=True)


def _crossing(rng: np.random.Generator, *, acute: bool) -> dict[str, Any]:
    row = _positive(rng); target = row["target"]; pivot = target[len(target) // 2]
    angle = float(rng.uniform(0.42, 0.58) if acute else rng.uniform(0.90, 1.18))
    dx = 43.0; dy = math.tan(angle) * dx
    distractor = _line((pivot[0] - dy * 0.45, pivot[1] - dx * 0.45), (pivot[0] + dy * 0.55, pivot[1] + dx * 0.55))
    row["distractors"].append(distractor); row["amplitudes"].append(0.86); row["topology"] = "acute_crossing" if acute else "x_crossing"; return row


def _x_crossing_correct(rng: np.random.Generator) -> dict[str, Any]: return _crossing(rng, acute=False)
def _acute_crossing_correct(rng: np.random.Generator) -> dict[str, Any]: return _crossing(rng, acute=True)


def _t_junction_continue(rng: np.random.Generator) -> dict[str, Any]:
    row = _positive(rng); junction = row["target"][len(row["target"]) // 3]
    arm = _line((junction[0], junction[1]), (np.clip(junction[0] - rng.uniform(18, 28), 4, 91), junction[1]))
    row["distractors"].append(arm); row["amplitudes"].append(0.78); row["topology"] = "t_junction"; return row


def _y_junction_continue(rng: np.random.Generator) -> dict[str, Any]:
    row = _positive(rng); junction = row["target"][0]
    arm = _line((junction[0], junction[1]), (np.clip(junction[0] + rng.choice((-1, 1)) * rng.uniform(16, 24), 4, 91), 92.0))
    row["distractors"].append(arm); row["amplitudes"].append(0.76); row["topology"] = "y_junction"; return row


def _weak_branch_continue(rng: np.random.Generator) -> dict[str, Any]:
    row = _positive(rng); competitor = _parallel(row["target"], rng.choice((-1, 1)) * rng.uniform(4.0, 6.0))
    row["distractors"].append(competitor); row["amplitudes"][1] = 0.46; row["amplitudes"].append(0.90); row["topology"] = "weak_branch"; return row


def _close_parallel_continue(rng: np.random.Generator) -> dict[str, Any]:
    row = _positive(rng); row["distractors"].append(_parallel(row["target"], rng.choice((-1, 1)) * rng.uniform(3.2, 5.2))); row["amplitudes"].append(0.84); row["topology"] = "close_parallel"; return row


def _low_contrast_continue(rng: np.random.Generator) -> dict[str, Any]:
    row = _positive(rng); row["amplitudes"][1] = 0.52; row["topology"] = "low_contrast"; return row


def _partial_occlusion_continue(rng: np.random.Generator) -> dict[str, Any]:
    row = _positive(rng); row["topology"] = "partial_occlusion"; row["occlusion"] = (54, 61); return row


def _multiple_plausible_correct(rng: np.random.Generator) -> dict[str, Any]:
    row = _positive(rng); competitor = _parallel(row["target"], rng.choice((-1, 1)) * rng.uniform(2.8, 4.2)); competitor[:, 0] += 0.035 * (competitor[:, 1] - competitor[0, 1])
    row["distractors"].append(competitor); row["amplitudes"].append(0.82); row["topology"] = "multiple_plausible"; return row


def _cluttered_corridor_continue(rng: np.random.Generator) -> dict[str, Any]:
    row = _positive(rng); row["topology"] = "cluttered_corridor"; row["clutter"] = True; return row


def _negative_base(rng: np.random.Generator, kind: str) -> dict[str, Any]:
    source, target_like, params = _base(rng, kind="straight" if kind == "independent_collinear" else "curve")
    distractors: list[np.ndarray] = []
    if kind == "none":
        pass
    elif kind == "parallel":
        distractors.append(_parallel(target_like, rng.choice((-1, 1)) * rng.uniform(3.0, 5.5)))
    elif kind in {"x", "t", "y"}:
        pivot = np.asarray((source[-1][0], 57.0)); angle = {"x": 1.05, "t": math.pi / 2, "y": 0.58}[kind]
        distractors.append(_line((pivot[0] - 18 * math.sin(angle), 50.0), (pivot[0] + 26 * math.sin(angle), 92.0)))
        if kind == "y": distractors.append(_line(tuple(pivot), (np.clip(pivot[0] - 22, 4, 91), 92.0)))
    elif kind == "independent_collinear":
        distractors.append(target_like)
    return {"source": source, "target": None, "distractors": distractors, "amplitudes": [0.90] + [0.84] * len(distractors), "params": params, "topology": f"negative_{kind}"}


def _none_isolated_end(rng: np.random.Generator) -> dict[str, Any]: return _negative_base(rng, "none")
def _parallel_wrong_only(rng: np.random.Generator) -> dict[str, Any]: return _negative_base(rng, "parallel")
def _x_wrong_only(rng: np.random.Generator) -> dict[str, Any]: return _negative_base(rng, "x")
def _t_wrong_only(rng: np.random.Generator) -> dict[str, Any]: return _negative_base(rng, "t")
def _y_wrong_only(rng: np.random.Generator) -> dict[str, Any]: return _negative_base(rng, "y")
def _independent_collinear_fault(rng: np.random.Generator) -> dict[str, Any]: return _negative_base(rng, "independent_collinear")


BUILDERS: dict[str, Callable[[np.random.Generator], dict[str, Any]]] = {
    name: globals()[f"_{name}"] for name in STRATA
}


def _deposit(field: np.ndarray, path: np.ndarray, value: float) -> None:
    points = np.rint(path).astype(int); valid = (points[:, 0] >= 1) & (points[:, 0] < SIZE - 1) & (points[:, 1] >= 1) & (points[:, 1] < SIZE - 1)
    np.maximum.at(field, (points[valid, 0], points[valid, 1]), value)


def _render(row: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    evidence = np.zeros((SIZE, SIZE), dtype=np.float64)
    paths = [row["source"]] + ([] if row["target"] is None else [row["target"]]) + row["distractors"]
    for path, amplitude in zip(paths, row["amplitudes"]): _deposit(evidence, path, float(amplitude))
    if "occlusion" in row:
        start, end = row["occlusion"]; evidence[:, start:end] *= 0.22
    fine = gaussian_filter(rng.normal(size=(SIZE, SIZE)), (0.8, 2.0)); broad = gaussian_filter(rng.normal(size=(SIZE, SIZE)), (5.0, 11.0))
    y, x = np.mgrid[:SIZE, :SIZE]; layers = np.sin(y * rng.uniform(0.14, 0.27) + 0.7 * np.sin(x * rng.uniform(0.02, 0.055)))
    image = 0.12 * fine + 0.08 * broad + 0.10 * layers + gaussian_filter(evidence, 1.05)
    if row.get("clutter"):
        image += 0.15 * gaussian_filter(rng.normal(size=image.shape), (0.5, 2.8))
    image = gaussian_filter(image, rng.uniform(0.2, 0.65)); image += rng.normal(0.0, rng.uniform(0.025, 0.065), image.shape)
    image = (image - image.mean()) / (image.std() + 1e-8); gy, gx = np.gradient(image)
    return np.stack((image, gx, gy)).astype(np.float32), evidence.astype(np.float32)


def _make_scene(split: str, index: int) -> dict[str, Any]:
    rng = _rng(split, index); stratum = STRATA[index % len(STRATA)]; geometry = BUILDERS[stratum](rng)
    model_input, visible_evidence = _render(geometry, rng)
    source = np.asarray(geometry["source"], dtype=np.float64); target = None if geometry["target"] is None else np.asarray(geometry["target"], dtype=np.float64)
    source_vector = source[-1] - source[max(0, len(source) - 6)]; source_vector /= max(float(np.linalg.norm(source_vector)), 1e-8)
    public = {
        "model_input": model_input,
        "source_query_yx": tuple(map(float, source[-1])),
        "source_tangent_yx": tuple(map(float, source_vector)),
        "relation_corridor_x": RELATION_CORRIDOR_X,
        "stratum": stratum,
        "split": split,
        "index": int(index),
    }
    truth = {
        "has_valid_continuation": stratum in POSITIVE_STRATA,
        "source_branch": source,
        "destination_branch": target,
        "distractor_branches": tuple(np.asarray(item, dtype=np.float64) for item in geometry["distractors"]),
        "topology": geometry["topology"],
        "destination_signal": None if target is None else float(geometry["amplitudes"][1]),
        "competitor_signals": tuple(map(float, geometry["amplitudes"][2 if target is not None else 1 :])),
        "visible_evidence": visible_evidence,
    }
    return {"input": public, "truth": truth}


def generate_scene(split: str, index: int, *, allow_confirm: bool = False) -> dict[str, Any]:
    if split == "confirm" and not allow_confirm:
        raise PermissionError("TRACEGRAPH_RELATION_V2 confirm is hash-only in V3-A")
    return _make_scene(split, index)


def scene_digest(scene: dict[str, Any]) -> bytes:
    digest = hashlib.sha256(); public = scene["input"]; truth = scene["truth"]
    digest.update(public["model_input"].tobytes()); digest.update(public["stratum"].encode()); digest.update(np.asarray(public["source_query_yx"], dtype=np.float64).tobytes())
    digest.update(str(bool(truth["has_valid_continuation"])).encode()); digest.update(truth["source_branch"].tobytes())
    if truth["destination_branch"] is not None: digest.update(truth["destination_branch"].tobytes())
    for branch in truth["distractor_branches"]: digest.update(branch.tobytes())
    return digest.digest()


def split_hash(split: str) -> str:
    digest = hashlib.sha256()
    for index in range(SPLIT_SIZES[split]):
        digest.update(index.to_bytes(4, "little")); digest.update(scene_digest(generate_scene(split, index, allow_confirm=True)))
    return digest.hexdigest()
