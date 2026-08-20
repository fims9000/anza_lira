"""Pre-gradient and final fail-closed validators for ANZA-FS H3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .geometry import axial_bank, frozen_foliation_geometry
from .hard_bench_v6 import CASES, SPLIT_BASE, SPLIT_SIZE, freeze_hard_bench, generate_hard_sample
from .kernels import kernel_centroids
from .model import VARIANTS, build_h3_model
from .foliation_conv import ANZAFoliationConv, FreeFoliationConv
from .protocol import H3_ROOT, PREGRADIENT_ROOT, canonical_hash, freeze_protocol


SOURCE_FILES = (
    "anza_fs/geometry.py", "anza_fs/kernels.py", "anza_fs/foliation_conv.py",
    "anza_fs/model.py", "anza_fs/hard_bench_v6.py", "anza_fs/metrics.py",
    "anza_fs/training.py", "anza_fs/protocol.py", "anza_fs/run_h3.py",
)


def source_manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    combined = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"files": files, "sha256": combined}


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_pregradient() -> dict[str, Any]:
    protocol = freeze_protocol()
    benchmark = freeze_hard_bench(PREGRADIENT_ROOT / "stressbench_v6_hard.json")
    code = source_manifest()
    checks: dict[str, bool] = {}
    checks["parent_h1_frozen_negative"] = protocol["parent_h1_status"] == "HYPERBOLIC_CONSTRAINT_NOT_INCREMENTAL"
    checks["benchmark_hash_nested"] = protocol["stressbench"]["sha256"] == benchmark["sha256"]
    checks["event_count_minimum"] = benchmark["negative_events_calibration_plus_development"] >= 1000 and benchmark["positive_events_calibration_plus_development"] >= 1000
    ranges = {name: set(range(base, base + SPLIT_SIZE[name])) for name, base in SPLIT_BASE.items()}
    checks["split_seed_disjoint"] = all(not (ranges[first] & ranges[second]) for index, first in enumerate(ranges) for second in list(ranges)[index + 1 :])
    first = generate_hard_sample("train", 0)
    again = generate_hard_sample("train", 0)
    checks["deterministic_generation"] = np.array_equal(first["image"], again["image"]) and np.array_equal(first["visible_fault_mask"], again["visible_fault_mask"])
    try:
        generate_hard_sample("confirm", 0)
        checks["confirm_locked"] = False
    except PermissionError:
        checks["confirm_locked"] = True
    checks["all_cases_declared"] = len(CASES) == 16
    angles, unstable, stable = axial_bank(8)
    checks["stable_unstable_orthogonal"] = bool(torch.allclose((unstable * stable).sum(-1), torch.zeros(8), atol=1e-7))
    sigma_u, sigma_s, delta_u, delta_s = frozen_foliation_geometry()
    checks["reciprocal_scales"] = abs(sigma_u * sigma_s - 1.5**2) < 1e-7
    checks["frozen_offsets"] = abs(delta_u - 1.5 * sigma_u) < 1e-7 and abs(delta_s - 1.5 * sigma_s) < 1e-7
    anza = ANZAFoliationConv(2)
    free = FreeFoliationConv(2)
    checks["free_reproduces_anza_at_init"] = bool(torch.allclose(anza.kernels(), free.kernels(), atol=1e-6))
    centers = kernel_centroids(anza.kernels())
    checks["five_lobes_distinct"] = bool((centers[:, 1].norm(dim=-1) > 1.0).all() and (centers[:, 3].norm(dim=-1) > 0.5).all())
    checks["gamma_zero_init"] = float(anza.gamma.detach()) == 0.0 and float(free.gamma.detach()) == 0.0
    smoke = {}
    for variant in VARIANTS:
        model = build_h3_model(variant).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 32, 32), return_aux=True)
        smoke[variant] = bool(output["visible_logits"].shape == (1, 1, 32, 32) and torch.isfinite(output["visible_logits"]).all())
    checks["cpu_vertical_smoke"] = all(smoke.values())
    checks["downstream_locks"] = not any(protocol[key] for key in ("confirm_opened", "cracks_accessed", "expert_accessed", "H4_opened", "lambda_tuned", "M_tuned", "base_scale_tuned"))
    passed = all(checks.values())
    result = {
        "research_status": "ANZA_FS_H3_PREGRADIENT_PASS" if passed else "ANZA_FS_H3_PREGRADIENT_FAIL",
        "training_authorized": passed,
        "protocol_sha256": canonical_hash(protocol),
        "stressbench_sha256": benchmark["sha256"],
        "code_sha256": code["sha256"],
        "checks": checks,
        "smoke": smoke,
        "confirm_opened": False,
        "cracks_accessed": False,
        "expert_accessed": False,
    }
    _write(PREGRADIENT_ROOT / "code_freeze.json", code)
    _write(PREGRADIENT_ROOT / "validator.json", result)
    if not passed:
        raise ValueError("ANZA-FS H3 pre-gradient validation failed")
    return result


def validate_final() -> dict[str, Any]:
    metrics_path = H3_ROOT / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("H3 metrics do not exist")
    metrics = json.loads(metrics_path.read_text())
    pregradient = json.loads((PREGRADIENT_ROOT / "validator.json").read_text())
    parent = json.loads((H3_ROOT / "parent_pregradient.json").read_text())
    checks = {
        "pregradient_pass": pregradient.get("research_status") == "ANZA_FS_H3_PREGRADIENT_PASS",
        "protocol_unchanged": metrics.get("protocol_sha256") == pregradient.get("protocol_sha256"),
        "source_unchanged": parent.get("code_sha256") == pregradient.get("code_sha256") == source_manifest()["sha256"],
        "four_seed41_runs": set(metrics.get("variants", {})) == set(VARIANTS) and metrics.get("seed") == 41,
        "exact_dev_event_counts": all(value["primary"]["overall"]["negative_event_count"] == SPLIT_SIZE["development"] for value in metrics.get("variants", {}).values()),
        "confirm_closed": metrics.get("confirm_opened") is False,
        "cracks_closed": metrics.get("cracks_accessed") is False,
        "expert_closed": metrics.get("expert_accessed") is False,
        "H4_closed": metrics.get("H4_opened") is False,
        "no_tuning": not any(metrics.get(key, True) for key in ("lambda_tuned", "M_tuned", "base_scale_tuned")),
    }
    passed = all(checks.values())
    result = {"validator_status": "PASS" if passed else "FAIL", "research_status": metrics.get("status"), "checks": checks}
    _write(H3_ROOT / "validator.json", result)
    if not passed:
        raise ValueError("ANZA-FS H3 final validation failed")
    return result
