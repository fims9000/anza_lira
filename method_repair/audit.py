"""Machine-checkable forensic audit of the frozen mode-resolved implementation.

This module deliberately does not import or mutate training checkpoints.  It
records why the already-frozen experiment is a negative control and protects
that evidence while a separate repaired candidate is developed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time
from typing import Any

import torch

from models.azconv_v2 import ModeResolvedAZConv2d
from models.segmentation_v2 import ComparableStructuralUNet, build_comparable_model
from synthetic.structural_losses import branch_transition_logits
from synthetic.training import LOSS_WEIGHTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_DEADLINE_ROOT = PROJECT_ROOT / "results" / "anza_v2_study" / "deadline_20260817"
FROZEN_FILES = {
    "deadline_zip": FROZEN_DEADLINE_ROOT
    / "packages"
    / "ANZA_LIRA_DEADLINE_FINAL_20260817_20260817T211049.zip",
    "thesis_numbers": FROZEN_DEADLINE_ROOT / "THESIS_NUMBERS.json",
    "main_cracks": FROZEN_DEADLINE_ROOT / "tables" / "main_cracks.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_membership(membership: torch.Tensor) -> torch.Tensor:
    membership = torch.as_tensor(membership)
    if membership.ndim < 1:
        raise ValueError("membership must have a mode dimension")
    if not torch.isfinite(membership).all() or torch.any(membership < 0):
        raise ValueError("membership must be finite and non-negative")
    denominator = membership.sum(dim=0, keepdim=True)
    if torch.any(denominator <= 0):
        raise ValueError("membership must have positive mass at every location")
    return membership / denominator


def current_membership_gain(membership: torch.Tensor, *, variant: str) -> torch.Tensor:
    """Feature gain before updates in the frozen v2 implementation.

    v2a creates ``mu*V`` and fuses with another ``mu``.  v2b creates two
    ``0.5*mu*V`` half states and fuses each with ``0.5*mu``.
    """
    mu = normalized_membership(membership)
    squared_mass = mu.square().sum(dim=0)
    if variant == "v2a":
        return squared_mass
    if variant == "v2b":
        return 0.5 * squared_mass
    raise ValueError("variant must be v2a or v2b")


def repaired_membership_gain(membership: torch.Tensor) -> torch.Tensor:
    """The repaired single-gate/sum-fusion path preserves feature amplitude."""
    return normalized_membership(membership).sum(dim=0)


def _source_facts() -> dict[str, Any]:
    forward = inspect.getsource(ModeResolvedAZConv2d.forward)
    route = inspect.getsource(branch_transition_logits)
    model = inspect.getsource(ComparableStructuralUNet.__init__)
    return {
        "H1_repeated_membership": {
            "v2a_source_gate": "geometry[\"membership\"].unsqueeze(2) * value.unsqueeze(1)" in forward,
            "v2a_fusion_gate": "geometry[\"membership\"].unsqueeze(2) * state" in forward,
            "v2b_source_half_factor": "0.5 * geometry[\"membership\"]" in forward,
            "v2b_fusion_half_factor": "fusion_weight = 0.5 * geometry[\"membership\"]" in forward,
        },
        "H2_route_identity_marginalized": {
            "v2a_sums_source_and_destination_modes": "transition.sum(dim=(1, 2))" in route,
            "v2b_sums_modes_and_halves": "transition.sum(dim=(1, 2, 3, 4))" in route,
        },
        "H3_replay_effective_weights": {
            "replay_interval": 3,
            "outer_replay_weight": 0.25,
            "synthetic_visible_effective_per_real_step": 0.25 / 3.0,
            "synthetic_route_effective_per_real_step": 0.25 * LOSS_WEIGHTS["route"] / 3.0,
        },
        "H4_persistent_half_modes": {
            "present": "modes * 2" in forward and "half_membership" in forward,
            "semantic_contract": "FAULT_TRACE_IS_AXIAL_THETA_EQUIVALENT_THETA_PLUS_PI",
        },
        "H5_declared_controls": {
            "junction_score_returned_as_diagnostic": '"junction_score": geometry["junction_score"]' in forward,
            "junction_score_changes_forward": "use_junction" in forward,
            "cone_flag_changes_forward": "use_cone" in forward,
        },
        "H6_router_scope": {
            "same_operator_in_enc1_enc2_enc3": all(
                token in model for token in ("self.enc1", "self.enc2", "self.enc3")
            ),
            "router_block_count": 3,
        },
    }


def _frozen_hashes() -> dict[str, str]:
    missing = [str(path) for path in FROZEN_FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen deadline evidence missing: {missing}")
    return {name: sha256_file(path) for name, path in FROZEN_FILES.items()}


def _conv_macs(model: torch.nn.Module, sample: torch.Tensor) -> int:
    total = 0

    def hook(module: torch.nn.Conv2d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        kernel_h, kernel_w = module.kernel_size
        operations_per_output = kernel_h * kernel_w * module.in_channels // module.groups
        total += int(output.numel() * operations_per_output)

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    try:
        with torch.inference_mode():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()
    return total


def _transport_macs_approx(model_name: str, image_size: int, *, modes: int = 4, steps: int = 2) -> int:
    """Approximate only state transport multiply-accumulates, kept separate from conv MACs."""
    if model_name not in {"anza_v2a", "anza_v2b"}:
        return 0
    halves = 2 if model_name == "anza_v2b" else 1
    total = 0
    for scale, channels in ((1, 16), (2, 32), (4, 64)):
        height = image_size // scale
        width = image_size // scale
        state_pairs = modes * halves * modes * halves
        total += steps * state_pairs * 9 * height * width * channels
    return int(total)


def profile_frozen_models(
    *, image_size: int = 256, repeats: int = 3, device: str | None = None
) -> dict[str, Any]:
    """Profile frozen comparison models without training or checkpoint access."""
    if image_size <= 0 or repeats <= 0:
        raise ValueError("image_size and repeats must be positive")
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(selected_device)
    profiles: dict[str, Any] = {}
    for model_name in ("anza_v1", "anza_v2a", "anza_v2b"):
        model = build_comparable_model(model_name).eval().to(torch_device)
        sample = torch.zeros(1, 3, image_size, image_size, device=torch_device)
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(torch_device)
        conv_macs = _conv_macs(model, sample)
        with torch.inference_mode():
            model(sample)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(repeats):
                model(sample)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        elapsed = (time.perf_counter() - started) / repeats
        peak_memory = (
            int(torch.cuda.max_memory_allocated(torch_device))
            if torch_device.type == "cuda"
            else None
        )
        profiles[model_name] = {
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "conv_macs_lower_bound": int(conv_macs),
            "transport_state_pair_macs_approx": _transport_macs_approx(model_name, image_size),
            "inference_seconds_mean": float(elapsed),
            "peak_cuda_memory_bytes": peak_memory,
        }
        del sample, model
    return {
        "device": str(torch_device),
        "image_shape": [1, 3, image_size, image_size],
        "repeats": repeats,
        "mac_definition": "conv MACs measured by hooks plus separate approximate transport state-pair MACs",
        "models": profiles,
    }


def run_forensic_audit(
    output_path: Path, *, include_profile: bool = False, profile_device: str | None = None
) -> dict[str, Any]:
    """Write an idempotent audit without loading data, checkpoints, or experts."""
    output_path = Path(output_path)
    uniform = torch.full((4,), 0.25)
    one_hot = torch.tensor([1.0, 0.0, 0.0, 0.0])
    payload: dict[str, Any] = {
        "status": "NEGATIVE_BASELINE_FROZEN_METHOD_REPAIR_AUTHORIZED",
        "scientific_scope": "forensic implementation audit; no new performance result",
        "expert_data_accessed": False,
        "training_started": False,
        "legacy_code_modified": False,
        "frozen_deadline_sha256": _frozen_hashes(),
        "membership_gain": {
            "four_uniform_modes": {
                "current_v2a": float(current_membership_gain(uniform, variant="v2a")),
                "current_v2b": float(current_membership_gain(uniform, variant="v2b")),
                "repaired_contract": float(repaired_membership_gain(uniform)),
            },
            "one_hot_mode": {
                "current_v2a": float(current_membership_gain(one_hot, variant="v2a")),
                "current_v2b": float(current_membership_gain(one_hot, variant="v2b")),
                "repaired_contract": float(repaired_membership_gain(one_hot)),
            },
        },
        "implementation_facts": _source_facts(),
        "cracks_white_semantics": {
            "status": "NOT_ESTABLISHED",
            "baseline_policy": "paper_like_preserved_as_historical_inferred_policy",
            "official_defined_colors": ["orange_certain_no_fault", "blue_certain_fault", "green_uncertain_fault"],
            "official_paper_experiment": "orange_not_used; certain_and_uncertain_fault_combined",
            "white_is_not_reinterpreted": True,
            "sources": [
                "https://arxiv.org/abs/2408.11185",
                "https://github.com/olivesgatech/CRACKS",
                "https://alregib.ece.gatech.edu/software-and-datasets/cracks-crowdsourcing-resources-for-analysis-and-categorization-of-key-subsurface-faults/",
            ],
        },
    }
    if include_profile:
        payload["runtime_profile"] = profile_frozen_models(device=profile_device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() == encoded:
        return payload
    output_path.write_text(encoded)
    return payload
