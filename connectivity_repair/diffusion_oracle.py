"""GT-connectivity feasibility test for finite-step restarted ANZA diffusion."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from affinity_repair.matrix import affinity_matrix
from affinity_repair.training import build_candidate, checkpoint_sha256, load_checkpoint
from models.azconv_affinity import LOCAL8_OFFSETS, _shift_tensor
from synthetic.affinity_targets import build_affinity_targets
from synthetic.crossing_trace_bench_v5 import generate_sample_v5
from synthetic.structural_metrics import _cldice, _false_bridge_rate


ORACLE_T_VALUES = (1, 2, 4, 6, 8)
ORACLE_ALPHA_VALUES = (0.4, 0.6, 0.8)
ORACLE_PROTOCOL = {
    "version": "gt_connectivity_diffusion_oracle_v1",
    "base": "frozen prior C1 CleanANZA checkpoint; no retraining",
    "h0": "sigmoid foreground probability from frozen CleanANZA visible head",
    "conductance": "sum_r mu_r(p)mu_r(q)G_r(p,q) times [eps_c+(1-eps_c)GT_connectivity]",
    "connectivity_offsets_xy": [list(offset) for offset in LOCAL8_OFFSETS],
    "connectivity_epsilon": 0.05,
    "t_values": list(ORACLE_T_VALUES),
    "alpha_values": list(ORACLE_ALPHA_VALUES),
    "split": "crossing_trace_bench_v5 validation[0:512]",
    "visible_threshold": "frozen prior C1 v4 validation threshold",
    "selection": "minimum passing T, then minimum alpha; no trained-model result",
    "gates": {
        "gap_recovery_min": 0.70,
        "false_bridge_max": 0.20,
        "visible_dice_loss_max": 0.005,
    },
    "test_v5": "LOCKED_UNOPENED",
    "cracks": "FORBIDDEN",
    "expert": "FORBIDDEN",
}


def transition_from_anza_and_connectivity(
    raw_anza_weights: torch.Tensor,
    connectivity: torch.Tensor,
    *,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """Build an eight-neighbour row-stochastic transition tensor."""

    if raw_anza_weights.ndim != 4 or raw_anza_weights.shape[2] != 9:
        raise ValueError("raw ANZA weights must have shape B,R,9,L")
    batch, _rules, _support, locations = raw_anza_weights.shape
    if connectivity.ndim != 4 or connectivity.shape[:2] != (batch, 8):
        raise ValueError("connectivity must have shape B,8,H,W")
    height, width = connectivity.shape[-2:]
    if height * width != locations:
        raise ValueError("ANZA locations and connectivity image shape disagree")
    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("connectivity epsilon must be in [0, 1]")
    if not torch.isfinite(raw_anza_weights).all() or not torch.isfinite(connectivity).all():
        raise ValueError("transition inputs must be finite")
    if torch.any(raw_anza_weights < 0) or torch.any((connectivity < 0) | (connectivity > 1)):
        raise ValueError("transition inputs must be nonnegative and connectivity bounded")
    positions = [index for index in range(9) if index != 4]
    base = raw_anza_weights[:, :, positions].sum(dim=1).reshape(batch, 8, height, width)
    conductance = base * (float(epsilon) + (1.0 - float(epsilon)) * connectivity)
    denominator = conductance.sum(dim=1, keepdim=True)
    if torch.any(denominator <= 0):
        raise ValueError("every pixel must have positive outgoing conductance")
    transition = conductance / denominator
    if not torch.allclose(transition.sum(dim=1), torch.ones_like(denominator[:, 0]), atol=1e-6):
        raise AssertionError("transition is not row stochastic")
    return transition


def diffusion_step(h0: torch.Tensor, state: torch.Tensor, transition: torch.Tensor, *, alpha: float) -> torch.Tensor:
    if h0.shape != state.shape or h0.ndim != 4 or h0.shape[1] != 1:
        raise ValueError("h0 and state must be matching B,1,H,W tensors")
    if transition.shape != (h0.shape[0], 8, h0.shape[2], h0.shape[3]):
        raise ValueError("transition must be B,8,H,W")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    propagated = torch.zeros_like(state)
    for channel, (dx, dy) in enumerate(LOCAL8_OFFSETS):
        neighbor, _valid = _shift_tensor(state, int(dx), int(dy))
        propagated = propagated + transition[:, channel : channel + 1] * neighbor
    result = (1.0 - float(alpha)) * h0 + float(alpha) * propagated
    if not torch.isfinite(result).all():
        raise ValueError("diffusion produced NaN or Inf")
    return result


def diffuse(h0: torch.Tensor, transition: torch.Tensor, *, steps: int, alpha: float) -> torch.Tensor:
    if int(steps) < 0:
        raise ValueError("steps must be nonnegative")
    state = h0
    for _ in range(int(steps)):
        state = diffusion_step(h0, state, transition, alpha=float(alpha))
    return state


def _dice(prediction: np.ndarray, truth: np.ndarray) -> float:
    predicted = np.asarray(prediction, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    denominator = int(predicted.sum() + target.sum())
    return 2.0 * int((predicted & target).sum()) / denominator if denominator else 1.0


def completion_gate_metrics(
    probabilities: np.ndarray,
    samples: Iterable[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("oracle evaluation requires samples")
    values = np.asarray(probabilities, dtype=np.float32)
    expected = (len(sample_list),) + tuple(np.asarray(sample_list[0]["visible_fault_mask"]).shape)
    if values.shape != expected:
        raise ValueError("probabilities must match samples")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("oracle probabilities must be finite in [0, 1]")
    predictions = values >= float(threshold)
    visible_dice = [
        _dice(prediction, sample["visible_fault_mask"])
        for prediction, sample in zip(predictions, sample_list)
    ]
    visible_cldice = [
        _cldice(prediction, np.asarray(sample["visible_fault_mask"], dtype=bool))
        for prediction, sample in zip(predictions, sample_list)
    ]
    gap_recovery: list[float] = []
    false_bridges = 0
    negative_gaps = 0
    for prediction, sample in zip(predictions, sample_list):
        gap = np.asarray(sample["positive_gap_mask"], dtype=bool)
        if gap.any():
            gap_recovery.append(float(np.logical_and(prediction, gap).sum() / gap.sum()))
        rate, bridge_count, gap_count = _false_bridge_rate(prediction, sample, 0.50)
        if gap_count:
            if not math.isclose(rate, bridge_count / gap_count):
                raise AssertionError("false-bridge aggregation drift")
            false_bridges += int(bridge_count)
            negative_gaps += int(gap_count)
    return {
        "sample_count": len(sample_list),
        "visible_dice": float(np.mean(visible_dice)),
        "visible_cldice": float(np.mean(visible_cldice)),
        "gap_recovery_rate": float(np.mean(gap_recovery)) if gap_recovery else 1.0,
        "positive_gap_count": len(gap_recovery),
        "false_bridge_rate": false_bridges / negative_gaps if negative_gaps else 0.0,
        "false_bridge_count": int(false_bridges),
        "negative_gap_count": int(negative_gaps),
    }


def _load_frozen_clean_anza(root: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    clean_spec = affinity_matrix()[1]
    run_dir = root / "results" / "affinity_repair" / "development" / f"C1-{clean_spec.run_hash}"
    status = json.loads((run_dir / "status.json").read_text())
    if status.get("status") != "COMPLETE" or status.get("candidate_id") != "C1":
        raise ValueError("frozen C1 checkpoint is not complete")
    for field, expected in (
        ("expert_data_accessed", False),
        ("legacy_test_samples_opened", 0),
        ("v4_test_samples_opened", 0),
        ("cracks_samples_opened", 0),
    ):
        if status.get(field) != expected:
            raise ValueError(f"frozen C1 lock violation: {field}")
    checkpoint = run_dir / "checkpoint-last.pt"
    model = build_candidate(clean_spec, widths=tuple(int(value) for value in status["widths"]))
    load_checkpoint(checkpoint, spec=clean_spec, model=model, clean_checkpoint_sha256=None)
    model.to(device).eval()
    validation = json.loads(
        (root / "results" / "affinity_repair" / "validation" / f"C1-{clean_spec.run_hash}.json").read_text()
    )
    return model, {
        "candidate_id": "C1",
        "run_hash": clean_spec.run_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "visible_threshold": float(validation["selected_visible_threshold"]),
        "image_size": int(status["image_size"]),
    }


def run_diffusion_oracle(
    root: Path,
    *,
    device: str = "cuda",
    sample_count: int = 512,
    batch_size: int = 8,
    t_values: tuple[int, ...] = ORACLE_T_VALUES,
    alpha_values: tuple[float, ...] = ORACLE_ALPHA_VALUES,
) -> dict[str, Any]:
    root = Path(root)
    torch_device = torch.device(device)
    model, frozen = _load_frozen_clean_anza(root, torch_device)
    image_size = int(frozen["image_size"])
    samples = [generate_sample_v5("validation", index, image_size=image_size) for index in range(int(sample_count))]
    h0_batches: list[torch.Tensor] = []
    transition_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(samples), int(batch_size)):
            batch_samples = samples[start : start + int(batch_size)]
            images = torch.stack([torch.as_tensor(sample["image"]) for sample in batch_samples]).to(torch_device)
            h0 = torch.sigmoid(model(images))
            raw, _valid, _mu, _gap, _interp = model.enc1.spatial._base_terms(images)
            connectivity = torch.stack([
                torch.as_tensor(build_affinity_targets(sample, LOCAL8_OFFSETS)["affinity_positive"])
                for sample in batch_samples
            ]).to(device=torch_device, dtype=h0.dtype)
            transition = transition_from_anza_and_connectivity(
                raw,
                connectivity,
                epsilon=float(ORACLE_PROTOCOL["connectivity_epsilon"]),
            )
            h0_batches.append(h0.cpu())
            transition_batches.append(transition.cpu())
    h0_all = torch.cat(h0_batches)
    transition_all = torch.cat(transition_batches)
    threshold = float(frozen["visible_threshold"])
    baseline = completion_gate_metrics(h0_all[:, 0].numpy(), samples, threshold=threshold)
    rows: list[dict[str, Any]] = []
    for steps in t_values:
        for alpha in alpha_values:
            states: list[torch.Tensor] = []
            for start in range(0, len(samples), int(batch_size)):
                h0 = h0_all[start : start + int(batch_size)].to(torch_device)
                transition = transition_all[start : start + int(batch_size)].to(torch_device)
                states.append(diffuse(h0, transition, steps=int(steps), alpha=float(alpha)).cpu())
            metrics = completion_gate_metrics(
                torch.cat(states)[:, 0].numpy(), samples, threshold=threshold
            )
            visible_loss = float(baseline["visible_dice"] - metrics["visible_dice"])
            checks = {
                "gap_recovery": metrics["gap_recovery_rate"] >= float(ORACLE_PROTOCOL["gates"]["gap_recovery_min"]),
                "false_bridge": metrics["false_bridge_rate"] <= float(ORACLE_PROTOCOL["gates"]["false_bridge_max"]),
                "visible_dice_safety": visible_loss <= float(ORACLE_PROTOCOL["gates"]["visible_dice_loss_max"]),
            }
            rows.append({
                "steps": int(steps),
                "alpha": float(alpha),
                **metrics,
                "visible_dice_loss": visible_loss,
                "all_gates_pass": all(checks.values()),
                **{f"check_{name}": value for name, value in checks.items()},
            })
    passing = [row for row in rows if row["all_gates_pass"]]
    selected = min(passing, key=lambda row: (row["steps"], row["alpha"])) if passing else None
    return {
        "status": "GT_CONNECTIVITY_DIFFUSION_ORACLE_PASS" if selected else "DIFFUSION_OPERATOR_INSUFFICIENT",
        "protocol": ORACLE_PROTOCOL,
        "frozen_clean_anza": frozen,
        "baseline": baseline,
        "rows": rows,
        "selected": {"steps": selected["steps"], "alpha": selected["alpha"]} if selected else None,
        "sample_count": len(samples),
        "test_v5_samples_opened": 0,
        "expert_data_accessed": False,
        "cracks_samples_opened": 0,
    }


def write_diffusion_oracle(output_root: Path, *, root: Path, device: str = "cuda") -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result = run_diffusion_oracle(root, device=device)
    (output_root / "gt_connectivity_diffusion_oracle.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    with (output_root / "gt_connectivity_diffusion_oracle.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    lines = [
        "# GT-connectivity diffusion oracle",
        "",
        f"Status: `{result['status']}`",
        "",
        "This is a validation-only feasibility upper bound using latent generator lineage, not a trained-model result.",
        "",
        f"Frozen C1 checkpoint SHA256: `{result['frozen_clean_anza']['checkpoint_sha256']}`",
        f"Frozen visible threshold: `{result['frozen_clean_anza']['visible_threshold']}`",
        "",
        "| T | alpha | visible Dice | visible clDice | gap recovery | false bridge | Dice loss | PASS |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['steps']} | {row['alpha']:.1f} | {row['visible_dice']:.4f} | "
            f"{row['visible_cldice']:.4f} | {row['gap_recovery_rate']:.4f} | "
            f"{row['false_bridge_rate']:.4f} | {row['visible_dice_loss']:.4f} | "
            f"{'yes' if row['all_gates_pass'] else 'no'} |"
        )
    lines.extend(["", f"Frozen selection: `{result['selected']}`."])
    (output_root / "GT_CONNECTIVITY_DIFFUSION_ORACLE.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    payload = write_diffusion_oracle(
        repository / "results" / "connectivity_repair" / "pretraining",
        root=repository,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))

