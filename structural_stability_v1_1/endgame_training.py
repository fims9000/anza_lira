"""Frozen SS2/SS3 training for the V1.1 twelve-run matrix."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import threading
import time
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from cracks_experiment.partial_labels import map_partial_annotation
from datasets.cracks import load_section_image
from lira_final.protocol import TRAIN_ANNOTATORS
from structural_stability_v1.perturb import apply_perturbation, transform_rgb_mask
from structural_stability_v1.agreement import crowd_agreement
from structural_stability_v1_1.amendment import sha256_file
from structural_stability_v1_1.geometry_targets import geometry_target
from structural_stability_v1_1.initialization import state_dict_sha256
from structural_stability_v1_1.metric_transport import (
    forward_jacobian_xy,
    transport_metric,
)
from structural_stability_v1_1.protocol import (
    PROTOCOL,
    RESULT_ROOT,
    ROOT,
    SEEDS,
    VARIANTS,
    protocol_hash,
)
from structural_stability_v1_1.train_variants import build_fresh_variant
import utils


EXPECTED_SPLIT_SHA = "43a3fb7716d5ff9e56c7da9a78f2127c20f8d13ba27d7e5576ac493176045671"
EXPECTED_NORMALIZATION_SHA = "013b16cc61ee8e1bc34a3221c5e7c26576e7dde8b4955e51adc65cc45f008630"
EXPECTED_MANIFESTS = {
    41: "188819937660d8412c0ff5c3551f4bd062c4af301a525d9f614ce99385245afe",
    42: "2864655d868ab97544b4f6d4510330031654ea9159d34212a720171afcf9fb54",
    43: "5e892d61ba1a99f22a8f0d7e2fc84ff7b99d3a8f1a101babeeb74f97f15e8087",
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": gpu,
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "pid": os.getpid(),
        "precision": "FP32",
        "deterministic_algorithms": True,
    }


def training_preflight(variant: str, seed: int, run_dir: Path) -> dict[str, Any]:
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("job is outside the frozen matrix")
    pretrain = RESULT_ROOT / "pretrain_freeze"
    validator = json.loads((pretrain / "validator.json").read_text())
    split = json.loads((RESULT_ROOT.parent / "anza_lira_ss_v1/s0_audit/split_manifest.json").read_text())
    normalization_path = pretrain / "TRAIN_ONLY_NORMALIZATION.json"
    normalization = json.loads(normalization_path.read_text())
    manifest_path = pretrain / f"manifests/train_pair_manifest_seed{seed}.jsonl"
    init_path = pretrain / f"initialization/backbone_init_s{seed}.pt"
    checks = {
        "ss1_5_pass": validator.get("status") == "SS1_5_PRETRAINING_FREEZE_PASS",
        "protocol": validator.get("protocol_sha256") == protocol_hash(),
        "split": split.get("sha256") == EXPECTED_SPLIT_SHA,
        "normalization": normalization.get("sha256") == EXPECTED_NORMALIZATION_SHA,
        "manifest": sha256_file(manifest_path) == EXPECTED_MANIFESTS[seed],
        "manifest_rows": sum(1 for _ in manifest_path.open()) == 7920,
        "init_exists": init_path.is_file(),
        "development_lock": not (RESULT_ROOT / "SS_DEVELOPMENT_AUTHORIZATION.json").exists(),
        "confirm_lock": not (RESULT_ROOT / "SS_V1_1_CONFIRM_AUTHORIZATION.json").exists(),
        "expert_lock": not (RESULT_ROOT / "expert_descriptive/EXPERT_ACCESS.json").exists(),
    }
    if "smoke" in run_dir.parts:
        checks["output_resumable"] = True
    elif run_dir.exists() and any(run_dir.iterdir()):
        checks["output_resumable"] = (run_dir / "checkpoint_recovery.pt").is_file() or (run_dir / "RUN_FINAL_VALIDATION.json").is_file()
    else:
        checks["output_resumable"] = True
    payload = {
        "status": "RUN_PREFLIGHT_PASS" if all(checks.values()) else "STOP_EXECUTION_PREFLIGHT_INTEGRITY",
        "variant": variant,
        "seed": seed,
        "checks": checks,
        "protocol_sha256": protocol_hash(),
        "split_sha256": split.get("sha256"),
        "normalization_sha256": normalization.get("sha256"),
        "manifest_sha256": sha256_file(manifest_path),
        "historical_H0_loaded": False,
        "severity3_training": False,
        "development_opened": False,
        "confirm_opened": False,
        "expert_opened": False,
    }
    if payload["status"] != "RUN_PREFLIGHT_PASS":
        raise PermissionError(json.dumps(payload, sort_keys=True))
    return payload


class _SectionCache:
    def __init__(self, normalization: dict[str, Any], variant: str, limit: int = 20) -> None:
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(normalization["std"], dtype=np.float32)[:, None, None]
        self.limit = limit
        self.variant = variant
        self.values: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.geometry_values: dict[int, dict[str, np.ndarray]] = {}
        self.lock = threading.Lock()

    def get(self, section_id: int) -> dict[str, Any]:
        with self.lock:
            if section_id in self.values:
                value = self.values.pop(section_id)
                self.values[section_id] = value
                return value
        name = f"section_{section_id:03d}.png"
        image = load_section_image(ROOT / "data/cracks/images" / name).transpose(2, 0, 1)
        image = ((image - self.mean) / np.maximum(self.std, 1e-6)).astype(np.float32)
        image = np.pad(image, ((0, 0), (0, 1), (0, 3)))
        masks: dict[str, np.ndarray] = {}
        all_masks = []
        for annotator in TRAIN_ANNOTATORS:
            path = ROOT / "data/cracks/annotations" / annotator / name
            if path.is_file():
                with Image.open(path) as handle:
                    mask = np.asarray(handle.convert("RGB"), dtype=np.uint8)
                masks[annotator] = mask
                all_masks.append(mask)
        with self.lock:
            geo = self.geometry_values.get(section_id)
        if geo is None:
            if self.variant == "B0":
                geo = {}
            elif self.variant == "B1":
                geo = {"agreement": crowd_agreement(all_masks)["agreement"]}
            else:
                computed = geometry_target(all_masks)
                geo = {key: computed[key] for key in ("target_c2", "target_s2", "target_d", "geometry_weight", "agreement")}
            with self.lock:
                self.geometry_values.setdefault(section_id, geo)
                geo = self.geometry_values[section_id]
        value = {"image": image, "masks": masks, "geometry": geo}
        with self.lock:
            existing = self.values.pop(section_id, None)
            self.values[section_id] = existing if existing is not None else value
            while len(self.values) > self.limit:
                self.values.popitem(last=False)
            return self.values[section_id]


def _crop_mask(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    padded = np.pad(mask, ((0, 1), (0, 3), (0, 0)), constant_values=255)
    return padded[top : top + 256, left : left + 256]


def _crop_field(field: np.ndarray, top: int, left: int) -> np.ndarray:
    padded = np.pad(field, ((0, 1), (0, 3)), mode="edge")
    return padded[top : top + 256, left : left + 256]


def _row_sample(row: dict[str, Any], cache: _SectionCache) -> dict[str, Any]:
    section_id = int(row["section_id"])
    top, left = int(row["crop_top"]), int(row["crop_left"])
    section = cache.get(section_id)
    image = section["image"][:, top : top + 256, left : left + 256]
    clean_masks = [_crop_mask(section["masks"][name], top, left) for name in row["annotators"]]
    crop_id = f"e{int(row['epoch']) - 1:02d}_p{int(row['order_position']):03d}_s{section_id:03d}_y{top}_x{left}"
    pert = apply_perturbation(image, section_id, crop_id, str(row["family"]), int(row["severity"]))
    if pert.seed != int(row["perturbation_seed"]):
        raise ValueError("perturbation seed drift")
    pert_masks = [transform_rgb_mask(mask, pert) for mask in clean_masks]

    def partial(masks: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        mapped = [map_partial_annotation(mask) for mask in masks]
        targets = torch.from_numpy(np.stack([item[0] for item in mapped])).unsqueeze(1)
        weights = torch.from_numpy(np.stack([item[1] for item in mapped])).unsqueeze(1)
        return targets, weights

    clean_targets, clean_weights = partial(clean_masks)
    pert_targets, pert_weights = partial(pert_masks)
    geo = section["geometry"]
    geometry = {
        key: torch.from_numpy(_crop_field(np.asarray(geo[key], dtype=np.float32), top, left))
        for key in geo
    }
    if pert.family == "warp":
        displacement = np.asarray(pert.displacement_yx, dtype=np.float32)
    else:
        displacement = np.zeros((2, 256, 256), dtype=np.float32)
    return {
        "clean_image": torch.from_numpy(np.ascontiguousarray(image)),
        "pert_image": torch.from_numpy(np.ascontiguousarray(pert.image)),
        "clean_targets": clean_targets,
        "clean_weights": clean_weights,
        "pert_targets": pert_targets,
        "pert_weights": pert_weights,
        "geometry": geometry,
        "displacement": displacement,
        "family": pert.family,
        "section_id": section_id,
    }


def _sample_tensor(tensor: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Sample clean BxCxHxW at output pixels using output-to-input displacement."""
    batch, _channels, height, width = tensor.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=tensor.device, dtype=tensor.dtype),
        torch.arange(width, device=tensor.device, dtype=tensor.dtype),
        indexing="ij",
    )
    dy = F.interpolate(displacement[:, 0:1], size=(height, width), mode="bilinear", align_corners=True)[:, 0]
    dx = F.interpolate(displacement[:, 1:2], size=(height, width), mode="bilinear", align_corners=True)[:, 0]
    dy = dy * (height / displacement.shape[-2])
    dx = dx * (width / displacement.shape[-1])
    x = xx[None] + dx
    y = yy[None] + dy

    def reflect(coordinates: torch.Tensor, size: int) -> torch.Tensor:
        if size <= 1:
            return torch.zeros_like(coordinates)
        period = 2.0 * (size - 1)
        folded = torch.remainder(coordinates, period)
        return torch.where(folded <= size - 1, folded, period - folded)

    x = reflect(x, width); y = reflect(y, height)
    x0 = torch.floor(x).long(); y0 = torch.floor(y).long()
    x1 = torch.clamp(x0 + 1, max=width - 1); y1 = torch.clamp(y0 + 1, max=height - 1)
    wx = (x - x0.to(x.dtype)).unsqueeze(1); wy = (y - y0.to(y.dtype)).unsqueeze(1)
    flat = tensor.reshape(batch, tensor.shape[1], height * width)

    def gather(yi: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        index = (yi * width + xi).reshape(batch, 1, height * width).expand(-1, tensor.shape[1], -1)
        return torch.gather(flat, 2, index).reshape(batch, tensor.shape[1], height, width)

    v00, v01 = gather(y0, x0), gather(y0, x1)
    v10, v11 = gather(y1, x0), gather(y1, x1)
    return (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11)


def _bernoulli_js(p: torch.Tensor, q: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    middle = 0.5 * (p + q)
    kl_p = p * torch.log(p / middle) + (1 - p) * torch.log((1 - p) / (1 - middle))
    kl_q = q * torch.log(q / middle) + (1 - q) * torch.log((1 - q) / (1 - middle))
    return ((0.5 * (kl_p + kl_q)) * weight).sum() / (weight.sum() + eps)


def _topology_consistency(p: torch.Tensor, q: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p * weight
    q = q * weight
    skel_p = utils._soft_skeletonize(p, num_iters=5)
    skel_q = utils._soft_skeletonize(q, num_iters=5)
    precision = (skel_p * q).sum() / (skel_p.sum() + eps)
    recall = (skel_q * p).sum() / (skel_q.sum() + eps)
    return 1.0 - (2 * precision * recall + eps) / (precision + recall + eps)


def _partial_supervision(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Exact per-annotator BCE+Dice+0.2*softClDice, vectorized then averaged."""
    if logits.shape != targets.shape or targets.shape != weights.shape:
        raise ValueError("partial supervision tensors must have identical Nx1xHxW shapes")
    dimensions = (1, 2, 3)
    bce_pixels = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weights
    bce = bce_pixels.sum(dim=dimensions) / weights.sum(dim=dimensions).clamp_min(1.0)
    probabilities = torch.sigmoid(logits) * weights
    weighted_targets = targets * weights
    dice = 1.0 - (2.0 * (probabilities * weighted_targets).sum(dim=dimensions) + eps) / (
        probabilities.sum(dim=dimensions) + weighted_targets.sum(dim=dimensions) + eps
    )
    skel_pred = utils._soft_skeletonize(probabilities, num_iters=5)
    skel_target = utils._soft_skeletonize(weighted_targets, num_iters=5)
    topology_precision = (skel_pred * weighted_targets).sum(dim=dimensions) / (skel_pred.sum(dim=dimensions) + eps)
    topology_recall = (skel_target * probabilities).sum(dim=dimensions) / (skel_target.sum(dim=dimensions) + eps)
    cldice_loss = 1.0 - (2.0 * topology_precision * topology_recall + eps) / (topology_precision + topology_recall + eps)
    return (bce + dice + 0.2 * cldice_loss).mean()


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (value * weight).sum() / (weight.sum() + eps)


def _spd_log_2x2(metric: torch.Tensor, minimum: float = 1e-4, maximum: float = 1e4) -> torch.Tensor:
    """Analytic 2x2 SPD logarithm, equivalent to eigh without batched cuSOLVER."""
    a, b, d = metric[:, 0, 0], metric[:, 0, 1], metric[:, 1, 1]
    half_trace = 0.5 * (a + d)
    radius = torch.sqrt((0.5 * (a - d)).square() + b.square()).clamp_min(0)
    high = (half_trace + radius).clamp(min=minimum, max=maximum)
    low = (half_trace - radius).clamp(min=minimum, max=maximum)
    gap = high - low
    log_high, log_low = torch.log(high), torch.log(low)
    alpha = torch.where(gap > 1e-6, (log_high - log_low) / gap.clamp_min(1e-12), 1.0 / half_trace.clamp(min=minimum, max=maximum))
    beta = torch.where(gap > 1e-6, (high * log_low - low * log_high) / gap.clamp_min(1e-12), torch.log(half_trace.clamp(min=minimum, max=maximum)) - alpha * half_trace)
    output = torch.empty_like(metric)
    output[:, 0, 0] = alpha * a + beta
    output[:, 0, 1] = alpha * b
    output[:, 1, 0] = alpha * b
    output[:, 1, 1] = alpha * d + beta
    return output


def _metric_equivariance_loss(predicted: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise FloatingPointError("non-finite SPD metric before matrix log")
    difference = _spd_log_2x2(predicted.float()) - _spd_log_2x2(target.detach().float())
    squared = difference.square().sum(dim=(1, 2))
    return (weight.float() * squared).sum() / (weight.float().sum() + eps)


def _geometry_losses(clean_geo: list[Any], pert_geo: list[Any], samples: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis_losses, strength_losses, equiv_losses = [], [], []
    displacement = torch.from_numpy(np.stack([sample["displacement"] for sample in samples])).to(device)
    for clean, pert in zip(clean_geo, pert_geo):
        size = clean.d.shape[-2:]
        c2 = torch.stack([sample["geometry"]["target_c2"] for sample in samples]).unsqueeze(1).to(device)
        s2 = torch.stack([sample["geometry"]["target_s2"] for sample in samples]).unsqueeze(1).to(device)
        d_target = torch.stack([sample["geometry"]["target_d"] for sample in samples]).unsqueeze(1).to(device)
        weight = torch.stack([sample["geometry"]["geometry_weight"] for sample in samples]).unsqueeze(1).to(device)
        target_cs = F.interpolate(torch.cat((c2, s2), dim=1), size=size, mode="bilinear", align_corners=False)
        target_cs = target_cs / torch.sqrt(target_cs.square().sum(1, keepdim=True) + 1e-8)
        d_target = F.interpolate(d_target, size=size, mode="bilinear", align_corners=False)[:, 0]
        weight_s = F.interpolate(weight, size=size, mode="area")[:, 0]
        axis = 1.0 - (clean.c2 * target_cs[:, 0] + clean.s2 * target_cs[:, 1])
        axis_losses.append(_weighted_mean(axis, weight_s))
        strength_losses.append(_weighted_mean(F.smooth_l1_loss(clean.d, d_target, reduction="none"), weight_s))

        clean_metric = clean.metric.reshape(clean.metric.shape[0], 4, *size)
        sampled = _sample_tensor(clean_metric, displacement).reshape(clean.metric.shape[0], 2, 2, *size)
        jacobians = []
        for sample in samples:
            if sample["family"] == "warp":
                forward = forward_jacobian_xy(sample["displacement"])
            else:
                forward = np.broadcast_to(np.eye(2), (256, 256, 2, 2)).copy()
            jacobians.append(forward)
        jac = torch.from_numpy(np.stack(jacobians).astype(np.float32)).to(device)
        jac = F.interpolate(jac.permute(0, 3, 4, 1, 2).reshape(len(samples), 4, 256, 256), size=size, mode="bilinear", align_corners=False)
        jac = jac.reshape(len(samples), 2, 2, *size).permute(0, 3, 4, 1, 2)
        target_metric = transport_metric(sampled.permute(0, 3, 4, 1, 2), jac).permute(0, 3, 4, 1, 2)
        pert_weight = _sample_tensor(weight, displacement)
        equiv_losses.append(_metric_equivariance_loss(pert.metric, target_metric, F.interpolate(pert_weight, size=size, mode="area")[:, 0]))
    return torch.stack(axis_losses).mean(), torch.stack(strength_losses).mean(), torch.stack(equiv_losses).mean()


def _batch_loss(model: torch.nn.Module, rows: list[dict[str, Any]], cache: _SectionCache, device: torch.device, executor: ThreadPoolExecutor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    samples = list(executor.map(lambda row: _row_sample(row, cache), rows)) if executor is not None else [_row_sample(row, cache) for row in rows]
    clean_images = torch.stack([sample["clean_image"] for sample in samples]).to(device)
    pert_images = torch.stack([sample["pert_image"] for sample in samples]).to(device)
    output = model(torch.cat((clean_images, pert_images)), return_geometry=model.variant in {"B2", "B3"})
    if isinstance(output, dict):
        logits = output["visible_logits"]
        geometry = output["geometry"]
    else:
        logits = output
        geometry = []
    count = len(samples)
    clean_logits, pert_logits = logits[:count], logits[count:]
    annotators = samples[0]["clean_targets"].shape[0]
    clean_target = torch.cat([sample["clean_targets"].to(device) for sample in samples])
    clean_weight = torch.cat([sample["clean_weights"].to(device) for sample in samples])
    pert_target = torch.cat([sample["pert_targets"].to(device) for sample in samples])
    pert_weight = torch.cat([sample["pert_weights"].to(device) for sample in samples])
    clean_repeated = clean_logits.repeat_interleave(annotators, dim=0)
    pert_repeated = pert_logits.repeat_interleave(annotators, dim=0)
    loss_sup = 0.5 * (_partial_supervision(clean_repeated, clean_target, clean_weight) + _partial_supervision(pert_repeated, pert_target, pert_weight))
    zero = loss_sup.new_zeros(())
    loss_prob = loss_topo = loss_axis = loss_strength = loss_equiv = zero
    if model.variant in {"B1", "B2", "B3"}:
        displacement = torch.from_numpy(np.stack([sample["displacement"] for sample in samples])).to(device)
        clean_p = torch.sigmoid(clean_logits)
        pert_p = torch.sigmoid(pert_logits)
        aligned_clean = _sample_tensor(clean_p, displacement)
        agreement = torch.stack([sample["geometry"]["agreement"] for sample in samples]).unsqueeze(1).to(device)
        aligned_weight = _sample_tensor(agreement, displacement).clamp(0, 1)
        loss_prob = _bernoulli_js(aligned_clean, pert_p, aligned_weight)
        loss_topo = _topology_consistency(aligned_clean, pert_p, aligned_weight)
    if model.variant in {"B2", "B3"}:
        clean_geo = [item[:count] if hasattr(item, "__getitem__") else item for item in ()]
        # GeometryOutput is a dataclass; split it explicitly.
        split_clean, split_pert = [], []
        for item in geometry:
            cls = type(item)
            split_clean.append(cls(item.c2[:count], item.s2[:count], item.d[:count], item.m[:count], item.metric[:count]))
            split_pert.append(cls(item.c2[count:], item.s2[count:], item.d[count:], item.m[count:], item.metric[count:]))
        loss_axis, loss_strength, loss_equiv = _geometry_losses(split_clean, split_pert, samples, device)
    total = loss_sup
    if model.variant in {"B1", "B2", "B3"}:
        total = total + 0.20 * loss_prob + 0.20 * loss_topo
    if model.variant in {"B2", "B3"}:
        total = total + 0.05 * loss_axis + 0.05 * loss_strength + 0.05 * loss_equiv
    logs = {
        "loss_total": float(total.detach()),
        "loss_sup": float(loss_sup.detach()),
        "loss_prob": float(loss_prob.detach()),
        "loss_topo": float(loss_topo.detach()),
        "loss_axis": float(loss_axis.detach()),
        "loss_strength": float(loss_strength.detach()),
        "loss_equiv": float(loss_equiv.detach()),
    }
    return total, logs


def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    values = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(values).norm(2)) if values else 0.0


def _checkpoint_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer, variant: str, seed: int, step: int, cursor: int) -> dict[str, Any]:
    return {
        "status": "IN_PROGRESS",
        "protocol_sha256": protocol_hash(),
        "variant": variant,
        "seed": seed,
        "optimizer_step": step,
        "manifest_cursor": cursor,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": _rng_state(),
        "historical_H0_loaded": False,
        "development_opened": False,
        "confirm_opened": False,
        "expert_opened": False,
    }


def run_training_job(variant: str, seed: int, *, device: str = "cuda", max_steps: int | None = None, smoke: bool = False) -> dict[str, Any]:
    run_dir = RESULT_ROOT / ("smoke" if smoke else "training") / variant / f"s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    preflight = training_preflight(variant, seed, run_dir)
    _write_json(run_dir / "RUN_PREFLIGHT.json", preflight)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    normalization = json.loads((RESULT_ROOT / "pretrain_freeze/TRAIN_ONLY_NORMALIZATION.json").read_text())
    manifest_path = RESULT_ROOT / f"pretrain_freeze/manifests/train_pair_manifest_seed{seed}.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    planned_steps = int(PROTOCOL["training"]["planned_optimizer_updates"])
    target_steps = min(planned_steps, int(max_steps)) if max_steps is not None else planned_steps
    model = build_fresh_variant(variant, seed, RESULT_ROOT / "pretrain_freeze/initialization").to(torch_device)
    initial_backbone_hash = state_dict_sha256(model.backbone.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(PROTOCOL["training"]["learning_rate"]))
    recovery = run_dir / "checkpoint_recovery.pt"
    start_step = 0
    cursor = 0
    if recovery.is_file() and not smoke:
        payload = torch.load(recovery, map_location="cpu", weights_only=False)
        required = (payload.get("protocol_sha256") == protocol_hash() and payload.get("variant") == variant and payload.get("seed") == seed)
        if not required:
            raise ValueError("recovery checkpoint provenance mismatch")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_step = int(payload["optimizer_step"])
        cursor = int(payload["manifest_cursor"])
        _restore_rng(payload["rng_state"])
    metadata = _runtime_metadata(torch_device)
    _write_json(run_dir / "NUMERICAL_EXECUTION_FREEZE.json", metadata)
    cache = _SectionCache(normalization, variant)
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ss-v1-1-pair")
    log_path = run_dir / "training_log.jsonl"
    if start_step and log_path.is_file():
        retained = [line for line in log_path.read_text().splitlines() if int(json.loads(line)["step"]) <= start_step]
        log_path.write_text("\n".join(retained) + ("\n" if retained else ""))
    started = time.time()
    peak_vram = 0
    start_validation: dict[str, Any] | None = None
    with log_path.open("a") as log_handle:
        for step in range(start_step, target_steps):
            batch_rows = rows[cursor : cursor + 4]
            if len(batch_rows) != 4:
                raise ValueError("manifest cursor did not yield effective batch four")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, logs = _batch_loss(model, batch_rows, cache, torch_device, executor)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            finite_grad = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
            backbone_grad = _grad_norm(model.backbone.parameters())
            sidecar_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("backbone.")]
            sidecar_grad = _grad_norm(sidecar_parameters)
            if not finite_grad or backbone_grad <= 0 or (variant in {"B2", "B3"} and sidecar_grad <= 0):
                raise FloatingPointError("invalid or missing gradient")
            if start_validation is None:
                metric_checks = {"B3_det_residual": None, "B2_logdet_std": None}
                if variant in {"B2", "B3"}:
                    with torch.no_grad():
                        model.eval()
                        probe = model(torch.stack([_row_sample(batch_rows[0], cache)["clean_image"]]).to(torch_device), return_geometry=True)
                        model.train()
                        metric = probe["geometry"][0].metric.permute(0, 3, 4, 1, 2)
                        determinant = torch.linalg.det(metric)
                        if variant == "B3":
                            metric_checks["B3_det_residual"] = float((determinant - 1).abs().max())
                        else:
                            metric_checks["B2_logdet_std"] = float(torch.log(determinant).std())
                start_validation = {
                    "status": "RUN_START_VALIDATION_PASS",
                    "initial_backbone_sha256": initial_backbone_hash,
                    "all_losses_finite": True,
                    "forward_backward_finite": finite_grad,
                    "grad_norm_backbone": backbone_grad,
                    "grad_norm_sidecar": sidecar_grad,
                    "historical_H0_loaded": False,
                    "development_confirm_expert_accessed": False,
                    **metric_checks,
                }
                _write_json(run_dir / "RUN_START_VALIDATION.json", start_validation)
            optimizer.step()
            cursor += 4
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated(torch_device) if torch_device.type == "cuda" else 0)
            row = {
                "step": step + 1,
                "epoch": int(batch_rows[0]["epoch"]),
                "manifest_row_start": cursor - 4,
                **logs,
                "grad_norm_backbone": backbone_grad,
                "grad_norm_sidecar": sidecar_grad,
                "finite": finite_grad,
                "elapsed_seconds": time.time() - started,
            }
            log_handle.write(json.dumps(row, sort_keys=True) + "\n")
            log_handle.flush()
            if not smoke and ((step + 1) % 100 == 0 or step + 1 == target_steps):
                torch.save(_checkpoint_payload(model, optimizer, variant, seed, step + 1, cursor), recovery)
            if (step + 1) % 10 == 0 or step + 1 == target_steps:
                print(f"phase=SS2_TRAIN variant={variant} seed={seed} step={step + 1}/{target_steps} loss={logs['loss_total']:.5f} dev=LOCKED confirm=LOCKED expert=LOCKED", flush=True)
    if smoke:
        executor.shutdown(wait=True)
        return {"status": "SMOKE_PASS", "variant": variant, "seed": seed, "steps": target_steps, "peak_vram_bytes": peak_vram}
    if target_steps != planned_steps:
        executor.shutdown(wait=True)
        return {"status": "IN_PROGRESS", "variant": variant, "seed": seed, "optimizer_step": target_steps}
    executor.shutdown(wait=True)
    final = _checkpoint_payload(model, optimizer, variant, seed, planned_steps, cursor)
    final["status"] = "COMPLETE"
    final_path = run_dir / "checkpoint_final.pt"
    torch.save(final, final_path)
    checkpoint_hash = sha256_file(final_path)
    cpu_model = build_fresh_variant(variant, seed, RESULT_ROOT / "pretrain_freeze/initialization")
    reload_payload = torch.load(final_path, map_location="cpu", weights_only=False)
    cpu_model.load_state_dict(reload_payload["model_state"])
    final_finite = all(torch.isfinite(value).all() for value in cpu_model.state_dict().values())
    final_metric_audit: dict[str, float | None] = {"B3_det_residual": None, "B2_m_parameter_abs_max": None}
    if variant == "B3":
        cpu_model.eval()
        with torch.no_grad():
            sample = _row_sample(rows[0], cache)["clean_image"].unsqueeze(0)
            geometry = cpu_model(sample, return_geometry=True)["geometry"]
            final_metric_audit["B3_det_residual"] = max(float((torch.linalg.det(item.metric.permute(0, 3, 4, 1, 2)) - 1).abs().max()) for item in geometry)
    elif variant == "B2":
        m_parameters = []
        for sidecar in (cpu_model.geometry_quarter, cpu_model.geometry_half):
            m_parameters.extend((sidecar.output.weight[3].detach().abs().max(), sidecar.output.bias[3].detach().abs()))
        final_metric_audit["B2_m_parameter_abs_max"] = float(torch.stack(m_parameters).max()) if m_parameters else 0.0
    validation = {
        "status": "RUN_FINAL_VALIDATION_PASS" if final_finite and cursor == 7920 else "STOP_TRAINING_FINAL_VALIDATION",
        "variant": variant,
        "seed": seed,
        "optimizer_step": planned_steps,
        "epoch": 36,
        "manifest_rows_consumed": cursor,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_cpu_reload": True,
        "final_state_finite": final_finite,
        "historical_H0_loaded": False,
        "development_opened": False,
        "confirm_opened": False,
        "expert_opened": False,
        "runtime_seconds": time.time() - started,
        "peak_vram_bytes": peak_vram,
        **final_metric_audit,
    }
    _write_json(run_dir / "RUN_FINAL_VALIDATION.json", validation)
    (run_dir / "checkpoint_final.sha256").write_text(checkpoint_hash + "\n")
    (run_dir / "TRAINING_REPORT.md").write_text(
        f"# {variant} seed {seed}\n\n- Final step: `1980`.\n- Checkpoint: `{checkpoint_hash}`.\n- Runtime seconds: `{validation['runtime_seconds']:.1f}`.\n- Peak VRAM bytes: `{peak_vram}`.\n- Development/confirm/expert: locked.\n\n{validation['status']}\n"
    )
    return validation


def complete_training_manifest() -> dict[str, Any]:
    records = []
    hashes = set()
    for seed in SEEDS:
        for variant in VARIANTS:
            path = RESULT_ROOT / f"training/{variant}/s{seed}/RUN_FINAL_VALIDATION.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            row = json.loads(path.read_text())
            if row.get("status") != "RUN_FINAL_VALIDATION_PASS" or row.get("optimizer_step") != 1980:
                raise ValueError(f"invalid final run: {variant} seed {seed}")
            hashes.add(row["checkpoint_sha256"])
            records.append(row)
    status = "SS2_SS3_TRAINING_COMPLETE" if len(records) == 12 and len(hashes) == 12 else "STOP_TWELVE_RUN_COMPLETION"
    payload = {"status": status, "jobs_planned": 12, "jobs_completed": len(records), "unique_checkpoint_hashes": len(hashes), "records": records, "development_opened": False, "confirm_opened": False, "expert_opened": False}
    _write_json(RESULT_ROOT / "TWELVE_RUN_COMPLETION_MANIFEST.json", payload)
    return payload
