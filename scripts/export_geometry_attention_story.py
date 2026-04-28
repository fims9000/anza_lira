#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils
from models.azconv import AZConv2d
from utils import build_model


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint at {path} must be a mapping.")
    return payload


def _find_run(results_dir: Path, run_name: str | None, variant: str | None) -> Path:
    if run_name:
        candidate = results_dir / run_name
        if not (candidate / "checkpoint_best.pt").exists():
            raise FileNotFoundError(f"Run not found or missing checkpoint: {candidate}")
        return candidate
    if not variant:
        raise ValueError("Either --run or --variant must be provided.")
    best: tuple[float, Path] | None = None
    for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        metrics_path = run_dir / "metrics.json"
        checkpoint_path = run_dir / "checkpoint_best.pt"
        if not metrics_path.exists() or not checkpoint_path.exists():
            continue
        metrics = _load_json(metrics_path)
        if str(metrics.get("variant")) != variant:
            continue
        score = float(metrics.get("test_dice", -1.0))
        if best is None or score > best[0]:
            best = (score, run_dir)
    if best is None:
        raise FileNotFoundError(f"No run found for variant={variant} under {results_dir}")
    return best[1]


def _load_model(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], str]:
    payload = _load_checkpoint(run_dir / "checkpoint_best.pt")
    cfg = payload.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError(f"Checkpoint in {run_dir} does not contain cfg.")
    metrics = _load_json(run_dir / "metrics.json")
    variant = str(payload.get("variant", metrics.get("variant", "baseline")))
    in_channels, num_outputs = utils.dataset_channels_and_outputs(cfg["dataset"])
    task = utils.task_for_dataset(cfg["dataset"])
    az_overrides = utils.resolve_azconv_config_kwargs(cfg)

    def _build_with_overrides(extra: dict[str, Any] | None = None) -> torch.nn.Module:
        merged = dict(az_overrides)
        if extra:
            merged.update(extra)
        return build_model(
            variant,
            num_outputs=num_outputs,
            in_channels=in_channels,
            num_rules=int(cfg.get("num_rules", 4)),
            task=task,
            widths=utils.parse_model_widths(cfg.get("model_widths")),
            model_kwargs=utils.resolve_segmentation_model_kwargs(cfg),
            az_cfg_kwargs=merged,
        ).to(device)

    model = _build_with_overrides()
    state = payload["model"]
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        # Backward compatibility:
        # old az_thesis checkpoints may use fixed_cat_map with raw_sigma_* params,
        # while newer defaults use local_hyperbolic with geometry_conv.
        state_keys = set(state.keys())
        has_old_sigma = any(key.endswith("raw_sigma_u") or key.endswith("raw_sigma_s") for key in state_keys)
        has_local_head = any("geometry_conv.weight" in key or "geometry_conv.bias" in key for key in state_keys)
        if variant == "az_thesis" and has_old_sigma and not has_local_head:
            legacy = _build_with_overrides({"geometry_mode": "fixed_cat_map", "learn_directions": False})
            legacy.load_state_dict(state)
            model = legacy
            cfg.setdefault("az_geometry_mode", "fixed_cat_map")
            cfg.setdefault("az_learn_directions", False)
        else:
            raise exc
    model.eval()
    return model, cfg, metrics, variant


def _build_eval_dataset(cfg: dict[str, Any]) -> tuple[Any, str]:
    dataset_name = utils.canonical_dataset_name(str(cfg["dataset"]))
    if dataset_name in utils.RETINAL_SEG_DATASETS:
        root = (PROJECT_ROOT / utils.retinal_dataset_root(cfg.get("data_root", "./data"), cfg["dataset"])).resolve()
        dataset = utils.DriveDataset(
            root=root,
            split="test",
            augment=False,
            use_fov_mask=bool(cfg.get("use_fov_mask", True)),
        )
        return dataset, dataset_name

    if dataset_name in utils.ARCADE_SEG_DATASETS:
        data_root = (PROJECT_ROOT / utils.arcade_dataset_root(cfg.get("data_root", "./data"))).resolve()
        objective = "stenosis" if dataset_name == "arcade_stenosis" else "syntax"
        if dataset_name == "arcade_syntax":
            objective = "syntax"
        elif dataset_name == "arcade_stenosis":
            objective = "stenosis"
        image_size = utils._parse_optional_hw(cfg.get("arcade_image_size"))
        dataset = utils.ArcadeVesselDataset(
            root=data_root,
            split="test",
            objective=objective,
            augment=False,
            crop_size=None,
            image_size=image_size,
        )
        return dataset, dataset_name

    if dataset_name in utils.GIS_SEG_DATASETS:
        data_root = (PROJECT_ROOT / utils.gis_dataset_root(cfg.get("data_root", "./data"), cfg["dataset"])).resolve()
        image_size = utils._parse_optional_hw(cfg.get("gis_image_size", cfg.get("road_image_size")))
        mask_downsample_mode = str(cfg.get("gis_mask_downsample_mode", "nearest"))
        mask_downsample_threshold = float(cfg.get("gis_mask_downsample_threshold", 0.5))

        if dataset_name in utils.GLOBAL_ROAD_DATASETS and (data_root / "train").exists():
            test_dir_name = str(cfg.get("gis_test_split", "in-domain-test"))
            dataset = utils.GISRoadDataset(
                root=data_root / test_dir_name,
                augment=False,
                crop_size=None,
                image_size=image_size,
                mask_downsample_mode=mask_downsample_mode,
                mask_downsample_threshold=mask_downsample_threshold,
            )
            return dataset, dataset_name

        dataset = utils.GISRoadDataset(
            root=data_root,
            augment=False,
            crop_size=None,
            image_size=image_size,
            mask_downsample_mode=mask_downsample_mode,
            mask_downsample_threshold=mask_downsample_threshold,
        )
        return dataset, dataset_name

    raise ValueError(f"Unsupported dataset for segmentation story: {cfg['dataset']}")


def _denorm_to_uint8(x_norm: torch.Tensor) -> np.ndarray:
    arr = x_norm.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    arr = arr * STD + MEAN
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _safe_log_ratio(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float32)
    den = np.asarray(denominator, dtype=np.float32)
    return np.log(np.maximum(num, eps) / np.maximum(den, eps))


def _robust_unit_scale(
    values: np.ndarray,
    valid_mask: np.ndarray,
    percentile: float = 95.0,
    eps: float = 1e-6,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask) > 0.5
    if np.any(valid):
        scale = float(np.percentile(np.abs(arr[valid]), percentile))
    else:
        scale = float(np.max(np.abs(arr)))
    scale = max(scale, eps)
    return np.clip(arr / scale, -1.0, 1.0)


def _probability_focus_gate(prob: np.ndarray, valid_mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    valid = valid_mask > 0.5
    arr = np.asarray(prob, dtype=np.float32)
    if np.any(valid):
        lo = float(np.percentile(arr[valid], 45.0))
        hi = float(np.percentile(arr[valid], 95.0))
    else:
        lo = float(np.percentile(arr, 45.0))
        hi = float(np.percentile(arr, 95.0))
    gate = (arr - lo) / max(hi - lo, eps)
    gate = np.clip(gate, 0.0, 1.0)
    return gate * valid.astype(np.float32)


def _angle_to_rgb(theta: np.ndarray) -> np.ndarray:
    phase = 2.0 * theta.astype(np.float32)
    r = 0.5 + 0.5 * np.cos(phase)
    g = 0.5 + 0.5 * np.cos(phase - (2.0 * np.pi / 3.0))
    b = 0.5 + 0.5 * np.cos(phase - (4.0 * np.pi / 3.0))
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _error_map(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    valid_b = valid.astype(bool)
    out = np.zeros((*pred_b.shape, 3), dtype=np.uint8)
    tp = pred_b & gt_b & valid_b
    fp = pred_b & (~gt_b) & valid_b
    fn = (~pred_b) & gt_b & valid_b
    tn = (~pred_b) & (~gt_b) & valid_b
    out[tp] = np.array([38, 201, 98], dtype=np.uint8)
    out[fp] = np.array([239, 83, 80], dtype=np.uint8)
    out[fn] = np.array([255, 202, 40], dtype=np.uint8)
    out[tn] = np.array([32, 40, 52], dtype=np.uint8)
    return out


def _mask_overlay(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.58) -> np.ndarray:
    out = image_rgb.astype(np.float32).copy()
    mk = mask.astype(bool)
    c = np.asarray(color, dtype=np.float32)
    out[mk] = out[mk] * (1.0 - alpha) + c * alpha
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _improvement_map(base_pred: np.ndarray, az_pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    b = base_pred.astype(bool)
    a = az_pred.astype(bool)
    g = gt.astype(bool)
    v = valid.astype(bool)
    out = np.zeros((*b.shape, 3), dtype=np.uint8)
    gain_vessel = a & g & (~b) & v
    loss_vessel = b & g & (~a) & v
    fp_removed = b & (~g) & (~a) & v
    fp_added = a & (~g) & (~b) & v
    stable = (a == b) & v
    out[stable] = np.array([42, 46, 58], dtype=np.uint8)
    out[gain_vessel] = np.array([67, 210, 99], dtype=np.uint8)
    out[loss_vessel] = np.array([244, 67, 54], dtype=np.uint8)
    out[fp_removed] = np.array([79, 195, 247], dtype=np.uint8)
    out[fp_added] = np.array([255, 167, 38], dtype=np.uint8)
    return out


def _direction_gain_confidence_maps(snapshot: dict[str, Any], valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.asarray(snapshot["mu_map"], dtype=np.float32)
    valid = valid_mask > 0.5
    dominant = mu.argmax(axis=0)
    rule_confidence = mu.max(axis=0)

    if "theta_map" in snapshot:
        theta_map = np.asarray(snapshot["theta_map"], dtype=np.float32)
        sigma_u_map = np.asarray(snapshot["sigma_u_map"], dtype=np.float32)
        sigma_s_map = np.asarray(snapshot["sigma_s_map"], dtype=np.float32)
        # Axis (undirected) aggregation from model outputs:
        # theta is periodic modulo pi -> use doubled-angle mean.
        cos2 = np.cos(2.0 * theta_map)
        sin2 = np.sin(2.0 * theta_map)
        wsum = np.maximum(mu.sum(axis=0), 1e-6)
        c = (mu * cos2).sum(axis=0) / wsum
        s = (mu * sin2).sum(axis=0) / wsum
        direction = 0.5 * np.arctan2(s, c)
        # consistency in [0,1]: 1 = all rules agree on one axis.
        axis_consistency = np.clip(np.sqrt(c * c + s * s), 0.0, 1.0)
        confidence = np.clip(rule_confidence * axis_consistency, 0.0, 1.0)

        log_ratio = _safe_log_ratio(sigma_u_map, sigma_s_map)
        gain_per_rule = np.tanh(log_ratio)
        signed_gain = ((mu * gain_per_rule).sum(axis=0) / wsum) * confidence
    else:
        u_vec = np.asarray(snapshot.get("u_vec"), dtype=np.float32)
        sigma_u = np.asarray(snapshot.get("sigma_u"), dtype=np.float32)
        sigma_s = np.asarray(snapshot.get("sigma_s"), dtype=np.float32)
        if u_vec.size == 0:
            zeros = np.zeros_like(rule_confidence, dtype=np.float32)
            return zeros, zeros, zeros
        angles = np.arctan2(u_vec[:, 1], u_vec[:, 0])
        rule_log_ratio = _safe_log_ratio(sigma_u, sigma_s)
        cos2 = np.cos(2.0 * angles)[:, None, None]
        sin2 = np.sin(2.0 * angles)[:, None, None]
        wsum = np.maximum(mu.sum(axis=0), 1e-6)
        c = (mu * cos2).sum(axis=0) / wsum
        s = (mu * sin2).sum(axis=0) / wsum
        direction = 0.5 * np.arctan2(s, c)
        axis_consistency = np.clip(np.sqrt(c * c + s * s), 0.0, 1.0)
        confidence = np.clip(rule_confidence * axis_consistency, 0.0, 1.0)

        gain_per_rule = np.tanh(rule_log_ratio)[:, None, None]
        signed_gain = ((mu * gain_per_rule).sum(axis=0) / wsum) * confidence

    target_h, target_w = valid.shape
    if direction.shape != (target_h, target_w):
        direction = np.asarray(
            Image.fromarray(direction.astype(np.float32), mode="F").resize((target_w, target_h), Image.Resampling.NEAREST),
            dtype=np.float32,
        )
        signed_gain = np.asarray(
            Image.fromarray(signed_gain.astype(np.float32), mode="F").resize((target_w, target_h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        confidence = np.asarray(
            Image.fromarray(confidence.astype(np.float32), mode="F").resize((target_w, target_h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )

    direction = direction * valid.astype(np.float32)
    signed_gain = signed_gain * valid.astype(np.float32)
    confidence = confidence * valid.astype(np.float32)
    return direction, np.clip(signed_gain, -1.0, 1.0), np.clip(confidence, 0.0, 1.0)


def _direction_overlay(
    image_rgb: np.ndarray,
    direction: np.ndarray,
    confidence: np.ndarray,
    valid_mask: np.ndarray,
    strength_map: np.ndarray | None = None,
    focus_gate: np.ndarray | None = None,
) -> np.ndarray:
    valid = valid_mask > 0.5
    if strength_map is None:
        strength_map = confidence
    scaled_strength = np.abs(_robust_unit_scale(np.asarray(strength_map, dtype=np.float32), valid))
    focus = np.clip(np.asarray(confidence, dtype=np.float32) * scaled_strength, 0.0, 1.0)
    if focus_gate is not None:
        focus = focus * np.clip(focus_gate, 0.0, 1.0)
    base = image_rgb.astype(np.float32)
    hue = _angle_to_rgb(direction) * 255.0
    alpha = np.clip(0.08 + 0.77 * focus, 0.0, 0.85)[..., None]
    overlay = base * (1.0 - alpha) + hue * alpha
    overlay[~valid] = np.array([20.0, 20.0, 20.0], dtype=np.float32)

    pil = Image.fromarray(np.clip(overlay, 0.0, 255.0).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    h, w = direction.shape
    step = max(16, min(h, w) // 22)
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if not valid[y, x]:
                continue
            c = float(focus[y, x])
            if c < 0.12:
                continue
            angle = float(direction[y, x])
            ln = 3.5 + 11.0 * c
            dx = math.cos(angle) * ln
            dy = math.sin(angle) * ln
            draw.line((x - dx, y - dy, x + dx, y + dy), fill=(248, 248, 248), width=2)
    return np.array(pil, dtype=np.uint8)


def _geometry_axis_overlay(
    image_rgb: np.ndarray,
    direction: np.ndarray,
    signed_gain: np.ndarray,
    confidence: np.ndarray,
    valid_mask: np.ndarray,
    focus_gate: np.ndarray | None = None,
) -> np.ndarray:
    valid = valid_mask > 0.5
    strength = np.abs(_robust_unit_scale(np.asarray(signed_gain, dtype=np.float32), valid))
    focus = np.clip(strength * np.clip(confidence, 0.0, 1.0), 0.0, 1.0)
    if focus_gate is not None:
        focus = focus * np.clip(focus_gate, 0.0, 1.0)

    base = image_rgb.astype(np.float32)
    warm = np.array([255.0, 171.0, 64.0], dtype=np.float32)
    alpha = (0.06 + 0.48 * focus)[..., None]
    overlay = base * (1.0 - alpha) + warm * alpha
    overlay[~valid] = np.array([20.0, 20.0, 20.0], dtype=np.float32)

    pil = Image.fromarray(np.clip(overlay, 0.0, 255.0).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    h, w = direction.shape
    step = max(18, min(h, w) // 18)
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if not valid[y, x]:
                continue
            s = float(focus[y, x])
            if s < 0.10:
                continue
            angle = float(direction[y, x])
            major = 5.0 + 13.0 * s
            minor = max(2.0, 0.32 * major)
            ux = math.cos(angle)
            uy = math.sin(angle)
            sx = -uy
            sy = ux

            # Long orange axis: sigma_u. Short cyan axis: compressed sigma_s.
            draw.line((x - ux * major, y - uy * major, x + ux * major, y + uy * major), fill=(255, 238, 210), width=3)
            draw.line((x - ux * major, y - uy * major, x + ux * major, y + uy * major), fill=(255, 151, 48), width=1)
            draw.line((x - sx * minor, y - sy * minor, x + sx * minor, y + sy * minor), fill=(68, 209, 255), width=2)

            points: list[tuple[float, float]] = []
            for k in range(20):
                t = (2.0 * math.pi * k) / 20.0
                px = x + math.cos(t) * major * ux + math.sin(t) * minor * sx
                py = y + math.cos(t) * major * uy + math.sin(t) * minor * sy
                points.append((px, py))
            draw.line(points + [points[0]], fill=(255, 255, 255), width=1)
    return np.array(pil, dtype=np.uint8)


def _gain_overlay(image_rgb: np.ndarray, signed_gain: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid = valid_mask > 0.5
    scaled_gain = _robust_unit_scale(np.asarray(signed_gain, dtype=np.float32), valid)
    magnitude = np.abs(scaled_gain)
    color = np.array([255.0, 144.0, 48.0], dtype=np.float32)
    alpha = (0.10 + 0.82 * magnitude)[..., None]
    base = image_rgb.astype(np.float32)
    overlay = base * (1.0 - alpha) + color.reshape(1, 1, 3) * alpha
    overlay[~valid] = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def _focused_gain_overlay(
    image_rgb: np.ndarray,
    signed_gain: np.ndarray,
    valid_mask: np.ndarray,
    focus_gate: np.ndarray,
) -> np.ndarray:
    return _gain_overlay(image_rgb, np.asarray(signed_gain, dtype=np.float32) * np.clip(focus_gate, 0.0, 1.0), valid_mask)


def _signed_overlay(
    image_rgb: np.ndarray,
    signed_map: np.ndarray,
    valid_mask: np.ndarray,
    pos_rgb: tuple[float, float, float],
    neg_rgb: tuple[float, float, float],
    min_alpha: float = 0.08,
    max_alpha: float = 0.86,
) -> np.ndarray:
    valid = valid_mask > 0.5
    scaled = _robust_unit_scale(np.asarray(signed_map, dtype=np.float32), valid)
    magnitude = np.abs(scaled)
    pos = np.clip(scaled, 0.0, 1.0)
    neg = np.clip(-scaled, 0.0, 1.0)

    pos_vec = np.asarray(pos_rgb, dtype=np.float32)
    neg_vec = np.asarray(neg_rgb, dtype=np.float32)
    color = pos[..., None] * pos_vec + neg[..., None] * neg_vec
    alpha = (min_alpha + (max_alpha - min_alpha) * magnitude)[..., None]
    base = image_rgb.astype(np.float32)
    overlay = base * (1.0 - alpha) + color * alpha
    overlay[~valid] = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def _mean_abs_delta(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> float:
    valid = valid_mask > 0.5
    if not np.any(valid):
        return float(np.mean(np.abs(a - b)))
    return float(np.mean(np.abs(a[valid] - b[valid])))


def _confidence_overlay(image_rgb: np.ndarray, confidence: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid = valid_mask > 0.5
    conf = np.clip(confidence, 0.0, 1.0)
    low_rgb = np.array([61.0, 90.0, 254.0], dtype=np.float32)
    high_rgb = np.array([255.0, 214.0, 10.0], dtype=np.float32)
    color = conf[..., None] * high_rgb + (1.0 - conf[..., None]) * low_rgb
    alpha = (0.12 + 0.72 * conf)[..., None]
    base = image_rgb.astype(np.float32)
    overlay = base * (1.0 - alpha) + color * alpha
    overlay[~valid] = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def _focused_confidence_overlay(
    image_rgb: np.ndarray,
    confidence: np.ndarray,
    valid_mask: np.ndarray,
    focus_gate: np.ndarray,
) -> np.ndarray:
    return _confidence_overlay(
        image_rgb,
        np.asarray(confidence, dtype=np.float32) * np.clip(focus_gate, 0.0, 1.0),
        valid_mask,
    )


def _legend_panel(size: tuple[int, int]) -> np.ndarray:
    w, h = size
    canvas = Image.new("RGB", (w, h), color=(247, 246, 241))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, w - 1, h - 1), outline=(204, 204, 198))
    draw.text((16, 16), "Geometry reading guide", fill=(25, 31, 42))
    draw.text((16, 48), "white/orange long axis = learned direction u", fill=(25, 31, 42))
    draw.text((16, 76), "cyan short axis = compressed cross-direction s", fill=(25, 31, 42))
    draw.text((16, 104), "orange heat = mu * anisotropy contribution", fill=(25, 31, 42))
    draw.text((16, 132), "yellow confidence = selected fuzzy regime", fill=(25, 31, 42))
    draw.text((16, 160), "maps are gated by proposed probability", fill=(25, 31, 42))

    cx, cy = int(w * 0.72), int(h * 0.52)
    angle = -0.45
    ux, uy = math.cos(angle), math.sin(angle)
    sx, sy = -uy, ux
    major, minor = 56.0, 16.0
    draw.line((cx - ux * major, cy - uy * major, cx + ux * major, cy + uy * major), fill=(255, 151, 48), width=5)
    draw.line((cx - sx * minor, cy - sy * minor, cx + sx * minor, cy + sy * minor), fill=(68, 209, 255), width=4)
    points: list[tuple[float, float]] = []
    for k in range(40):
        t = (2.0 * math.pi * k) / 40.0
        px = cx + math.cos(t) * major * ux + math.sin(t) * minor * sx
        py = cy + math.cos(t) * major * uy + math.sin(t) * minor * sy
        points.append((px, py))
    draw.line(points + [points[0]], fill=(70, 70, 70), width=2)
    return np.array(canvas, dtype=np.uint8)


def _label_panel(image: np.ndarray, title: str, size: tuple[int, int], bar_height: int = 38) -> Image.Image:
    panel = Image.fromarray(image.astype(np.uint8))
    panel.thumbnail(size, Image.Resampling.LANCZOS)
    framed = Image.new("RGB", (size[0], size[1] + bar_height), color=(244, 243, 239))
    px = (size[0] - panel.width) // 2
    py = bar_height + (size[1] - panel.height) // 2
    framed.paste(panel, (px, py))
    draw = ImageDraw.Draw(framed)
    draw.rectangle((0, 0, framed.width, bar_height), fill=(27, 34, 46))
    draw.text((12, 10), title, fill=(255, 255, 255))
    return framed


def _dice(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    p = pred.astype(bool) & valid.astype(bool)
    g = gt.astype(bool) & valid.astype(bool)
    inter = float(np.logical_and(p, g).sum())
    denom = float(p.sum() + g.sum())
    if denom <= 0.0:
        return 1.0
    return float((2.0 * inter) / denom)


def _predict_prob(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        output = model(x)
        logits, _, _ = utils.unpack_segmentation_outputs(output)
        return torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)


def _choose_best_improvement_sample(
    dataset: Any,
    az_model: torch.nn.Module,
    az_thr: float,
    baseline_model: torch.nn.Module,
    baseline_thr: float,
    device: torch.device,
) -> int:
    best_idx = 0
    best_delta = -1e9
    for idx in range(len(dataset)):
        x_norm, y, valid = dataset[idx]
        x = x_norm.unsqueeze(0).to(device)
        gt = y[0].numpy().astype(np.float32)
        v = valid[0].numpy().astype(np.float32)
        base_prob = _predict_prob(baseline_model, x)
        az_prob = _predict_prob(az_model, x)
        base_pred = (base_prob >= baseline_thr).astype(np.float32)
        az_pred = (az_prob >= az_thr).astype(np.float32)
        delta = _dice(az_pred, gt, v) - _dice(base_pred, gt, v)
        if delta > best_delta:
            best_delta = delta
            best_idx = idx
    return best_idx


def _sample_id(dataset: Any, idx: int) -> str:
    raw = getattr(dataset, "samples", None)
    if isinstance(raw, list) and raw:
        item = raw[idx]
        if isinstance(item, tuple) and item:
            return Path(item[0]).stem
        if isinstance(item, dict) and "image_path" in item:
            return Path(str(item["image_path"])).stem
    return f"sample_{idx:03d}"


def _stage_label(order_idx: int, total: int, layer_pos: int) -> str:
    if order_idx == 0:
        stage = "Early"
    elif order_idx == total - 1:
        stage = "Late"
    else:
        stage = "Mid"
    return f"{stage}\nLayer L{layer_pos + 1}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export layer-wise geometry attention story (direction/gain/confidence).")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--run", type=str, default=None, help="AZ run directory name.")
    parser.add_argument("--variant", type=str, default="az_thesis", help="Used if --run is not provided.")
    parser.add_argument("--baseline-run", type=str, default=None, help="Optional baseline run for improvement map and auto sample picking.")
    parser.add_argument("--sample-index", type=int, default=-1, help="Use -1 to auto-pick best improvement sample when baseline is provided.")
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="article_assets/final_figures")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    device = torch.device(args.device)

    az_run_dir = _find_run(results_dir, args.run, args.variant)
    az_model, az_cfg, az_metrics, az_variant = _load_model(az_run_dir, device)
    dataset, dataset_name = _build_eval_dataset(az_cfg)
    az_thr = float(az_metrics.get("selected_threshold", az_cfg.get("seg_threshold", 0.5)))

    baseline_model = None
    baseline_thr = None
    baseline_run_dir: Path | None = None
    if args.baseline_run:
        baseline_run_dir = _find_run(results_dir, args.baseline_run, None)
        baseline_model, _, baseline_metrics, _ = _load_model(baseline_run_dir, device)
        baseline_thr = float(baseline_metrics.get("selected_threshold", az_cfg.get("seg_threshold", 0.5)))

    if args.sample_index >= 0:
        sample_index = min(max(int(args.sample_index), 0), len(dataset) - 1)
    elif baseline_model is not None and baseline_thr is not None:
        sample_index = _choose_best_improvement_sample(dataset, az_model, az_thr, baseline_model, baseline_thr, device)
    else:
        sample_index = 0

    x_norm, y, valid = dataset[sample_index]
    image_rgb = _denorm_to_uint8(x_norm)
    gt = y[0].numpy().astype(np.float32)
    valid_np = valid[0].numpy().astype(np.float32)
    x = x_norm.unsqueeze(0).to(device)

    az_prob = _predict_prob(az_model, x)
    az_pred = (az_prob >= az_thr).astype(np.float32) * valid_np
    object_focus = _probability_focus_gate(az_prob, valid_np)

    baseline_pred = None
    if baseline_model is not None and baseline_thr is not None:
        base_prob = _predict_prob(baseline_model, x)
        baseline_pred = (base_prob >= baseline_thr).astype(np.float32) * valid_np

    az_layers: list[tuple[str, dict[str, Any]]] = []
    for name, module in az_model.named_modules():
        if isinstance(module, AZConv2d):
            snap = module.interpretation_snapshot()
            if snap:
                az_layers.append((name, snap))
    if not az_layers:
        raise SystemExit("No AZConv2d interpretation snapshots found. Ensure the run uses AZ layers.")

    layer_maps: list[dict[str, Any]] = []
    for layer_pos, (name, snap) in enumerate(az_layers):
        direction, signed_gain, confidence = _direction_gain_confidence_maps(snap, valid_np)
        valid_bool = valid_np > 0.5
        mean_conf = float((confidence * valid_bool).sum() / max(1.0, float(valid_bool.sum())))
        mean_abs_gain = float((np.abs(signed_gain) * valid_bool).sum() / max(1.0, float(valid_bool.sum())))
        layer_maps.append(
            {
                "layer_pos": int(layer_pos),
                "layer_name": name,
                "direction": direction,
                "signed_gain": signed_gain,
                "confidence": confidence,
                "mean_confidence": mean_conf,
                "mean_abs_contribution": mean_abs_gain,
            }
        )

    if len(layer_maps) <= args.max_layers:
        chosen_layer_order = list(range(len(layer_maps)))
    else:
        # Prefer layers with the most distinct contribution patterns for visible evolution.
        selected = {0, len(layer_maps) - 1}
        while len(selected) < int(args.max_layers):
            best_idx = None
            best_score = -1.0
            for cand in range(len(layer_maps)):
                if cand in selected:
                    continue
                cand_gain = layer_maps[cand]["signed_gain"]
                nearest = min(
                    _mean_abs_delta(cand_gain, layer_maps[s]["signed_gain"], valid_np)
                    for s in selected
                )
                if nearest > best_score:
                    best_score = nearest
                    best_idx = cand
            if best_idx is None:
                break
            selected.add(int(best_idx))
        chosen_layer_order = sorted(selected)

    cell_size = (300, 300)
    rows: list[list[tuple[str, np.ndarray]]] = []
    row_labels: list[str] = []
    gt_overlay = _mask_overlay(image_rgb, gt > 0.5, color=(50, 220, 80))
    az_overlay = _mask_overlay(image_rgb, az_pred > 0.5, color=(255, 110, 60))
    top_panels: list[tuple[str, np.ndarray]] = [
        ("Input", image_rgb),
        ("Ground Truth", gt_overlay),
        ("Proposed Prediction", az_overlay),
        ("How to Read Geometry", _legend_panel(cell_size)),
    ]
    if baseline_pred is not None:
        baseline_overlay = _mask_overlay(image_rgb, baseline_pred > 0.5, color=(255, 180, 40))
        top_panels.insert(2, ("Baseline Prediction", baseline_overlay))
        top_panels.insert(
            4,
            (
                "Difference Map vs Baseline",
                _improvement_map(baseline_pred > 0.5, az_pred > 0.5, gt > 0.5, valid_np > 0.5),
            ),
        )
    rows.append(top_panels)
    row_labels.append("Prediction\nComparison")

    layer_summary: list[dict[str, Any]] = []
    prev_signed_gain: np.ndarray | None = None
    total_selected = len(chosen_layer_order)
    for selected_rank, layer_idx in enumerate(chosen_layer_order):
        layer = layer_maps[layer_idx]
        layer_pos = int(layer["layer_pos"])
        name = str(layer["layer_name"])
        direction = np.asarray(layer["direction"], dtype=np.float32)
        signed_gain = np.asarray(layer["signed_gain"], dtype=np.float32)
        confidence = np.asarray(layer["confidence"], dtype=np.float32)
        axis_img = _geometry_axis_overlay(image_rgb, direction, signed_gain, confidence, valid_np, focus_gate=object_focus)
        dir_img = _direction_overlay(
            image_rgb,
            direction,
            confidence,
            valid_np,
            strength_map=np.abs(signed_gain),
            focus_gate=object_focus,
        )
        gain_img = _focused_gain_overlay(image_rgb, signed_gain, valid_np, object_focus)
        if prev_signed_gain is None:
            delta_gain = np.zeros_like(signed_gain, dtype=np.float32)
            mean_delta = 0.0
        else:
            delta_gain = signed_gain - prev_signed_gain
            mean_delta = _mean_abs_delta(signed_gain, prev_signed_gain, valid_np)
        delta_img = _signed_overlay(
            image_rgb,
            delta_gain * object_focus,
            valid_np,
            pos_rgb=(255.0, 214.0, 70.0),  # stronger than previous layer
            neg_rgb=(64.0, 170.0, 255.0),  # weaker than previous layer
            min_alpha=0.08,
            max_alpha=0.88,
        )
        conf_img = _focused_confidence_overlay(image_rgb, confidence, valid_np, object_focus)
        rows.append(
            [
                (f"L{layer_pos + 1}: Axes u/s", axis_img),
                (f"L{layer_pos + 1}: Direction Field", dir_img),
                (f"L{layer_pos + 1}: Contribution", gain_img),
                (f"L{layer_pos + 1}: Layer Change", delta_img),
                (f"L{layer_pos + 1}: Regime Confidence", conf_img),
            ]
        )
        row_labels.append(_stage_label(selected_rank, total_selected, layer_pos))
        layer_summary.append(
            {
                "layer_index": int(layer_pos),
                "layer_name": name,
                "mean_confidence": float(layer["mean_confidence"]),
                "mean_abs_contribution": float(layer["mean_abs_contribution"]),
                "mean_abs_delta_vs_prev_selected": float(mean_delta),
            }
        )
        prev_signed_gain = signed_gain.copy()

    labeled_rows: list[list[Image.Image]] = [[_label_panel(image, title, cell_size) for title, image in row] for row in rows]
    columns = max(len(row) for row in labeled_rows)
    cell_w = max(panel.width for row in labeled_rows for panel in row)
    cell_h = max(panel.height for row in labeled_rows for panel in row)
    row_label_w = 250
    canvas = Image.new("RGB", (row_label_w + columns * cell_w, len(labeled_rows) * cell_h), color=(232, 232, 228))
    for row_idx, row in enumerate(labeled_rows):
        y0 = row_idx * cell_h
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, y0, row_label_w, y0 + cell_h), fill=(245, 244, 239), outline=(210, 210, 205))
        label_text = row_labels[row_idx] if row_idx < len(row_labels) else f"Row {row_idx}"
        draw.text((16, y0 + 24), label_text, fill=(28, 28, 28))
        for col_idx, panel in enumerate(row):
            x0 = row_label_w + col_idx * cell_w
            canvas.paste(panel, (x0, y0))

    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_tag = _sample_id(dataset, sample_index)
    stem = f"figure_geometry_attention_story_{dataset_name}_{sample_tag}"
    png_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"
    canvas.save(png_path)

    payload = {
        "dataset": dataset_name,
        "sample_index": int(sample_index),
        "sample_id": sample_tag,
        "az_run": az_run_dir.name,
        "az_variant": az_variant,
        "az_threshold": az_thr,
        "baseline_run": baseline_run_dir.name if baseline_run_dir else None,
        "baseline_threshold": baseline_thr,
        "selection_mode": "diverse_contribution_patterns",
        "visual_encoding": {
            "axes_u_s": "orange/white long axis is sigma_u direction; cyan short axis is compressed sigma_s direction",
            "contribution": "mu * tanh(log(sigma_u / sigma_s)), gated by proposed probability map",
            "layer_change": "difference from previous selected layer after the same probability gate",
            "regime_confidence": "dominant fuzzy membership confidence gated by proposed probability map",
        },
        "chosen_layers": layer_summary,
        "output_image": str(png_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Exported geometry attention story: {png_path}")
    print(f"Metadata: {json_path}")


if __name__ == "__main__":
    main()
