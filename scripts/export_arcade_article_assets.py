#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from drive_viewer import _normalize_map, metrics_from_prob_map
from models.azconv import AZConv2d
from utils import build_model


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


def _to_uint8_rgb(image_chw: torch.Tensor) -> np.ndarray:
    arr = image_chw.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _mask_overlay(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.58) -> np.ndarray:
    out = image_rgb.astype(np.float32).copy()
    mk = mask.astype(bool)
    c = np.asarray(color, dtype=np.float32)
    out[mk] = out[mk] * (1.0 - alpha) + c * alpha
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


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


def _angle_to_rgb(theta: np.ndarray) -> np.ndarray:
    phase = 2.0 * theta.astype(np.float32)
    r = 0.5 + 0.5 * np.cos(phase)
    g = 0.5 + 0.5 * np.cos(phase - (2.0 * np.pi / 3.0))
    b = 0.5 + 0.5 * np.cos(phase - (4.0 * np.pi / 3.0))
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _best_rule_for_vessel(mu_map: np.ndarray, gt_mask: np.ndarray) -> int:
    gt = gt_mask > 0.5
    bg = ~gt
    if not np.any(gt):
        return 0
    vessel_mean = mu_map[:, gt].mean(axis=1)
    bg_mean = mu_map[:, bg].mean(axis=1) if np.any(bg) else np.zeros_like(vessel_mean)
    return int(np.argmax(vessel_mean - bg_mean))


def _geometry_direction_and_gain(snapshot: dict[str, Any], mu_map: np.ndarray, rule_idx: int, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = valid_mask > 0.5
    theta_map = snapshot.get("theta_map")
    sigma_u_map = snapshot.get("sigma_u_map")
    sigma_s_map = snapshot.get("sigma_s_map")

    if theta_map is None or sigma_u_map is None or sigma_s_map is None:
        h, w = valid.shape
        zeros = np.zeros((h, w), dtype=np.float32)
        return zeros, zeros, zeros

    theta = np.asarray(theta_map, dtype=np.float32)[rule_idx]
    sigma_u = np.asarray(sigma_u_map, dtype=np.float32)[rule_idx]
    sigma_s = np.asarray(sigma_s_map, dtype=np.float32)[rule_idx]
    weight = np.asarray(mu_map, dtype=np.float32)[rule_idx]

    log_ratio = _safe_log_ratio(sigma_u, sigma_s)
    gain = weight * np.tanh(log_ratio)
    strength = weight * np.tanh(np.abs(log_ratio))
    theta = theta * valid.astype(np.float32)
    gain = gain * valid.astype(np.float32)
    strength = strength * valid.astype(np.float32)
    return theta, np.clip(strength, 0.0, 1.0), np.clip(gain, -1.0, 1.0)


def _geometry_direction_overlay(image_rgb: np.ndarray, direction: np.ndarray, strength: np.ndarray, valid: np.ndarray) -> np.ndarray:
    valid_b = valid > 0.5
    scaled_strength = np.abs(_robust_unit_scale(np.asarray(strength, dtype=np.float32), valid_b))
    base = image_rgb.astype(np.float32)
    hue = _angle_to_rgb(direction) * 255.0
    alpha = np.clip(0.10 + 0.75 * scaled_strength, 0.0, 0.85)[..., None]
    overlay = base * (1.0 - alpha) + hue * alpha
    overlay[~valid_b] = np.array([18.0, 18.0, 18.0], dtype=np.float32)

    pil = Image.fromarray(np.clip(overlay, 0.0, 255.0).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    h, w = direction.shape
    step = max(16, min(h, w) // 18)
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if not valid_b[y, x]:
                continue
            s = float(scaled_strength[y, x])
            if s < 0.12:
                continue
            ang = float(direction[y, x])
            ln = 3.5 + 10.5 * s
            dx = np.cos(ang) * ln
            dy = np.sin(ang) * ln
            draw.line((x - dx, y - dy, x + dx, y + dy), fill=(250, 250, 250), width=2)
    return np.array(pil, dtype=np.uint8)


def _geometry_gain_overlay(image_rgb: np.ndarray, gain: np.ndarray, valid: np.ndarray) -> np.ndarray:
    valid_b = valid > 0.5
    score = _robust_unit_scale(np.asarray(gain, dtype=np.float32), valid_b)
    magnitude = np.abs(score)
    pos = np.clip(score, 0.0, 1.0)
    neg = np.clip(-score, 0.0, 1.0)

    pos_rgb = np.array([255.0, 128.0, 54.0], dtype=np.float32)
    neg_rgb = np.array([67.0, 126.0, 255.0], dtype=np.float32)
    color = pos[..., None] * pos_rgb + neg[..., None] * neg_rgb
    alpha = (0.10 + 0.82 * magnitude)[..., None]
    base = image_rgb.astype(np.float32)
    overlay = base * (1.0 - alpha) + color * alpha
    overlay[~valid_b] = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def _label_panel(image: np.ndarray, title: str, bar_height: int = 36) -> Image.Image:
    panel = Image.fromarray(image.astype(np.uint8))
    labeled = Image.new("RGB", (panel.width, panel.height + bar_height), color=(247, 247, 244))
    labeled.paste(panel, (0, bar_height))
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, labeled.width, bar_height), fill=(26, 32, 44))
    draw.text((12, 9), title, fill=(255, 255, 255))
    return labeled


def _save_grid(panels: list[tuple[str, np.ndarray]], out_path: Path, columns: int = 3) -> None:
    labeled = [_label_panel(image, title) for title, image in panels]
    cell_w = max(panel.width for panel in labeled)
    cell_h = max(panel.height for panel in labeled)
    rows = (len(labeled) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), color=(236, 236, 232))
    for idx, panel in enumerate(labeled):
        row = idx // columns
        col = idx % columns
        x = col * cell_w + (cell_w - panel.width) // 2
        y = row * cell_h + (cell_h - panel.height) // 2
        canvas.paste(panel, (x, y))
    canvas.save(out_path)


def _find_best_run(results_dir: Path, variant: str) -> Path:
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
        raise FileNotFoundError(f"No run found for variant={variant} in {results_dir}")
    return best[1]


def _load_model(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    payload = _load_checkpoint(run_dir / "checkpoint_best.pt")
    cfg = payload.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError(f"Checkpoint in {run_dir} does not contain cfg payload.")
    metrics = _load_json(run_dir / "metrics.json")
    in_channels, num_outputs = utils.dataset_channels_and_outputs(cfg["dataset"])
    model = build_model(
        str(payload.get("variant", metrics.get("variant", "baseline"))),
        num_outputs=num_outputs,
        in_channels=in_channels,
        num_rules=int(cfg.get("num_rules", 4)),
        task=utils.task_for_dataset(cfg["dataset"]),
        widths=utils.parse_model_widths(cfg.get("model_widths")),
        model_kwargs=utils.resolve_segmentation_model_kwargs(cfg),
        az_cfg_kwargs=utils.resolve_azconv_config_kwargs(cfg),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg, metrics


def _predict(model: torch.nn.Module, x: torch.Tensor) -> tuple[np.ndarray, dict[str, Any] | None]:
    with torch.no_grad():
        output = model(x)
        logits, _, _ = utils.unpack_segmentation_outputs(output)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

    first_snapshot = None
    for module in model.modules():
        if isinstance(module, AZConv2d):
            snap = module.interpretation_snapshot()
            if snap:
                first_snapshot = snap
                break
    return prob, first_snapshot


def _parse_indices(text: str) -> list[int]:
    out = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not out:
        raise ValueError("At least one sample index is required.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ARCADE comparison visuals (baseline vs az_thesis).")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--baseline-run", type=str, default=None)
    parser.add_argument("--az-run", type=str, default=None)
    parser.add_argument("--samples", type=str, default="0,5,10")
    parser.add_argument("--output-dir", type=str, default="article_assets/exports_arcade")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    baseline_dir = results_dir / args.baseline_run if args.baseline_run else _find_best_run(results_dir, "baseline")
    az_dir = results_dir / args.az_run if args.az_run else _find_best_run(results_dir, "az_thesis")

    device = torch.device(args.device)
    baseline_model, baseline_cfg, baseline_metrics = _load_model(baseline_dir, device)
    az_model, az_cfg, az_metrics = _load_model(az_dir, device)
    if utils.canonical_dataset_name(str(baseline_cfg["dataset"])) != utils.canonical_dataset_name(str(az_cfg["dataset"])):
        raise ValueError("Baseline and AZ runs must use the same dataset.")

    dataset_name = utils.canonical_dataset_name(str(az_cfg["dataset"]))
    objective = "stenosis" if dataset_name == "arcade_stenosis" else "syntax"
    data_root = utils.arcade_dataset_root(az_cfg.get("data_root", "./data"))
    dataset = utils.ArcadeVesselDataset(root=(PROJECT_ROOT / data_root).resolve(), split="test", objective=objective, augment=False, crop_size=None, image_size=None)

    base_thr = float(baseline_metrics.get("selected_threshold", baseline_cfg.get("seg_threshold", 0.5)))
    az_thr = float(az_metrics.get("selected_threshold", az_cfg.get("seg_threshold", 0.5)))
    sample_indices = _parse_indices(args.samples)

    out_root = (PROJECT_ROOT / args.output_dir / f"{baseline_dir.name}_vs_{az_dir.name}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "results_dir": str(results_dir),
        "baseline_run": baseline_dir.name,
        "az_run": az_dir.name,
        "dataset": dataset_name,
        "thresholds": {"baseline": base_thr, "az_thesis": az_thr},
        "samples": [],
    }

    for raw_idx in sample_indices:
        idx = min(max(int(raw_idx), 0), len(dataset.samples) - 1)
        sample = dataset.samples[idx]
        image = dataset._load_rgb(sample["image_path"])
        mask = dataset._build_mask(sample["height"], sample["width"], sample["polygons"])
        valid = torch.ones_like(mask)

        x = dataset._normalize(image).unsqueeze(0).to(device)
        base_prob, _ = _predict(baseline_model, x)
        az_prob, az_snapshot = _predict(az_model, x)

        gt = mask[0].numpy().astype(np.float32)
        valid_np = valid[0].numpy().astype(np.float32)
        image_rgb = _to_uint8_rgb(image)
        base_pred = (base_prob >= base_thr).astype(np.float32) * valid_np
        az_pred = (az_prob >= az_thr).astype(np.float32) * valid_np

        gt_overlay = _mask_overlay(image_rgb, gt > 0.5, color=(50, 220, 80))
        base_overlay = _mask_overlay(image_rgb, base_pred > 0.5, color=(255, 180, 40))
        az_overlay = _mask_overlay(image_rgb, az_pred > 0.5, color=(255, 110, 60))
        base_error = _error_map(base_pred > 0.5, gt > 0.5, valid_np > 0.5)
        az_error = _error_map(az_pred > 0.5, gt > 0.5, valid_np > 0.5)
        improve = _improvement_map(base_pred > 0.5, az_pred > 0.5, gt > 0.5, valid_np > 0.5)

        panels: list[tuple[str, np.ndarray]] = [
            ("Input", image_rgb),
            ("Ground Truth", gt_overlay),
            ("Baseline U-Net Prediction", base_overlay),
            ("Proposed Geometry-Aware Prediction", az_overlay),
            ("Baseline Error", base_error),
            ("Proposed Method Error", az_error),
            ("Geometry Contribution vs Baseline", improve),
        ]

        if az_snapshot is not None and "mu_map" in az_snapshot:
            mu_map = np.asarray(az_snapshot["mu_map"], dtype=np.float32)
            best_rule = _best_rule_for_vessel(mu_map, gt)
            direction, strength, gain = _geometry_direction_and_gain(az_snapshot, mu_map, best_rule, valid_np)
            direction_rgb = _geometry_direction_overlay(image_rgb, direction, strength, valid_np)
            gain_rgb = _geometry_gain_overlay(image_rgb, gain, valid_np)
            panels.extend([("Geometry Direction Field", direction_rgb), ("Geometry Contribution Map", gain_rgb)])

        sample_dir = out_root / f"{idx:03d}_{sample['image_path'].stem}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        _save_grid(panels, sample_dir / "article_grid.png", columns=3)
        for title, panel in panels:
            panel_name = title.lower().replace(" ", "_")
            Image.fromarray(panel.astype(np.uint8)).save(sample_dir / f"{panel_name}.png")

        base_metrics = metrics_from_prob_map(base_prob, gt, valid_np, threshold=base_thr)
        az_metrics_row = metrics_from_prob_map(az_prob, gt, valid_np, threshold=az_thr)
        with open(sample_dir / "metrics_compare.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "baseline": base_metrics,
                    "az_thesis": az_metrics_row,
                    "delta_az_minus_baseline": {k: float(az_metrics_row[k] - base_metrics[k]) for k in az_metrics_row.keys()},
                },
                handle,
                indent=2,
            )

        manifest["samples"].append(
            {
                "index": idx,
                "image_file": str(sample["image_path"].name),
                "output_dir": str(sample_dir),
                "baseline_metrics": base_metrics,
                "az_metrics": az_metrics_row,
            }
        )

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported ARCADE comparison assets to {out_root}")


if __name__ == "__main__":
    main()
