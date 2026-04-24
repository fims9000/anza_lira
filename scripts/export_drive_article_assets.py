#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils
from drive_viewer import (
    _error_map,
    _geometry_visualization,
    _heat_overlay,
    _mask_to_overlay,
    _normalize_map,
    _normalize_for_model,
    _scalar_map_to_rgb,
    _to_uint8_rgb,
    discover_drive_runs,
    metrics_from_prob_map,
    recommended_threshold_for_run,
)
from models.azconv import AZConv2d
from utils import build_model


RULE_PALETTE = np.asarray(
    [
        [231, 76, 60],
        [52, 152, 219],
        [46, 204, 113],
        [241, 196, 15],
        [155, 89, 182],
        [230, 126, 34],
        [26, 188, 156],
        [236, 112, 99],
    ],
    dtype=np.uint8,
)


def _angle_to_rgb(theta: np.ndarray) -> np.ndarray:
    angle = np.asarray(theta, dtype=np.float32)
    phase = 2.0 * angle
    r = 0.5 + 0.5 * np.cos(phase)
    g = 0.5 + 0.5 * np.cos(phase - (2.0 * np.pi / 3.0))
    b = 0.5 + 0.5 * np.cos(phase - (4.0 * np.pi / 3.0))
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _safe_log_ratio(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float32)
    den = np.asarray(denominator, dtype=np.float32)
    return np.log(np.maximum(num, eps) / np.maximum(den, eps))


def _geometry_direction_and_gain_maps(
    snapshot: dict[str, Any],
    mu_map: np.ndarray,
    rule_idx: int,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.asarray(mu_map, dtype=np.float32)
    valid = valid_mask > 0.5
    theta_map = snapshot.get("theta_map")
    sigma_u_map = snapshot.get("sigma_u_map")
    sigma_s_map = snapshot.get("sigma_s_map")

    if theta_map is not None:
        theta = np.asarray(theta_map, dtype=np.float32)[rule_idx]
        weight = mu[rule_idx]
        if sigma_u_map is not None and sigma_s_map is not None:
            log_ratio = _safe_log_ratio(
                np.asarray(sigma_u_map, dtype=np.float32)[rule_idx],
                np.asarray(sigma_s_map, dtype=np.float32)[rule_idx],
            )
        else:
            log_ratio = np.zeros_like(theta, dtype=np.float32)
        gain = weight * np.tanh(log_ratio)
        strength = weight * np.tanh(np.abs(log_ratio))
        direction = theta
    else:
        u_vec = snapshot.get("u_vec")
        sigma_u = snapshot.get("sigma_u")
        sigma_s = snapshot.get("sigma_s")
        if u_vec is None or sigma_u is None or sigma_s is None:
            h, w = mu.shape[-2:]
            zeros = np.zeros((h, w), dtype=np.float32)
            return zeros, zeros, zeros
        u = np.asarray(u_vec, dtype=np.float32)
        sigma_u_arr = np.asarray(sigma_u, dtype=np.float32)
        sigma_s_arr = np.asarray(sigma_s, dtype=np.float32)
        dominant = mu.argmax(axis=0)
        confidence = mu.max(axis=0)
        angles = np.arctan2(u[:, 1], u[:, 0]).astype(np.float32)
        log_ratio_rules = _safe_log_ratio(sigma_u_arr, sigma_s_arr)
        direction = angles[dominant]
        gain = confidence * np.tanh(log_ratio_rules[dominant])
        strength = confidence * np.tanh(np.abs(log_ratio_rules[dominant]))

    direction = direction * valid.astype(np.float32)
    gain = gain * valid.astype(np.float32)
    strength = strength * valid.astype(np.float32)
    return direction, np.clip(strength, 0.0, 1.0), np.clip(gain, -1.0, 1.0)


def _geometry_direction_overlay(
    image_rgb: np.ndarray,
    direction: np.ndarray,
    strength: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    valid = valid_mask > 0.5
    base = image_rgb.astype(np.float32)
    hue = _angle_to_rgb(direction) * 255.0
    alpha = np.clip(0.25 + 0.65 * strength, 0.0, 0.85)[..., None]
    overlay = base * (1.0 - alpha) + hue * alpha
    overlay[~valid] = np.array([20.0, 20.0, 20.0], dtype=np.float32)

    pil = Image.fromarray(np.clip(overlay, 0.0, 255.0).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    h, w = direction.shape
    step = max(18, min(h, w) // 18)
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if not valid[y, x]:
                continue
            s = float(strength[y, x])
            if s < 0.25:
                continue
            angle = float(direction[y, x])
            length = 5.0 + 11.0 * s
            dx = np.cos(angle) * length
            dy = np.sin(angle) * length
            x0 = x - dx
            y0 = y - dy
            x1 = x + dx
            y1 = y + dy
            draw.line((x0, y0, x1, y1), fill=(248, 248, 248), width=2)
    return np.array(pil, dtype=np.uint8)


def _geometry_gain_overlay(
    image_rgb: np.ndarray,
    gain: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    valid = valid_mask > 0.5
    score = np.asarray(gain, dtype=np.float32)
    magnitude = np.clip(np.abs(score), 0.0, 1.0)
    pos = np.clip(score, 0.0, 1.0)
    neg = np.clip(-score, 0.0, 1.0)

    pos_rgb = np.array([255.0, 128.0, 54.0], dtype=np.float32)
    neg_rgb = np.array([67.0, 126.0, 255.0], dtype=np.float32)
    color = pos[..., None] * pos_rgb + neg[..., None] * neg_rgb
    alpha = (0.15 + 0.75 * magnitude)[..., None]

    base = image_rgb.astype(np.float32)
    overlay = base * (1.0 - alpha) + color * alpha
    overlay[~valid] = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def _parse_indices(text: str) -> list[int]:
    out = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not out:
        raise ValueError("At least one sample index must be provided.")
    return out


def _checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _load_model_for_run(run, config_path: Path, device_name: str) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    payload = _checkpoint_payload(run.checkpoint_path)
    cfg = payload.get("cfg") or utils.load_config(str(config_path))
    utils.set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))
    in_channels, num_outputs = utils.dataset_channels_and_outputs(cfg["dataset"])
    device = torch.device(device_name)
    model = build_model(
        payload.get("variant", run.variant),
        num_outputs=num_outputs,
        in_channels=in_channels,
        num_rules=int(cfg.get("num_rules", 4)),
        task=utils.task_for_dataset(cfg["dataset"]),
        widths=utils.parse_model_widths(cfg.get("model_widths")),
        model_kwargs=utils.resolve_segmentation_model_kwargs(cfg),
        az_cfg_kwargs=utils.resolve_azconv_config_kwargs(cfg),
    )
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, cfg, device


def _prediction_for_sample(
    run,
    model: torch.nn.Module,
    dataset: utils.DriveDataset,
    idx: int,
    device: torch.device,
) -> dict[str, Any]:
    image_path, mask_path, fov_path = dataset.samples[idx]
    image_tensor = dataset._load_rgb(image_path)
    mask_tensor = dataset._load_mask(mask_path)
    valid_tensor = dataset._load_mask(fov_path) if dataset.use_fov_mask else torch.ones_like(mask_tensor)

    image_rgb = np.transpose(image_tensor.numpy(), (1, 2, 0))
    image_rgb_u8 = _to_uint8_rgb(image_rgb)
    normalized = _normalize_for_model(image_rgb_u8).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(normalized)
        logits, _aux_logits, _boundary_logits = utils.unpack_segmentation_outputs(output)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

    layer_infos: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, AZConv2d):
            snapshot = module.interpretation_snapshot()
            if snapshot:
                layer_infos.append({"name": name, "snapshot": snapshot})

    return {
        "sample_id": image_path.stem,
        "image": image_rgb_u8,
        "mask": mask_tensor[0].numpy().astype(np.float32),
        "valid": valid_tensor[0].numpy().astype(np.float32),
        "prob": prob,
        "layers": layer_infos,
    }


def _label_panel(image: np.ndarray, title: str, bar_height: int = 36) -> Image.Image:
    panel = Image.fromarray(image.astype(np.uint8))
    labeled = Image.new("RGB", (panel.width, panel.height + bar_height), color=(247, 247, 244))
    labeled.paste(panel, (0, bar_height))
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, labeled.width, bar_height), fill=(26, 32, 44))
    draw.text((12, 9), title, fill=(255, 255, 255))
    return labeled


def _rule_palette(num_rules: int) -> np.ndarray:
    repeats = (num_rules + len(RULE_PALETTE) - 1) // len(RULE_PALETTE)
    return np.tile(RULE_PALETTE, (repeats, 1))[:num_rules]


def _rule_partition_rgb(mu_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu_map, dtype=np.float32)
    valid = valid_mask > 0.5
    num_rules = int(mu.shape[0])
    palette = _rule_palette(num_rules)
    dominant = mu.argmax(axis=0)
    confidence = mu.max(axis=0)
    out = np.zeros((*dominant.shape, 3), dtype=np.uint8)
    out[valid] = palette[dominant[valid]]
    scale = (0.55 + 0.45 * _normalize_map(confidence))[..., None]
    out = np.clip(out.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    out[~valid] = np.array([20, 20, 20], dtype=np.uint8)
    return out


def _rule_entropy_overlay(image_rgb: np.ndarray, mu_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu_map, dtype=np.float32)
    valid = valid_mask > 0.5
    safe = np.clip(mu, 1e-8, 1.0)
    entropy = -(safe * np.log(safe)).sum(axis=0) / np.log(float(mu.shape[0]))
    entropy = entropy * valid.astype(np.float32)
    overlay = _heat_overlay(image_rgb, entropy, alpha=0.7)
    overlay[~valid] = np.array([20, 20, 20], dtype=np.uint8)
    return overlay


def _rule_stats(mu_map: np.ndarray, mask: np.ndarray, valid_mask: np.ndarray) -> dict[str, list[float]]:
    mu = np.asarray(mu_map, dtype=np.float32)
    vessel = (mask > 0.5) & (valid_mask > 0.5)
    background = (mask <= 0.5) & (valid_mask > 0.5)

    vessel_mean = mu[:, vessel].mean(axis=1) if vessel.any() else np.zeros(mu.shape[0], dtype=np.float32)
    background_mean = mu[:, background].mean(axis=1) if background.any() else np.zeros(mu.shape[0], dtype=np.float32)
    delta = vessel_mean - background_mean
    return {
        "vessel_mean": vessel_mean.tolist(),
        "background_mean": background_mean.tolist(),
        "delta": delta.tolist(),
    }


def _best_vessel_rule(stats: dict[str, list[float]]) -> int:
    delta = np.asarray(stats["delta"], dtype=np.float32)
    return int(delta.argmax()) if delta.size else 0


def _draw_rule_stats_chart(stats: dict[str, list[float]], num_rules: int) -> np.ndarray:
    palette = _rule_palette(num_rules)
    width = 720
    height = 420
    margin_left = 84
    margin_right = 36
    margin_top = 72
    margin_bottom = 52
    usable_w = width - margin_left - margin_right
    row_h = (height - margin_top - margin_bottom) / max(num_rules, 1)
    bar_h = max(10, int(row_h * 0.26))

    image = Image.new("RGB", (width, height), color=(246, 245, 241))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 54), fill=(27, 34, 46))
    draw.text((16, 16), "Rule statistics: vessel vs background", fill=(255, 255, 255))
    draw.text((16, 56), "mean membership inside FOV", fill=(60, 60, 60))

    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill=(110, 110, 110), width=2)
    for tick in range(6):
        x = margin_left + usable_w * (tick / 5.0)
        draw.line((x, height - margin_bottom, x, margin_top - 12), fill=(222, 222, 216), width=1)
        draw.text((x - 10, height - margin_bottom + 8), f"{tick * 0.2:.1f}", fill=(70, 70, 70))

    vessel_values = stats["vessel_mean"]
    background_values = stats["background_mean"]
    for idx in range(num_rules):
        y_center = margin_top + (idx + 0.5) * row_h
        label = f"R{idx + 1}"
        draw.text((18, y_center - 8), label, fill=(30, 30, 30))
        swatch = tuple(int(v) for v in palette[idx])
        draw.rectangle((46, y_center - 8, 62, y_center + 8), fill=swatch)

        bg_w = usable_w * float(background_values[idx])
        vessel_w = usable_w * float(vessel_values[idx])
        bg_box = (margin_left, y_center - bar_h - 3, margin_left + bg_w, y_center - 3)
        vessel_box = (margin_left, y_center + 3, margin_left + vessel_w, y_center + bar_h + 3)
        draw.rounded_rectangle(bg_box, radius=4, fill=(170, 180, 195))
        draw.rounded_rectangle(vessel_box, radius=4, fill=swatch)

    legend_y = height - 36
    draw.rectangle((margin_left, legend_y - 10, margin_left + 18, legend_y + 8), fill=(170, 180, 195))
    draw.text((margin_left + 28, legend_y - 10), "background", fill=(50, 50, 50))
    draw.rectangle((margin_left + 160, legend_y - 10, margin_left + 178, legend_y + 8), fill=(52, 152, 219))
    draw.text((margin_left + 188, legend_y - 10), "vessel", fill=(50, 50, 50))
    return np.array(image, dtype=np.uint8)


def _save_grid(panels: Sequence[tuple[str, np.ndarray]], out_path: Path, columns: int = 4) -> None:
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


def _panel_filename(title: str) -> str:
    static_aliases = {
        "Rule Membership Map": "mu_map",
        "Geometry Kernel Response": "kernel",
        "Compatibility Map": "compat",
        "Geometry Overlay": "geometry",
        "Local Geometry Regimes": "rule_partition",
        "Rule Uncertainty": "rule_uncertainty",
        "Geometry Direction Field": "direction_field",
        "Geometry Contribution Map": "anisotropy_gain",
        "Rule Statistics": "rule_stats",
        "Input": "input",
        "Ground Truth": "ground_truth",
        "Prediction": "prediction",
        "Error Map": "error_map",
    }
    if title.startswith("Vessel-Focused Regime "):
        suffix = title.split("Vessel-Focused Regime ", 1)[1].strip().lower().replace(" ", "")
        return f"vessel_rule_{suffix}"
    return static_aliases.get(title, title.lower().replace(" ", "_"))


def _resolve_run(results_dir: Path, run_name: str | None, variant: str | None):
    runs = discover_drive_runs(results_dir)
    if not runs:
        raise SystemExit(f"No DRIVE runs with checkpoints found under {results_dir}")
    if run_name is not None:
        for run in runs:
            if run.name == run_name:
                return run
        raise SystemExit(f"Run '{run_name}' was not found under {results_dir}")
    if variant is not None:
        for run in runs:
            if run.variant == variant:
                return run
        raise SystemExit(f"No run for variant '{variant}' was found under {results_dir}")
    return runs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export article-ready segmentation and interpretation figures for a retinal vessel segmentation run.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--config", type=str, default="configs/drive_benchmark.yaml")
    parser.add_argument("--run", type=str, default=None, help="Explicit run directory name.")
    parser.add_argument("--variant", type=str, default=None, help="Pick the best run for this variant if --run is not set.")
    parser.add_argument("--split", type=str, default="test", choices=("test", "training"))
    parser.add_argument("--samples", type=str, default="0", help="Comma-separated sample indices.")
    parser.add_argument("--output-dir", type=str, default="article_assets/exports")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--rule-index", type=int, default=0)
    args = parser.parse_args()

    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    config_path = (PROJECT_ROOT / args.config).resolve()
    run = _resolve_run(results_dir, args.run, args.variant)
    model, cfg, device = _load_model_for_run(run, config_path, args.device)

    data_root = Path(cfg.get("data_root", "./data"))
    dataset_root = (PROJECT_ROOT / utils.retinal_dataset_root(data_root, cfg["dataset"])).resolve()
    dataset = utils.DriveDataset(
        root=dataset_root,
        split=args.split,
        augment=False,
        use_fov_mask=bool(cfg.get("use_fov_mask", True)),
    )
    threshold = float(args.threshold) if args.threshold is not None else recommended_threshold_for_run(run, default_threshold=0.6)
    sample_indices = _parse_indices(args.samples)

    out_root = (PROJECT_ROOT / args.output_dir / run.name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "run_name": run.name,
        "variant": run.variant,
        "split": args.split,
        "threshold": threshold,
        "device": str(device),
        "samples": [],
    }

    for raw_idx in sample_indices:
        idx = min(max(int(raw_idx), 0), len(dataset.samples) - 1)
        sample = _prediction_for_sample(run, model, dataset, idx, device)
        pred = sample["prob"] >= threshold
        gt = sample["mask"] > 0.5
        valid = sample["valid"] > 0.5

        original = sample["image"]
        gt_overlay = _mask_to_overlay(original, gt & valid, color=(50, 220, 80))
        pred_overlay = _mask_to_overlay(original, pred & valid, color=(255, 180, 40))
        error_rgb = _error_map(pred, gt, valid)
        metrics = metrics_from_prob_map(sample["prob"], sample["mask"], sample["valid"], threshold=threshold)

        panels: list[tuple[str, np.ndarray]] = [
            ("Input", original),
            ("Ground Truth", gt_overlay),
            ("Prediction", pred_overlay),
            ("Error Map", error_rgb),
        ]

        layer_infos = sample["layers"]
        if layer_infos:
            layer_idx = min(max(int(args.layer_index), 0), len(layer_infos) - 1)
            layer_info = layer_infos[layer_idx]
            snapshot = layer_info["snapshot"]
            rule_idx = min(max(int(args.rule_index), 0), int(snapshot.get("num_rules", 1)) - 1)
            mu_map = np.asarray(snapshot["mu_map"])
            kernel_map = np.asarray(snapshot["kernel_map"])
            compat_map = np.asarray(snapshot["compat_map"])
            mu_overlay = _heat_overlay(original, mu_map[rule_idx])
            kernel_rgb = _scalar_map_to_rgb(kernel_map[rule_idx])
            compat_rgb = _scalar_map_to_rgb(compat_map[rule_idx])
            geometry_rgb = _geometry_visualization(original, snapshot, rule_idx)
            rule_partition_rgb = _rule_partition_rgb(mu_map, sample["valid"])
            rule_entropy_rgb = _rule_entropy_overlay(original, mu_map, sample["valid"])
            rule_stats = _rule_stats(mu_map, sample["mask"], sample["valid"])
            best_rule_idx = _best_vessel_rule(rule_stats)
            vessel_rule_overlay = _heat_overlay(original, mu_map[best_rule_idx], alpha=0.75)
            rule_stats_rgb = _draw_rule_stats_chart(rule_stats, int(snapshot.get("num_rules", mu_map.shape[0])))
            direction_map, direction_strength, gain_map = _geometry_direction_and_gain_maps(
                snapshot,
                mu_map,
                best_rule_idx,
                sample["valid"],
            )
            geometry_direction_rgb = _geometry_direction_overlay(
                original,
                direction_map,
                direction_strength,
                sample["valid"],
            )
            geometry_gain_rgb = _geometry_gain_overlay(
                original,
                gain_map,
                sample["valid"],
            )
            panels.extend(
                [
                    ("Rule Membership Map", mu_overlay),
                    ("Geometry Kernel Response", kernel_rgb),
                    ("Compatibility Map", compat_rgb),
                    ("Geometry Overlay", geometry_rgb),
                    ("Local Geometry Regimes", rule_partition_rgb),
                    ("Rule Uncertainty", rule_entropy_rgb),
                    (f"Vessel-Focused Regime R{best_rule_idx + 1}", vessel_rule_overlay),
                    ("Geometry Direction Field", geometry_direction_rgb),
                    ("Geometry Contribution Map", geometry_gain_rgb),
                    ("Rule Statistics", rule_stats_rgb),
                ]
            )
            layer_name = layer_info["name"]
        else:
            layer_name = None
            rule_idx = None
            rule_stats = None
            best_rule_idx = None

        sample_dir = out_root / f"{idx:03d}_{sample['sample_id']}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        _save_grid(panels, sample_dir / "article_grid.png")
        for title, image in panels:
            safe_title = _panel_filename(title)
            Image.fromarray(image.astype(np.uint8)).save(sample_dir / f"{safe_title}.png")
        Image.fromarray((pred.astype(np.uint8) * 255)).save(sample_dir / "prediction_mask.png")
        Image.fromarray((gt.astype(np.uint8) * 255)).save(sample_dir / "ground_truth_mask.png")
        Image.fromarray((valid.astype(np.uint8) * 255)).save(sample_dir / "valid_mask.png")
        with open(sample_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        if rule_stats is not None:
            with open(sample_dir / "rule_stats.json", "w", encoding="utf-8") as handle:
                json.dump(rule_stats, handle, indent=2)
            with open(sample_dir / "rule_summary.json", "w", encoding="utf-8") as handle:
                json.dump({"best_vessel_rule_index": int(best_rule_idx), "best_vessel_rule_name": f"R{int(best_rule_idx) + 1}"}, handle, indent=2)

        manifest["samples"].append(
            {
                "index": idx,
                "sample_id": sample["sample_id"],
                "metrics": metrics,
                "layer_name": layer_name,
                "rule_index": rule_idx,
                "best_vessel_rule_index": best_rule_idx,
                "output_dir": str(sample_dir),
            }
        )

    with open(out_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Exported article assets to {out_root}")


if __name__ == "__main__":
    main()
