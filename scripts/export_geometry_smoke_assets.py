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
from drive_viewer import _heat_overlay, _mask_to_overlay, _normalize_for_model, _scalar_map_to_rgb, _to_uint8_rgb
from models.azconv import AZConv2d
from scripts.export_drive_article_assets import (
    _geometry_direction_and_gain_maps,
    _geometry_direction_overlay,
    _geometry_gain_overlay,
    _rule_entropy_overlay,
    _rule_partition_rgb,
    _rule_stats,
)


def _crop_box_around_mask(mask: np.ndarray, crop_size: int) -> tuple[int, int, int, int]:
    h, w = mask.shape
    ys, xs = np.where(mask > 0.5)
    if ys.size:
        cy = int(np.median(ys))
        cx = int(np.median(xs))
    else:
        cy, cx = h // 2, w // 2
    crop_h = min(int(crop_size), h)
    crop_w = min(int(crop_size), w)
    top = min(max(cy - crop_h // 2, 0), h - crop_h)
    left = min(max(cx - crop_w // 2, 0), w - crop_w)
    return top, left, top + crop_h, left + crop_w


def _crop_chw(tensor: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
    top, left, bottom, right = box
    return tensor[:, top:bottom, left:right]


def _label_panel(image: np.ndarray, title: str, bar_height: int = 34) -> Image.Image:
    panel = Image.fromarray(image.astype(np.uint8))
    labeled = Image.new("RGB", (panel.width, panel.height + bar_height), color=(244, 244, 240))
    labeled.paste(panel, (0, bar_height))
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, labeled.width, bar_height), fill=(27, 34, 46))
    draw.text((10, 8), title, fill=(255, 255, 255))
    return labeled


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


def _simple_rule_stats_chart(stats: dict[str, list[float]], num_rules: int) -> np.ndarray:
    width, height = 720, 360
    margin_left, margin_right, margin_top, row_h = 110, 30, 54, 34
    canvas = Image.new("RGB", (width, height), color=(246, 245, 241))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 40), fill=(27, 34, 46))
    draw.text((14, 12), "Rule statistics: vessel vs background", fill=(255, 255, 255))
    usable_w = width - margin_left - margin_right
    vessel = stats.get("vessel_mean", [0.0] * num_rules)
    background = stats.get("background_mean", [0.0] * num_rules)
    for idx in range(num_rules):
        y = margin_top + idx * row_h
        draw.text((16, y + 4), f"R{idx + 1}", fill=(25, 31, 42))
        bg_w = int(usable_w * float(background[idx]))
        v_w = int(usable_w * float(vessel[idx]))
        draw.rectangle((margin_left, y, margin_left + bg_w, y + 10), fill=(160, 172, 190))
        draw.rectangle((margin_left, y + 14, margin_left + v_w, y + 24), fill=(52, 152, 219))
    draw.text((margin_left, height - 32), "upper: background mean, lower: vessel mean", fill=(60, 60, 60))
    return np.asarray(canvas, dtype=np.uint8)


def _panel_slug(title: str) -> str:
    return title.lower().replace("/", "_").replace(" ", "_").replace("-", "_")


def _scalar_panel(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    rgb = _scalar_map_to_rgb(np.asarray(values, dtype=np.float32))
    if rgb.shape[0] == size[1] and rgb.shape[1] == size[0]:
        return rgb
    return np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.NEAREST), dtype=np.uint8)


def _resize_rule_maps(rule_maps: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = np.asarray(rule_maps, dtype=np.float32)
    if arr.shape[-2:] == (height, width):
        return arr
    resized = []
    for idx in range(arr.shape[0]):
        image = Image.fromarray(arr[idx].astype(np.float32), mode="F")
        resized.append(np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32))
    out = np.stack(resized, axis=0)
    denom = out.sum(axis=0, keepdims=True)
    if np.all(denom > 1e-8):
        out = out / np.maximum(denom, 1e-8)
    return out


def _build_model(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    variant = str(cfg.get("variant", "az_cat"))
    in_channels, num_outputs = utils.dataset_channels_and_outputs(cfg["dataset"])
    model = utils.build_model(
        variant,
        num_outputs=num_outputs,
        in_channels=in_channels,
        num_rules=int(cfg.get("num_rules", 6)),
        task=utils.task_for_dataset(cfg["dataset"]),
        widths=utils.parse_model_widths(cfg.get("model_widths")),
        model_kwargs=utils.resolve_segmentation_model_kwargs(cfg),
        az_cfg_kwargs=utils.resolve_azconv_config_kwargs(cfg),
    ).to(device)
    model.eval()
    return model


def _architecture_summary(model: torch.nn.Module) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, AZConv2d):
            row = {"name": name, "rules": int(module.R), "kernel_size": int(module.k)}
            row.update(module.metric_tensor_summary())
            snapshot = module.interpretation_snapshot()
            mu_rule_mean = snapshot.get("mu_rule_mean")
            if isinstance(mu_rule_mean, torch.Tensor):
                probs = mu_rule_mean.float().clamp_min(1e-8)
                probs = probs / probs.sum().clamp_min(1e-8)
                row["rule_usage_entropy_norm"] = float((-(probs * probs.log()).sum() / np.log(max(int(module.R), 2))).cpu())
            compat_map = snapshot.get("compat_map")
            if isinstance(compat_map, torch.Tensor):
                row["compat_mass"] = float(compat_map.float().sum().cpu())
            rows.append(row)
    conds = [float(row["metric_condition"]) for row in rows if "metric_condition" in row]
    gaps = [float(row["anisotropy_gap"]) for row in rows if "anisotropy_gap" in row]
    return {
        "az_layer_count": len(rows),
        "az_layers": rows,
        "metric_condition_mean": float(np.mean(conds)) if conds else None,
        "metric_condition_min": float(np.min(conds)) if conds else None,
        "metric_condition_max": float(np.max(conds)) if conds else None,
        "anisotropy_gap_mean": float(np.mean(gaps)) if gaps else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CPU-only AZ geometry smoke-test assets on a tiny labeled split.")
    parser.add_argument("--config", type=str, default="configs/drive_az_thesis_article_current.yaml")
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/geometry_smoke_drive")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = utils.load_config(str(PROJECT_ROOT / args.config))
    device = torch.device(args.device)
    utils.set_seed(int(cfg.get("seed", 42)), deterministic=True)

    dataset_root = PROJECT_ROOT / utils.retinal_dataset_root(cfg.get("data_root", "./data"), cfg["dataset"])
    dataset = utils.DriveDataset(dataset_root, split=args.split, augment=False, use_fov_mask=bool(cfg.get("use_fov_mask", True)))
    idx = min(max(int(args.sample_index), 0), len(dataset.samples) - 1)
    image_path, mask_path, fov_path = dataset.samples[idx]
    image = dataset._load_rgb(image_path)
    mask = dataset._load_mask(mask_path)
    valid = dataset._load_mask(fov_path) if dataset.use_fov_mask else torch.ones_like(mask)

    box = _crop_box_around_mask(mask[0].numpy(), int(args.crop_size))
    image = _crop_chw(image, box)
    mask = _crop_chw(mask, box)
    valid = _crop_chw(valid, box)
    image_rgb = _to_uint8_rgb(np.transpose(image.numpy(), (1, 2, 0)))
    x = _normalize_for_model(image_rgb).unsqueeze(0).to(device)

    model = _build_model(cfg, device)
    with torch.no_grad():
        _ = model(x)

    layers: list[tuple[str, dict[str, Any]]] = []
    for name, module in model.named_modules():
        if isinstance(module, AZConv2d):
            snap = module.interpretation_snapshot()
            if snap:
                layers.append((name, snap))
    if not layers:
        raise SystemExit("No AZConv2d snapshots were produced.")
    layer_idx = min(max(int(args.layer_index), 0), len(layers) - 1)
    layer_name, snapshot = layers[layer_idx]

    valid_np = valid[0].numpy().astype(np.float32)
    mask_np = mask[0].numpy().astype(np.float32)
    mu_map = _resize_rule_maps(np.asarray(snapshot["mu_map"], dtype=np.float32), *valid_np.shape)
    stats = _rule_stats(mu_map, mask_np, valid_np)
    best_rule = int(np.asarray(stats["delta"], dtype=np.float32).argmax())
    direction_map, direction_strength, gain_map = _geometry_direction_and_gain_maps(snapshot, mu_map, best_rule, valid_np)

    panels: list[tuple[str, np.ndarray]] = [
        ("Input crop", image_rgb),
        ("Ground truth", _mask_to_overlay(image_rgb, (mask_np > 0.5) & (valid_np > 0.5), color=(50, 220, 80))),
        ("Rule partition", _rule_partition_rgb(mu_map, valid_np)),
        ("Rule entropy", _rule_entropy_overlay(image_rgb, mu_map, valid_np)),
        (f"Rule R{best_rule + 1} membership", _heat_overlay(image_rgb, mu_map[best_rule])),
        ("Kernel response", _scalar_panel(np.asarray(snapshot["kernel_map"], dtype=np.float32)[best_rule], image_rgb.shape[1::-1])),
        ("Compatibility map", _scalar_panel(np.asarray(snapshot["compat_map"], dtype=np.float32)[best_rule], image_rgb.shape[1::-1])),
        ("Direction field", _geometry_direction_overlay(image_rgb, direction_map, direction_strength, valid_np)),
        ("Geometry gain", _geometry_gain_overlay(image_rgb, gain_map, valid_np)),
        ("Rule statistics", _simple_rule_stats_chart(stats, int(snapshot.get("num_rules", mu_map.shape[0])))),
    ]

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for title, image_arr in panels:
        Image.fromarray(image_arr.astype(np.uint8)).save(out_dir / f"{_panel_slug(title)}.png")
    _save_grid(panels, out_dir / "geometry_smoke_grid.png", columns=3)

    manifest = {
        "purpose": "CPU-only geometry smoke test, no training and no checkpoint metrics.",
        "config": str(args.config),
        "dataset": str(cfg["dataset"]),
        "split": args.split,
        "sample_index": idx,
        "sample_id": image_path.stem,
        "crop_box_top_left_bottom_right": list(box),
        "layer_index": layer_idx,
        "layer_name": layer_name,
        "best_rule_index": best_rule,
        "best_rule_name": f"R{best_rule + 1}",
        "rule_stats": stats,
        "architecture_summary": _architecture_summary(model),
        "output_dir": str(out_dir),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest["architecture_summary"], indent=2))
    print(f"Exported geometry smoke assets to {out_dir}")


if __name__ == "__main__":
    main()
