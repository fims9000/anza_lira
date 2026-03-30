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
    _normalize_for_model,
    _scalar_map_to_rgb,
    _to_uint8_rgb,
    discover_drive_runs,
    metrics_from_prob_map,
    recommended_threshold_for_run,
)
from models.azconv import AZConv2d
from utils import build_model


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
    parser = argparse.ArgumentParser(description="Export article-ready segmentation and interpretation figures for a DRIVE run.")
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
    drive_root = (PROJECT_ROOT / data_root / "DRIVE").resolve()
    dataset = utils.DriveDataset(
        root=drive_root,
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
            panels.extend(
                [
                    ("mu Map", mu_overlay),
                    ("Kernel", kernel_rgb),
                    ("Compat", compat_rgb),
                    ("Geometry", geometry_rgb),
                ]
            )
            layer_name = layer_info["name"]
        else:
            layer_name = None
            rule_idx = None

        sample_dir = out_root / f"{idx:03d}_{sample['sample_id']}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        _save_grid(panels, sample_dir / "article_grid.png")
        for title, image in panels:
            safe_title = title.lower().replace(" ", "_")
            Image.fromarray(image.astype(np.uint8)).save(sample_dir / f"{safe_title}.png")
        with open(sample_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        manifest["samples"].append(
            {
                "index": idx,
                "sample_id": sample["sample_id"],
                "metrics": metrics,
                "layer_name": layer_name,
                "rule_index": rule_idx,
                "output_dir": str(sample_dir),
            }
        )

    with open(out_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Exported article assets to {out_root}")


if __name__ == "__main__":
    main()
