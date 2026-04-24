from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train
import utils


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    return train.load_checkpoint_payload(path)


def _resolve_threshold(cfg: dict[str, Any], checkpoint_path: Path, override: str) -> float:
    if override and override.lower() != "auto":
        return float(override)
    metrics_path = checkpoint_path.parent / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "selected_threshold" in metrics:
            return float(metrics["selected_threshold"])
    return float(cfg.get("seg_threshold", 0.5))


def _load_model(cfg: dict[str, Any], payload: dict[str, Any], device: torch.device, tta_mode: str) -> torch.nn.Module:
    in_channels, num_outputs = utils.dataset_channels_and_outputs(cfg["dataset"])
    model = utils.build_model(
        payload.get("variant", cfg.get("variant", "az_thesis")),
        num_outputs=num_outputs,
        in_channels=in_channels,
        num_rules=int(cfg.get("num_rules", 4)),
        task=utils.task_for_dataset(cfg["dataset"]),
        widths=utils.parse_model_widths(cfg.get("model_widths")),
        model_kwargs=utils.resolve_segmentation_model_kwargs(cfg),
        az_cfg_kwargs=utils.resolve_azconv_config_kwargs(cfg),
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.eval()
    if str(tta_mode).lower().strip() not in {"", "none", "off"}:
        model = utils.SegmentationTTAWrapper(model, mode=tta_mode).to(device)
    return model


def _predict_prob(model: torch.nn.Module, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        logits, _, _ = utils.unpack_segmentation_outputs(output)
        return torch.sigmoid(logits)[0, 0].detach().cpu()


def _save_gray(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(array, 0.0, 1.0)
    Image.fromarray((clipped * 255.0).astype(np.uint8), mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build false-negative hard-mining maps for retinal vessel datasets.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-tta", type=str, default="flips")
    parser.add_argument("--threshold", type=str, default="auto")
    parser.add_argument("--thin-weight", type=float, default=1.5)
    parser.add_argument("--spread-kernel", type=int, default=1)
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    checkpoint_path = (PROJECT_ROOT / args.checkpoint).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()

    cfg = utils.load_config(str(config_path))
    payload = _checkpoint_payload(checkpoint_path)
    if payload.get("cfg"):
        cfg.update(payload["cfg"])
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    threshold = _resolve_threshold(cfg, checkpoint_path, args.threshold)

    dataset_root = utils.retinal_dataset_root(cfg.get("data_root", "./data"), cfg["dataset"])
    dataset = utils.DriveDataset(
        dataset_root,
        split=args.split,
        augment=False,
        use_fov_mask=bool(cfg.get("use_fov_mask", True)),
        crop_size=None,
        input_mode=str(cfg.get("retinal_input_mode", "rgb")),
        green_blend_alpha=float(cfg.get("retinal_green_blend_alpha", 0.35)),
    )
    model = _load_model(cfg, payload, device, args.eval_tta)

    manifest: list[dict[str, Any]] = []
    for image_path, mask_path, fov_path in dataset.samples:
        sample_id = utils._retinal_sample_id(image_path)
        image = dataset._load_rgb(image_path)
        mask = dataset._load_mask(mask_path)[0]
        fov = dataset._load_mask(fov_path)[0] if dataset.use_fov_mask else torch.ones_like(mask)
        prob = _predict_prob(model, dataset._normalize(image), device)

        vessel = ((mask > 0.5) & (fov > 0.5)).float()
        base_error = (threshold - prob).clamp(min=0.0) * vessel
        if float(args.thin_weight) > 1.0:
            thin = dataset._thin_vessel_candidates(mask.unsqueeze(0), fov.unsqueeze(0))[0].float()
            base_error = base_error * (1.0 + (float(args.thin_weight) - 1.0) * thin)
        spread_kernel = max(1, int(args.spread_kernel))
        if spread_kernel > 1:
            if spread_kernel % 2 == 0:
                raise ValueError("--spread-kernel must be odd.")
            base_error = F.max_pool2d(
                base_error.unsqueeze(0).unsqueeze(0),
                kernel_size=spread_kernel,
                stride=1,
                padding=spread_kernel // 2,
            )[0, 0]
        if base_error.max().item() > 0.0:
            norm = base_error / base_error.max().item()
        else:
            norm = base_error

        _save_gray(output_dir / f"{sample_id}.png", norm.numpy())
        manifest.append(
            {
                "sample_id": sample_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "fov_path": str(fov_path),
                "threshold": threshold,
                "tta_mode": str(args.eval_tta),
                "thin_weight": float(args.thin_weight),
                "spread_kernel": spread_kernel,
                "hard_pixels": int((norm > 0).sum().item()),
                "hard_mass": float(norm.sum().item()),
                "max_weight": float(norm.max().item()),
            }
        )
        print(f"[{args.split}] {sample_id}: hard_pixels={manifest[-1]['hard_pixels']} hard_mass={manifest[-1]['hard_mass']:.2f}")

    summary = {
        "dataset": cfg["dataset"],
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "eval_tta": str(args.eval_tta),
        "thin_weight": float(args.thin_weight),
        "spread_kernel": spread_kernel,
        "num_samples": len(manifest),
        "output_dir": str(output_dir),
        "samples": manifest,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
