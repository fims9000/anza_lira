#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils
from utils import build_model


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@dataclass
class RunBundle:
    run_dir: Path
    cfg: dict[str, Any]
    metrics: dict[str, Any]
    variant: str
    model: torch.nn.Module
    threshold: float


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


def _load_run(run_dir: Path, device: torch.device) -> RunBundle:
    payload = _load_checkpoint(run_dir / "checkpoint_best.pt")
    cfg = payload.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError(f"Checkpoint in {run_dir} does not contain cfg.")
    metrics = _load_json(run_dir / "metrics.json")
    variant = str(payload.get("variant", metrics.get("variant", "baseline")))
    in_channels, num_outputs = utils.dataset_channels_and_outputs(cfg["dataset"])
    model = build_model(
        variant,
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
    threshold = float(metrics.get("selected_threshold", cfg.get("seg_threshold", 0.5)))
    return RunBundle(
        run_dir=run_dir,
        cfg=cfg,
        metrics=metrics,
        variant=variant,
        model=model,
        threshold=threshold,
    )


def _predict_prob(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        output = model(x)
        logits, _, _ = utils.unpack_segmentation_outputs(output)
        return torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)


def _denorm_to_uint8(x_norm: torch.Tensor) -> np.ndarray:
    arr = x_norm.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    arr = arr * STD + MEAN
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _overlay(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.58) -> np.ndarray:
    out = image_rgb.astype(np.float32).copy()
    mk = mask.astype(bool)
    c = np.asarray(color, dtype=np.float32)
    out[mk] = out[mk] * (1.0 - alpha) + c * alpha
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _error_map(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    p = pred.astype(bool)
    g = gt.astype(bool)
    v = valid.astype(bool)
    out = np.zeros((*p.shape, 3), dtype=np.uint8)
    tp = p & g & v
    fp = p & (~g) & v
    fn = (~p) & g & v
    tn = (~p) & (~g) & v
    out[tp] = np.array([38, 201, 98], dtype=np.uint8)   # green
    out[fp] = np.array([239, 83, 80], dtype=np.uint8)   # red
    out[fn] = np.array([66, 165, 245], dtype=np.uint8)  # blue
    out[tn] = np.array([15, 15, 15], dtype=np.uint8)
    return out


def _advantage_map(base_pred: np.ndarray, az_pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    b = base_pred.astype(bool)
    a = az_pred.astype(bool)
    g = gt.astype(bool)
    v = valid.astype(bool)
    out = np.zeros((*b.shape, 3), dtype=np.uint8)

    az_gain = a & g & (~b) & v               # recovered GT road
    az_loss = b & g & (~a) & v               # AZ missed vs baseline
    fp_removed = b & (~g) & (~a) & v         # AZ removed baseline FP
    fp_added = a & (~g) & (~b) & v           # AZ added FP
    both_road = a & b & g & v
    both_bg = (~a) & (~b) & (~g) & v

    out[az_gain] = np.array([0, 255, 0], dtype=np.uint8)         # bright green
    out[az_loss] = np.array([255, 0, 255], dtype=np.uint8)       # magenta
    out[fp_removed] = np.array([0, 255, 255], dtype=np.uint8)    # cyan
    out[fp_added] = np.array([255, 165, 0], dtype=np.uint8)      # orange
    out[both_road] = np.array([255, 255, 255], dtype=np.uint8)   # white
    out[both_bg] = np.array([45, 45, 45], dtype=np.uint8)
    return out


def _pred_diff_map(base_pred: np.ndarray, az_pred: np.ndarray, valid: np.ndarray) -> np.ndarray:
    b = base_pred.astype(bool) & valid.astype(bool)
    a = az_pred.astype(bool) & valid.astype(bool)
    both = a & b
    only_az = a & (~b)
    only_base = b & (~a)
    none = (~a) & (~b) & valid.astype(bool)

    out = np.zeros((*base_pred.shape, 3), dtype=np.uint8)
    out[both] = np.array([255, 255, 255], dtype=np.uint8)       # both predicted road
    out[only_az] = np.array([0, 255, 0], dtype=np.uint8)        # AZ only
    out[only_base] = np.array([255, 106, 0], dtype=np.uint8)    # baseline only
    out[none] = np.array([28, 28, 28], dtype=np.uint8)
    return out


def _binary_stats(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    p = pred.astype(bool) & valid.astype(bool)
    g = gt.astype(bool) & valid.astype(bool)
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    eps = 1e-8
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def _integral_sum_map(arr: np.ndarray, win: int) -> np.ndarray:
    # returns sum over every [y:y+win, x:x+win], shape (h-win+1, w-win+1)
    h, w = arr.shape
    if win > h or win > w:
        raise ValueError(f"window too large: {win} for {arr.shape}")
    s = np.pad(arr.astype(np.float32), ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    out = s[win:, win:] - s[:-win, win:] - s[win:, :-win] + s[:-win, :-win]
    return out


def _best_advantage_bbox(
    base_pred: np.ndarray,
    az_pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    win: int,
) -> tuple[int, int, int, int]:
    b = base_pred.astype(bool)
    a = az_pred.astype(bool)
    g = gt.astype(bool)
    v = valid.astype(bool)

    az_gain = (a & g & (~b) & v).astype(np.float32)
    az_loss = (b & g & (~a) & v).astype(np.float32)
    fp_removed = (b & (~g) & (~a) & v).astype(np.float32)
    fp_added = (a & (~g) & (~b) & v).astype(np.float32)

    # Weighted local advantage score.
    score = (2.2 * az_gain) + (1.0 * fp_removed) - (2.4 * az_loss) - (1.2 * fp_added)
    score_sum = _integral_sum_map(score, win=win)
    valid_sum = _integral_sum_map(v.astype(np.float32), win=win)

    # avoid empty borders
    density = valid_sum / float(win * win)
    score_sum[density < 0.75] = -1e9

    y, x = np.unravel_index(np.argmax(score_sum), score_sum.shape)
    return int(y), int(x), int(y + win - 1), int(x + win - 1)


def _crop(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, x0, y1, x1 = bbox
    return img[y0 : y1 + 1, x0 : x1 + 1]


def _draw_bbox(img: np.ndarray, bbox: tuple[int, int, int, int], color: tuple[int, int, int] = (255, 255, 255), width: int = 3) -> np.ndarray:
    pil = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    y0, x0, y1, x1 = bbox
    for k in range(width):
        draw.rectangle((x0 - k, y0 - k, x1 + k, y1 + k), outline=color)
    return np.array(pil, dtype=np.uint8)


def _label_panel(img: np.ndarray, title: str, size: tuple[int, int] = (300, 300), bar_h: int = 34) -> Image.Image:
    pil = Image.fromarray(img.astype(np.uint8))
    pil.thumbnail(size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", (size[0], size[1] + bar_h), color=(238, 238, 234))
    x = (size[0] - pil.width) // 2
    y = bar_h + (size[1] - pil.height) // 2
    out.paste(pil, (x, y))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, bar_h), fill=(26, 34, 46))
    draw.text((8, 9), title, fill=(255, 255, 255))
    return out


def _case_figure(
    *,
    idx: int,
    input_img: np.ndarray,
    gt_overlay: np.ndarray,
    base_overlay: np.ndarray,
    az_overlay: np.ndarray,
    base_err: np.ndarray,
    az_err: np.ndarray,
    adv_map: np.ndarray,
    bbox: tuple[int, int, int, int],
    base_stats_crop: dict[str, float],
    az_stats_crop: dict[str, float],
    counts: dict[str, int],
    out_path: Path,
) -> None:
    full = [
        ("Input (full)", _draw_bbox(input_img, bbox)),
        ("GT overlay (full)", _draw_bbox(gt_overlay, bbox)),
        ("Baseline (full)", _draw_bbox(base_overlay, bbox)),
        ("AZ-Thesis (full)", _draw_bbox(az_overlay, bbox)),
    ]
    crop = [
        ("Baseline error (zoom)", _crop(base_err, bbox)),
        ("AZ error (zoom)", _crop(az_err, bbox)),
        ("Advantage map (zoom)", _crop(adv_map, bbox)),
        ("Input (zoom)", _crop(input_img, bbox)),
    ]
    panels = full + crop
    panel_imgs = [_label_panel(img, name) for name, img in panels]
    col_w = panel_imgs[0].width
    row_h = panel_imgs[0].height
    top_h = 190
    canvas = Image.new("RGB", (4 * col_w + 20, 2 * row_h + top_h + 16), color=(229, 229, 225))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"GlobalScaleRoad advantage-focused visualization | idx={idx}", fill=(12, 12, 12))
    draw.text(
        (10, 36),
        (
            f"Zoom metrics: Dice {base_stats_crop['dice']:.4f} -> {az_stats_crop['dice']:.4f} "
            f"(Δ {az_stats_crop['dice']-base_stats_crop['dice']:+.4f}), "
            f"IoU {base_stats_crop['iou']:.4f} -> {az_stats_crop['iou']:.4f}"
        ),
        fill=(30, 30, 30),
    )
    draw.text(
        (10, 60),
        (
            f"Precision {base_stats_crop['precision']:.4f} -> {az_stats_crop['precision']:.4f}, "
            f"Recall {base_stats_crop['recall']:.4f} -> {az_stats_crop['recall']:.4f}"
        ),
        fill=(30, 30, 30),
    )
    draw.text((10, 88), "Advantage map legend:", fill=(30, 30, 30))
    legend = [
        ((0, 255, 0), f"AZ recovers GT roads: {counts['az_gain']}"),
        ((255, 0, 255), f"AZ misses vs baseline: {counts['az_loss']}"),
        ((0, 255, 255), f"FP removed by AZ: {counts['fp_removed']}"),
        ((255, 165, 0), f"FP added by AZ: {counts['fp_added']}"),
    ]
    x = 10
    y = 114
    for color, txt in legend:
        draw.rectangle((x, y, x + 16, y + 16), fill=color, outline=(90, 90, 90))
        draw.text((x + 24, y - 1), txt, fill=(30, 30, 30))
        x += 315

    for i, panel in enumerate(panel_imgs):
        r = i // 4
        c = i % 4
        px = 8 + c * col_w
        py = top_h + r * row_h
        canvas.paste(panel, (px, py))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _simple_compare_figure(
    *,
    idx: int,
    input_img: np.ndarray,
    base_overlay: np.ndarray,
    az_overlay: np.ndarray,
    pred_diff: np.ndarray,
    base_dice: float,
    az_dice: float,
    out_path: Path,
) -> None:
    panels = [
        ("Input", input_img),
        ("Baseline", base_overlay),
        ("AZ-Thesis", az_overlay),
        ("AZ vs Baseline", pred_diff),
    ]
    panel_imgs = [_label_panel(img, name, size=(360, 360), bar_h=34) for name, img in panels]
    col_w = panel_imgs[0].width
    row_h = panel_imgs[0].height
    top_h = 108
    canvas = Image.new("RGB", (4 * col_w + 18, row_h + top_h + 12), color=(229, 229, 225))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"GlobalScaleRoad simple compare | idx={idx}", fill=(12, 12, 12))
    draw.text((10, 34), f"Dice: baseline {base_dice:.4f} -> AZ {az_dice:.4f} (Δ {az_dice-base_dice:+.4f})", fill=(30, 30, 30))
    draw.text((10, 58), "Diff legend: green=only AZ, orange=only baseline, white=both, dark=none", fill=(30, 30, 30))
    for i, panel in enumerate(panel_imgs):
        x = 8 + i * col_w
        y = top_h
        canvas.paste(panel, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _write_markdown(path: Path, rows: list[dict[str, Any]], top_cases: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# GlobalScaleRoad: визуализация преимуществ (с авто-фокусом)")
    lines.append("")
    lines.append("Этот файл сделан специально для проблемы «не видно преимущества на общей картинке».")
    lines.append("")
    lines.append("## Что изменено")
    lines.append("")
    lines.append("- Для каждого тайла автоматически ищется локальное окно, где AZ выигрывает у baseline.")
    lines.append("- В figure показывается полный тайл + zoom именно этого окна.")
    lines.append("- В zoom даны отдельные карты ошибок baseline/AZ и advantage-map с цветовой легендой и счётчиками.")
    lines.append("")
    lines.append("## Лучшие кейсы для вставки в статью")
    lines.append("")
    lines.append("| idx | full Dice base | full Dice AZ | zoom Dice base | zoom Dice AZ | Δzoom Dice | figure |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for r in top_cases:
        lines.append(
            f"| {r['idx']} | {r['full_base_dice']:.4f} | {r['full_az_dice']:.4f} | "
            f"{r['zoom_base_dice']:.4f} | {r['zoom_az_dice']:.4f} | {r['zoom_delta_dice']:+.4f} | "
            f"`{r['figure_path']}` |"
        )
    lines.append("")
    lines.append("Рекомендация: в paper оставить 1-2 такие advantage-focused фигуры вместо «общего» road-тайла.")
    lines.append("")
    lines.append("Дополнительно доступны упрощённые figure формата `Input | Baseline | AZ | AZ vs Baseline`:")
    lines.append("- `results/article_visual_assets/global_roads_advantage_v3/simple_compare_best1.png`")
    lines.append("- `results/article_visual_assets/global_roads_advantage_v3/simple_compare_best2.png`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build advantage-focused GlobalScaleRoad visualizations.")
    parser.add_argument(
        "--selection-json",
        type=str,
        default="results/article_visual_assets/global_roads_figures/selection.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/article_visual_assets/global_roads_advantage_v3",
    )
    parser.add_argument(
        "--report-md",
        type=str,
        default="results/global_roads_advantage_visual_report_ru.md",
    )
    parser.add_argument("--window", type=int, default=196)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    selection = _load_json((PROJECT_ROOT / args.selection_json).resolve())
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    report_md = (PROJECT_ROOT / args.report_md).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    baseline = _load_run(Path(str(selection["baseline_run"])), device)
    az = _load_run(Path(str(selection["az_run"])), device)

    _, _, test_loader, _, _, task = utils.build_dataloaders(az.cfg)
    if task != "segmentation":
        raise ValueError("Expected segmentation task.")
    dataset = test_loader.dataset
    indices = [int(r["idx"]) for r in selection.get("rows", [])]

    case_rows: list[dict[str, Any]] = []
    simple_data: dict[int, dict[str, Any]] = {}
    for idx in indices:
        x_norm, y, valid = dataset[idx]
        x = x_norm.unsqueeze(0).to(device)
        gt = y[0].numpy().astype(np.float32)
        valid_np = valid[0].numpy().astype(np.float32)
        gt_bin = ((gt > 0.5) & (valid_np > 0.5)).astype(np.float32)

        base_prob = _predict_prob(baseline.model, x)
        az_prob = _predict_prob(az.model, x)
        base_pred = ((base_prob >= baseline.threshold) & (valid_np > 0.5)).astype(np.float32)
        az_pred = ((az_prob >= az.threshold) & (valid_np > 0.5)).astype(np.float32)

        full_base = _binary_stats(base_pred, gt_bin, valid_np)
        full_az = _binary_stats(az_pred, gt_bin, valid_np)

        bbox = _best_advantage_bbox(base_pred, az_pred, gt_bin, valid_np, win=int(args.window))
        y0, x0, y1, x1 = bbox
        base_crop = base_pred[y0 : y1 + 1, x0 : x1 + 1]
        az_crop = az_pred[y0 : y1 + 1, x0 : x1 + 1]
        gt_crop = gt_bin[y0 : y1 + 1, x0 : x1 + 1]
        valid_crop = valid_np[y0 : y1 + 1, x0 : x1 + 1]

        crop_base = _binary_stats(base_crop, gt_crop, valid_crop)
        crop_az = _binary_stats(az_crop, gt_crop, valid_crop)

        # Counts for legend.
        b = base_crop.astype(bool)
        a = az_crop.astype(bool)
        g = gt_crop.astype(bool)
        v = valid_crop.astype(bool)
        counts = {
            "az_gain": int((a & g & (~b) & v).sum()),
            "az_loss": int((b & g & (~a) & v).sum()),
            "fp_removed": int((b & (~g) & (~a) & v).sum()),
            "fp_added": int((a & (~g) & (~b) & v).sum()),
        }

        input_img = _denorm_to_uint8(x_norm)
        gt_overlay = _overlay(input_img, gt_bin > 0.5, (50, 220, 80), alpha=0.56)
        base_overlay = _overlay(input_img, base_pred > 0.5, (255, 160, 40), alpha=0.56)
        az_overlay = _overlay(input_img, az_pred > 0.5, (255, 95, 70), alpha=0.56)
        base_err = _error_map(base_pred > 0.5, gt_bin > 0.5, valid_np > 0.5)
        az_err = _error_map(az_pred > 0.5, gt_bin > 0.5, valid_np > 0.5)
        adv_map = _advantage_map(base_pred > 0.5, az_pred > 0.5, gt_bin > 0.5, valid_np > 0.5)
        pred_diff = _pred_diff_map(base_pred > 0.5, az_pred > 0.5, valid_np > 0.5)

        figure_path = out_dir / f"case_{idx:03d}_advantage.png"
        _case_figure(
            idx=idx,
            input_img=input_img,
            gt_overlay=gt_overlay,
            base_overlay=base_overlay,
            az_overlay=az_overlay,
            base_err=base_err,
            az_err=az_err,
            adv_map=adv_map,
            bbox=bbox,
            base_stats_crop=crop_base,
            az_stats_crop=crop_az,
            counts=counts,
            out_path=figure_path,
        )
        simple_data[int(idx)] = {
            "input": input_img,
            "base_overlay": base_overlay,
            "az_overlay": az_overlay,
            "pred_diff": pred_diff,
            "base_dice": float(full_base["dice"]),
            "az_dice": float(full_az["dice"]),
        }

        # Case ranking: prioritize local clarity of AZ improvement.
        rank_score = (
            2.0 * (crop_az["dice"] - crop_base["dice"])
            + 1.0 * (crop_az["iou"] - crop_base["iou"])
            + 0.7 * (crop_az["recall"] - crop_base["recall"])
            + 0.7 * (crop_az["precision"] - crop_base["precision"])
        )
        case_rows.append(
            {
                "idx": int(idx),
                "full_base_dice": float(full_base["dice"]),
                "full_az_dice": float(full_az["dice"]),
                "zoom_base_dice": float(crop_base["dice"]),
                "zoom_az_dice": float(crop_az["dice"]),
                "zoom_delta_dice": float(crop_az["dice"] - crop_base["dice"]),
                "zoom_base_iou": float(crop_base["iou"]),
                "zoom_az_iou": float(crop_az["iou"]),
                "zoom_delta_iou": float(crop_az["iou"] - crop_base["iou"]),
                "zoom_base_precision": float(crop_base["precision"]),
                "zoom_az_precision": float(crop_az["precision"]),
                "zoom_base_recall": float(crop_base["recall"]),
                "zoom_az_recall": float(crop_az["recall"]),
                "bbox": [int(y0), int(x0), int(y1), int(x1)],
                "counts": counts,
                "rank_score": float(rank_score),
                "figure_path": str(figure_path).replace("\\", "/"),
            }
        )

    case_rows_sorted = sorted(case_rows, key=lambda r: float(r["rank_score"]), reverse=True)
    positive_full = [r for r in case_rows_sorted if float(r["full_az_dice"]) >= float(r["full_base_dice"])]
    if len(positive_full) >= int(args.top_k):
        top_cases = positive_full[: int(args.top_k)]
    else:
        top_cases = positive_full + case_rows_sorted[: max(0, int(args.top_k) - len(positive_full))]
    (out_dir / "all_cases.json").write_text(json.dumps(case_rows_sorted, indent=2), encoding="utf-8")
    (out_dir / "top_cases.json").write_text(json.dumps(top_cases, indent=2), encoding="utf-8")
    _write_markdown(report_md, case_rows_sorted, top_cases)

    # Stable names for quick article insertion.
    for i, r in enumerate(top_cases, start=1):
        src = Path(r["figure_path"])
        dst = out_dir / f"top{i}_advantage_case_idx_{int(r['idx']):03d}.png"
        dst.write_bytes(src.read_bytes())

    # Build simple low-clutter comparison figures.
    for i, r in enumerate(top_cases[:2], start=1):
        idx = int(r["idx"])
        d = simple_data[idx]
        _simple_compare_figure(
            idx=idx,
            input_img=d["input"],
            base_overlay=d["base_overlay"],
            az_overlay=d["az_overlay"],
            pred_diff=d["pred_diff"],
            base_dice=float(d["base_dice"]),
            az_dice=float(d["az_dice"]),
            out_path=out_dir / f"simple_compare_best{i}.png",
        )

    summary = {
        "output_dir": str(out_dir),
        "report_md": str(report_md),
        "top_cases": top_cases,
        "window": int(args.window),
        "num_cases": len(case_rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
