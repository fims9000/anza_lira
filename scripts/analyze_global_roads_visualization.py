#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils
from utils import build_model


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
CC_STRUCT = np.ones((3, 3), dtype=np.uint8)


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
    out[tp] = np.array([38, 201, 98], dtype=np.uint8)
    out[fp] = np.array([239, 83, 80], dtype=np.uint8)
    out[fn] = np.array([66, 165, 245], dtype=np.uint8)
    out[tn] = np.array([20, 20, 20], dtype=np.uint8)
    return out


def _improvement_map(base_pred: np.ndarray, az_pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    b = base_pred.astype(bool)
    a = az_pred.astype(bool)
    g = gt.astype(bool)
    v = valid.astype(bool)
    out = np.zeros((*b.shape, 3), dtype=np.uint8)

    az_fix = (a == g) & (b != g) & v
    base_fix = (b == g) & (a != g) & v
    both_wrong = (a != g) & (b != g) & v
    both_road_correct = a & b & g & v
    both_bg_correct = (~a) & (~b) & (~g) & v

    out[az_fix] = np.array([76, 175, 80], dtype=np.uint8)         # green
    out[base_fix] = np.array([229, 57, 53], dtype=np.uint8)       # red
    out[both_road_correct] = np.array([255, 255, 255], dtype=np.uint8)  # white
    out[both_wrong] = np.array([0, 0, 0], dtype=np.uint8)         # black
    out[both_bg_correct] = np.array([58, 58, 58], dtype=np.uint8) # dark gray
    return out


def _binary_stats(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    p = pred.astype(bool) & valid.astype(bool)
    g = gt.astype(bool) & valid.astype(bool)
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    tn = float(np.logical_and(~p, ~g).sum())
    eps = 1e-8
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    fg_ratio = (float(p.sum()) + eps) / (float(valid.astype(bool).sum()) + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(acc),
        "pred_fg_ratio": float(fg_ratio),
    }


def _component_stats(mask: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    m = mask.astype(bool) & valid.astype(bool)
    labeled, num = ndi.label(m.astype(np.uint8), structure=CC_STRUCT)
    if num <= 0:
        return {"components": 0.0, "largest_cc_ratio": 0.0}
    counts = np.bincount(labeled.ravel())[1:]
    total = float(m.sum())
    largest = float(counts.max()) if counts.size > 0 else 0.0
    return {
        "components": float(num),
        "largest_cc_ratio": float(largest / max(total, 1.0)),
    }


def _skeleton_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, num_iters: int = 10) -> dict[str, float]:
    pred_t = torch.from_numpy(pred.astype(np.float32))[None, None]
    gt_t = torch.from_numpy(gt.astype(np.float32))[None, None]
    valid_t = torch.from_numpy(valid.astype(np.float32))[None, None]
    s_pot, s_pred, s_top, s_target = utils.skeleton_confusion_counts(
        pred_t,
        gt_t,
        valid_t,
        num_iters=int(num_iters),
    )
    m = utils.skeleton_metrics_from_counts(s_pot, s_pred, s_top, s_target)
    return {
        "cldice": float(m["cldice"]),
        "skeleton_precision": float(m["skeleton_precision"]),
        "skeleton_recall": float(m["skeleton_recall"]),
    }


def _skeleton_improvement_map(base_pred: np.ndarray, az_pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    v = valid.astype(bool)
    gt_skel = skeletonize((gt.astype(bool) & v).astype(np.uint8)) > 0
    base_hit = gt_skel & base_pred.astype(bool)
    az_hit = gt_skel & az_pred.astype(bool)

    az_gain = az_hit & (~base_hit)
    base_gain = base_hit & (~az_hit)
    both = base_hit & az_hit
    miss = gt_skel & (~base_hit) & (~az_hit)

    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[az_gain] = np.array([76, 175, 80], dtype=np.uint8)    # green
    out[base_gain] = np.array([229, 57, 53], dtype=np.uint8)  # red
    out[both] = np.array([255, 255, 255], dtype=np.uint8)     # white
    out[miss] = np.array([66, 165, 245], dtype=np.uint8)      # blue
    out[v & (~gt_skel)] = np.array([35, 35, 35], dtype=np.uint8)
    return out


def _road_band_mask(gt: np.ndarray, base_pred: np.ndarray, az_pred: np.ndarray, valid: np.ndarray, radius: int = 6) -> np.ndarray:
    union = (gt.astype(bool) | base_pred.astype(bool) | az_pred.astype(bool)) & valid.astype(bool)
    if not union.any():
        return valid.astype(bool)
    distance = ndi.distance_transform_edt(~union)
    return (distance <= float(radius)) & valid.astype(bool)


def _masked_entropy(entropy: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if not np.any(m):
        return float(np.mean(entropy))
    return float(np.mean(entropy[m]))


def _text_panel(width: int, height: int, lines: list[str]) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)
    y = 14
    for line in lines:
        draw.text((14, y), line, fill=(28, 28, 28))
        y += 22
    return img


def _label_panel(img: np.ndarray, title: str, size: tuple[int, int] = (360, 360), bar_h: int = 36) -> Image.Image:
    pil = Image.fromarray(img.astype(np.uint8))
    pil.thumbnail(size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", (size[0], size[1] + bar_h), color=(240, 240, 236))
    x = (size[0] - pil.width) // 2
    y = bar_h + (size[1] - pil.height) // 2
    out.paste(pil, (x, y))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, bar_h), fill=(30, 38, 50))
    draw.text((10, 10), title, fill=(255, 255, 255))
    return out


def _grid_2x4(
    panels: list[tuple[str, np.ndarray]],
    title: str,
    subtitle: str,
    out_path: Path,
) -> None:
    assert len(panels) == 8
    panel_imgs = [_label_panel(img, name) for name, img in panels]
    col_w = panel_imgs[0].width
    row_h = panel_imgs[0].height
    top_h = 176
    canvas = Image.new("RGB", (4 * col_w + 30, 2 * row_h + top_h + 20), color=(231, 231, 227))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill=(15, 15, 15))
    draw.text((12, 40), subtitle, fill=(35, 35, 35))
    draw.text((12, 68), "Error map: TP=green, FP=red, FN=blue, TN=black", fill=(35, 35, 35))
    draw.text((12, 92), "Difference map: AZ-better=green, Base-better=red, both-road-correct=white, both-wrong=black", fill=(35, 35, 35))
    draw.text((12, 116), "Skeleton map: AZ-recovers=green, Base-recovers=red, both-hit=white, both-miss=blue", fill=(35, 35, 35))
    # color chips for quick reading
    chips = [
        ((76, 175, 80), "AZ+"),
        ((229, 57, 53), "Base+"),
        ((255, 255, 255), "Both OK road"),
        ((66, 165, 245), "FN/Both miss"),
        ((0, 0, 0), "Both wrong"),
        ((58, 58, 58), "Both OK bg"),
    ]
    x = 12
    y = 140
    for color, label in chips:
        draw.rectangle((x, y, x + 18, y + 18), fill=color, outline=(80, 80, 80))
        draw.text((x + 24, y + 1), label, fill=(35, 35, 35))
        x += 150
    for i, panel in enumerate(panel_imgs):
        r = i // 4
        c = i % 4
        x = 10 + c * col_w
        y = top_h + r * row_h
        canvas.paste(panel, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _largest_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    m = mask.astype(bool)
    if not np.any(m):
        return None
    labeled, num = ndi.label(m.astype(np.uint8), structure=CC_STRUCT)
    if num <= 0:
        return None
    counts = np.bincount(labeled.ravel())[1:]
    if counts.size == 0:
        return None
    k = int(np.argmax(counts) + 1)
    ys, xs = np.where(labeled == k)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _expand_to_square_bbox(
    bbox: tuple[int, int, int, int],
    h: int,
    w: int,
    pad: int = 26,
    min_side: int = 220,
) -> tuple[int, int, int, int]:
    y0, x0, y1, x1 = bbox
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(h - 1, y1 + pad)
    x1 = min(w - 1, x1 + pad)
    bh = y1 - y0 + 1
    bw = x1 - x0 + 1
    side = max(min_side, bh, bw)
    cy = 0.5 * (y0 + y1)
    cx = 0.5 * (x0 + x1)
    ny0 = int(round(cy - side / 2))
    nx0 = int(round(cx - side / 2))
    ny1 = ny0 + side - 1
    nx1 = nx0 + side - 1
    if ny0 < 0:
        ny1 += -ny0
        ny0 = 0
    if nx0 < 0:
        nx1 += -nx0
        nx0 = 0
    if ny1 >= h:
        shift = ny1 - (h - 1)
        ny0 = max(0, ny0 - shift)
        ny1 = h - 1
    if nx1 >= w:
        shift = nx1 - (w - 1)
        nx0 = max(0, nx0 - shift)
        nx1 = w - 1
    return ny0, nx0, ny1, nx1


def _draw_bbox(img: np.ndarray, bbox: tuple[int, int, int, int], color: tuple[int, int, int] = (255, 255, 255), width: int = 3) -> np.ndarray:
    pil = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    y0, x0, y1, x1 = bbox
    for k in range(width):
        draw.rectangle((x0 - k, y0 - k, x1 + k, y1 + k), outline=color)
    return np.array(pil, dtype=np.uint8)


def _crop(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, x0, y1, x1 = bbox
    return img[y0 : y1 + 1, x0 : x1 + 1]


def _zoom_story_2x5(
    *,
    input_img: np.ndarray,
    gt_img: np.ndarray,
    base_img: np.ndarray,
    az_img: np.ndarray,
    diff_img: np.ndarray,
    focus_bbox: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    out_path: Path,
) -> None:
    full_panels = [
        ("Input (full)", _draw_bbox(input_img, focus_bbox)),
        ("GT (full)", _draw_bbox(gt_img, focus_bbox)),
        ("Baseline (full)", _draw_bbox(base_img, focus_bbox)),
        ("AZ-Thesis (full)", _draw_bbox(az_img, focus_bbox)),
        ("Difference (full)", _draw_bbox(diff_img, focus_bbox)),
    ]
    zoom_panels = [
        ("Input (zoom)", _crop(input_img, focus_bbox)),
        ("GT (zoom)", _crop(gt_img, focus_bbox)),
        ("Baseline (zoom)", _crop(base_img, focus_bbox)),
        ("AZ-Thesis (zoom)", _crop(az_img, focus_bbox)),
        ("Difference (zoom)", _crop(diff_img, focus_bbox)),
    ]
    panel_imgs = [_label_panel(img, name, size=(320, 320)) for name, img in (full_panels + zoom_panels)]
    col_w = panel_imgs[0].width
    row_h = panel_imgs[0].height
    top_h = 116
    canvas = Image.new("RGB", (5 * col_w + 24, 2 * row_h + top_h + 16), color=(231, 231, 227))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(15, 15, 15))
    draw.text((10, 36), subtitle, fill=(35, 35, 35))
    draw.text((10, 64), "Difference: green=AZ better, red=baseline better, white=both road-correct, black=both wrong", fill=(35, 35, 35))
    for i, panel in enumerate(panel_imgs):
        r = i // 5
        c = i % 5
        x = 8 + c * col_w
        y = top_h + r * row_h
        canvas.paste(panel, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _extract_sample_id(dataset: Any, idx: int) -> str:
    raw = getattr(dataset, "samples", None)
    if isinstance(raw, list) and 0 <= idx < len(raw):
        item = raw[idx]
        if isinstance(item, tuple) and len(item) > 0:
            return Path(item[0]).stem
    return f"sample_{idx:03d}"


def _write_md_report(
    out_md: Path,
    *,
    selection: dict[str, Any],
    rows: list[dict[str, Any]],
    top_win: list[dict[str, Any]],
    top_fail: list[dict[str, Any]],
    selected_idx: int,
    figure_paths: list[Path],
) -> None:
    selected = next((r for r in rows if int(r["idx"]) == int(selected_idx)), None)
    if selected is None:
        selected = rows[0]
    wins = sum(1 for r in rows if float(r["delta_dice"]) > 0.0)
    fails = sum(1 for r in rows if float(r["delta_dice"]) < 0.0)
    mean_base = float(np.mean([float(r["baseline_dice"]) for r in rows]))
    mean_az = float(np.mean([float(r["az_dice"]) for r in rows]))
    mean_delta = mean_az - mean_base

    lines: list[str] = []
    lines.append("# GlobalScaleRoad: визуализация и проверка интерпретируемости")
    lines.append("")
    lines.append("Дата: 2026-04-26")
    lines.append("")
    lines.append("## 1) Что было проблемой")
    lines.append("")
    lines.append("Текущая картинка могла быть непонятной для рецензента по двум причинам: (1) неявная легенда цветов, (2) отсутствие численной расшифровки по конкретному тайлу.")
    lines.append("")
    lines.append("## 2) Явная легенда (для новой версии рисунка)")
    lines.append("")
    lines.append("- `Error map` (для baseline и AZ): зелёный = TP, красный = FP, синий = FN, чёрный = TN.")
    lines.append("- `Difference map (AZ vs Baseline)`:")
    lines.append("  - зелёный = AZ прав, baseline ошибся")
    lines.append("  - красный = baseline прав, AZ ошибся")
    lines.append("  - белый = оба верно на пикселях дороги")
    lines.append("  - чёрный = оба ошиблись")
    lines.append("  - тёмно-серый = оба верно на фоне")
    lines.append("- `Skeleton improvement map`:")
    lines.append("  - зелёный = AZ восстанавливает GT-скелет, где baseline пропустил")
    lines.append("  - красный = baseline лучше AZ по GT-скелету")
    lines.append("  - белый = оба попали в GT-скелет")
    lines.append("  - синий = оба пропустили GT-скелет")
    lines.append("- Примечание по clDice: на этом наборе и при этих порогах clDice-семейство почти не различает baseline/AZ, поэтому основной акцент для road-cases делаем на Dice/IoU/Precision/Recall и связности.")
    lines.append("")
    lines.append("## 2.1) Общая картина по 30 тайлам")
    lines.append("")
    lines.append(f"- Тайлов, где AZ лучше baseline по Dice: `{wins}` из `{len(rows)}`")
    lines.append(f"- Тайлов, где baseline лучше AZ по Dice: `{fails}` из `{len(rows)}`")
    lines.append(f"- Средний Dice: baseline `{mean_base:.4f}` -> AZ `{mean_az:.4f}` (Δ `{mean_delta:+.4f}`)")
    lines.append("")
    lines.append("## 3) Числа по текущему выбранному тайлу")
    lines.append("")
    lines.append(f"- Индекс тайла: `{int(selected['idx'])}`")
    lines.append(f"- Dice: baseline `{selected['baseline_dice']:.4f}` -> AZ `{selected['az_dice']:.4f}` (Δ `{selected['delta_dice']:+.4f}`)")
    lines.append(f"- IoU: baseline `{selected['baseline_iou']:.4f}` -> AZ `{selected['az_iou']:.4f}` (Δ `{selected['delta_iou']:+.4f}`)")
    lines.append(f"- Precision: baseline `{selected['baseline_precision']:.4f}` -> AZ `{selected['az_precision']:.4f}`")
    lines.append(f"- Recall: baseline `{selected['baseline_recall']:.4f}` -> AZ `{selected['az_recall']:.4f}`")
    lines.append(f"- clDice: baseline `{selected['baseline_cldice']:.4f}` -> AZ `{selected['az_cldice']:.4f}`")
    lines.append(f"- Компоненты связности: baseline `{int(selected['baseline_components'])}` vs AZ `{int(selected['az_components'])}`")
    lines.append(f"- Largest connected component ratio: baseline `{selected['baseline_lcc_ratio']:.4f}` vs AZ `{selected['az_lcc_ratio']:.4f}`")
    lines.append("")
    lines.append("## 4) Лучшие/провальные кейсы для статьи")
    lines.append("")
    lines.append("### Топ-5, где AZ лучше baseline (по Dice)")
    lines.append("")
    lines.append("| idx | Dice base | Dice AZ | ΔDice | Recall base | Recall AZ | clDice base | clDice AZ |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in top_win:
        lines.append(
            f"| {int(r['idx'])} | {r['baseline_dice']:.4f} | {r['az_dice']:.4f} | {r['delta_dice']:+.4f} | "
            f"{r['baseline_recall']:.4f} | {r['az_recall']:.4f} | {r['baseline_cldice']:.4f} | {r['az_cldice']:.4f} |"
        )
    lines.append("")
    lines.append("### Топ-3 failure cases (где baseline лучше)")
    lines.append("")
    lines.append("| idx | Dice base | Dice AZ | ΔDice | Recall base | Recall AZ | clDice base | clDice AZ |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in top_fail:
        lines.append(
            f"| {int(r['idx'])} | {r['baseline_dice']:.4f} | {r['az_dice']:.4f} | {r['delta_dice']:+.4f} | "
            f"{r['baseline_recall']:.4f} | {r['az_recall']:.4f} | {r['baseline_cldice']:.4f} | {r['az_cldice']:.4f} |"
        )
    lines.append("")
    lines.append("## 5) Новые визуализации (готово к вставке)")
    lines.append("")
    for p in figure_paths:
        lines.append(f"- `{p.as_posix()}`")
    lines.append("- `results/article_visual_assets/global_roads_figures_v2/positive_gain_zoom_grid.png`")
    lines.append("- `results/article_visual_assets/global_roads_figures_v2/failure_loss_zoom_grid.png`")
    lines.append("")
    lines.append("Рекомендация: в основную статью поставить 1 positive case + 1 failure case с одинаковым 8-панельным форматом. Это сразу снимает претензию про «непонятную» визуализацию и добавляет честный анализ ограничений.")
    lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GlobalScaleRoad visualization cases with interpretable metrics.")
    parser.add_argument(
        "--selection-json",
        type=str,
        default="results/article_visual_assets/global_roads_figures/selection.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/article_visual_assets/global_roads_figures_v2",
    )
    parser.add_argument(
        "--report-md",
        type=str,
        default="results/global_roads_visualization_audit_v2_ru.md",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    selection_path = (PROJECT_ROOT / args.selection_json).resolve()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    report_md = (PROJECT_ROOT / args.report_md).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    selection = _load_json(selection_path)
    baseline_run = Path(str(selection["baseline_run"]))
    az_run = Path(str(selection["az_run"]))
    if not baseline_run.exists() or not az_run.exists():
        raise FileNotFoundError("Run directories from selection.json are missing.")

    baseline = _load_run(baseline_run, device)
    az = _load_run(az_run, device)
    if utils.canonical_dataset_name(str(az.cfg["dataset"])) not in utils.GIS_SEG_DATASETS:
        raise ValueError(f"Expected GIS dataset, got: {az.cfg['dataset']}")

    _, _, test_loader, _, _, task = utils.build_dataloaders(az.cfg)
    if task != "segmentation":
        raise ValueError("Expected segmentation task.")
    dataset = test_loader.dataset

    # Normalize indices from the prior selection pass.
    rows_in = selection.get("rows", [])
    indices = [int(r["idx"]) for r in rows_in] if rows_in else list(range(len(dataset)))

    all_rows: list[dict[str, Any]] = []
    per_case_images: dict[int, dict[str, np.ndarray]] = {}
    num_iters = int(az.cfg.get("topology_num_iters", 10))

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

        b_core = _binary_stats(base_pred, gt_bin, valid_np)
        a_core = _binary_stats(az_pred, gt_bin, valid_np)
        b_skel = _skeleton_metrics(base_pred, gt_bin, valid_np, num_iters=num_iters)
        a_skel = _skeleton_metrics(az_pred, gt_bin, valid_np, num_iters=num_iters)
        b_cc = _component_stats(base_pred, valid_np)
        a_cc = _component_stats(az_pred, valid_np)
        gt_cc = _component_stats(gt_bin, valid_np)

        row = {
            "idx": int(idx),
            "sample_id": _extract_sample_id(dataset, idx),
            "baseline_dice": b_core["dice"],
            "az_dice": a_core["dice"],
            "delta_dice": a_core["dice"] - b_core["dice"],
            "baseline_iou": b_core["iou"],
            "az_iou": a_core["iou"],
            "delta_iou": a_core["iou"] - b_core["iou"],
            "baseline_precision": b_core["precision"],
            "az_precision": a_core["precision"],
            "baseline_recall": b_core["recall"],
            "az_recall": a_core["recall"],
            "baseline_cldice": b_skel["cldice"],
            "az_cldice": a_skel["cldice"],
            "baseline_skeleton_recall": b_skel["skeleton_recall"],
            "az_skeleton_recall": a_skel["skeleton_recall"],
            "baseline_components": b_cc["components"],
            "az_components": a_cc["components"],
            "gt_components": gt_cc["components"],
            "baseline_lcc_ratio": b_cc["largest_cc_ratio"],
            "az_lcc_ratio": a_cc["largest_cc_ratio"],
            "gt_lcc_ratio": gt_cc["largest_cc_ratio"],
            "baseline_fg_ratio": b_core["pred_fg_ratio"],
            "az_fg_ratio": a_core["pred_fg_ratio"],
            "gt_fg_ratio": float(gt_bin.sum() / max(1.0, float((valid_np > 0.5).sum()))),
        }
        all_rows.append(row)

        image_rgb = _denorm_to_uint8(x_norm)
        gt_overlay = _overlay(image_rgb, gt_bin > 0.5, (50, 220, 80), alpha=0.56)
        base_overlay = _overlay(image_rgb, base_pred > 0.5, (255, 160, 40), alpha=0.56)
        az_overlay = _overlay(image_rgb, az_pred > 0.5, (255, 95, 70), alpha=0.56)
        base_err = _error_map(base_pred > 0.5, gt_bin > 0.5, valid_np > 0.5)
        az_err = _error_map(az_pred > 0.5, gt_bin > 0.5, valid_np > 0.5)
        diff_map = _improvement_map(base_pred > 0.5, az_pred > 0.5, gt_bin > 0.5, valid_np > 0.5)
        skel_map = _skeleton_improvement_map(base_pred > 0.5, az_pred > 0.5, gt_bin > 0.5, valid_np > 0.5)
        per_case_images[int(idx)] = {
            "input": image_rgb,
            "gt": gt_overlay,
            "base": base_overlay,
            "az": az_overlay,
            "base_err": base_err,
            "az_err": az_err,
            "diff": diff_map,
            "skel": skel_map,
        }

    all_rows.sort(key=lambda r: int(r["idx"]))
    ranked = sorted(all_rows, key=lambda r: float(r["delta_dice"]), reverse=True)
    top_win = ranked[:5]
    top_fail = list(reversed(ranked[-3:]))
    selected_idx = int(selection.get("selected", {}).get("idx", top_win[0]["idx"]))

    # Export detailed tables.
    (out_dir / "global_roads_tile_metrics_full.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    _write_csv(out_dir / "global_roads_tile_metrics_full.csv", all_rows)
    (out_dir / "global_roads_top_win_fail.json").write_text(
        json.dumps({"top_win": top_win, "top_fail": top_fail}, indent=2),
        encoding="utf-8",
    )

    # Build 8-panel visualizations for one selected case, one strongest win, one strongest fail.
    showcase = [int(selected_idx), int(top_win[0]["idx"]), int(top_fail[0]["idx"])]
    showcase = list(dict.fromkeys(showcase))
    figure_paths: list[Path] = []

    for idx in showcase:
        row = next(r for r in all_rows if int(r["idx"]) == idx)
        imgs = per_case_images[idx]
        subtitle = (
            f"idx={idx} | Dice base={row['baseline_dice']:.4f} AZ={row['az_dice']:.4f} (Δ={row['delta_dice']:+.4f}) | "
            f"Recall base={row['baseline_recall']:.4f} AZ={row['az_recall']:.4f}"
        )
        panels = [
            ("Input", imgs["input"]),
            ("Ground Truth Overlay", imgs["gt"]),
            ("Baseline Prediction", imgs["base"]),
            ("AZ-Thesis Prediction", imgs["az"]),
            ("Baseline Error Map", imgs["base_err"]),
            ("AZ Error Map", imgs["az_err"]),
            ("Difference Map (AZ vs Base)", imgs["diff"]),
            ("Skeleton Improvement Map", imgs["skel"]),
        ]
        out_png = out_dir / f"case_{idx:03d}_visual_grid.png"
        _grid_2x4(
            panels,
            title="GlobalScaleRoad visualization audit (interpretable layout)",
            subtitle=subtitle,
            out_path=out_png,
        )
        figure_paths.append(out_png)

    # Add zoomed stories to make local advantage/limitation immediately visible.
    def _focus_bbox(case_idx: int, prefer_gain: bool) -> tuple[int, int, int, int]:
        diff = per_case_images[case_idx]["diff"]
        green = np.all(diff == np.array([76, 175, 80], dtype=np.uint8), axis=-1)
        red = np.all(diff == np.array([229, 57, 53], dtype=np.uint8), axis=-1)
        target = green if prefer_gain else red
        bbox = _largest_component_bbox(target)
        if bbox is None:
            bbox = _largest_component_bbox(red if prefer_gain else green)
        if bbox is None:
            h, w = diff.shape[:2]
            bbox = (h // 4, w // 4, 3 * h // 4, 3 * w // 4)
        h, w = diff.shape[:2]
        return _expand_to_square_bbox(bbox, h=h, w=w)

    pos_idx = int(top_win[0]["idx"])
    pos_row = next(r for r in all_rows if int(r["idx"]) == pos_idx)
    pos_bbox = _focus_bbox(pos_idx, prefer_gain=True)
    pos_imgs = per_case_images[pos_idx]
    pos_zoom = out_dir / f"case_{pos_idx:03d}_gain_zoom_grid.png"
    _zoom_story_2x5(
        input_img=pos_imgs["input"],
        gt_img=pos_imgs["gt"],
        base_img=pos_imgs["base"],
        az_img=pos_imgs["az"],
        diff_img=pos_imgs["diff"],
        focus_bbox=pos_bbox,
        title="GlobalScaleRoad positive focus (where AZ is better)",
        subtitle=f"idx={pos_idx} | Dice base={pos_row['baseline_dice']:.4f} -> AZ={pos_row['az_dice']:.4f}",
        out_path=pos_zoom,
    )
    shutil.copyfile(pos_zoom, out_dir / "positive_gain_zoom_grid.png")

    fail_idx = int(top_fail[0]["idx"])
    fail_row = next(r for r in all_rows if int(r["idx"]) == fail_idx)
    fail_bbox = _focus_bbox(fail_idx, prefer_gain=False)
    fail_imgs = per_case_images[fail_idx]
    fail_zoom = out_dir / f"case_{fail_idx:03d}_loss_zoom_grid.png"
    _zoom_story_2x5(
        input_img=fail_imgs["input"],
        gt_img=fail_imgs["gt"],
        base_img=fail_imgs["base"],
        az_img=fail_imgs["az"],
        diff_img=fail_imgs["diff"],
        focus_bbox=fail_bbox,
        title="GlobalScaleRoad failure focus (where baseline is better)",
        subtitle=f"idx={fail_idx} | Dice base={fail_row['baseline_dice']:.4f} -> AZ={fail_row['az_dice']:.4f}",
        out_path=fail_zoom,
    )
    shutil.copyfile(fail_zoom, out_dir / "failure_loss_zoom_grid.png")

    _write_md_report(
        report_md,
        selection=selection,
        rows=all_rows,
        top_win=top_win,
        top_fail=top_fail,
        selected_idx=selected_idx,
        figure_paths=figure_paths,
    )

    summary = {
        "selection_json": str(selection_path),
        "baseline_run": str(baseline.run_dir),
        "az_run": str(az.run_dir),
        "baseline_threshold": baseline.threshold,
        "az_threshold": az.threshold,
        "num_samples_analyzed": len(all_rows),
        "selected_idx": selected_idx,
        "top_win_idx": int(top_win[0]["idx"]),
        "top_fail_idx": int(top_fail[0]["idx"]),
        "output_dir": str(out_dir),
        "report_md": str(report_md),
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
