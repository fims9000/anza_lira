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
import scripts.export_geometry_attention_story as story


def _seg_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    p = (pred > 0.5) & (valid > 0.5)
    g = (gt > 0.5) & (valid > 0.5)
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    eps = 1e-8
    dice = (2.0 * tp) / max(2.0 * tp + fp + fn, eps)
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    iou = tp / max(tp + fp + fn, eps)
    return {"dice": dice, "precision": precision, "recall": recall, "iou": iou}


def _to_skeleton(mask: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(mask.astype(np.float32))[None, None]
    with torch.no_grad():
        sk = utils._soft_skeletonize(x, num_iters=12)[0, 0].numpy()
    return sk > 0.15


def _prune_small_components(binary: np.ndarray, min_size: int = 24) -> np.ndarray:
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    out = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            comp: list[tuple[int, int]] = []
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if ny < 0 or nx < 0 or ny >= h or nx >= w:
                            continue
                        if visited[ny, nx] or (not binary[ny, nx]):
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) >= min_size:
                for py, px in comp:
                    out[py, px] = True
    return out


def _sample_skeleton_points(skeleton: np.ndarray, valid: np.ndarray, max_points: int = 5000) -> list[tuple[int, int]]:
    ys, xs = np.where(skeleton & valid)
    if ys.size == 0:
        return []
    idx = np.arange(ys.size)
    if ys.size > max_points:
        take = np.linspace(0, ys.size - 1, max_points).astype(np.int64)
        idx = take
    points = [(int(ys[i]), int(xs[i])) for i in idx]
    return points


def _grid_filter_points(points: list[tuple[int, int]], cell: int = 10) -> list[tuple[int, int]]:
    if not points:
        return []
    taken: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for y, x in points:
        key = (y // max(cell, 1), x // max(cell, 1))
        if key in taken:
            continue
        taken.add(key)
        out.append((y, x))
    return out


def _neighbors8(skeleton: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    h, w = skeleton.shape
    out: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = y + dy, x + dx
            if yy < 0 or xx < 0 or yy >= h or xx >= w:
                continue
            if skeleton[yy, xx]:
                out.append((yy, xx))
    return out


def _local_tangent_angle(skeleton: np.ndarray, y: int, x: int) -> float | None:
    nbs = _neighbors8(skeleton, y, x)
    if not nbs:
        return None
    # Smooth local tangent via PCA over a tiny neighborhood around the skeleton.
    r = 3
    h, w = skeleton.shape
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    patch = skeleton[y0:y1, x0:x1]
    py, px = np.where(patch)
    if py.size >= 4:
        pts = np.stack([px.astype(np.float32), py.astype(np.float32)], axis=1)
        pts -= pts.mean(axis=0, keepdims=True)
        cov = pts.T @ pts / max(float(len(pts) - 1), 1.0)
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, int(np.argmax(vals))]
        return float(math.atan2(v[1], v[0]))
    if len(nbs) == 1:
        yy, xx = nbs[0]
        return float(math.atan2(float(yy - y), float(xx - x)))
    (y1n, x1n), (y2n, x2n) = nbs[0], nbs[1]
    return float(math.atan2(float(y2n - y1n), float(x2n - x1n)))


def _overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.52) -> np.ndarray:
    out = image_rgb.astype(np.float32).copy()
    m = mask > 0.5
    c = np.asarray(color, dtype=np.float32)
    out[m] = (1.0 - alpha) * out[m] + alpha * c
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _difference_map(baseline: np.ndarray, proposed: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    b = baseline > 0.5
    a = proposed > 0.5
    g = gt > 0.5
    v = valid > 0.5
    canvas = np.zeros((*baseline.shape, 3), dtype=np.uint8)
    canvas[:] = np.array([28, 32, 42], dtype=np.uint8)
    # Error-centric map (relative to GT), easier to interpret than raw mask difference.
    # green: AZ fixed baseline false negative
    # blue: AZ removed baseline false positive
    # orange: AZ added false positive
    # red: AZ missed vessel that baseline had
    az_fix_fn = (~b) & g & a & v
    az_fix_fp = b & (~g) & (~a) & v
    az_add_fp = (~b) & (~g) & a & v
    az_new_fn = b & g & (~a) & v
    both_pred = a & b & v

    canvas[both_pred] = np.array([215, 215, 215], dtype=np.uint8)
    canvas[az_fix_fn] = np.array([72, 194, 110], dtype=np.uint8)
    canvas[az_fix_fp] = np.array([56, 128, 196], dtype=np.uint8)
    canvas[az_add_fp] = np.array([232, 166, 35], dtype=np.uint8)
    canvas[az_new_fn] = np.array([220, 78, 72], dtype=np.uint8)
    return canvas


def _draw_direction_arrows(
    image_rgb: np.ndarray,
    direction: np.ndarray,
    skeleton: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray | None = None,
    step: int = 14,
) -> np.ndarray:
    base = image_rgb.astype(np.float32)
    dim = (base * 0.78).astype(np.uint8)
    pil = Image.fromarray(dim)
    draw = ImageDraw.Draw(pil)
    points = _sample_skeleton_points(skeleton, valid > 0.5, max_points=10000)
    points = _grid_filter_points(points, cell=4)
    # Optional local axial smoothing from model-only orientation:
    # smooth cos(2*theta), sin(2*theta) and map back via 0.5*atan2.
    mask = ((skeleton > 0.5) & (valid > 0.5)).astype(np.float32)
    theta2_c = np.cos(2.0 * direction).astype(np.float32) * mask
    theta2_s = np.sin(2.0 * direction).astype(np.float32) * mask
    k = 5
    r = k // 2
    pad_mode = "edge"
    c_pad = np.pad(theta2_c, ((r, r), (r, r)), mode=pad_mode)
    s_pad = np.pad(theta2_s, ((r, r), (r, r)), mode=pad_mode)
    m_pad = np.pad(mask, ((r, r), (r, r)), mode=pad_mode)
    c_blur = np.zeros_like(theta2_c)
    s_blur = np.zeros_like(theta2_s)
    m_blur = np.zeros_like(mask)
    for yy in range(theta2_c.shape[0]):
        y0 = yy
        y1 = yy + k
        for xx in range(theta2_c.shape[1]):
            x0 = xx
            x1 = xx + k
            msum = float(m_pad[y0:y1, x0:x1].sum())
            if msum <= 1e-6:
                continue
            c_blur[yy, xx] = float(c_pad[y0:y1, x0:x1].sum()) / msum
            s_blur[yy, xx] = float(s_pad[y0:y1, x0:x1].sum()) / msum
            m_blur[yy, xx] = msum
    direction_smooth = 0.5 * np.arctan2(s_blur, c_blur)
    use_smooth = m_blur > 1e-3
    def _axial_diff(a: float, b: float) -> float:
        d = abs(a - b)
        while d > math.pi:
            d -= math.pi
        return min(d, math.pi - d)

    def _axial_blend(a: float, b: float, wa: float, wb: float) -> float:
        # Blend undirected angles through doubled-angle representation.
        ca = math.cos(2.0 * a)
        sa = math.sin(2.0 * a)
        cb = math.cos(2.0 * b)
        sb = math.sin(2.0 * b)
        c = wa * ca + wb * cb
        s = wa * sa + wb * sb
        return 0.5 * math.atan2(s, c)

    for i, (y, x) in enumerate(points):
        if confidence is not None:
            c = float(confidence[y, x])
            if c < 0.03:
                continue
        else:
            c = 1.0
        # Model-native orientation axis from AZ (theta map / rule mixture).
        ang_model = float(direction_smooth[y, x] if use_smooth[y, x] else direction[y, x])
        # Keep only clearly inconsistent outliers away from object tangent;
        # keep threshold relaxed to preserve coverage on curved/branch regions.
        ang_tan = _local_tangent_angle(skeleton, y, x)
        if ang_tan is None:
            continue
        if _axial_diff(ang_model, ang_tan) > 1.15:  # ~66 deg
            continue
        ang = ang_model
        ln = 4.8 + 5.8 * c
        dx = math.cos(ang) * ln
        dy = math.sin(ang) * ln
        x1, y1 = x - dx, y - dy
        x2, y2 = x + dx, y + dy
        # Draw centered orientation segment (axis, not one-way arrow).
        draw.line((x1, y1, x2, y2), fill=(20, 20, 20), width=2)
        draw.line((x1, y1, x2, y2), fill=(245, 245, 245), width=1)
        # End-caps improve readability in print and indicate an undirected axis.
        ux, uy = math.cos(ang + math.pi / 2.0), math.sin(ang + math.pi / 2.0)
        cap = 1.0
        draw.line((x1 - ux * cap, y1 - uy * cap, x1 + ux * cap, y1 + uy * cap), fill=(240, 240, 240), width=1)
        draw.line((x2 - ux * cap, y2 - uy * cap, x2 + ux * cap, y2 + uy * cap), fill=(240, 240, 240), width=1)
    return np.asarray(pil, dtype=np.uint8)


def _draw_anisotropy_support(
    image_rgb: np.ndarray,
    direction: np.ndarray,
    signed_gain: np.ndarray,
    skeleton: np.ndarray,
    valid: np.ndarray,
    step: int = 16,
) -> np.ndarray:
    base = (image_rgb.astype(np.float32) * 0.82).astype(np.uint8)
    rgba = np.dstack([base, np.full(base.shape[:2], 255, dtype=np.uint8)])
    pil = Image.fromarray(rgba, mode="RGBA")
    draw = ImageDraw.Draw(pil, "RGBA")
    gain = np.clip(np.abs(signed_gain), 0.0, 1.0)
    points = _sample_skeleton_points(skeleton, valid > 0.5, max_points=4000)
    points = _grid_filter_points(points, cell=7)
    for i, (y, x) in enumerate(points):
        g = float(gain[y, x])
        # keep low-gain points visible too; paper figure should be interpretable.
        g = max(g, 0.12)
        ang_tan = _local_tangent_angle(skeleton, y, x)
        if ang_tan is None:
            continue
        major = 8.0 + 12.0 * g
        minor = 2.0 + 1.4 * (1.0 - g)
        ux, uy = math.cos(ang_tan), math.sin(ang_tan)
        sx, sy = -uy, ux
        pts = []
        for k in range(24):
            t = (2.0 * math.pi * k) / 24.0
            px = x + math.cos(t) * major * ux + math.sin(t) * minor * sx
            py = y + math.cos(t) * major * uy + math.sin(t) * minor * sy
            pts.append((px, py))
        fill_alpha = int(88 + 95 * min(g, 1.0))
        draw.polygon(pts, fill=(236, 184, 82, fill_alpha))
        draw.line((x - ux * major, y - uy * major, x + ux * major, y + uy * major), fill=(255, 242, 214, 210), width=2)
        draw.line((x - sx * minor, y - sy * minor, x + sx * minor, y + sy * minor), fill=(86, 180, 233, 235), width=2)
    return np.asarray(pil.convert("RGB"), dtype=np.uint8)


def _draw_anisotropy_strength_map(
    image_rgb: np.ndarray,
    signed_gain: np.ndarray,
    object_mask: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    base = (image_rgb.astype(np.float32) * 0.78).astype(np.uint8)
    out = base.astype(np.float32)
    m = (object_mask > 0.5) & (valid > 0.5)
    if not np.any(m):
        return base

    g = np.abs(np.asarray(signed_gain, dtype=np.float32))
    lo = float(np.percentile(g[m], 10))
    hi = float(np.percentile(g[m], 95))
    scale = np.clip((g - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    # blue -> orange ramp (colorblind-friendly)
    c0 = np.array([56.0, 108.0, 176.0], dtype=np.float32)
    c1 = np.array([240.0, 168.0, 52.0], dtype=np.float32)
    color = (1.0 - scale[..., None]) * c0 + scale[..., None] * c1
    alpha = 0.14 + 0.72 * scale
    out[m] = (1.0 - alpha[m, None]) * out[m] + alpha[m, None] * color[m]

    pil = Image.fromarray(np.clip(out, 0.0, 255.0).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    # tiny inline legend
    x0, y0, w, h = 18, 18, 170, 14
    for i in range(w):
        t = i / max(w - 1, 1)
        cc = tuple(np.clip((1.0 - t) * c0 + t * c1, 0, 255).astype(np.uint8).tolist())
        d.line((x0 + i, y0, x0 + i, y0 + h), fill=cc, width=1)
    d.rectangle((x0 - 1, y0 - 1, x0 + w + 1, y0 + h + 1), outline=(235, 235, 235), width=1)
    d.text((x0, y0 + h + 6), "anisotropy strength: low -> high", fill=(235, 235, 235))
    return np.asarray(pil, dtype=np.uint8)


def _panel(img: np.ndarray, title: str, w: int, h: int) -> Image.Image:
    top_bar = 62
    bottom_bar = 44
    panel = Image.fromarray(img.astype(np.uint8)).resize((w, h), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (w, h + top_bar + bottom_bar), color=(244, 243, 239))
    out.paste(panel, (0, top_bar))
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, w, top_bar), fill=(22, 30, 44))
    d.rectangle((0, h + top_bar, w, h + top_bar + bottom_bar), fill=(18, 26, 36))
    words = title.split()
    lines: list[str] = []
    cur = ""
    max_w = w - 20
    for token in words:
        cand = token if not cur else f"{cur} {token}"
        if d.textlength(cand) <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = token
    if cur:
        lines.append(cur)
    lines = lines[:2]
    y = 9
    for ln in lines:
        d.text((10, y), ln, fill=(255, 255, 255))
        y += 20
    return out


def _panel_with_bottom_legend(
    img: np.ndarray,
    title: str,
    w: int,
    h: int,
    legend_items: list[tuple[tuple[int, int, int], str]],
) -> Image.Image:
    top_bar = 62
    bottom_bar = 44
    panel = Image.fromarray(img.astype(np.uint8)).resize((w, h), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (w, h + top_bar + bottom_bar), color=(244, 243, 239))
    out.paste(panel, (0, top_bar))
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, w, top_bar), fill=(22, 30, 44))
    d.rectangle((0, h + top_bar, w, h + top_bar + bottom_bar), fill=(18, 26, 36))

    words = title.split()
    lines: list[str] = []
    cur = ""
    max_w = w - 20
    for token in words:
        cand = token if not cur else f"{cur} {token}"
        if d.textlength(cand) <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = token
    if cur:
        lines.append(cur)
    lines = lines[:2]
    y = 9
    for ln in lines:
        d.text((10, y), ln, fill=(255, 255, 255))
        y += 20

    cursor_x = 10
    y_box = h + top_bar + 14
    for color, label in legend_items:
        d.rectangle((cursor_x, y_box, cursor_x + 12, y_box + 12), fill=color)
        d.text((cursor_x + 18, y_box - 1), label, fill=(218, 226, 235))
        cursor_x += 150
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export clean, human-readable geometry figure for paper.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--baseline-run", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument("--output-dir", type=str, default="results/a3_final_package/fig_geometry_clean")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    az_run_dir = story._find_run(results_dir, args.run, None)
    baseline_run_dir = story._find_run(results_dir, args.baseline_run, None)
    az_model, az_cfg, az_metrics, _ = story._load_model(az_run_dir, device)
    baseline_model, _, baseline_metrics, _ = story._load_model(baseline_run_dir, device)
    dataset, dataset_name = story._build_eval_dataset(az_cfg)
    az_thr = float(az_metrics.get("selected_threshold", az_cfg.get("seg_threshold", 0.5)))
    base_thr = float(baseline_metrics.get("selected_threshold", az_cfg.get("seg_threshold", 0.5)))

    if args.sample_index >= 0:
        idx = min(max(int(args.sample_index), 0), len(dataset) - 1)
    else:
        idx = story._choose_best_improvement_sample(dataset, az_model, az_thr, baseline_model, base_thr, device)

    x_norm, y, valid = dataset[idx]
    image_rgb = story._denorm_to_uint8(x_norm)
    gt = y[0].numpy().astype(np.float32)
    v = valid[0].numpy().astype(np.float32)
    x = x_norm.unsqueeze(0).to(device)

    az_prob = story._predict_prob(az_model, x)
    base_prob = story._predict_prob(baseline_model, x)
    az_pred = (az_prob >= az_thr).astype(np.float32) * v
    base_pred = (base_prob >= base_thr).astype(np.float32) * v

    az_layers: list[tuple[str, dict[str, Any]]] = []
    for name, module in az_model.named_modules():
        if isinstance(module, story.AZConv2d):
            snap = module.interpretation_snapshot()
            if snap:
                az_layers.append((name, snap))
    if not az_layers:
        raise SystemExit("No AZ layers found.")
    # use the latest AZ layer for most semantic direction field
    _, snap = az_layers[-1]
    direction, signed_gain, conf = story._direction_gain_confidence_maps(snap, v)

    # For geometry visualization, rely on model prediction support plus GT fallback.
    object_mask = ((az_pred > 0.5) | (gt > 0.5)) & (v > 0.5)
    skeleton = _to_skeleton(object_mask.astype(np.float32))
    skeleton = _prune_small_components(skeleton, min_size=24)
    skeleton = skeleton & object_mask

    panel_a = _overlay_mask(image_rgb, gt, (86, 180, 233), alpha=0.44)
    panel_b = _difference_map(base_pred, az_pred, gt, v)
    panel_c = _draw_direction_arrows(
        image_rgb=image_rgb,
        direction=direction,
        skeleton=skeleton,
        valid=v > 0.5,
        confidence=conf,
        step=14,
    )
    panel_d = _draw_anisotropy_strength_map(
        image_rgb=image_rgb,
        signed_gain=signed_gain * conf,
        object_mask=object_mask.astype(np.float32),
        valid=(v > 0.5).astype(np.float32),
    )

    w, h = 620, 620
    p1 = _panel(panel_a, "Input + Ground Truth", w, h)
    p2 = _panel_with_bottom_legend(
        panel_b,
        "Baseline vs AZ error difference vs GT",
        w,
        h,
        legend_items=[
            ((72, 194, 110), "green: fix FN"),
            ((56, 128, 196), "blue: remove FP"),
            ((232, 166, 35), "orange: add FP"),
            ((220, 78, 72), "red: new FN"),
        ],
    )
    p3 = _panel(panel_c, "AZ Orientation Axis (model theta map)", w, h)
    p4 = _panel(panel_d, "Anisotropy Strength Map", w, h)

    top_h = max(p1.size[1], p2.size[1])
    bot_h = max(p3.size[1], p4.size[1])
    canvas = Image.new("RGB", (2 * w, top_h + bot_h), color=(232, 232, 228))
    canvas.paste(p1, (0, 0))
    canvas.paste(p2, (w, 0))
    canvas.paste(p3, (0, top_h))
    canvas.paste(p4, (w, top_h))

    sample_id = story._sample_id(dataset, idx)
    out_png = out_dir / f"geometry_clean_{dataset_name}_{sample_id}.png"
    out_json = out_dir / f"geometry_clean_{dataset_name}_{sample_id}.json"
    canvas.save(out_png)
    out_json.write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "sample_index": int(idx),
                "sample_id": sample_id,
                "az_run": az_run_dir.name,
                "baseline_run": baseline_run_dir.name,
                "az_threshold": az_thr,
                "baseline_threshold": base_thr,
                "output_image": str(out_png),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Exported: {out_png}")
    print(f"Meta: {out_json}")


if __name__ == "__main__":
    main()
