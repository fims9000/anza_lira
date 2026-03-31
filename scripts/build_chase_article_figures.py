#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from build_drawio_article_figures import (
    DrawIoBuilder,
    _crop_image,
    _label_panel,
    _load_png,
    _thumbnail_with_boxes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMPARISON_SAMPLE = "002_Image_06L"
COMPARISON_CROPS = [
    {"label": "A", "box": (381, 308, 220), "caption": "central bifurcation and side branches"},
    {"label": "B", "box": (496, 271, 220), "caption": "thin lateral branches recovered by the proposed method"},
]
XAI_CROP = {"label": "X", "box": (350, 250, 260), "caption": "optic-disc neighborhood with dominant-rule switch"}


def _gain_map(gt_mask: np.ndarray, baseline_mask: np.ndarray, thesis_mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    fixed = (baseline_mask != gt_mask) & (thesis_mask == gt_mask) & valid_mask
    lost = (baseline_mask == gt_mask) & (thesis_mask != gt_mask) & valid_mask
    gain = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    gain[fixed] = np.array([64, 224, 208], dtype=np.uint8)
    gain[lost] = np.array([255, 67, 163], dtype=np.uint8)
    return gain


def _save_comparison_assets(input_dir: Path, baseline_dir: Path, thesis_dir: Path) -> None:
    input_image = _load_png(input_dir / "input.png")
    gt_image = _load_png(input_dir / "ground_truth.png")
    baseline_image = _load_png(baseline_dir / "prediction.png")
    thesis_image = _load_png(thesis_dir / "prediction.png")
    full_thumb = _thumbnail_with_boxes(input_image, COMPARISON_CROPS, size=(360, 360))
    full_thumb.save(input_dir / "comparison_reference.png")

    gt_mask = np.asarray(Image.open(input_dir / "ground_truth_mask.png").convert("L"), dtype=np.uint8) > 127
    base_mask = np.asarray(Image.open(baseline_dir / "prediction_mask.png").convert("L"), dtype=np.uint8) > 127
    thesis_mask = np.asarray(Image.open(thesis_dir / "prediction_mask.png").convert("L"), dtype=np.uint8) > 127
    valid_mask = np.asarray(Image.open(input_dir / "valid_mask.png").convert("L"), dtype=np.uint8) > 127
    gain = _gain_map(gt_mask, base_mask, thesis_mask, valid_mask)
    Image.fromarray(gain).save(thesis_dir / "gain_full.png")
    _thumbnail_with_boxes(gt_image, COMPARISON_CROPS, size=(360, 360)).save(input_dir / "comparison_gt_reference.png")
    _thumbnail_with_boxes(baseline_image, COMPARISON_CROPS, size=(360, 360)).save(baseline_dir / "comparison_prediction_reference.png")
    _thumbnail_with_boxes(thesis_image, COMPARISON_CROPS, size=(360, 360)).save(thesis_dir / "comparison_prediction_reference.png")
    _thumbnail_with_boxes(Image.fromarray(gain), COMPARISON_CROPS, size=(360, 360)).save(thesis_dir / "comparison_gain_reference.png")

    for spec in COMPARISON_CROPS:
        label = str(spec["label"]).lower()
        box = spec["box"]
        _crop_image(input_image, box, (250, 250)).save(input_dir / f"crop_{label}_input.png")
        _crop_image(gt_image, box, (250, 250)).save(input_dir / f"crop_{label}_gt.png")
        _crop_image(baseline_image, box, (250, 250)).save(baseline_dir / f"crop_{label}_prediction.png")
        _crop_image(thesis_image, box, (250, 250)).save(thesis_dir / f"crop_{label}_prediction.png")
        _crop_image(Image.fromarray(gain), box, (250, 250), nearest=True).save(thesis_dir / f"gain_{label}.png")


def _save_xai_assets(sample_dir: Path) -> str:
    input_image = _load_png(sample_dir / "input.png")
    prediction_image = _load_png(sample_dir / "prediction.png")
    partition_image = _load_png(sample_dir / "rule_partition.png")
    summary = json.loads((sample_dir / "rule_summary.json").read_text(encoding="utf-8"))
    vessel_rule_name = summary.get("best_vessel_rule_name", "R?")
    vessel_rule_image = _load_png(sample_dir / f"vessel_rule_{vessel_rule_name.lower()}.png")
    full_thumb = _thumbnail_with_boxes(input_image, [XAI_CROP], size=(360, 360))
    full_thumb.save(sample_dir / "xai_reference.png")
    box = XAI_CROP["box"]
    _crop_image(input_image, box, (260, 260)).save(sample_dir / "xai_crop_input.png")
    _crop_image(prediction_image, box, (260, 260)).save(sample_dir / "xai_crop_prediction.png")
    _crop_image(partition_image, box, (260, 260), nearest=True).save(sample_dir / "xai_crop_rule_partition.png")
    _crop_image(vessel_rule_image, box, (260, 260)).save(sample_dir / f"xai_crop_vessel_rule_{vessel_rule_name.lower()}.png")
    return vessel_rule_name


def build_comparison_png(input_dir: Path, baseline_dir: Path, thesis_dir: Path, out_path: Path) -> None:
    note = Image.new("RGB", (360, 190), color=(246, 245, 241))
    draw = ImageDraw.Draw(note)
    draw.rounded_rectangle((0, 0, 359, 189), radius=14, outline=(139, 151, 173), width=2, fill=(246, 245, 241))
    note_text = (
        "Top row shows full-image predictions.\n"
        "Rows A/B zoom into the two regions with the clearest local differences.\n\n"
        "Improvement map:\n"
        "turquoise = baseline errors fixed by the proposed method\n"
        "magenta = new errors introduced by the proposed method"
    )
    draw.multiline_text((18, 24), note_text, fill=(28, 28, 28), spacing=10)

    left_col = Image.new("RGB", (380, 600), color=(232, 232, 228))
    left_col.paste(_label_panel(_load_png(input_dir / "comparison_reference.png"), "Reference Image", (360, 360)), (10, 10))
    left_col.paste(_label_panel(note, "Comparison Notes", (360, 190)), (10, 390))

    overview_panel_size = (180, 180)
    overview_headers = ["Input", "Ground Truth", "Baseline Full", "Proposed method Full", "Improvement map Full"]
    overview_images = [
        _load_png(input_dir / "comparison_reference.png"),
        _load_png(input_dir / "comparison_gt_reference.png"),
        _load_png(baseline_dir / "comparison_prediction_reference.png"),
        _load_png(thesis_dir / "comparison_prediction_reference.png"),
        _load_png(thesis_dir / "comparison_gain_reference.png"),
    ]
    overview_labeled = [_label_panel(image, title, overview_panel_size) for title, image in zip(overview_headers, overview_images)]
    overview_row_w = sum(item.width for item in overview_labeled)
    overview_row_h = max(item.height for item in overview_labeled)
    overview_row = Image.new("RGB", (overview_row_w, overview_row_h + 46), color=(232, 232, 228))
    draw = ImageDraw.Draw(overview_row)
    draw.text((10, 10), "Whole-image view: predictions over the entire retinal field", fill=(28, 28, 28))
    x = 0
    for panel in overview_labeled:
        overview_row.paste(panel, (x, 46))
        x += panel.width

    panel_size = (210, 210)
    headers = ["Input Crop", "Ground Truth", "Baseline", "Proposed method", "Improvement map"]
    rows: list[Image.Image] = [overview_row]
    for spec in COMPARISON_CROPS:
        label = str(spec["label"]).lower()
        row_items = [
            (headers[0], _load_png(input_dir / f"crop_{label}_input.png")),
            (headers[1], _load_png(input_dir / f"crop_{label}_gt.png")),
            (headers[2], _load_png(baseline_dir / f"crop_{label}_prediction.png")),
            (headers[3], _load_png(thesis_dir / f"crop_{label}_prediction.png")),
            (headers[4], _load_png(thesis_dir / f"gain_{label}.png")),
        ]
        labeled = [_label_panel(image, title, panel_size) for title, image in row_items]
        row_w = sum(item.width for item in labeled)
        row_h = max(item.height for item in labeled)
        row = Image.new("RGB", (row_w, row_h + 46), color=(232, 232, 228))
        draw = ImageDraw.Draw(row)
        draw.text((10, 10), f"Crop {spec['label']}: {spec['caption']}", fill=(28, 28, 28))
        x = 0
        for panel in labeled:
            row.paste(panel, (x, 46))
            x += panel.width
        rows.append(row)

    right_w = max(row.width for row in rows)
    right_h = sum(row.height for row in rows) + 18
    right = Image.new("RGB", (right_w, right_h), color=(232, 232, 228))
    y = 0
    for row in rows:
        right.paste(row, (0, y))
        y += row.height + 18

    canvas = Image.new("RGB", (left_col.width + 24 + right.width, max(left_col.height, right.height)), color=(232, 232, 228))
    canvas.paste(left_col, (0, 0))
    canvas.paste(right, (left_col.width + 24, 0))
    canvas.save(out_path)


def build_comparison_drawio(input_dir: Path, baseline_dir: Path, thesis_dir: Path, out_path: Path) -> None:
    builder = DrawIoBuilder(width=2100, height=1500)
    builder.add_text(120, 30, 1860, 40, "Figure 2. Local comparison of Baseline U-Net and the proposed method on CHASE_DB1", font_size=26, bold=True)
    builder.add_image(40, 140, 360, 360, input_dir / "comparison_reference.png")
    builder.add_text(40, 110, 360, 24, "Reference image with crop locations", font_size=18, bold=True)
    builder.add_box(
        420,
        150,
        560,
        130,
        "Top row shows full-image predictions. Rows A/B zoom into the two regions with the clearest local differences.<br><br>Improvement map: turquoise = baseline errors fixed by the proposed method; magenta = new errors introduced by the proposed method.",
        fill="#f6f7fb",
        stroke="#8b97ad",
    )

    x_positions = [430, 740, 1050, 1360, 1670]
    titles = ["Input", "Ground Truth", "Baseline Full", "Proposed method Full", "Improvement map Full"]
    for title, x in zip(titles, x_positions):
        builder.add_text(x, 300, 250, 24, title, font_size=18, bold=True)
    builder.add_text(40, 300, 340, 24, "Whole-image view", font_size=18, bold=True)
    builder.add_image(430, 330, 250, 250, input_dir / "comparison_reference.png")
    builder.add_image(740, 330, 250, 250, input_dir / "comparison_gt_reference.png")
    builder.add_image(1050, 330, 250, 250, baseline_dir / "comparison_prediction_reference.png")
    builder.add_image(1360, 330, 250, 250, thesis_dir / "comparison_prediction_reference.png")
    builder.add_image(1670, 330, 250, 250, thesis_dir / "comparison_gain_reference.png")

    for row_idx, spec in enumerate(COMPARISON_CROPS):
        y = 660 + row_idx * 360
        label = str(spec["label"]).lower()
        builder.add_text(40, y + 130, 340, 40, f"Crop {spec['label']}: {spec['caption']}", font_size=18, bold=True)
        builder.add_image(430, y, 250, 250, input_dir / f"crop_{label}_input.png")
        builder.add_image(740, y, 250, 250, input_dir / f"crop_{label}_gt.png")
        builder.add_image(1050, y, 250, 250, baseline_dir / f"crop_{label}_prediction.png")
        builder.add_image(1360, y, 250, 250, thesis_dir / f"crop_{label}_prediction.png")
        builder.add_image(1670, y, 250, 250, thesis_dir / f"gain_{label}.png")
    builder.save(out_path)


def build_xai_png(sample_dir: Path, vessel_rule_name: str, out_path: Path) -> None:
    panel_size = (270, 270)
    panels = [
        ("Input Crop", _load_png(sample_dir / "xai_crop_input.png")),
        ("Prediction Crop", _load_png(sample_dir / "xai_crop_prediction.png")),
        ("Rule Partition", _load_png(sample_dir / "xai_crop_rule_partition.png")),
        (f"Vessel Rule {vessel_rule_name}", _load_png(sample_dir / f"xai_crop_vessel_rule_{vessel_rule_name.lower()}.png")),
        ("Rule Statistics", _load_png(sample_dir / "rule_stats.png").resize(panel_size, Image.Resampling.BICUBIC)),
    ]

    note = Image.new("RGB", panel_size, color=(246, 245, 241))
    draw = ImageDraw.Draw(note)
    draw.rounded_rectangle((0, 0, panel_size[0] - 1, panel_size[1] - 1), radius=14, outline=(139, 151, 173), width=2, fill=(246, 245, 241))
    note_text = (
        f"Crop {XAI_CROP['label']}: optic-disc neighborhood.\n\n"
        "Rule partition shows the dominant fuzzy regime.\n"
        f"{vessel_rule_name} is the most vessel-preferring rule here."
    )
    draw.multiline_text((18, 18), note_text, fill=(28, 28, 28), spacing=10)

    labeled = [_label_panel(image, title, panel_size) for title, image in panels]
    note_panel = _label_panel(note, "Interpretation Notes", panel_size)
    ref_panel = _label_panel(_load_png(sample_dir / "xai_reference.png"), "Reference Image", (360, 360))

    canvas = Image.new("RGB", (1290, 800), color=(232, 232, 228))
    canvas.paste(ref_panel, (20, 20))
    canvas.paste(labeled[0], (410, 20))
    canvas.paste(labeled[1], (700, 20))
    canvas.paste(labeled[2], (410, 410))
    canvas.paste(labeled[3], (700, 410))
    canvas.paste(labeled[4], (990, 20))
    canvas.paste(note_panel, (990, 410))
    canvas.save(out_path)


def build_xai_drawio(sample_dir: Path, vessel_rule_name: str, out_path: Path) -> None:
    builder = DrawIoBuilder(width=1820, height=1180)
    builder.add_text(120, 30, 1560, 40, "Figure 3. Local XAI view of fuzzy rule partition on CHASE_DB1", font_size=26, bold=True)
    builder.add_text(40, 110, 360, 24, "Reference image and XAI crop", font_size=18, bold=True)
    builder.add_image(40, 140, 360, 360, sample_dir / "xai_reference.png")
    builder.add_text(440, 110, 270, 24, "Input Crop", font_size=18, bold=True)
    builder.add_image(440, 140, 270, 270, sample_dir / "xai_crop_input.png")
    builder.add_text(740, 110, 270, 24, "Prediction Crop", font_size=18, bold=True)
    builder.add_image(740, 140, 270, 270, sample_dir / "xai_crop_prediction.png")
    builder.add_text(1040, 110, 270, 24, "Rule Statistics", font_size=18, bold=True)
    builder.add_image(1040, 140, 270, 270, sample_dir / "rule_stats.png")
    builder.add_text(440, 470, 270, 24, "Rule Partition", font_size=18, bold=True)
    builder.add_image(440, 500, 270, 270, sample_dir / "xai_crop_rule_partition.png")
    builder.add_text(740, 470, 270, 24, f"Vessel Rule {vessel_rule_name}", font_size=18, bold=True)
    builder.add_image(740, 500, 270, 270, sample_dir / f"xai_crop_vessel_rule_{vessel_rule_name.lower()}.png")
    note = (
        f"Crop {XAI_CROP['label']}: {XAI_CROP['caption']}<br><br>"
        "The rule partition shows where the first AZ block switches its dominant rule.<br>"
        f"{vessel_rule_name} is the strongest vessel-preferring rule in this sample.<br>"
        "The rule-statistics panel compares vessel and background memberships."
    )
    builder.add_box(1040, 500, 520, 250, note, fill="#f6f7fb", stroke="#8b97ad")
    builder.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CHASE_DB1 article figures from exported model assets.")
    parser.add_argument("--thesis-root", type=str, default="article_assets/exports_chase_transfer/chase_az_thesis_from_fives16_probe_ft20")
    parser.add_argument("--baseline-root", type=str, default="article_assets/exports_chase_transfer/chase_baseline_dice80_p512_fg07")
    parser.add_argument("--sample", type=str, default="012_Image_11L")
    parser.add_argument("--output-dir", type=str, default="article_assets/final_figures_chase_transfer")
    args = parser.parse_args()

    thesis_root = (PROJECT_ROOT / args.thesis_root).resolve()
    baseline_root = (PROJECT_ROOT / args.baseline_root).resolve()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    thesis_sample = thesis_root / args.sample
    baseline_sample = baseline_root / args.sample
    if not thesis_sample.exists():
        raise SystemExit(f"Missing thesis sample export: {thesis_sample}")
    if not baseline_sample.exists():
        raise SystemExit(f"Missing baseline sample export: {baseline_sample}")

    _save_comparison_assets(thesis_sample, baseline_sample, thesis_sample)
    vessel_rule_name = _save_xai_assets(thesis_sample)

    build_comparison_png(thesis_sample, baseline_sample, thesis_sample, out_dir / "figure2_chase_examples.png")
    build_comparison_drawio(thesis_sample, baseline_sample, thesis_sample, out_dir / "figure2_chase_examples.drawio")
    build_xai_png(thesis_sample, vessel_rule_name, out_dir / "figure3_chase_xai.png")
    build_xai_drawio(thesis_sample, vessel_rule_name, out_dir / "figure3_chase_xai.drawio")
    print(f"Built CHASE article figures under {out_dir}")


if __name__ == "__main__":
    main()
