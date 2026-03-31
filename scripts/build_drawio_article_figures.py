#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import shutil
import textwrap
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEGMENTATION_SAMPLE = "000_01_test"
SEGMENTATION_CROPS = [
    {"label": "A", "box": (288, 128, 128), "caption": "branch junctions and thin links"},
    {"label": "B", "box": (320, 192, 128), "caption": "peripheral thin vessels"},
]
XAI_SAMPLE = "000_01_test"
XAI_CROP = {"label": "X", "box": (32, 160, 128), "caption": "optic-disc neighborhood with mixed rules"}


def _load_png(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _fit_size(image: Image.Image, max_w: int, max_h: int) -> tuple[int, int]:
    scale = min(max_w / image.width, max_h / image.height)
    return max(1, int(image.width * scale)), max(1, int(image.height * scale))


def _label_panel(image: Image.Image, title: str, size: tuple[int, int], bar_height: int = 36) -> Image.Image:
    panel = image.copy()
    panel.thumbnail(size, Image.Resampling.LANCZOS)
    framed = Image.new("RGB", (size[0], size[1] + bar_height), color=(244, 243, 239))
    px = (size[0] - panel.width) // 2
    py = bar_height + (size[1] - panel.height) // 2
    framed.paste(panel, (px, py))
    draw = ImageDraw.Draw(framed)
    draw.rectangle((0, 0, framed.width, bar_height), fill=(27, 34, 46))
    draw.text((12, 9), title, fill=(255, 255, 255))
    return framed


def _save_grid(rows: list[tuple[str, list[tuple[str, Image.Image]]]], out_path: Path, cell_size: tuple[int, int]) -> None:
    labeled_rows = [[_label_panel(image, title, cell_size) for title, image in panels] for _, panels in rows]
    columns = max(len(row) for row in labeled_rows)
    cell_w = max(panel.width for row in labeled_rows for panel in row)
    cell_h = max(panel.height for row in labeled_rows for panel in row)
    row_label_w = 180
    canvas = Image.new("RGB", (row_label_w + columns * cell_w, len(rows) * cell_h), color=(232, 232, 228))
    for row_idx, row in enumerate(labeled_rows):
        label_draw = ImageDraw.Draw(canvas)
        y = row_idx * cell_h
        label_draw.rectangle((0, y, row_label_w, y + cell_h), fill=(245, 244, 239), outline=(210, 210, 205))
        label_draw.text((18, y + 24), rows[row_idx][0], fill=(28, 28, 28))
        for col_idx, panel in enumerate(row):
            x = row_label_w + col_idx * cell_w
            canvas.paste(panel, (x, y))
    canvas.save(out_path)


def _crop_image(image: Image.Image, box: tuple[int, int, int], size: tuple[int, int], nearest: bool = False) -> Image.Image:
    x, y, patch = box
    crop = image.crop((x, y, x + patch, y + patch))
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
    return crop.resize(size, resample)


def _thumbnail_with_boxes(image: Image.Image, boxes: list[dict[str, object]], size: tuple[int, int]) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    scale_x = thumb.width / image.width
    scale_y = thumb.height / image.height
    canvas = Image.new("RGB", size, color=(244, 243, 239))
    px = (size[0] - thumb.width) // 2
    py = (size[1] - thumb.height) // 2
    canvas.paste(thumb, (px, py))
    draw = ImageDraw.Draw(canvas)
    colors = ["#4fd1c5", "#f6ad55", "#63b3ed"]
    for idx, spec in enumerate(boxes):
        x, y, patch = spec["box"]
        x0 = int(px + x * scale_x)
        y0 = int(py + y * scale_y)
        x1 = int(px + (x + patch) * scale_x)
        y1 = int(py + (y + patch) * scale_y)
        color = colors[idx % len(colors)]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
        draw.rounded_rectangle((x0 + 4, y0 + 4, x0 + 34, y0 + 30), radius=7, fill=color)
        draw.text((x0 + 13, y0 + 9), str(spec["label"]), fill=(20, 20, 20))
    return canvas


@dataclass
class DrawIoBuilder:
    width: int
    height: int

    def __post_init__(self) -> None:
        self._next_id = 2
        self.mxfile = ET.Element(
            "mxfile",
            {
                "host": "app.diagrams.net",
                "modified": "2026-03-31T00:00:00.000Z",
                "agent": "Codex",
                "version": "26.0.0",
            },
        )
        self.diagram = ET.SubElement(self.mxfile, "diagram", {"id": str(uuid.uuid4())[:8], "name": "Page-1"})
        self.model = ET.SubElement(
            self.diagram,
            "mxGraphModel",
            {
                "dx": "1600",
                "dy": "900",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(self.width),
                "pageHeight": str(self.height),
                "math": "0",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})

    def _id(self) -> str:
        value = str(self._next_id)
        self._next_id += 1
        return value

    def add_text(self, x: int, y: int, w: int, h: int, text: str, font_size: int = 18, bold: bool = False) -> None:
        style = (
            "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
            f"whiteSpace=wrap;fontSize={font_size};fontStyle={'1' if bold else '0'};"
        )
        cell = ET.SubElement(self.root, "mxCell", {"id": self._id(), "value": text, "style": style, "vertex": "1", "parent": "1"})
        ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})

    def add_box(self, x: int, y: int, w: int, h: int, text: str, fill: str, stroke: str = "#3a4b66") -> str:
        style = (
            "rounded=1;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};fontSize=18;fontStyle=1;arcSize=12;"
        )
        cell_id = self._id()
        cell = ET.SubElement(self.root, "mxCell", {"id": cell_id, "value": text, "style": style, "vertex": "1", "parent": "1"})
        ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
        return cell_id

    def add_arrow(self, src: str, dst: str, text: str = "") -> None:
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
            "html=1;endArrow=block;endFill=1;strokeWidth=2;strokeColor=#44556f;"
        )
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": self._id(),
                "value": text,
                "style": style,
                "edge": "1",
                "parent": "1",
                "source": src,
                "target": dst,
            },
        )
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})

    def add_image(self, x: int, y: int, w: int, h: int, image_path: Path) -> None:
        payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        style = (
            "shape=image;html=1;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;"
            f"image=data:image/png;base64,{payload};"
        )
        cell = ET.SubElement(self.root, "mxCell", {"id": self._id(), "value": "", "style": style, "vertex": "1", "parent": "1"})
        ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})

    def save(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(self.mxfile)
        tree.write(out_path, encoding="utf-8", xml_declaration=True)


def build_pipeline_drawio(out_path: Path) -> None:
    builder = DrawIoBuilder(width=1600, height=900)
    builder.add_text(120, 40, 1360, 40, "Figure 1. Segmentation pipeline based on ANZA/LIRA blocks", font_size=26, bold=True)
    box1 = builder.add_box(90, 270, 180, 110, "Input retinal\nimage", fill="#fde8d9")
    box2 = builder.add_box(340, 250, 230, 150, "Encoder\nAZ residual blocks\n+ skip features", fill="#d8ecff")
    box3 = builder.add_box(650, 250, 220, 150, "Bottleneck\nAZ / ASPP\nrefinement", fill="#e2d8ff")
    box4 = builder.add_box(950, 250, 230, 150, "Decoder\nresidual fusion\n+ AZ guidance", fill="#d9f3e6")
    box5 = builder.add_box(1260, 250, 190, 150, "Probability map\np(x)", fill="#fff0bf")
    box6 = builder.add_box(1260, 500, 190, 110, "Threshold\nτ", fill="#ffe1bf")
    box7 = builder.add_box(1260, 670, 190, 110, "Binary vessel\nmask", fill="#ffd7d7")
    callout = builder.add_box(
        450,
        520,
        520,
        180,
        "Inside one AZ block:\nvalue projection + fuzzy memberships μr\n+ anisotropic geometry κr\n+ normalized local aggregation",
        fill="#f6f7fb",
        stroke="#8b97ad",
    )
    builder.add_arrow(box1, box2)
    builder.add_arrow(box2, box3)
    builder.add_arrow(box3, box4)
    builder.add_arrow(box4, box5)
    builder.add_arrow(box5, box6)
    builder.add_arrow(box6, box7)
    builder.add_arrow(box2, callout, "detail")
    builder.save(out_path)


def build_qualitative_png(sample_dirs: list[Path], out_path: Path) -> None:
    rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []
    for sample_dir in sample_dirs:
        row_label = sample_dir.name
        panels = [
            ("Input", _load_png(sample_dir / "input.png")),
            ("Ground Truth", _load_png(sample_dir / "ground_truth.png")),
            ("Prediction", _load_png(sample_dir / "prediction.png")),
            ("Error Map", _load_png(sample_dir / "error_map.png")),
        ]
        rows.append((row_label, panels))
    _save_grid(rows, out_path, cell_size=(280, 280))


def build_comparison_png(input_dir: Path, baseline_dir: Path, thesis_dir: Path, out_path: Path) -> None:
    note = Image.new("RGB", (340, 130), color=(246, 245, 241))
    draw = ImageDraw.Draw(note)
    draw.rounded_rectangle((0, 0, 339, 129), radius=14, outline=(139, 151, 173), width=2, fill=(246, 245, 241))
    note_text = (
        "Turquoise pixels in the improvement map = baseline errors fixed by the proposed method\n"
        "Magenta pixels in the improvement map = errors introduced relative to baseline"
    )
    draw.multiline_text((18, 24), note_text, fill=(28, 28, 28), spacing=10)

    left_col = Image.new("RGB", (360, 520), color=(232, 232, 228))
    left_col.paste(_label_panel(_load_png(input_dir / "comparison_reference.png"), "Reference Image", (340, 340)), (10, 10))
    left_col.paste(_label_panel(note, "Comparison Notes", (340, 130)), (10, 380))

    panel_size = (210, 210)
    headers = ["Input Crop", "Ground Truth", "Baseline", "Proposed method", "Improvement map"]
    rows: list[Image.Image] = []
    for spec in SEGMENTATION_CROPS:
        row_items = [
            (headers[0], _load_png(input_dir / f"crop_{str(spec['label']).lower()}_input.png")),
            (headers[1], _load_png(input_dir / f"crop_{str(spec['label']).lower()}_gt.png")),
            (headers[2], _load_png(baseline_dir / f"crop_{str(spec['label']).lower()}_prediction.png")),
            (headers[3], _load_png(thesis_dir / f"crop_{str(spec['label']).lower()}_prediction.png")),
            (headers[4], _load_png(thesis_dir / f"gain_{str(spec['label']).lower()}.png")),
        ]
        labeled = [_label_panel(image, title, panel_size) for title, image in row_items]
        row_w = sum(item.width for item in labeled)
        row_h = max(item.height for item in labeled)
        row = Image.new("RGB", (row_w, row_h + 44), color=(232, 232, 228))
        draw = ImageDraw.Draw(row)
        draw.text((10, 8), f"Crop {spec['label']}: {spec['caption']}", fill=(28, 28, 28))
        x = 0
        for panel in labeled:
            row.paste(panel, (x, 44))
            x += panel.width
        rows.append(row)

    right_w = max(row.width for row in rows)
    right_h = sum(row.height for row in rows) + 20
    right = Image.new("RGB", (right_w, right_h), color=(232, 232, 228))
    y = 0
    for row in rows:
        right.paste(row, (0, y))
        y += row.height + 20

    canvas = Image.new("RGB", (left_col.width + 24 + right.width, max(left_col.height, right.height)), color=(232, 232, 228))
    canvas.paste(left_col, (0, 0))
    canvas.paste(right, (left_col.width + 24, 0))
    canvas.save(out_path)


def build_xai_png(sample_dir: Path, out_path: Path) -> None:
    import json

    summary = json.loads((sample_dir / "rule_summary.json").read_text(encoding="utf-8"))
    vessel_rule_name = summary.get("best_vessel_rule_name", "R?")
    note = Image.new("RGB", (320, 320), color=(246, 245, 241))
    draw = ImageDraw.Draw(note)
    draw.rectangle((0, 0, 320, 54), fill=(27, 34, 46))
    draw.text((14, 16), "Interpretation Notes", fill=(255, 255, 255))
    body = textwrap.dedent(
        """
        - sample: 04_test
        - model: az_thesis_seed42
        - layer: enc1.body.0
        - dominant rule map shows how fuzzy rules partition the retina
        - vessel rule overlay shows the most vessel-preferring rule
        - rule statistics compare vessel vs background memberships
        """
    ).strip()
    draw.multiline_text((18, 78), body, fill=(28, 28, 28), spacing=10)
    panels = [
        ("Input", _load_png(sample_dir / "input.png")),
        ("Prediction", _load_png(sample_dir / "prediction.png")),
        ("Error Map", _load_png(sample_dir / "error_map.png")),
        ("Rule Partition", _load_png(sample_dir / "rule_partition.png")),
        (f"Vessel Rule {vessel_rule_name}", _load_png(sample_dir / f"vessel_rule_{vessel_rule_name.lower()}.png")),
        ("Rule Stats", _load_png(sample_dir / "rule_stats.png")),
        ("Interpretation Notes", note),
    ]
    labeled = [_label_panel(image, title, (320, 320)) for title, image in panels]
    cell_w = max(panel.width for panel in labeled)
    cell_h = max(panel.height for panel in labeled)
    canvas = Image.new("RGB", (3 * cell_w, 3 * cell_h), color=(232, 232, 228))
    for idx, panel in enumerate(labeled):
        row = idx // 3
        col = idx % 3
        canvas.paste(panel, (col * cell_w, row * cell_h))
    canvas.save(out_path)


def build_xai_local_png(sample_dir: Path, out_path: Path) -> None:
    summary = json.loads((sample_dir / "rule_summary.json").read_text(encoding="utf-8"))
    vessel_rule_name = summary.get("best_vessel_rule_name", "R?")
    input_image = _load_png(sample_dir / "input.png")
    prediction_image = _load_png(sample_dir / "prediction.png")
    partition_image = _load_png(sample_dir / "rule_partition.png")
    vessel_rule_image = _load_png(sample_dir / f"vessel_rule_{vessel_rule_name.lower()}.png")
    stats_image = _load_png(sample_dir / "rule_stats.png")

    panel_size = (250, 250)
    panels = [
        ("Input Crop", _load_png(sample_dir / "xai_crop_input.png")),
        ("Prediction Crop", _load_png(sample_dir / "xai_crop_prediction.png")),
        ("Rule Partition", _load_png(sample_dir / "xai_crop_rule_partition.png")),
        (f"Vessel Rule {vessel_rule_name}", _load_png(sample_dir / f"xai_crop_vessel_rule_{vessel_rule_name.lower()}.png")),
        ("Rule Statistics", stats_image.copy().resize(panel_size, Image.Resampling.BICUBIC)),
    ]

    note = Image.new("RGB", panel_size, color=(246, 245, 241))
    draw = ImageDraw.Draw(note)
    draw.rounded_rectangle((0, 0, panel_size[0] - 1, panel_size[1] - 1), radius=14, outline=(139, 151, 173), width=2, fill=(246, 245, 241))
    note_text = (
        f"Crop {XAI_CROP['label']} is near the optic disc.\n\n"
        "Rule partition shows where the first AZ block switches dominant rule.\n"
        f"{vessel_rule_name} is the most vessel-preferring rule here.\n"
        "Rule stats compare vessel vs background memberships."
    )
    draw.multiline_text((18, 20), note_text, fill=(28, 28, 28), spacing=10)

    labeled = [_label_panel(image, title, panel_size) for title, image in panels]
    note_panel = _label_panel(note, "Interpretation Notes", panel_size)
    top_left = _label_panel(_load_png(sample_dir / "xai_reference.png"), "Reference Image", (340, 340))

    canvas = Image.new("RGB", (1220, 760), color=(232, 232, 228))
    canvas.paste(top_left, (20, 20))
    canvas.paste(labeled[0], (390, 20))
    canvas.paste(labeled[1], (660, 20))
    canvas.paste(labeled[2], (390, 390))
    canvas.paste(labeled[3], (660, 390))
    canvas.paste(labeled[4], (20, 390))
    canvas.paste(note_panel, (930, 390))
    canvas.save(out_path)


def build_qualitative_drawio(sample_dirs: list[Path], out_path: Path) -> None:
    builder = DrawIoBuilder(width=1800, height=1600)
    builder.add_text(120, 30, 1560, 40, "Figure 2. Qualitative vessel segmentation on DRIVE test images", font_size=26, bold=True)
    headers = ["Input", "Ground Truth", "Prediction", "Error Map"]
    x_positions = [220, 560, 900, 1240]
    for title, x in zip(headers, x_positions):
        builder.add_text(x, 110, 260, 30, title, font_size=18, bold=True)

    y = 180
    panel_w = 260
    panel_h = 260
    for sample_dir in sample_dirs:
        metrics_path = sample_dir / "metrics.json"
        metrics = metrics_path.read_text(encoding="utf-8")
        dice_fragment = "Dice"
        try:
            import json

            dice_value = json.loads(metrics)["dice"]
            dice_fragment = f"{sample_dir.name} | Dice={dice_value:.3f}"
        except Exception:
            dice_fragment = sample_dir.name
        builder.add_text(20, y + 110, 170, 40, dice_fragment, font_size=16, bold=True)
        for image_name, x in zip(["input.png", "ground_truth.png", "prediction.png", "error_map.png"], x_positions):
            builder.add_image(x, y, panel_w, panel_h, sample_dir / image_name)
        y += 360
    builder.save(out_path)


def build_comparison_drawio(input_dir: Path, baseline_dir: Path, thesis_dir: Path, out_path: Path) -> None:
    builder = DrawIoBuilder(width=2100, height=1200)
    builder.add_text(120, 30, 1860, 40, "Figure 2. Local comparison of Baseline U-Net and the proposed method on hard DRIVE regions", font_size=26, bold=True)
    builder.add_image(40, 140, 360, 360, input_dir / "comparison_reference.png")
    builder.add_text(40, 110, 360, 24, "Reference image with crop locations", font_size=18, bold=True)
    builder.add_box(420, 150, 540, 90, "Improvement map: turquoise = baseline errors fixed by the proposed method; magenta = errors introduced by the proposed method", fill="#f6f7fb", stroke="#8b97ad")

    x_positions = [430, 740, 1050, 1360, 1670]
    titles = ["Input Crop", "Ground Truth", "Baseline", "Proposed method", "Improvement map"]
    for title, x in zip(titles, x_positions):
        builder.add_text(x, 120, 250, 24, title, font_size=18, bold=True)

    for row_idx, spec in enumerate(SEGMENTATION_CROPS):
        y = 220 + row_idx * 430
        builder.add_text(40, y + 130, 340, 40, f"Crop {spec['label']}: {spec['caption']}", font_size=18, bold=True)
        builder.add_image(430, y, 250, 250, input_dir / f"crop_{spec['label'].lower()}_input.png")
        builder.add_image(740, y, 250, 250, input_dir / f"crop_{spec['label'].lower()}_gt.png")
        builder.add_image(1050, y, 250, 250, baseline_dir / f"crop_{spec['label'].lower()}_prediction.png")
        builder.add_image(1360, y, 250, 250, thesis_dir / f"crop_{spec['label'].lower()}_prediction.png")
        builder.add_image(1670, y, 250, 250, thesis_dir / f"gain_{spec['label'].lower()}.png")
    builder.save(out_path)


def build_xai_drawio(sample_dir: Path, out_path: Path) -> None:
    summary = json.loads((sample_dir / "rule_summary.json").read_text(encoding="utf-8"))
    vessel_rule_name = summary.get("best_vessel_rule_name", "R?")
    builder = DrawIoBuilder(width=1800, height=1600)
    builder.add_text(120, 30, 1560, 40, "Figure 3. XAI view of the first AZ block on a representative DRIVE image", font_size=26, bold=True)
    titles = ["Input", "Prediction", "Error Map", "Rule Partition", f"Vessel Rule {vessel_rule_name}", "Rule Stats"]
    files = ["input.png", "prediction.png", "error_map.png", "rule_partition.png", f"vessel_rule_{vessel_rule_name.lower()}.png", "rule_stats.png"]
    positions = [(120, 120), (560, 120), (1000, 120), (120, 560), (560, 560), (1000, 560)]
    for title, (x, y), filename in zip(titles, positions, files):
        builder.add_text(x, y - 34, 320, 28, title, font_size=18, bold=True)
        builder.add_image(x, y, 320, 320, sample_dir / filename)
    note = (
        "Interpretation\n\n"
        "- sample: 04_test\n"
        "- model: az_thesis_seed42\n"
        "- layer: enc1.body.0\n"
        "- dominant-rule map visualizes fuzzy partition of the retina\n"
        f"- {vessel_rule_name} is the strongest vessel-preferring rule in this sample\n"
        "- rule-stats panel shows which rules prefer vessels vs background"
    )
    builder.add_box(640, 1020, 520, 250, note.replace("\n", "<br>"), fill="#f6f7fb", stroke="#8b97ad")
    builder.save(out_path)


def build_xai_local_drawio(sample_dir: Path, out_path: Path) -> None:
    summary = json.loads((sample_dir / "rule_summary.json").read_text(encoding="utf-8"))
    vessel_rule_name = summary.get("best_vessel_rule_name", "R?")
    builder = DrawIoBuilder(width=1800, height=1200)
    builder.add_text(120, 30, 1560, 40, "Figure 3. Local XAI view of fuzzy rule partition in the first AZ block", font_size=26, bold=True)
    builder.add_text(40, 110, 340, 24, "Reference image and XAI crop", font_size=18, bold=True)
    builder.add_image(40, 140, 340, 340, sample_dir / "xai_reference.png")
    builder.add_text(420, 110, 250, 24, "Input Crop", font_size=18, bold=True)
    builder.add_image(420, 140, 250, 250, sample_dir / "xai_crop_input.png")
    builder.add_text(710, 110, 250, 24, "Prediction Crop", font_size=18, bold=True)
    builder.add_image(710, 140, 250, 250, sample_dir / "xai_crop_prediction.png")
    builder.add_text(1000, 110, 250, 24, "Rule Partition", font_size=18, bold=True)
    builder.add_image(1000, 140, 250, 250, sample_dir / "xai_crop_rule_partition.png")
    builder.add_text(420, 470, 250, 24, f"Vessel Rule {vessel_rule_name}", font_size=18, bold=True)
    builder.add_image(420, 500, 250, 250, sample_dir / f"xai_crop_vessel_rule_{vessel_rule_name.lower()}.png")
    builder.add_text(710, 470, 250, 24, "Rule Statistics", font_size=18, bold=True)
    builder.add_image(710, 500, 250, 250, sample_dir / "rule_stats.png")
    note = (
        f"Crop {XAI_CROP['label']}: {XAI_CROP['caption']}<br><br>"
        "Rule partition shows where the first AZ block switches dominant rule.<br>"
        f"{vessel_rule_name} is the strongest vessel-preferring rule here.<br>"
        "Rule statistics compare vessel and background memberships."
    )
    builder.add_box(1000, 500, 520, 250, note, fill="#f6f7fb", stroke="#8b97ad")
    builder.save(out_path)


def _save_comparison_assets(input_dir: Path, baseline_dir: Path, thesis_dir: Path) -> None:
    input_image = _load_png(input_dir / "input.png")
    gt_image = _load_png(input_dir / "ground_truth.png")
    baseline_image = _load_png(baseline_dir / "prediction.png")
    thesis_image = _load_png(thesis_dir / "prediction.png")
    full_thumb = _thumbnail_with_boxes(input_image, SEGMENTATION_CROPS, size=(340, 340))
    full_thumb.save(input_dir / "comparison_reference.png")
    for spec in SEGMENTATION_CROPS:
        label = str(spec["label"]).lower()
        box = spec["box"]
        _crop_image(input_image, box, (250, 250)).save(input_dir / f"crop_{label}_input.png")
        _crop_image(gt_image, box, (250, 250)).save(input_dir / f"crop_{label}_gt.png")
        _crop_image(baseline_image, box, (250, 250)).save(baseline_dir / f"crop_{label}_prediction.png")
        _crop_image(thesis_image, box, (250, 250)).save(thesis_dir / f"crop_{label}_prediction.png")
        gt_mask = np.asarray(Image.open(input_dir / "ground_truth_mask.png").convert("L"), dtype=np.uint8) > 127
        base_mask = np.asarray(Image.open(baseline_dir / "prediction_mask.png").convert("L"), dtype=np.uint8) > 127
        thesis_mask = np.asarray(Image.open(thesis_dir / "prediction_mask.png").convert("L"), dtype=np.uint8) > 127
        valid_mask = np.asarray(Image.open(input_dir / "valid_mask.png").convert("L"), dtype=np.uint8) > 127
        gain = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        fixed = (base_mask != gt_mask) & (thesis_mask == gt_mask) & valid_mask
        lost = (base_mask == gt_mask) & (thesis_mask != gt_mask) & valid_mask
        gain[fixed] = np.array([64, 224, 208], dtype=np.uint8)
        gain[lost] = np.array([255, 67, 163], dtype=np.uint8)
        _crop_image(Image.fromarray(gain), box, (250, 250), nearest=True).save(thesis_dir / f"gain_{label}.png")


def _save_xai_assets(sample_dir: Path) -> None:
    input_image = _load_png(sample_dir / "input.png")
    prediction_image = _load_png(sample_dir / "prediction.png")
    partition_image = _load_png(sample_dir / "rule_partition.png")
    summary = json.loads((sample_dir / "rule_summary.json").read_text(encoding="utf-8"))
    vessel_rule_name = summary.get("best_vessel_rule_name", "R?")
    vessel_rule_image = _load_png(sample_dir / f"vessel_rule_{vessel_rule_name.lower()}.png")
    full_thumb = _thumbnail_with_boxes(input_image, [XAI_CROP], size=(340, 340))
    full_thumb.save(sample_dir / "xai_reference.png")
    box = XAI_CROP["box"]
    _crop_image(input_image, box, (250, 250)).save(sample_dir / "xai_crop_input.png")
    _crop_image(prediction_image, box, (250, 250)).save(sample_dir / "xai_crop_prediction.png")
    _crop_image(partition_image, box, (250, 250), nearest=True).save(sample_dir / "xai_crop_rule_partition.png")
    _crop_image(vessel_rule_image, box, (250, 250)).save(sample_dir / f"xai_crop_vessel_rule_{vessel_rule_name.lower()}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact 3-figure article pack in draw.io format.")
    parser.add_argument("--export-root", type=str, default="article_assets/exports/az_thesis_seed42")
    parser.add_argument("--baseline-root", type=str, default="article_assets/exports/baseline_seed43")
    parser.add_argument("--output-dir", type=str, default="article_assets/final_figures")
    args = parser.parse_args()

    export_root = (PROJECT_ROOT / args.export_root).resolve()
    baseline_root = (PROJECT_ROOT / args.baseline_root).resolve()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_input = export_root / SEGMENTATION_SAMPLE
    comparison_baseline = baseline_root / SEGMENTATION_SAMPLE
    comparison_thesis = export_root / SEGMENTATION_SAMPLE
    xai_sample = export_root / XAI_SAMPLE
    required_dirs = [comparison_input, comparison_baseline, comparison_thesis, xai_sample]
    for sample_dir in required_dirs:
        if not sample_dir.exists():
            raise SystemExit(f"Missing exported sample directory: {sample_dir}")

    _save_comparison_assets(comparison_input, comparison_baseline, comparison_thesis)
    _save_xai_assets(xai_sample)

    build_pipeline_drawio(out_dir / "figure1_pipeline.drawio")
    shutil.copyfile(PROJECT_ROOT / "article_assets" / "segmentation_pipeline.svg", out_dir / "figure1_pipeline.svg")
    build_comparison_png(comparison_input, comparison_baseline, comparison_thesis, out_dir / "figure2_drive_examples.png")
    build_comparison_drawio(comparison_input, comparison_baseline, comparison_thesis, out_dir / "figure2_drive_examples.drawio")
    build_xai_local_png(xai_sample, out_dir / "figure3_xai.png")
    build_xai_local_drawio(xai_sample, out_dir / "figure3_xai.drawio")
    print(f"Built 3 article figures under {out_dir}")


if __name__ == "__main__":
    main()
