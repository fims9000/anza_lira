"""Article-ready, deterministic figures from frozen ANZA-LIRA v2 artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cracks_experiment.finetuning import verify_setting_a_complete
from cracks_experiment.human import _normalized_image, tiled_v2_uncertainty
from cracks_experiment.matrix import PROJECT_ROOT, setting_a_matrix
from cracks_experiment.training import build_real_model, load_real_checkpoint
from cracks_experiment.validation import tiled_probability
from datasets.cracks import fuse_crowd_masks, load_rgb_mask, load_section_image, map_mask_rgb
from models.segmentation_v2 import build_comparable_model
from synthetic.crossing_trace_bench import generate_sample
from synthetic.experiment_matrix import development_matrix
from synthetic.training import load_checkpoint
from trace_extraction.skeleton import skeletonize_mask


def _save(fig: plt.Figure, output_root: Path, index: int) -> None:
    fig.patch.set_facecolor("white")
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(output_root / f"figure_{index}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def _hide(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def _operator_figure(output_root: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    for axis, title in zip(axes, ("ANZA-LIRA V1: early fusion", "ANZA-LIRA V2: delayed mode fusion")):
        axis.set_xlim(0, 10)
        axis.set_ylim(0, 6)
        axis.axis("off")
        axis.set_title(title, fontsize=11)
    axes[0].plot([1, 4], [1, 5], color="#2878B5", lw=4)
    axes[0].plot([1, 4], [5, 1], color="#D95319", lw=4)
    axes[0].annotate("", xy=(8.5, 3), xytext=(4.5, 3), arrowprops={"arrowstyle": "->", "lw": 2})
    axes[0].add_patch(plt.Circle((8.7, 3), 0.65, color="#777777"))
    axes[0].text(6.5, 3.45, "single fused response", ha="center", fontsize=9)
    axes[1].plot([1, 4], [1, 5], color="#2878B5", lw=4)
    axes[1].plot([1, 4], [5, 1], color="#D95319", lw=4)
    axes[1].annotate("", xy=(7.5, 4.7), xytext=(4.5, 3.7), arrowprops={"arrowstyle": "->", "color": "#2878B5", "lw": 2})
    axes[1].annotate("", xy=(7.5, 1.3), xytext=(4.5, 2.3), arrowprops={"arrowstyle": "->", "color": "#D95319", "lw": 2})
    axes[1].plot([7.5, 9], [4.7, 5.3], color="#2878B5", lw=4)
    axes[1].plot([7.5, 9], [1.3, 0.7], color="#D95319", lw=4)
    axes[1].text(6.2, 3.0, "normalized mode transport", ha="center", fontsize=9)
    _save(fig, output_root, 1)


def _median_row(path: Path, metric: str, *, predicate: Any | None = None) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if predicate is not None:
        rows = [row for row in rows if predicate(row)]
    values = np.asarray([float(row[metric]) for row in rows])
    median = float(np.median(values))
    return min(rows, key=lambda row: (abs(float(row[metric]) - median), int(row["index"]) if "index" in row else int(row["section_id"])))


def _synthetic_figure(study_root: Path, output_root: Path, device: torch.device) -> dict[str, Any]:
    spec_by_id = {spec.candidate_id: spec for spec in development_matrix()}
    c3 = spec_by_id["C3"]
    validation_path = study_root / "synthetic" / "validation" / f"C3-{c3.run_hash}.csv"
    selected = _median_row(
        validation_path,
        "visible_dice",
        predicate=lambda row: "nontrivial_pairing" in row["strata"],
    )
    index = int(selected["index"])
    sample = generate_sample("validation", index, image_size=128)
    predictions = {}
    for candidate_id in ("C0", "C3"):
        spec = spec_by_id[candidate_id]
        model = build_comparable_model(spec.model).to(device)
        checkpoint = study_root / "synthetic" / "development" / f"{candidate_id}-{spec.run_hash}" / "checkpoint-last.pt"
        load_checkpoint(checkpoint, expected_hash=spec.run_hash, model=model)
        model.eval()
        with torch.no_grad():
            logits = model(torch.as_tensor(sample["image"], device=device).unsqueeze(0))
        threshold = json.loads(
            (study_root / "synthetic" / "validation" / f"{candidate_id}-{spec.run_hash}.json").read_text()
        )["selected_visible_threshold"]
        predictions[candidate_id] = (torch.sigmoid(logits)[0, 0].cpu().numpy() >= threshold)
    instance_overlay = np.zeros((*sample["latent_fault_mask"].shape, 3), dtype=float)
    colors = np.asarray(((0.15, 0.47, 0.71), (0.85, 0.33, 0.10), (0.3, 0.7, 0.3)))
    for instance_index, mask in enumerate(sample["instance_masks"]):
        instance_overlay[np.asarray(mask, dtype=bool)] += colors[instance_index % len(colors)]
    instance_overlay = np.clip(instance_overlay, 0.0, 1.0)
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0), constrained_layout=True)
    axes[0].imshow(np.moveaxis(sample["image"], 0, -1))
    axes[1].imshow(instance_overlay)
    axes[2].imshow(predictions["C0"], cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(predictions["C3"], cmap="gray", vmin=0, vmax=1)
    for axis, title in zip(axes, ("Synthetic input", "Latent instances", "ANZA V1", "ANZA V2 frozen")):
        _hide(axis, title)
    fig.suptitle("Controlled validation example selected by median V2 visible Dice", fontsize=11)
    _save(fig, output_root, 2)
    return {"split": "validation", "sample_index": index, "selection": "median C3 visible Dice within nontrivial_pairing"}


def _load_real_model(spec: Any, setting_a_root: Path, device: torch.device) -> torch.nn.Module:
    model = build_real_model(spec).to(device)
    run_dir = setting_a_root / f"{spec.run_id}-{spec.run_hash}"
    load_real_checkpoint(run_dir / "checkpoint-last.pt", spec.run_hash, model)
    model.eval()
    return model


def _real_figures(study_root: Path, output_root: Path, device: torch.device) -> dict[str, Any]:
    setting_a_root = study_root / "cracks" / "setting_a"
    expert_root = study_root / "cracks" / "setting_a_expert"
    verify_setting_a_complete(setting_a_root, expert_root)
    specs = {
        spec.run_id: spec for spec in setting_a_matrix()
        if spec.comparison_family == "main" and spec.seed == 42
    }
    v2_spec = specs["v2_s42"]
    v2_rows = expert_root / f"{v2_spec.run_id}-{v2_spec.run_hash}.csv"
    selected = _median_row(v2_rows, "dice", predicate=lambda row: row["policy"] == "paper_like")
    section_id = int(selected["section_id"])
    image = load_section_image(PROJECT_ROOT / "data" / "cracks" / "images" / f"section_{section_id:03d}.png")
    expert, valid, _ = map_mask_rgb(
        load_rgb_mask(PROJECT_ROOT / "data" / "cracks" / "annotations" / "expert" / f"section_{section_id:03d}.png"),
        "paper_like",
    )
    normalized = _normalized_image(section_id)
    predictions = {}
    for run_id in ("unet_s42", "v1_s42", "v2_s42"):
        spec = specs[run_id]
        model = _load_real_model(spec, setting_a_root, device)
        threshold = json.loads((setting_a_root / f"{spec.run_id}-{spec.run_hash}" / "crowd_validation.json").read_text())["selected_threshold"]
        predictions[run_id] = tiled_probability(model, normalized).numpy()[:255, :701] >= threshold
    fig, axes = plt.subplots(5, 1, figsize=(12.0, 8.5), constrained_layout=True)
    axes[0].imshow(image)
    axes[1].imshow(np.ma.masked_where(~valid, expert), cmap="gray", vmin=0, vmax=1)
    for axis, run_id in zip(axes[2:], ("unet_s42", "v1_s42", "v2_s42")):
        axis.imshow(predictions[run_id], cmap="gray", vmin=0, vmax=1)
    for axis, title in zip(axes, ("Seismic section", "Available expert annotation", "U-Net", "ANZA V1", "ANZA V2")):
        _hide(axis, title)
    fig.suptitle("Deterministic median V2 Setting A section", fontsize=11)
    _save(fig, output_root, 3)

    v2_model = _load_real_model(v2_spec, setting_a_root, device)
    maps = tiled_v2_uncertainty(v2_model, normalized)
    cropped = {key: value[:255, :701] for key, value in maps.items()}
    threshold = json.loads((setting_a_root / f"{v2_spec.run_id}-{v2_spec.run_hash}" / "crowd_validation.json").read_text())["selected_threshold"]
    skeleton = skeletonize_mask(cropped["probability"] >= threshold)
    fig, axes = plt.subplots(4, 1, figsize=(12.0, 7.3), constrained_layout=True)
    displays = (
        (cropped["orientation"], "twilight", "Axial orientation modes"),
        (cropped["junction_score"], "magma", "Junction score"),
        (cropped["routing_entropy"], "viridis", "Incoming routing entropy"),
        (skeleton, "gray", "Extracted candidate trace skeleton"),
    )
    for axis, (data, cmap, title) in zip(axes, displays):
        image_artist = axis.imshow(data, cmap=cmap, aspect="auto")
        _hide(axis, title)
        if title != "Extracted candidate trace skeleton":
            fig.colorbar(image_artist, ax=axis, fraction=0.018, pad=0.01)
    _save(fig, output_root, 4)

    section_rows_path = study_root / "cracks" / "disagreement" / "section_metrics.csv"
    with section_rows_path.open(newline="") as handle:
        disagreement_rows = list(csv.DictReader(handle))
    entropy_values = np.asarray([float(row["mean_human_entropy"]) for row in disagreement_rows])
    target_entropy = float(np.median(entropy_values))
    disagreement_row = min(disagreement_rows, key=lambda row: (abs(float(row["mean_human_entropy"]) - target_entropy), int(row["section_id"])))
    disagreement_id = int(disagreement_row["section_id"])
    disagreement_image = _normalized_image(disagreement_id)
    disagreement_maps = tiled_v2_uncertainty(v2_model, disagreement_image)
    name = f"section_{disagreement_id:03d}.png"
    annotation_root = PROJECT_ROOT / "data" / "cracks" / "annotations"
    annotators = sorted(path.name for path in annotation_root.iterdir() if path.is_dir() and path.name != "expert" and (path / name).exists())
    crowd = fuse_crowd_masks([load_rgb_mask(annotation_root / item / name) for item in annotators], annotators, "paper_like")
    expert_target, expert_valid, _ = map_mask_rgb(load_rgb_mask(annotation_root / "expert" / name), "paper_like")
    probability = disagreement_maps["probability"][:255, :701]
    model_error = (probability >= threshold) != (expert_target >= 0.5)
    fig, axes = plt.subplots(4, 1, figsize=(12.0, 7.3), constrained_layout=True)
    panels = (
        (np.ma.masked_where(~expert_valid, expert_target), "gray", "Available expert annotation"),
        (crowd["human_entropy"], "magma", "Crowd disagreement entropy"),
        (disagreement_maps["routing_entropy"][:255, :701], "viridis", "V2 incoming routing entropy"),
        (model_error, "Reds", "V2 segmentation error"),
    )
    for axis, (data, cmap, title) in zip(axes, panels):
        artist = axis.imshow(data, cmap=cmap, aspect="auto")
        _hide(axis, title)
        if title not in {"Available expert annotation", "V2 segmentation error"}:
            fig.colorbar(artist, ax=axis, fraction=0.018, pad=0.01)
    summary = json.loads((study_root / "cracks" / "disagreement" / "summary.json").read_text())
    established = any(
        values.get("status") == "COMPLETE" and values.get("ci95_low") is not None
        and (values["ci95_low"] > 0 or values["ci95_high"] < 0)
        for values in summary["correlations"].values()
    )
    fig.suptitle(
        "Section-level uncertainty comparison" + ("" if established else " (no robust association established)"),
        fontsize=11,
    )
    _save(fig, output_root, 5)
    return {
        "real_median_section_id": section_id,
        "real_selection": "closest to median v2_s42 Setting A paper_like Dice",
        "disagreement_median_section_id": disagreement_id,
        "meaningful_disagreement_association": established,
    }


def generate_figures(study_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    study_root = Path(study_root)
    output_root = study_root / "cracks" / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        expected = [output_root / f"figure_{index}.{suffix}" for index in range(1, 6) for suffix in ("png", "svg", "pdf")]
        if existing.get("status") == "COMPLETE" and all(path.exists() and path.stat().st_size for path in expected):
            return {**existing, "action": "SKIP"}
    torch_device = torch.device(device)
    _operator_figure(output_root)
    synthetic = _synthetic_figure(study_root, output_root, torch_device)
    real = _real_figures(study_root, output_root, torch_device)
    payload = {
        "status": "COMPLETE",
        "action": "RUN",
        "formats": ["png_300dpi", "svg", "pdf"],
        "background": "white",
        "synthetic": synthetic,
        **real,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
