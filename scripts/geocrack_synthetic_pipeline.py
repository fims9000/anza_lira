#!/usr/bin/env python3
"""Run the complete GeoCrack software stack on generated, non-scientific data."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import csv
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch

import train
import utils
from datasets.geocrack import GeoCrackDataset, compute_train_normalization, discover_pairs, sha256_file
from scripts.check_geocrack_split import assert_no_source_leakage, freeze_or_verify_test_split
from scripts.geocrack_study import (
    _native_geometry,
    _pixel_metrics,
    cluster_bootstrap_delta,
    now_iso,
    write_json,
    write_report_provenance,
)
from tests.fixtures.geocrack_synthetic import generate_synthetic_geocrack
from trace_extraction.export import traces_to_geojson, write_geojson
from trace_extraction.geometry import local_pca_orientation
from trace_extraction.graph import extract_trace_graph
from trace_extraction.metrics import compute_trace_metrics
from trace_extraction.skeleton import skeletonize_mask


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "geocrack_study" / "prepared" / "synthetic_pipeline"
CSV_FIELDS = ("patch_id", "source_image_id", "image_path", "mask_path")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _prepare_fixture(root: Path, split_dir: Path) -> dict[str, Any]:
    fixture = generate_synthetic_geocrack(root, variants_per_case=3)
    rows = discover_pairs(root)
    splits = {
        "train": [row for row in rows if row["source_image_id"].endswith("_00")],
        "val": [row for row in rows if row["source_image_id"].endswith("_01")],
        "test": [row for row in rows if row["source_image_id"].endswith("_02")],
    }
    sources = {name: {row["source_image_id"] for row in part} for name, part in splits.items()}
    assert_no_source_leakage(sources["train"], sources["val"], sources["test"])
    for name, part in splits.items():
        _write_csv(split_dir / f"geocrack_small_v1_{name}.csv", part)
    test_hash = freeze_or_verify_test_split(
        split_dir / "geocrack_small_v1_test.csv", split_dir / "test_split.sha256"
    )
    normalization = compute_train_normalization(
        root,
        split_dir / "geocrack_small_v1_train.csv",
        split_dir / "train_normalization.json",
    )
    manifest = {
        "scientific_result": False,
        "fixture": fixture,
        "split_counts": {name: len(part) for name, part in splits.items()},
        "source_leakage": 0,
        "frozen_test_csv_sha256": test_hash,
        "normalization": normalization,
    }
    write_json(split_dir / "synthetic_split_manifest.json", manifest)
    return manifest


def _config(data_root: Path, split_dir: Path, output_root: Path, variant: str) -> dict[str, Any]:
    return {
        "dataset": "geocrack",
        "task": "segmentation",
        "data_root": str(data_root.parent),
        "geocrack_dirname": data_root.name,
        "geocrack_split_dir": str(split_dir),
        "geocrack_normalization": str(split_dir / "train_normalization.json"),
        "geocrack_augment": False,
        "geocrack_brightness_jitter": 0.0,
        "geocrack_contrast_jitter": 0.0,
        "image_size": 224,
        "variant": variant,
        "num_rules": 2,
        "model_widths": [4, 8, 12, 16],
        "epochs": 1,
        "batch_size": 3,
        "lr": 0.001,
        "weight_decay": 0.0,
        "seed": 42,
        "deterministic": True,
        "num_workers": 0,
        "bce_weight": 1.0,
        "dice_weight": 1.0,
        "overlap_mode": "dice",
        "seg_threshold": 0.5,
        "eval_threshold_sweep": True,
        "eval_threshold_metric": "dice",
        "eval_threshold_start": 0.3,
        "eval_threshold_end": 0.7,
        "eval_threshold_step": 0.2,
        "aux_loss_weight": 0.2,
        "boundary_loss_weight": 0.0,
        "topology_loss_weight": 0.0,
        "topology_num_iters": 2,
        "bce_pos_weight": "auto",
        "bce_pos_weight_min": 1.0,
        "bce_pos_weight_max": 25.0,
        "reg_membership_entropy": 0.0005,
        "reg_membership_smoothness": 0.0005,
        "reg_geometry_smoothness": 0.0005,
        "reg_hyperbolicity": 0.001,
        "reg_anisotropy_gap": 0.0005,
        "results_dir": str(output_root / "runs"),
        "run_name": f"{variant}_seed42",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "timing_warmup": 0,
        "timing_iters": 1,
        "synthetic_only": True,
    }


def _cldice(predicted: np.ndarray, target: np.ndarray) -> float:
    pred_skeleton = skeletonize_mask(predicted)
    target_skeleton = skeletonize_mask(target)
    precision = float(target[pred_skeleton].mean()) if pred_skeleton.any() else (1.0 if not target.any() else 0.0)
    recall = float(predicted[target_skeleton].mean()) if target_skeleton.any() else (1.0 if not predicted.any() else 0.0)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _load_trained_model(run_dir: Path) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = train.load_checkpoint_payload(run_dir / "checkpoint_best.pt")
    cfg = dict(checkpoint["cfg"])
    variant = str(checkpoint["variant"])
    model = utils.build_model(
        variant,
        num_outputs=1,
        in_channels=3,
        num_rules=int(cfg["num_rules"]),
        task="segmentation",
        widths=utils.parse_model_widths(cfg["model_widths"]),
        model_kwargs=utils.resolve_segmentation_model_kwargs(cfg),
        az_cfg_kwargs=utils.resolve_azconv_config_kwargs(cfg),
    ).to(torch.device(cfg["device"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg, json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def _evaluate_variant(run_dir: Path, output_root: Path) -> list[dict[str, Any]]:
    model, cfg, metrics = _load_trained_model(run_dir)
    device = torch.device(cfg["device"])
    dataset = GeoCrackDataset(
        Path(cfg["data_root"]) / cfg["geocrack_dirname"],
        Path(cfg["geocrack_split_dir"]) / "geocrack_small_v1_test.csv",
        normalization_path=cfg["geocrack_normalization"],
        augment=False,
    )
    threshold = float(metrics["selected_threshold"])
    normalization = json.loads(Path(cfg["geocrack_normalization"]).read_text(encoding="utf-8"))
    mean = np.asarray(normalization["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(normalization["std"], dtype=np.float32)[:, None, None]
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            image, target_tensor, sample = dataset[index]
            logits, _, _ = utils.unpack_segmentation_outputs(model(image.unsqueeze(0).to(device)))
            probability = torch.sigmoid(logits[0, 0]).cpu().numpy()
            predicted = probability >= threshold
            target = target_tensor[0].numpy().astype(bool)
            pred_skeleton = skeletonize_mask(predicted)
            target_skeleton = skeletonize_mask(target)
            native = _native_geometry(model, predicted.shape)
            if native is None or any(np.asarray(item).ndim != 2 for item in native):
                orientation = local_pca_orientation(pred_skeleton)
                coherence = np.ones_like(probability)
                anisotropy = np.zeros_like(probability)
                geometry_source = "skeleton_pca"
            else:
                orientation, coherence, anisotropy = native
                geometry_source = "model_native"
            for value in (probability, coherence, anisotropy):
                if not np.isfinite(value).all() or np.min(value) < 0 or np.max(value) > 1:
                    raise ValueError("Synthetic inference produced invalid probability/coherence/anisotropy range")
            graph = extract_trace_graph(pred_skeleton, border_margin=5)
            row = {
                "model": cfg["variant"],
                "seed": cfg["seed"],
                "source_image_id": sample["source_image_id"],
                "patch_id": sample["patch_id"],
                "threshold": threshold,
                "geometry_source": geometry_source,
                **_pixel_metrics(predicted, target),
                "cldice": _cldice(predicted, target),
                **compute_trace_metrics(
                    pred_skeleton,
                    target_skeleton,
                    pred_orientation=orientation,
                    border_margin=5,
                ),
            }
            rows.append(row)
            geojson = traces_to_geojson(
                graph.segments,
                source_image_id=sample["source_image_id"],
                patch_id=sample["patch_id"],
                model=cfg["variant"],
                seed=cfg["seed"],
                probability=probability,
                coherence=coherence,
                anisotropy=anisotropy,
            )
            write_geojson(output_root / "traces" / cfg["variant"] / f"{sample['patch_id']}.geojson", geojson)
            (output_root / "artifacts" / cfg["variant"]).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_root / "artifacts" / cfg["variant"] / f"{sample['patch_id']}.npz",
                input=np.clip(image.numpy() * std + mean, 0.0, 1.0).transpose(1, 2, 0),
                target=target,
                predicted=predicted,
                probability=probability,
                orientation=orientation,
                coherence=coherence,
                anisotropy=anisotropy,
                pred_skeleton=pred_skeleton,
            )
    fields = list(rows[0])
    path = run_dir / "per_patch_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        run_dir / "evaluation_summary.json",
        {
            **{
                field: float(np.mean([float(row[field]) for row in rows]))
                for field in fields
                if field not in {"model", "source_image_id", "patch_id", "geometry_source"}
            },
            "patch_count": len(rows),
            "checkpoint_reloaded": True,
        },
    )
    return rows


def _statistics(output_root: Path, rows_by_model: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    table_dir = output_root / "tables"
    summary_rows = []
    for model, rows in rows_by_model.items():
        summary_rows.append(
            {
                "model": model,
                **{
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in ("dice", "iou", "precision", "recall", "cldice", "trace_f1", "endpoint_f1")
                },
            }
        )
    summary_path = table_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    baseline = {row["source_image_id"]: float(row["dice"]) for row in rows_by_model["baseline"]}
    az = {row["source_image_id"]: float(row["dice"]) for row in rows_by_model["az_thesis"]}
    bootstrap = cluster_bootstrap_delta(baseline, az, replicates=300, seed=2026)
    bootstrap_path = table_dir / "cluster_bootstrap.csv"
    with bootstrap_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bootstrap), lineterminator="\n")
        writer.writeheader()
        writer.writerow(bootstrap)
    return {"summary_path": summary_path, "bootstrap_path": bootstrap_path, "bootstrap": bootstrap}


def _figures(output_root: Path, rows_by_model: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    from scripts.geocrack_study import _save_figure

    def assert_no_text_clipping(fig) -> None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        canvas = fig.bbox
        for artist in fig.findobj(match=Text):
            if not artist.get_visible() or not artist.get_text().strip():
                continue
            if artist.axes is not None and not artist.axes.axison:
                continue
            try:
                float(artist.get_text().replace("−", "-"))
            except ValueError:
                pass
            else:
                continue
            bounds = artist.get_window_extent(renderer=renderer)
            if bounds.x0 < canvas.x0 - 1 or bounds.y0 < canvas.y0 - 1 or bounds.x1 > canvas.x1 + 1 or bounds.y1 > canvas.y1 + 1:
                raise ValueError(f"Synthetic figure text is clipped: {artist.get_text()!r}")

    figures = output_root / "figures"
    patch_id = next(row["patch_id"] for row in rows_by_model["baseline"] if row["patch_id"])
    baseline = np.load(output_root / "artifacts" / "baseline" / f"{patch_id}.npz")
    az = np.load(output_root / "artifacts" / "az_thesis" / f"{patch_id}.npz")
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), constrained_layout=True)
    for axis, image, title, cmap in zip(
        axes,
        (baseline["input"], baseline["target"], baseline["predicted"], az["predicted"]),
        ("Synthetic input", "Ground truth", "Baseline", "ANZA-LIRA"),
        (None, "gray", "gray", "gray"),
    ):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    assert_no_text_clipping(fig)
    _save_figure(fig, figures / "synthetic_pipeline_overview")
    plt.close(fig)

    summary = _read_csv(output_root / "tables" / "summary.csv")
    fig, axis = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    positions = np.arange(3)
    width = 0.35
    for index, row in enumerate(summary):
        axis.bar(positions + index * width, [float(row[key]) for key in ("dice", "cldice", "trace_f1")], width, label=row["model"])
    axis.set_xticks(positions + width / 2, ["Dice", "clDice", "trace F1"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Synthetic integration metric")
    legend = axis.legend(loc="upper right")
    if legend is None:
        raise ValueError("Synthetic figure legend is missing")
    if legend._loc != 1:  # Matplotlib's stable code for upper-right.
        raise ValueError("Synthetic figure legend moved from its declared position")
    assert_no_text_clipping(fig)
    _save_figure(fig, figures / "synthetic_metric_comparison")
    plt.close(fig)
    stems = ["synthetic_pipeline_overview", "synthetic_metric_comparison"]
    for stem in stems:
        for suffix in ("png", "svg", "pdf"):
            path = figures / f"{stem}.{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Synthetic figure export missing: {path}")
        with Image.open(figures / f"{stem}.png") as image:
            dpi = image.info.get("dpi", (0, 0))
            if min(dpi) < 299:
                raise ValueError(f"Synthetic PNG DPI below 300: {dpi}")
    overview_svg = (figures / "synthetic_pipeline_overview.svg").read_text(encoding="utf-8")
    comparison_svg = (figures / "synthetic_metric_comparison.svg").read_text(encoding="utf-8")
    if not all(label in overview_svg for label in ("Synthetic input", "Ground truth", "Baseline", "ANZA-LIRA")):
        raise ValueError("Synthetic overview labels are missing from SVG")
    if not all(label in comparison_svg for label in ("Dice", "clDice", "trace F1", "baseline", "az_thesis")):
        raise ValueError("Synthetic comparison labels/legend are missing from SVG")
    return stems


def _report(output_root: Path, stats: Mapping[str, Any], figure_stems: list[str]) -> None:
    summary = _read_csv(Path(stats["summary_path"]))
    bootstrap = _read_csv(Path(stats["bootstrap_path"]))[0]
    thesis = {
        "scientific_result": False,
        "fixture_only": True,
        "warning": "SYNTHETIC INTEGRATION VALUES MUST NEVER BE USED IN THE ARTICLE",
        "training": {"epochs": 1, "seed": 42, "models": [row["model"] for row in summary]},
        "summary": {row["model"]: row for row in summary},
        "cluster_bootstrap": bootstrap,
        "machine_sources": {
            "summary_csv_sha256": sha256_file(Path(stats["summary_path"])),
            "bootstrap_csv_sha256": sha256_file(Path(stats["bootstrap_path"])),
        },
        "figures": figure_stems,
    }
    thesis_path = output_root / "THESIS_NUMBERS.json"
    write_json(thesis_path, thesis)
    baseline, az = thesis["summary"]["baseline"], thesis["summary"]["az_thesis"]
    report = f"""# Synthetic GeoCrack pipeline report

This is a software integration fixture, not a scientific GeoCrack result.

The baseline and ANZA-LIRA models each trained for {thesis['training']['epochs']} epoch with seed {thesis['training']['seed']}. Checkpoints were saved and reloaded before inference.

Baseline Dice: {float(baseline['dice']):.4f}. ANZA-LIRA Dice: {float(az['dice']):.4f}.

Baseline clDice: {float(baseline['cldice']):.4f}. ANZA-LIRA clDice: {float(az['cldice']):.4f}.

Cluster bootstrap source count: {bootstrap['source_count']}; replicates: {bootstrap['replicates']}.

All values above were rendered from `THESIS_NUMBERS.json`, which was generated only from the machine CSV tables.
"""
    report_path = output_root / "FINAL_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    write_report_provenance(thesis_path, report_path, output_root / "REPORT_PROVENANCE.json")


def run_synthetic_pipeline(output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="geocrack-synthetic-") as temp:
        data_root = Path(temp) / "geocrack_synthetic"
        split_dir = Path(temp) / "splits"
        fixture = _prepare_fixture(data_root, split_dir)
        rows_by_model: dict[str, list[dict[str, Any]]] = {}
        for variant in ("baseline", "az_thesis"):
            run_dir = output_root / "runs" / f"{variant}_seed42"
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = _config(data_root, split_dir, output_root, variant)
            with (run_dir / "train.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                metrics = train.run_training(cfg, variant, run_dir)
            write_json(
                run_dir / "run_metadata.json",
                {
                    "status": "COMPLETE",
                    "model": variant,
                    "seed": 42,
                    "scientific_result": False,
                    "selected_threshold": metrics["selected_threshold"],
                    "checkpoint_best_sha256": sha256_file(run_dir / "checkpoint_best.pt"),
                    "checkpoint_last_sha256": sha256_file(run_dir / "checkpoint_last.pt"),
                },
            )
            rows_by_model[variant] = _evaluate_variant(run_dir, output_root)
        stats = _statistics(output_root, rows_by_model)
        figures = _figures(output_root, rows_by_model)
        _report(output_root, stats, figures)
    required = [
        output_root / "runs" / model / filename
        for model in ("baseline_seed42", "az_thesis_seed42")
        for filename in ("checkpoint_best.pt", "checkpoint_last.pt", "metrics.json", "evaluation_summary.json")
    ]
    if not all(path.is_file() for path in required):
        raise ValueError("Synthetic pipeline did not produce the complete checkpoint/evaluation contract")
    status = {
        "status": "PASS",
        "scientific_result": False,
        "completed_at": now_iso(),
        "fixture_sample_count": fixture["fixture"]["sample_count"],
        "steps": {
            name: "PASS"
            for name in (
                "dataset",
                "split",
                "normalization",
                "training",
                "validation_threshold",
                "checkpoint_save_reload",
                "test_inference",
                "orientation_anisotropy",
                "skeleton_graph_traces",
                "trace_metrics",
                "geojson",
                "cluster_bootstrap",
                "tables",
                "figures_png_svg_pdf",
                "thesis_numbers",
                "final_report",
                "report_consistency",
            )
        },
    }
    write_json(output_root / "synthetic_pipeline_status.json", status)
    print("SYNTHETIC PIPELINE: PASS")
    print("SCIENTIFIC RESULT: FALSE")
    return status


def main() -> int:
    run_synthetic_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
