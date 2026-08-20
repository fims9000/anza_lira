#!/usr/bin/env python3
"""Modular, resumable automation for the ANZA-LIRA GeoCrack study."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train
import utils
from datasets.geocrack import GeoCrackDataset, extract_source_image_id, sha256_file
from models.azconv import AZConv2d
from scripts.check_geocrack_split import assert_no_source_leakage, freeze_or_verify_test_split, load_sources
from trace_extraction.export import traces_to_geojson, write_geojson
from trace_extraction.geometry import geometry_from_interpretation, local_pca_orientation
from trace_extraction.graph import extract_trace_graph
from trace_extraction.metrics import compute_trace_metrics
from trace_extraction.skeleton import skeletonize_mask


STUDY_ROOT = PROJECT_ROOT / "results" / "geocrack_study"
SPLIT_DIR = PROJECT_ROOT / "data" / "geocrack" / "splits"
RUN_MATRIX = (
    ("baseline", 41),
    ("baseline", 42),
    ("baseline", 43),
    ("az_thesis", 41),
    ("az_thesis", 42),
    ("az_thesis", 43),
    ("az_no_fuzzy", 42),
    ("az_no_aniso", 42),
    ("attention_unet", 42),
)
TRACE_METRICS = (
    "trace_f1",
    "endpoint_f1",
    "junction_f1",
    "symmetric_skeleton_distance",
    "orientation_error_mean_deg",
    "trace_length_error",
)
BOOTSTRAP_METRICS = ("dice", "iou", "precision", "recall", "cldice", "trace_f1", "endpoint_f1", "junction_f1")
PROTOCOL_KEYS = (
    "dataset",
    "task",
    "geocrack_split_dir",
    "geocrack_normalization",
    "geocrack_augment",
    "geocrack_brightness_jitter",
    "geocrack_contrast_jitter",
    "image_size",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "deterministic",
    "num_workers",
    "bce_weight",
    "dice_weight",
    "overlap_mode",
    "aux_loss_weight",
    "boundary_loss_weight",
    "topology_loss_weight",
    "eval_threshold_sweep",
    "eval_threshold_metric",
    "eval_threshold_start",
    "eval_threshold_end",
    "eval_threshold_step",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout.strip()


def command_result(command: Sequence[str]) -> tuple[int, str]:
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.returncode, result.stdout.strip()


def git_commit() -> str:
    return command_output(["git", "rev-parse", "HEAD"])


def split_bundle_hash(split_dir: Path = SPLIT_DIR) -> str:
    paths = [split_dir / f"geocrack_small_v1_{split}.csv" for split in ("train", "val", "test")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen GeoCrack split files: {missing}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def check_split(split_dir: Path = SPLIT_DIR) -> str:
    train_csv = split_dir / "geocrack_small_v1_train.csv"
    val_csv = split_dir / "geocrack_small_v1_val.csv"
    test_csv = split_dir / "geocrack_small_v1_test.csv"
    assert_no_source_leakage(load_sources(train_csv), load_sources(val_csv), load_sources(test_csv))
    manifest = json.loads((split_dir / "geocrack_small_v1_manifest.json").read_text(encoding="utf-8"))
    current_test_hash = sha256_file(test_csv)
    if manifest.get("frozen_test_csv_sha256") != current_test_hash:
        raise ValueError("Frozen GeoCrack test CSV hash changed")
    contract_hash = freeze_or_verify_test_split(test_csv, split_dir / "test_split.sha256")
    if contract_hash != current_test_hash:
        raise ValueError("Standalone GeoCrack test split contract disagrees with the manifest")
    return split_bundle_hash(split_dir)


def resolve_geocrack_data_root() -> Path:
    manifest_path = PROJECT_ROOT / "data" / "geocrack" / "manual_import_manifest.json"
    if manifest_path.is_file():
        root = Path(_json_file(manifest_path).get("data_root", ""))
        if root.is_dir():
            return root.resolve()
    return (PROJECT_ROOT / "data" / "geocrack").resolve()


def ensure_real_split() -> None:
    required = [SPLIT_DIR / f"geocrack_small_v1_{name}.csv" for name in ("train", "val", "test")]
    if not all(path.is_file() for path in required):
        from scripts.prepare_geocrack_split import prepare

        prepare(resolve_geocrack_data_root(), SPLIT_DIR)
    check_split()


def capture_environment(output_root: Path = STUDY_ROOT) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        torchvision_version = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        torchvision_version = "MISSING"
    cuda_available = torch.cuda.is_available()
    torchvision_import_code, torchvision_import_output = command_result(
        [sys.executable, "-c", "import torchvision; print(torchvision.__version__)"]
    )
    lines = [
        f"captured_at: {now_iso()}",
        f"python: {sys.version.replace(os.linesep, ' ')}",
        f"executable: {sys.executable}",
        f"torch: {torch.__version__}",
        f"torchvision_distribution: {torchvision_version}",
        f"torchvision_import_status: {'PASS' if torchvision_import_code == 0 else 'FAIL'}",
        f"torchvision_import_output: {torchvision_import_output}",
        f"cuda_available: {cuda_available}",
        f"torch_cuda: {torch.version.cuda}",
        f"gpu: {torch.cuda.get_device_name(0) if cuda_available else 'NONE'}",
        f"gpu_total_memory: {torch.cuda.get_device_properties(0).total_memory if cuda_available else 0}",
        f"os: {platform.platform()}",
        "",
        "pip freeze:",
        command_output([sys.executable, "-m", "pip", "freeze"]),
    ]
    (output_root / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    reproducibility = {
        "captured_at": now_iso(),
        "repo_commit": git_commit(),
        "branch": command_output(["git", "branch", "--show-current"]),
        "remote": command_output(["git", "remote", "get-url", "origin"]),
        "python_executable": sys.executable,
        "rtk_version": command_output(["rtk", "--version"]) if shutil_which("rtk") else "MISSING",
    }
    write_json(output_root / "reproducibility.json", reproducibility)
    if shutil_which("rtk"):
        (output_root / "rtk_before.txt").write_text(command_output(["rtk", "gain"]) + "\n", encoding="utf-8")
    print("ENVIRONMENT: PASS")


def shutil_which(command: str) -> str | None:
    from shutil import which

    return which(command)


def _run_config_hash(cfg: Mapping[str, Any], variant: str, seed: int, split_hash: str) -> str:
    normalized = {key: value for key, value in cfg.items() if key not in {"run_name", "resume_checkpoint", "_resolved_test_split"}}
    return stable_hash({"config": normalized, "variant": variant, "seed": seed, "split_hash": split_hash})


def protocol_contract(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Fields that must be identical across every architecture comparison."""
    missing = [key for key in PROTOCOL_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"GeoCrack config lacks shared protocol fields: {missing}")
    return {
        "optimizer": "Adam",
        "loss": {key: cfg[key] for key in PROTOCOL_KEYS if "weight" in key or key == "overlap_mode"},
        "split": {key: cfg[key] for key in ("geocrack_split_dir", "geocrack_normalization")},
        "batch_size": cfg["batch_size"],
        "epoch_budget": cfg["epochs"],
        "threshold_procedure": {
            key: cfg[key] for key in PROTOCOL_KEYS if key.startswith("eval_threshold_")
        },
        "evaluation_code": "scripts/geocrack_study.py:evaluate_run",
        "all_shared_config": {key: cfg[key] for key in PROTOCOL_KEYS},
    }


def dry_run_matrix(config_path: Path) -> dict[str, Any]:
    cfg = utils.load_config(str(config_path))
    protocol = protocol_contract(cfg)
    protocol_hash = stable_hash(protocol)
    payload = {
        "run_count": len(RUN_MATRIX),
        "runs": [{"model": model, "seed": seed} for model, seed in RUN_MATRIX],
        "protocol_hash": protocol_hash,
        "protocol": protocol,
        "fairness": "PASS",
    }
    for model, seed in RUN_MATRIX:
        print(f"{model} seed {seed}")
    print(f"PROTOCOL HASH: {protocol_hash[:16]}")
    print("PROTOCOL FAIRNESS: PASS")
    return payload


def resolve_run_action(
    metadata: Mapping[str, Any] | None,
    *,
    config_hash: str,
    split_hash: str,
    checkpoint_last: bool,
    checkpoint_best: bool = False,
    metrics_present: bool = False,
) -> str:
    """Pure resume decision used by training and synthetic contract tests."""
    if metadata is None:
        return "START"
    same_config = metadata.get("config_hash") == config_hash
    same_split = metadata.get("split_hash") == split_hash
    if metadata.get("status") == "COMPLETE" and same_config and same_split and checkpoint_best and metrics_present:
        return "SKIP"
    if same_config and same_split and checkpoint_last:
        return "RESUME"
    if not same_config or not same_split:
        return "NEW_RUN_ID"
    return "START"


def _preflight_batch_size(cfg: Mapping[str, Any], candidates: Sequence[int]) -> int:
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        return int(candidates[0])
    for batch_size in candidates:
        try:
            model = utils.build_model(
                "az_thesis",
                num_outputs=1,
                in_channels=3,
                num_rules=int(cfg.get("num_rules", 4)),
                task="segmentation",
                widths=utils.parse_model_widths(cfg.get("model_widths")),
                model_kwargs=utils.resolve_segmentation_model_kwargs(dict(cfg)),
                az_cfg_kwargs=utils.resolve_azconv_config_kwargs(dict(cfg)),
            ).to(device)
            sample = torch.zeros((batch_size, 3, 224, 224), device=device)
            logits, _, _ = utils.unpack_segmentation_outputs(model(sample))
            logits.mean().backward()
            del model, sample, logits
            torch.cuda.empty_cache()
            return int(batch_size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    raise RuntimeError("GeoCrack preflight OOM even at batch_size=1")


def run_training_matrix(config_path: Path, *, smoke: bool = False) -> None:
    split_hash = check_split()
    base_cfg = utils.load_config(str(config_path))
    active_root = resolve_geocrack_data_root()
    base_cfg.update({"data_root": str(active_root.parent), "geocrack_dirname": active_root.name})
    requested = int(base_cfg["batch_size"])
    candidates = [value for value in (requested, 4, 2, 1) if value <= requested]
    selected_batch = _preflight_batch_size(base_cfg, tuple(dict.fromkeys(candidates)))
    matrix = (("baseline", 42), ("az_thesis", 42)) if smoke else RUN_MATRIX
    root = STUDY_ROOT / ("smoke" if smoke else "runs")
    root.mkdir(parents=True, exist_ok=True)
    for variant, seed in matrix:
        canonical_name = f"{variant}_seed{seed}"
        cfg = dict(base_cfg)
        cfg.update({"variant": variant, "seed": seed, "batch_size": selected_batch, "run_name": canonical_name})
        cfg_hash = _run_config_hash(cfg, variant, seed, split_hash)
        run_dir = root / canonical_name
        metadata_path = run_dir / "run_metadata.json"
        previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
        action = resolve_run_action(
            previous,
            config_hash=cfg_hash,
            split_hash=split_hash,
            checkpoint_last=(run_dir / "checkpoint_last.pt").is_file(),
            checkpoint_best=(run_dir / "checkpoint_best.pt").is_file(),
            metrics_present=(run_dir / "metrics.json").is_file(),
        )
        if action == "SKIP":
            print(f"SKIP {canonical_name}: COMPLETE with matching config and split hashes")
            continue
        if action == "NEW_RUN_ID":
            run_dir = root / f"{canonical_name}_{cfg_hash[:8]}"
            metadata_path = run_dir / "run_metadata.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg["run_name"] = run_dir.name
        if action == "RESUME":
            cfg["resume_checkpoint"] = str(run_dir / "checkpoint_last.pt")
        metadata = {
            "run": run_dir.name,
            "model": variant,
            "seed": seed,
            "batch_size": selected_batch,
            "config_hash": cfg_hash,
            "commit_hash": git_commit(),
            "split_hash": split_hash,
            "started_at": now_iso(),
            "finished_at": None,
            "status": "RUNNING",
        }
        write_json(metadata_path, metadata)
        print(f"RUN START {run_dir.name}")
        try:
            with (run_dir / "train.log").open("a", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                metrics = train.run_training(cfg, variant, run_dir)
            metadata.update(
                {
                    "finished_at": now_iso(),
                    "status": "COMPLETE",
                    "metrics_sha256": sha256_file(run_dir / "metrics.json"),
                    "checkpoint_sha256": sha256_file(run_dir / "checkpoint_best.pt"),
                }
            )
            write_json(metadata_path, metadata)
            print(f"RUN COMPLETE {run_dir.name} val_best={metrics.get('best_val_dice', metrics.get('best_val_accuracy'))}")
        except Exception as exc:
            metadata.update({"finished_at": now_iso(), "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
            write_json(metadata_path, metadata)
            print(f"RUN FAILED {run_dir.name}: {type(exc).__name__}: {exc}")
            raise


def _load_model(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], str]:
    checkpoint = train.load_checkpoint_payload(run_dir / "checkpoint_best.pt")
    cfg = dict(checkpoint["cfg"])
    variant = str(checkpoint.get("variant", cfg.get("variant")))
    model = utils.build_model(
        variant,
        num_outputs=1,
        in_channels=3,
        num_rules=int(cfg.get("num_rules", 4)),
        task="segmentation",
        widths=utils.parse_model_widths(cfg.get("model_widths")),
        model_kwargs=utils.resolve_segmentation_model_kwargs(cfg),
        az_cfg_kwargs=utils.resolve_azconv_config_kwargs(cfg),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return model, cfg, metrics, variant


def _native_geometry(model: torch.nn.Module, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for _name, module in model.named_modules():
        if not isinstance(module, AZConv2d):
            continue
        snapshot = module.interpretation_snapshot()
        if snapshot and tuple(snapshot["mu_map"].shape[-2:]) == shape and "theta_map" in snapshot:
            geometry = geometry_from_interpretation(snapshot)
            return geometry.orientation, geometry.coherence, geometry.anisotropy
    return None


def _pixel_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    tn = float(np.logical_and(~pred, ~target).sum())
    metrics = utils.segmentation_metrics_from_counts(tp, fp, tn, fn)
    return {key: metrics[key] for key in ("dice", "iou", "precision", "recall", "specificity", "balanced_accuracy")}


def _binary_cldice(predicted: np.ndarray, target: np.ndarray) -> float:
    pred_skeleton = skeletonize_mask(predicted)
    target_skeleton = skeletonize_mask(target)
    precision = float(target[pred_skeleton].mean()) if pred_skeleton.any() else (1.0 if not target.any() else 0.0)
    recall = float(predicted[target_skeleton].mean()) if target_skeleton.any() else (1.0 if not predicted.any() else 0.0)
    return _safe_ratio_f1(precision, recall)


def _safe_ratio_f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_run(run_dir: Path, *, artifact_root: Path = STUDY_ROOT / "artifacts") -> None:
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "COMPLETE":
        raise ValueError(f"Run is not COMPLETE: {run_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, run_metrics, variant = _load_model(run_dir, device)
    split_dir = Path(cfg.get("geocrack_split_dir", SPLIT_DIR))
    dataset = GeoCrackDataset(
        PROJECT_ROOT / cfg.get("data_root", "./data") / cfg.get("geocrack_dirname", "geocrack"),
        split_dir / "geocrack_small_v1_test.csv",
        normalization_path=Path(cfg.get("geocrack_normalization", split_dir / "train_normalization.json")),
        augment=False,
    )
    limit = cfg.get("geocrack_test_limit")
    indices = list(range(min(len(dataset), int(limit)))) if limit else list(range(len(dataset)))
    threshold = float(run_metrics["selected_threshold"])
    normalization = json.loads(Path(cfg.get("geocrack_normalization", split_dir / "train_normalization.json")).read_text(encoding="utf-8"))
    mean = np.asarray(normalization["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(normalization["std"], dtype=np.float32)[:, None, None]
    rows: list[dict[str, Any]] = []
    artifact_dir = artifact_root / run_dir.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index in indices:
            image, target_tensor, sample = dataset[index]
            output = model(image.unsqueeze(0).to(device))
            logits, _, _ = utils.unpack_segmentation_outputs(output)
            probability = torch.sigmoid(logits[0, 0]).cpu().numpy()
            predicted = probability >= threshold
            target = target_tensor[0].numpy().astype(bool)
            pred_skeleton = skeletonize_mask(predicted)
            native = _native_geometry(model, predicted.shape)
            if native is None:
                orientation = local_pca_orientation(pred_skeleton)
                coherence = np.ones(predicted.shape, dtype=np.float64)
                anisotropy = np.zeros(predicted.shape, dtype=np.float64)
                geometry_source = "skeleton_pca_baseline"
            else:
                orientation, coherence, anisotropy = native
                geometry_source = "first_full_resolution_azconv"
            pixel_metrics = _pixel_metrics(predicted, target)
            row = {
                "model": variant,
                "seed": int(cfg["seed"]),
                "source_image_id": sample["source_image_id"],
                "patch_id": sample["patch_id"],
                "threshold": threshold,
                "geometry_source": geometry_source,
                **pixel_metrics,
            }
            rows.append(row)
            input_rgb = np.clip(image.numpy() * std + mean, 0.0, 1.0).transpose(1, 2, 0)
            np.savez_compressed(
                artifact_dir / f"{sample['patch_id']}.npz",
                input=input_rgb,
                target=target,
                predicted=predicted,
                probability=probability,
                pred_skeleton=pred_skeleton,
                orientation=orientation,
                coherence=coherence,
                anisotropy=anisotropy,
            )
    fields = list(rows[0])
    with (run_dir / "per_patch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    numeric_fields = [field for field in fields if field not in {"model", "source_image_id", "patch_id", "geometry_source"}]
    summary = {field: float(np.mean([float(row[field]) for row in rows])) for field in numeric_fields}
    summary.update({"model": variant, "seed": int(cfg["seed"]), "patch_count": len(rows), "generated_at": now_iso()})
    write_json(run_dir / "evaluation_summary.json", summary)
    print(f"EVALUATION COMPLETE {run_dir.name}: {len(rows)} patches")


def _complete_run_dirs(root: Path) -> list[Path]:
    output = []
    for metadata_path in sorted(root.glob("*/run_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") == "COMPLETE":
            output.append(metadata_path.parent)
    return output


def evaluate_all(*, smoke: bool = False) -> None:
    root = STUDY_ROOT / ("smoke" if smoke else "runs")
    runs = _complete_run_dirs(root)
    expected = 2 if smoke else 9
    if len(runs) < expected:
        raise ValueError(f"Expected {expected} complete runs under {root}, found {len(runs)}")
    artifact_root = STUDY_ROOT / ("smoke_artifacts" if smoke else "artifacts")
    for run_dir in runs:
        evaluate_run(run_dir, artifact_root=artifact_root)
    if smoke:
        import matplotlib.pyplot as plt

        baseline_dir = next(path for path in runs if _json_file(path / "run_metadata.json")["model"] == "baseline")
        az_dir = next(path for path in runs if _json_file(path / "run_metadata.json")["model"] == "az_thesis")
        patch_id = _read_csv(baseline_dir / "per_patch_metrics.csv")[0]["patch_id"]
        baseline = np.load(artifact_root / baseline_dir.name / f"{patch_id}.npz")
        az = np.load(artifact_root / az_dir.name / f"{patch_id}.npz")
        fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), constrained_layout=True)
        for axis, image, title, cmap in zip(
            axes,
            (baseline["input"], baseline["target"], baseline["predicted"], az["predicted"]),
            ("Input", "Ground truth", "Baseline", "ANZA-LIRA"),
            (None, "gray", "gray", "gray"),
        ):
            axis.imshow(image, cmap=cmap)
            axis.set_title(title)
            axis.axis("off")
        smoke_figure = STUDY_ROOT / "smoke_vertical_slice.png"
        fig.savefig(smoke_figure, dpi=150, bbox_inches="tight")
        plt.close(fig)
        (STUDY_ROOT / "smoke_vertical_slice.md").write_text(
            "# GeoCrack vertical smoke\n\n"
            "The baseline and az_thesis one-epoch runs completed forward, backward, validation threshold selection, "
            "checkpoint reload, test inference, saved arrays, and figure output. Trace completion is recorded separately.\n",
            encoding="utf-8",
        )
        write_json(
            STUDY_ROOT / "smoke_test_report.json",
            {
                "status": "INFERENCE_PASS",
                "run_count": len(runs),
                "runs": [path.name for path in runs],
                "figure": smoke_figure.as_posix(),
                "report": (STUDY_ROOT / "smoke_vertical_slice.md").as_posix(),
                "completed_at": now_iso(),
            },
        )


def build_traces_for_run(run_dir: Path, *, artifact_root: Path, traces_root: Path) -> None:
    """Rebuild trace objects/metrics from saved inference arrays without a model."""
    metadata = _json_file(run_dir / "run_metadata.json")
    existing_rows = {row["patch_id"]: row for row in _read_csv(run_dir / "per_patch_metrics.csv")}
    artifact_dir = artifact_root / run_dir.name
    artifact_paths = sorted(artifact_dir.glob("*.npz"))
    if not artifact_paths:
        raise FileNotFoundError(f"No saved inference artifacts under {artifact_dir}; run evaluate first")
    rows: list[dict[str, Any]] = []
    output_dir = traces_root / run_dir.name
    for artifact_path in artifact_paths:
        patch_id = artifact_path.stem
        artifact = np.load(artifact_path)
        predicted = artifact["predicted"].astype(bool)
        target = artifact["target"].astype(bool)
        pred_skeleton = artifact["pred_skeleton"].astype(bool)
        target_skeleton = skeletonize_mask(target)
        graph = extract_trace_graph(pred_skeleton, border_margin=5)
        trace_metrics = compute_trace_metrics(
            pred_skeleton,
            target_skeleton,
            pred_orientation=artifact["orientation"],
            border_margin=5,
        )
        source_id = existing_rows.get(patch_id, {}).get("source_image_id") or extract_source_image_id(patch_id)
        base = dict(existing_rows.get(patch_id, {}))
        base.update(
            {
                "model": metadata["model"],
                "seed": int(metadata["seed"]),
                "source_image_id": source_id,
                "patch_id": patch_id,
                **_pixel_metrics(predicted, target),
                "cldice": _binary_cldice(predicted, target),
                **trace_metrics,
            }
        )
        rows.append(base)
        geojson = traces_to_geojson(
            graph.segments,
            source_image_id=source_id,
            patch_id=patch_id,
            model=metadata["model"],
            seed=int(metadata["seed"]),
            probability=artifact["probability"],
            coherence=artifact["coherence"],
            anisotropy=artifact["anisotropy"],
        )
        write_geojson(output_dir / f"{patch_id}.geojson", geojson)
    fields = list(rows[0])
    _write_csv(run_dir / "per_patch_metrics.csv", rows, fields)
    numeric_fields = [
        field
        for field in fields
        if field not in {"model", "source_image_id", "patch_id", "geometry_source"}
        and all(_is_float(row.get(field)) for row in rows)
    ]
    summary = {field: float(np.mean([float(row[field]) for row in rows])) for field in numeric_fields}
    summary.update(
        {"model": metadata["model"], "seed": int(metadata["seed"]), "patch_count": len(rows), "generated_at": now_iso()}
    )
    write_json(run_dir / "evaluation_summary.json", summary)
    write_json(
        run_dir / "traces_complete.json",
        {"status": "PASS", "artifact_count": len(rows), "model_loaded": False, "generated_at": now_iso()},
    )
    print(f"TRACES COMPLETE {run_dir.name}: {len(rows)} patches")


def _is_float(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def build_all_traces(*, smoke: bool = False) -> None:
    run_root = STUDY_ROOT / ("smoke" if smoke else "runs")
    artifact_root = STUDY_ROOT / ("smoke_artifacts" if smoke else "artifacts")
    traces_root = STUDY_ROOT / ("smoke_traces" if smoke else "traces")
    runs = _complete_run_dirs(run_root)
    expected = 2 if smoke else 9
    if len(runs) < expected:
        raise ValueError(f"Expected {expected} complete runs under {run_root}, found {len(runs)}")
    for run_dir in runs:
        build_traces_for_run(run_dir, artifact_root=artifact_root, traces_root=traces_root)
    if smoke:
        report = STUDY_ROOT / "smoke_test_report.json"
        previous = _json_file(report) if report.is_file() else {}
        write_json(
            report,
            {
                **previous,
                "status": "PASS",
                "trace_stage_independent": True,
                "trace_runs": len(runs),
                "completed_at": now_iso(),
            },
        )


def run_full_pipeline(config: Path, smoke_config: Path) -> None:
    """Linux/Windows common business logic; downstream stages never train."""
    capture_environment()
    ensure_real_split()
    test = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT)
    if test.returncode:
        raise RuntimeError(f"pytest failed with exit code {test.returncode}")
    run_training_matrix(smoke_config, smoke=True)
    evaluate_all(smoke=True)
    build_all_traces(smoke=True)
    run_training_matrix(config)
    evaluate_all()
    build_all_traces()
    build_statistics()
    build_figures()
    build_report()
    validation = subprocess.run(
        [sys.executable, "scripts/validate_geocrack_study.py", "--phase", "final"], cwd=PROJECT_ROOT
    )
    if validation.returncode:
        raise RuntimeError(f"final validator failed with exit code {validation.returncode}")


def _json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cluster_bootstrap_delta(
    baseline_by_source: Mapping[str, float],
    az_by_source: Mapping[str, float],
    *,
    replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, float | int]:
    sources = sorted(set(baseline_by_source) & set(az_by_source))
    if not sources:
        raise ValueError("No paired source_image_id values for bootstrap")
    deltas = np.asarray([az_by_source[source] - baseline_by_source[source] for source in sources], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(deltas, size=(replicates, len(deltas)), replace=True).mean(axis=1)
    return {
        "source_count": len(sources),
        "replicates": replicates,
        "mean_delta_az_minus_baseline": float(deltas.mean()),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }


def cluster_bootstrap_from_patch_frame(
    frame: Any,
    *,
    metric: str,
    baseline_model: str = "baseline",
    az_model: str = "az_thesis",
    replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Aggregate patches, then seeds, before resampling source-image clusters."""
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    required = {"model", "seed", "source_image_id", metric}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Patch metric frame lacks columns: {sorted(missing)}")
    source_seed = (
        frame.groupby(["model", "seed", "source_image_id"], as_index=False, sort=True)[metric]
        .mean()
    )
    source_model = (
        source_seed.groupby(["model", "source_image_id"], as_index=False, sort=True)[metric]
        .mean()
    )
    baseline = {
        str(row.source_image_id): float(getattr(row, metric))
        for row in source_model[source_model["model"] == baseline_model].itertuples(index=False)
    }
    az = {
        str(row.source_image_id): float(getattr(row, metric))
        for row in source_model[source_model["model"] == az_model].itertuples(index=False)
    }
    result = cluster_bootstrap_delta(baseline, az, replicates=replicates, seed=seed)
    return {
        **result,
        "resampling_unit": "source_image_id",
        "patch_row_count": int(len(frame)),
        "source_seed_row_count": int(len(source_seed)),
    }


def build_statistics() -> None:
    run_dirs = _complete_run_dirs(STUDY_ROOT / "runs")
    if len(run_dirs) < 9:
        raise ValueError(f"Statistics require 9 complete runs, found {len(run_dirs)}")
    seed_rows = []
    patch_rows: list[dict[str, str]] = []
    for run_dir in run_dirs:
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        evaluation = json.loads((run_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
        per_patch = _read_csv(run_dir / "per_patch_metrics.csv")
        patch_rows.extend(per_patch)
        seed_rows.append(
            {
                "model": metadata["model"],
                "seed": metadata["seed"],
                "dice": metrics["test_dice"],
                "iou": metrics["test_iou"],
                "precision": metrics["test_precision"],
                "recall": metrics["test_recall"],
                "cldice": metrics["test_cldice"],
                "skeleton_precision": metrics["test_skeleton_precision"],
                "skeleton_recall": metrics["test_skeleton_recall"],
                "trace_f1": evaluation["trace_f1"],
                "endpoint_f1": evaluation["endpoint_f1"],
                "junction_f1": evaluation["junction_f1"],
                "orientation_error_deg": evaluation["orientation_error_mean_deg"],
                "trace_length_error": evaluation["trace_length_error"],
                "params": metrics["num_parameters"],
                "inference_ms": 1000.0 * metrics["seconds_per_forward_batch"] / metrics["batch_size"],
            }
        )
    tables = STUDY_ROOT / "tables"
    seed_fields = list(seed_rows[0])
    _write_csv(tables / "summary_by_seed.csv", seed_rows, seed_fields)
    _write_csv(tables / "trace_metrics.csv", patch_rows, list(patch_rows[0]))

    mean_rows = []
    for model in sorted({row["model"] for row in seed_rows}):
        selected = [row for row in seed_rows if row["model"] == model]
        output: dict[str, Any] = {"model": model, "seed_count": len(selected)}
        for field in seed_fields[2:]:
            values = np.asarray([float(row[field]) for row in selected])
            output[f"{field}_mean"] = float(values.mean())
            output[f"{field}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        mean_rows.append(output)
    _write_csv(tables / "summary_mean_std.csv", mean_rows, list(mean_rows[0]))

    bootstrap_rows = []
    import pandas as pd

    patch_frame = pd.DataFrame(patch_rows)
    for metric in BOOTSTRAP_METRICS:
        patch_frame[metric] = patch_frame[metric].astype(float)
    patch_frame["seed"] = patch_frame["seed"].astype(int)
    for metric in BOOTSTRAP_METRICS:
        bootstrap_rows.append({"metric": metric, **cluster_bootstrap_from_patch_frame(patch_frame, metric=metric)})
    _write_csv(tables / "bootstrap_comparison.csv", bootstrap_rows, list(bootstrap_rows[0]))
    print("STATISTICS: PASS")


def _save_figure(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def _patch_delta_selection() -> dict[str, str]:
    baseline = {
        row["patch_id"]: row
        for row in _read_csv(STUDY_ROOT / "runs" / "baseline_seed42" / "per_patch_metrics.csv")
    }
    az = {row["patch_id"]: row for row in _read_csv(STUDY_ROOT / "runs" / "az_thesis_seed42" / "per_patch_metrics.csv")}
    paired = sorted((float(az[key]["dice"]) - float(baseline[key]["dice"]), key) for key in set(baseline) & set(az))
    if not paired:
        raise ValueError("No paired baseline/AZ patch artifacts for figures")
    return {"worst": paired[0][1], "median": paired[len(paired) // 2][1], "best": paired[-1][1]}


def _load_artifact(run: str, patch_id: str) -> Mapping[str, np.ndarray]:
    return np.load(STUDY_ROOT / "artifacts" / run / f"{patch_id}.npz")


def build_figures() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    selections = _patch_delta_selection()
    figures = STUDY_ROOT / "figures"

    def segmentation_figure(patch_id: str, stem: str) -> None:
        baseline = _load_artifact("baseline_seed42", patch_id)
        az = _load_artifact("az_thesis_seed42", patch_id)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
        for axis, image, title, cmap in zip(
            axes,
            (baseline["input"], baseline["target"], baseline["predicted"], az["predicted"]),
            ("Input", "Ground truth", "Baseline", "ANZA-LIRA"),
            (None, "gray", "gray", "gray"),
        ):
            axis.imshow(image, cmap=cmap)
            axis.set_title(title)
            axis.axis("off")
        _save_figure(fig, figures / stem)
        plt.close(fig)

    segmentation_figure(selections["median"], "fig_segmentation_median")
    segmentation_figure(selections["best"], "fig_best_case")
    segmentation_figure(selections["worst"], "fig_worst_case")

    patch_id = selections["median"]
    baseline = _load_artifact("baseline_seed42", patch_id)
    az = _load_artifact("az_thesis_seed42", patch_id)
    target, base_pred, az_pred = baseline["target"].astype(bool), baseline["predicted"].astype(bool), az["predicted"].astype(bool)
    classes = np.zeros(target.shape, dtype=np.uint8)
    classes[target & az_pred] = 1
    classes[target & ~base_pred & az_pred] = 2
    classes[~target & base_pred & ~az_pred] = 3
    classes[~target & ~base_pred & az_pred] = 4
    classes[target & base_pred & ~az_pred] = 5
    labels = ("background", "true positive", "fixed false negative", "removed false positive", "new false positive", "new false negative")
    colors = ("white", "#2b8cbe", "#31a354", "#756bb1", "#de2d26", "#fd8d3c")
    fig, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    axis.imshow(classes, cmap=ListedColormap(colors), vmin=0, vmax=5)
    axis.axis("off")
    axis.set_title("ANZA-LIRA changes vs baseline")
    axis.legend([Patch(color=color, label=label) for color, label in zip(colors[1:], labels[1:])], labels[1:], loc="center left", bbox_to_anchor=(1.02, 0.5))
    _save_figure(fig, figures / "fig_error_median")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3), constrained_layout=True)
    axes[0].imshow(az["input"])
    axes[0].contour(az["target"], levels=[0.5], colors="cyan", linewidths=0.6)
    axes[0].set_title("Input + GT")
    step = 14
    yy, xx = np.mgrid[0 : az["orientation"].shape[0] : step, 0 : az["orientation"].shape[1] : step]
    theta = az["orientation"][::step, ::step]
    axes[1].imshow(az["input"], alpha=0.55)
    axes[1].quiver(xx, yy, np.cos(theta), np.sin(theta), angles="xy", scale_units="xy", scale=0.12, color="navy")
    axes[1].set_title("Native orientation axes")
    anisotropy_image = axes[2].imshow(az["anisotropy"], cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Anisotropy")
    fig.colorbar(anisotropy_image, ax=axes[2], fraction=0.046)
    graph = extract_trace_graph(az["pred_skeleton"])
    axes[3].imshow(az["input"], alpha=0.35)
    for segment in graph.segments:
        y, x = zip(*segment.pixels)
        axes[3].plot(x, y, linewidth=1)
        axes[3].text(x[len(x) // 2], y[len(y) // 2], str(segment.trace_id), fontsize=5)
    axes[3].set_title("Extracted trace segments")
    for axis in axes:
        axis.axis("off")
    _save_figure(fig, figures / "fig_geometry_traces")
    plt.close(fig)

    summary = _read_csv(STUDY_ROOT / "tables" / "summary_mean_std.csv")
    metrics = ("dice", "cldice", "trace_f1")
    fig, axes = plt.subplots(1, len(metrics), figsize=(11, 3.5), constrained_layout=True)
    for axis, metric in zip(axes, metrics):
        for index, row in enumerate(summary):
            axis.errorbar(index, float(row[f"{metric}_mean"]), yerr=float(row[f"{metric}_std"]), fmt="o", capsize=3)
        axis.set_xticks(range(len(summary)), [row["model"] for row in summary], rotation=35, ha="right")
        axis.set_title(metric)
        axis.grid(alpha=0.25)
    _save_figure(fig, figures / "fig_model_comparison")
    plt.close(fig)
    write_json(figures / "example_selection.json", selections)
    print("FIGURES: PASS")


NUMERIC_TOKEN = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?")


def _report_allowed_numeric_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    if isinstance(value, bool) or value is None:
        return tokens
    if isinstance(value, (int, float)):
        number = float(value)
        tokens.update({str(value), f"{number:.1f}", f"{number:.2f}", f"{number:.3f}", f"{number:.4f}", f"{number:.6f}"})
        if number.is_integer():
            tokens.add(str(int(number)))
        return tokens
    if isinstance(value, str):
        tokens.update(NUMERIC_TOKEN.findall(value))
        try:
            number = float(value)
        except ValueError:
            return tokens
        tokens.update({f"{number:.1f}", f"{number:.2f}", f"{number:.3f}", f"{number:.4f}", f"{number:.6f}"})
        if number.is_integer():
            tokens.add(str(int(number)))
        return tokens
    if isinstance(value, Mapping):
        for item in value.values():
            tokens.update(_report_allowed_numeric_tokens(item))
        return tokens
    if isinstance(value, Sequence):
        for item in value:
            tokens.update(_report_allowed_numeric_tokens(item))
    return tokens


def write_report_provenance(thesis_path: Path, report_path: Path, provenance_path: Path) -> dict[str, Any]:
    thesis = _json_file(thesis_path)
    report = report_path.read_text(encoding="utf-8")
    report_tokens = sorted(set(NUMERIC_TOKEN.findall(report)))
    allowed = _report_allowed_numeric_tokens(thesis)
    untraced = sorted(set(report_tokens) - allowed)
    if untraced:
        raise ValueError(f"FINAL_REPORT contains numeric tokens absent from THESIS_NUMBERS: {untraced}")
    payload = {
        "status": "PASS",
        "thesis_numbers_sha256": sha256_file(thesis_path),
        "report_sha256": sha256_file(report_path),
        "numeric_tokens": report_tokens,
        "untraced_numeric_tokens": [],
    }
    write_json(provenance_path, payload)
    return payload


def verify_report_consistency(thesis_path: Path, report_path: Path, provenance_path: Path) -> None:
    provenance = _json_file(provenance_path)
    if provenance.get("thesis_numbers_sha256") != sha256_file(thesis_path):
        raise ValueError("THESIS_NUMBERS changed after FINAL_REPORT generation")
    if provenance.get("report_sha256") != sha256_file(report_path):
        raise ValueError("FINAL_REPORT changed after provenance generation")
    write_report_provenance(thesis_path, report_path, provenance_path)


def build_report() -> None:
    manifest = json.loads((SPLIT_DIR / "geocrack_small_v1_manifest.json").read_text(encoding="utf-8"))
    summary = _read_csv(STUDY_ROOT / "tables" / "summary_mean_std.csv")
    bootstrap = _read_csv(STUDY_ROOT / "tables" / "bootstrap_comparison.csv")
    by_model = {row["model"]: row for row in summary}
    if "baseline" not in by_model or "az_thesis" not in by_model:
        raise ValueError("Report requires baseline and az_thesis summary rows")
    baseline, az = by_model["baseline"], by_model["az_thesis"]
    delta = {
        metric: float(az[f"{metric}_mean"]) - float(baseline[f"{metric}_mean"])
        for metric in ("dice", "iou", "precision", "recall", "cldice", "trace_f1")
    }
    thesis = {
        "dataset": {
            "name": "GeoCrack",
            "doi": "10.7910/DVN/E4OXHQ",
            "pair_count": manifest["dataset_pair_count"],
            "patch_size": [224, 224],
        },
        "split": manifest,
        "training": {"required_runs": 9, "seeds": [41, 42, 43]},
        "baseline": baseline,
        "anza_lira": az,
        "delta": delta,
        "ablations": {row["model"]: row for row in summary if row["model"] not in {"baseline", "az_thesis"}},
        "bootstrap_ci": {row["metric"]: row for row in bootstrap},
        "trace_extraction": {
            "object": "fracture trace segment",
            "threshold_source": "validation",
            "connectivity": 8,
            "border_margin_px": 5,
        },
        "confidence_level_percent": 95,
        "provenance": {"checksum_algorithm": "SHA-256"},
        "runtime": {"baseline_inference_ms": baseline["inference_ms_mean"], "az_inference_ms": az["inference_ms_mean"]},
        "limitations": [
            "GeoCrack masks are binary and do not provide geological instance IDs.",
            "Images are outcrop photogrammetry, not satellite lineaments.",
            "Trace merging and segmentation thresholds are validation-selected.",
            "Confidence intervals use source-image clusters rather than independent patches.",
        ],
    }
    write_json(STUDY_ROOT / "THESIS_NUMBERS.json", thesis)
    bootstrap_lines = "\n".join(
        f"- {row['metric']}: delta={float(row['mean_delta_az_minus_baseline']):.4f}, "
        f"95% CI [{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}]"
        for row in bootstrap
    )
    report = f"""# ANZA-LIRA GeoCrack Study

## Что проверяли

Проверялась сегментация и трассировка следов геологических трещин на фотограмметрических изображениях обнажений.

## Почему GeoCrack

Официальный набор DOI `10.7910/DVN/E4OXHQ` содержит реальные 224x224 image-mask patches и сохраняет источник patch в имени.

## Что считается fracture trace

Объект — непрерывный fracture trace segment между концом и/или кластером узла. Это не подтверждённый тектонический разлом.

## Как сформирован split

Split seed 2026, группировка по `source_image_id`, source leakage 0. Test CSV заморожен SHA-256.

## Что такое baseline

Существующая U-Net repository baseline при тех же split, loss, augmentation, epochs, batch и threshold grid.

## Что изменено в ANZA-LIRA

Ядро AZConv не переписывалось; добавлен downstream перевод вероятности и native geometry в skeleton graph и trace segments.

## Что означает orientation field

Осевое поле агрегировано через doubled angles, поэтому theta и theta+pi эквивалентны.

## Как mask превращается в trace objects

Validation threshold -> binary mask -> one-pixel skeleton -> 8-connected graph -> endpoint/junction chains -> GeoJSON LineString.

## Training protocol

Baseline и az_thesis: seeds 41/42/43. Ablations az_no_fuzzy, az_no_aniso и attention_unet: seed 42. Test не использовался для выбора параметров.

## Pixel metrics

Baseline Dice mean: {float(baseline['dice_mean']):.4f}; ANZA-LIRA: {float(az['dice_mean']):.4f}; delta: {delta['dice']:.4f}.

## Topology metrics

Baseline clDice mean: {float(baseline['cldice_mean']):.4f}; ANZA-LIRA: {float(az['cldice_mean']):.4f}; delta: {delta['cldice']:.4f}.

## Trace metrics

Baseline trace F1 mean: {float(baseline['trace_f1_mean']):.4f}; ANZA-LIRA: {float(az['trace_f1_mean']):.4f}; delta: {delta['trace_f1']:.4f}.

## Ablations

Полные значения генерируются в `tables/summary_mean_std.csv`; отсутствующие или отрицательные результаты не скрываются.

## Statistical uncertainty

{bootstrap_lines}

## Best/median/worst examples

Выбраны автоматически по patch-level delta Dice; основной рисунок использует median, best/worst сохранены отдельно.

## Где ANZA-LIRA помогает

Поддерживаемые данными положительные delta и CI перечислены выше; CI через ноль не трактуется как подтверждённое преимущество.

## Где ухудшает результат

Все отрицательные delta, включая precision/recall/speed trade-offs, сохранены в таблицах и `THESIS_NUMBERS.json`.

## Ограничения

- Binary masks не содержат instance-ID отдельных геологических объектов.
- Данные относятся к обнажениям и фотограмметрии, не к спутниковой съёмке.
- Малое source-grouped subset ограничивает статистическую мощность.

## Что можно утверждать в тезисах

Можно утверждать только измеренный результат сегментации, связности и trace extraction при зафиксированном протоколе.

## Что нельзя утверждать

Нельзя называть сегменты тектоническими разломами, а оператор — доказанно эргодическим или системой Аносова.

## Exact reproduction commands

```powershell
scripts/run_geocrack_full_study.ps1
python scripts/validate_geocrack_study.py
```
"""
    report_path = STUDY_ROOT / "FINAL_REPORT.md"
    thesis_path = STUDY_ROOT / "THESIS_NUMBERS.json"
    report_path.write_text(report, encoding="utf-8")
    write_report_provenance(thesis_path, report_path, STUDY_ROOT / "REPORT_PROVENANCE.json")
    print("REPORT: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("environment")
    training = sub.add_parser("training")
    training.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "geocrack_small.yaml")
    training.add_argument("--smoke", action="store_true")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--smoke", action="store_true")
    traces = sub.add_parser("traces")
    traces.add_argument("--smoke", action="store_true")
    sub.add_parser("statistics")
    sub.add_parser("figures")
    sub.add_parser("report")
    dry_run = sub.add_parser("dry-run")
    dry_run.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "geocrack_small.yaml")
    full = sub.add_parser("full")
    full.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "geocrack_small.yaml")
    full.add_argument("--smoke-config", type=Path, default=PROJECT_ROOT / "configs" / "geocrack_smoke.yaml")
    full.add_argument("--dry-run", action="store_true")
    synthetic = sub.add_parser("synthetic")
    synthetic.add_argument(
        "--output-root",
        type=Path,
        default=STUDY_ROOT / "prepared" / "synthetic_pipeline",
    )
    args = parser.parse_args()
    if args.command == "environment":
        capture_environment()
    elif args.command == "training":
        run_training_matrix(args.config, smoke=args.smoke)
    elif args.command == "evaluate":
        evaluate_all(smoke=args.smoke)
    elif args.command == "traces":
        build_all_traces(smoke=args.smoke)
    elif args.command == "statistics":
        build_statistics()
    elif args.command == "figures":
        build_figures()
    elif args.command == "report":
        build_report()
    elif args.command == "dry-run":
        dry_run_matrix(args.config)
    elif args.command == "full":
        if args.dry_run:
            dry_run_matrix(args.config)
        else:
            run_full_pipeline(args.config, args.smoke_config)
    elif args.command == "synthetic":
        from scripts.geocrack_synthetic_pipeline import run_synthetic_pipeline

        run_synthetic_pipeline(args.output_root)


if __name__ == "__main__":
    main()
