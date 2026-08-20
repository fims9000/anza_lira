"""Reproducible CRACKS compute/VRAM microbenchmark for the main models."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from cracks_experiment.matrix import PROJECT_ROOT, setting_a_matrix
from cracks_experiment.training import NORMALIZATION, build_real_model, load_real_checkpoint
from cracks_experiment.validation import tiled_probability
from datasets.cracks import CRACKSSectionDataset
import utils


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_efficiency_audit(
    setting_a_root: Path,
    output_root: Path,
    *,
    device: str = "cuda",
    repetitions: int = 5,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "efficiency.json"
    rows_path = output_root / "efficiency.csv"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("status") == "COMPLETE" and existing.get("repetitions") == repetitions:
            return {**existing, "action": "SKIP"}
    protocol = json.loads((PROJECT_ROOT / "results" / "anza_v2_study" / "protocol.json").read_text())
    section_id = int(protocol["setting_a"]["training_section_ids"][0])
    dataset = CRACKSSectionDataset(
        PROJECT_ROOT / "data" / "cracks" / "images",
        PROJECT_ROOT / "data" / "cracks" / "crowd_targets" / "paper_like" / "train",
        [section_id],
        mean=NORMALIZATION["mean"],
        std=NORMALIZATION["std"],
        crop_size=256,
        foreground_probability=0.7,
        seed=123,
    )
    crop = dataset[0]
    full = CRACKSSectionDataset(
        PROJECT_ROOT / "data" / "cracks" / "images",
        PROJECT_ROOT / "data" / "cracks" / "crowd_targets" / "paper_like" / "heldout",
        [int(protocol["setting_a"]["held_out_validation_section_ids"][0])],
        mean=NORMALIZATION["mean"],
        std=NORMALIZATION["std"],
    )[0]
    torch_device = torch.device(device)
    specs = [
        spec for spec in setting_a_matrix()
        if spec.comparison_family == "main" and spec.seed == 42
    ]
    rows = []
    for spec in specs:
        run_dir = Path(setting_a_root) / f"{spec.run_id}-{spec.run_hash}"
        model = build_real_model(spec).to(torch_device)
        load_real_checkpoint(run_dir / "checkpoint-last.pt", spec.run_hash, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        image = crop["image"].unsqueeze(0).to(torch_device)
        target = crop["target"].unsqueeze(0).to(torch_device)
        valid = crop["valid"].unsqueeze(0).float().to(torch_device)
        if torch_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch_device)
        training_times = []
        model.train()
        for _ in range(int(repetitions) + 1):
            optimizer.zero_grad(set_to_none=True)
            _synchronize(torch_device)
            started = time.perf_counter()
            logits = model(image)
            loss, _logs, _ = utils.segmentation_objective(
                logits, target, valid, topology_weight=0.2, topology_num_iters=5
            )
            loss.backward()
            optimizer.step()
            _synchronize(torch_device)
            training_times.append((time.perf_counter() - started) * 1000.0)
        peak = (
            torch.cuda.max_memory_allocated(torch_device) / (1024**2)
            if torch_device.type == "cuda"
            else 0.0
        )
        model.eval()
        inference_times = []
        for _ in range(int(repetitions) + 1):
            _synchronize(torch_device)
            started = time.perf_counter()
            tiled_probability(model, full["image"])
            _synchronize(torch_device)
            inference_times.append((time.perf_counter() - started) * 1000.0)
        train_ms = float(np.median(training_times[1:]))
        inference_ms = float(np.median(inference_times[1:]))
        rows.append(
            {
                "model": spec.model,
                "run_id": spec.run_id,
                "parameter_count": parameter_count(model),
                "peak_vram_mib": peak,
                "train_step_ms_256": train_ms,
                "estimated_sec_per_393_section_epoch": train_ms * 393 / 1000.0,
                "tiled_inference_ms_256x704": inference_ms,
                "repetitions": repetitions,
            }
        )
        print(f"phase=efficiency model={spec.model} inference_ms={inference_ms:.2f} status=COMPLETE")
    baseline = next(row for row in rows if row["model"] == "anza_v1")
    v2 = next(row for row in rows if row["model"] == "anza_v2b")
    ratios = {
        "v2_to_v1_peak_vram": float(v2["peak_vram_mib"] / baseline["peak_vram_mib"]) if baseline["peak_vram_mib"] else None,
        "v2_to_v1_inference_time": float(v2["tiled_inference_ms_256x704"] / baseline["tiled_inference_ms_256x704"]),
    }
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "COMPLETE",
        "action": "RUN",
        "device": str(torch_device),
        "repetitions": repetitions,
        "warmup_repetitions": 1,
        "training_timing_kind": "controlled single-crop forward-backward-step microbenchmark",
        "inference_timing_kind": "full padded section via four 256px tiles",
        "ratios": ratios,
        "v2_exceeds_3x_vram": ratios["v2_to_v1_peak_vram"] is not None and ratios["v2_to_v1_peak_vram"] > 3.0,
        "v2_exceeds_3x_inference": ratios["v2_to_v1_inference_time"] > 3.0,
    }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
