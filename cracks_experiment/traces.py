"""Frozen Setting A predictions to candidate fault-trace GeoJSON."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cracks_experiment.finetuning import verify_setting_a_complete
from cracks_experiment.human import _normalized_image, tiled_v2_uncertainty
from cracks_experiment.matrix import PROJECT_ROOT, setting_a_matrix
from cracks_experiment.training import build_real_model, load_real_checkpoint
from cracks_experiment.validation import tiled_probability
from trace_extraction.export import traces_to_geojson, write_geojson
from trace_extraction.graph import extract_trace_graph
from trace_extraction.skeleton import skeletonize_mask


def export_setting_a_traces(
    setting_a_root: Path,
    setting_a_expert_root: Path,
    output_root: Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    receipt = verify_setting_a_complete(setting_a_root, setting_a_expert_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        files = list(output_root.glob("*/*.geojson"))
        if existing.get("status") == "COMPLETE" and len(files) == existing.get("geojson_count"):
            return {**existing, "action": "SKIP"}
    protocol = json.loads((PROJECT_ROOT / "results" / "anza_v2_study" / "protocol.json").read_text())
    section_ids = list(protocol["setting_a"]["expert_evaluation_sections"])
    specs = [
        spec for spec in setting_a_matrix()
        if spec.comparison_family == "main" and spec.seed == 42
    ]
    torch_device = torch.device(device)
    records = []
    for spec in specs:
        run_dir = Path(setting_a_root) / f"{spec.run_id}-{spec.run_hash}"
        threshold = float(json.loads((run_dir / "crowd_validation.json").read_text())["selected_threshold"])
        model = build_real_model(spec).to(torch_device)
        load_real_checkpoint(run_dir / "checkpoint-last.pt", spec.run_hash, model)
        model.eval()
        model_root = output_root / spec.run_id
        model_root.mkdir(exist_ok=True)
        for position, section_id in enumerate(section_ids):
            image = _normalized_image(section_id)
            if spec.model.startswith("anza_v2"):
                maps = tiled_v2_uncertainty(model, image)
                probability = maps["probability"][:255, :701]
                coherence = maps["rho"][:255, :701]
                anisotropy = maps["anisotropy"][:255, :701]
            else:
                probability = tiled_probability(model, image).numpy()[:255, :701]
                coherence = np.ones_like(probability)
                anisotropy = np.zeros_like(probability)
            skeleton = skeletonize_mask(probability >= threshold)
            graph = extract_trace_graph(skeleton, border_margin=5)
            payload = traces_to_geojson(
                graph.segments,
                source_image_id=f"section_{section_id:03d}",
                patch_id="full_section_255x701",
                model=spec.model,
                seed=spec.seed,
                probability=probability,
                coherence=coherence,
                anisotropy=anisotropy,
                confidence=probability,
            )
            path = model_root / f"section_{section_id:03d}.geojson"
            write_geojson(path, payload)
            records.append(
                {
                    "run_id": spec.run_id,
                    "run_hash": spec.run_hash,
                    "section_id": section_id,
                    "threshold": threshold,
                    "trace_count": len(graph.segments),
                    "geojson": str(path.relative_to(output_root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            print(
                f"phase=cracks_traces model={spec.run_id} section={position + 1}/{len(section_ids)} "
                f"traces={len(graph.segments)} status=RUNNING"
            )
    payload = {
        "status": "COMPLETE",
        "action": "RUN",
        "setting_a_receipt_sha256": receipt["sha256"],
        "model_count": len(specs),
        "section_count_per_model": len(section_ids),
        "geojson_count": len(records),
        "object_name": "candidate fault trace branch",
        "records": records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
