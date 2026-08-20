"""Load and cache the immutable three-seed T1 U-Net mean ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cracks_experiment.partial_label_evaluation import _load_t1_model
from cracks_experiment.partial_label_training import T1RunSpec
from cracks_experiment.training import NORMALIZATION
from cracks_experiment.validation import tiled_probability
from datasets.cracks import load_section_image
from lira_final.io import sha256
from lira_final.protocol import ROOT


def checkpoint_manifest() -> list[dict[str, object]]:
    rows = []
    for seed in (41, 42, 43):
        spec = T1RunSpec(f"t1_unet_s{seed}", "unet", seed)
        path = ROOT / "results/final_practical_cycle/cracks_t1" / f"{spec.run_id}-{spec.run_hash}" / "checkpoint-last.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"seed": seed, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    return rows


def _normalized_image(section_id: int) -> torch.Tensor:
    image = load_section_image(ROOT / "data/cracks/images" / f"section_{section_id:03d}.png")
    tensor = torch.from_numpy(image.transpose(2, 0, 1))
    mean = torch.tensor(NORMALIZATION["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(NORMALIZATION["std"], dtype=torch.float32).view(3, 1, 1).clamp_min(1e-6)
    return torch.nn.functional.pad((tensor - mean) / std, (0, 3, 0, 1))


def cache_ensemble(section_ids: list[int], cache_root: Path, *, device: str) -> dict[str, object]:
    cache_root.mkdir(parents=True, exist_ok=True)
    missing = [section_id for section_id in section_ids if not (cache_root / f"section_{section_id:03d}.npy").is_file()]
    if missing:
        models = []
        for seed in (41, 42, 43):
            model, _path, _hash = _load_t1_model(T1RunSpec(f"t1_unet_s{seed}", "unet", seed), device)
            models.append(model.eval())
        for index, section_id in enumerate(missing):
            image = _normalized_image(section_id)
            probability = np.mean([tiled_probability(model, image).numpy()[:255, :701] for model in models], axis=0)
            np.save(cache_root / f"section_{section_id:03d}.npy", probability.astype(np.float16), allow_pickle=False)
            if (index + 1) % 20 == 0 or index + 1 == len(missing):
                print(f"phase=F1_DENSE section={index + 1}/{len(missing)}", flush=True)
    return {"sections": len(section_ids), "new_sections": len(missing), "checkpoints": checkpoint_manifest(), "ensemble": "arithmetic mean", "expert_accessed": False}


def load_probability(cache_root: Path, section_id: int) -> np.ndarray:
    return np.load(cache_root / f"section_{section_id:03d}.npy", allow_pickle=False).astype(np.float32)

