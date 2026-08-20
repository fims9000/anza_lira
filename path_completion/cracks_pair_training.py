"""One frozen real-domain CRACKS endpoint-pair experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from connectivity_repair.balanced_metrics import balanced_matched_pair_metrics
from cracks_experiment.partial_label_evaluation import T1_ROOT
from cracks_experiment.partial_label_training import (
    T1_PROTOCOL,
    _model,
    load_t1_checkpoint,
    t1_matrix,
)
from cracks_experiment.partial_labels import map_partial_annotation
from cracks_experiment.training import NORMALIZATION
from cracks_experiment.validation import _sha256, tiled_probability
from datasets.cracks import load_section_image
from path_completion.cracks_pairs import (
    MATCH_TOLERANCES,
    matched_section_pairs,
    oriented_real_pair_crop,
    split_sections,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAIR_ROOT = PROJECT_ROOT / "results" / "final_practical_cycle" / "cracks_pairs"
_ANZA_SPEC = next(spec for spec in t1_matrix() if spec.model == "anza_v1" and spec.seed == 42)
_ANZA_RUN = T1_ROOT / f"{_ANZA_SPEC.run_id}-{_ANZA_SPEC.run_hash}"
_ANZA_EVALUATION = T1_ROOT / "evaluation" / _ANZA_SPEC.run_id / "evaluation.json"
PAIR_PROTOCOL: dict[str, Any] = {
    "version": "cracks_real_endpoint_pairs_v1",
    "source": "CRACKS crowd TRAIN annotations only; expert forbidden",
    "segmentation_checkpoint": str(_ANZA_RUN / "checkpoint-last.pt"),
    "segmentation_run_hash": _ANZA_SPEC.run_hash,
    "segmentation_seed": 42,
    "segmentation_threshold_source": str(_ANZA_EVALUATION),
    "feature_channels": [
        "rgb_r", "rgb_g", "rgb_b", "base_probability_with_candidate_gap",
        "endpoint_markers", "anza_cos2theta", "anza_sin2theta", "anza_anisotropy",
    ],
    "pair_crop_hw": [33, 49],
    "section_split": "stable_sha256_two_thirds_train_one_third_validation",
    "train_matched_pairs": 256,
    "validation_matched_pairs": 120,
    "validation_capacity_audit": "strict generator capacity 125 across 107 frozen validation sections; quota fixed to 120 before model scores",
    "pairs_per_section_max": 2,
    "annotators_per_section_search_max": 8,
    "balance": "one positive and one descriptor-matched negative per pair",
    "matching": ["endpoint_distance", "tangent_angle", "local_contrast", "spatial_depth"],
    "matching_max_abs_delta": MATCH_TOLERANCES.tolist(),
    "classifier": "five-convolution corridor encoder, mean+max readout",
    "epochs": 100,
    "batch_pairs": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "ranking_weight": 0.5,
    "ranking_temperature": 0.2,
    "seed": 42,
    "operating_point": "validation max TPR subject FPR<=0.05; ties highest threshold",
    "gates": {"auroc": 0.85, "balanced_auprc": 0.85, "fpr_max": 0.05, "tpr_min": 0.70},
    "expert": "LOCKED_NOT_ACCESSED",
    "source_sha256": {
        "path_completion/cracks_pairs.py": hashlib.sha256(
            (PROJECT_ROOT / "path_completion" / "cracks_pairs.py").read_bytes()
        ).hexdigest(),
        "path_completion/cracks_pair_training.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    },
}


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class CRACKSRealPairClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(inputs)
        pooled = torch.cat((features.mean(dim=(-2, -1)), features.amax(dim=(-2, -1))), dim=1)
        return self.head(pooled).squeeze(1)


def _load_frozen_anza(device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    evaluation = json.loads(_ANZA_EVALUATION.read_text())
    if (
        evaluation.get("status") != "COMPLETE"
        or evaluation.get("expert_data_accessed") is not False
        or int(evaluation.get("section_count", 0)) != len(T1_PROTOCOL["heldout_section_ids"])
    ):
        raise PermissionError("Frozen T1 ANZA crowd validation is missing or not expert-locked")
    model = _model(_ANZA_SPEC).to(torch.device(device))
    load_t1_checkpoint(_ANZA_RUN / "checkpoint-last.pt", _ANZA_SPEC, model)
    model.eval()
    return model, evaluation


@torch.inference_mode()
def _anza_fields(model: torch.nn.Module, image_hwc: np.ndarray) -> dict[str, np.ndarray]:
    raw = torch.from_numpy(np.asarray(image_hwc, dtype=np.float32).transpose(2, 0, 1))
    mean = torch.tensor(NORMALIZATION["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(NORMALIZATION["std"], dtype=torch.float32).view(3, 1, 1)
    normalized = F.pad((raw - mean) / std, (0, 3, 0, 1))
    probability = tiled_probability(model, normalized).numpy()[:255, :701]
    device = next(model.parameters()).device
    spatial = model.enc1.spatial
    if not hasattr(spatial, "geometry_conv") or spatial.geometry_conv is None:
        raise TypeError("Frozen legacy ANZA lacks local geometry fields")
    inputs = normalized.unsqueeze(0).to(device)
    memberships = torch.softmax(spatial.gate_conv(inputs), dim=1)
    theta, _raw_base, raw_hyper = torch.chunk(spatial.geometry_conv(inputs), 3, dim=1)
    hyper = F.softplus(raw_hyper).clamp_max(float(spatial.cfg.max_hyperbolicity))
    cos2 = (memberships * torch.cos(2.0 * theta)).sum(dim=1)
    sin2 = (memberships * torch.sin(2.0 * theta)).sum(dim=1)
    norm = torch.sqrt(cos2.square() + sin2.square()).clamp_min(1e-6)
    anisotropy = (memberships * hyper).sum(dim=1).clamp(0.0, 1.0)
    return {
        "image": np.asarray(image_hwc, dtype=np.float32).transpose(2, 0, 1),
        "base_probability": probability.astype(np.float32),
        "cos2theta": (cos2 / norm)[0, :255, :701].cpu().numpy().astype(np.float32),
        "sin2theta": (sin2 / norm)[0, :255, :701].cpu().numpy().astype(np.float32),
        "anisotropy": anisotropy[0, :255, :701].cpu().numpy().astype(np.float32),
    }


def build_real_pair_dataset(*, device: str = "cuda") -> dict[str, Any]:
    PAIR_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_path = PAIR_ROOT / "pairs.npz"
    manifest_path = PAIR_ROOT / "manifest.json"
    protocol = dict(PAIR_PROTOCOL)
    protocol["segmentation_checkpoint_sha256"] = _sha256(_ANZA_RUN / "checkpoint-last.pt")
    protocol["segmentation_evaluation_sha256"] = _sha256(_ANZA_EVALUATION)
    protocol_hash = _canonical_hash(protocol)
    if dataset_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("protocol_sha256") != protocol_hash or manifest.get("expert_data_accessed") is not False:
            raise ValueError("Real-pair dataset provenance drift")
        return {**manifest, "action": "SKIP"}

    model, evaluation = _load_frozen_anza(device)
    train_sections, validation_sections = split_sections(T1_PROTOCOL["training_section_ids"])
    if set(train_sections) & set(validation_sections):
        raise AssertionError("Real-pair train/validation sections overlap")
    annotation_root = PROJECT_ROOT / "data" / "cracks" / "annotations"
    image_root = PROJECT_ROOT / "data" / "cracks" / "images"
    split_payload: dict[str, list[np.ndarray]] = {"train": [], "validation": []}
    rows: list[dict[str, Any]] = []
    for split, sections in (("train", train_sections), ("validation", validation_sections)):
        quota = int(protocol[f"{split}_matched_pairs"])
        for section_id in sections:
            if len(split_payload[split]) >= quota:
                break
            name = f"section_{section_id:03d}.png"
            image = load_section_image(image_root / name)
            section_examples = []
            for annotator in T1_PROTOCOL["training_annotators"][: int(protocol["annotators_per_section_search_max"])]:
                path = annotation_root / annotator / name
                if not path.is_file():
                    continue
                with Image.open(path) as handle:
                    target, weight = map_partial_annotation(np.asarray(handle.convert("RGB"), dtype=np.uint8))
                examples = matched_section_pairs(
                    (target > 0.5) & (weight > 0),
                    image.transpose(2, 0, 1),
                    max_pairs=int(protocol["pairs_per_section_max"]),
                )
                for positive, negative in examples:
                    section_examples.append((annotator, positive, negative))
                    if len(section_examples) >= int(protocol["pairs_per_section_max"]):
                        break
                if len(section_examples) >= int(protocol["pairs_per_section_max"]):
                    break
            if not section_examples:
                continue
            fields = _anza_fields(model, image)
            for annotator, positive, negative in section_examples:
                pair = np.stack(
                    [oriented_real_pair_crop(fields, positive), oriented_real_pair_crop(fields, negative)]
                ).astype(np.float32)
                split_payload[split].append(pair)
                rows.append({
                    "split": split,
                    "section_id": section_id,
                    "annotator": annotator,
                    "positive_first": [int(value) for value in positive.first],
                    "positive_second": [int(value) for value in positive.second],
                    "negative_first": [int(value) for value in negative.first],
                    "negative_second": [int(value) for value in negative.second],
                    "positive_descriptor": positive.descriptor.tolist(),
                    "negative_descriptor": negative.descriptor.tolist(),
                })
                if len(split_payload[split]) >= quota:
                    break
            if len(split_payload[split]) % 32 < int(protocol["pairs_per_section_max"]):
                print(
                    f"phase=cracks_real_pairs split={split} pairs={len(split_payload[split])}/{quota} "
                    "expert=LOCKED status=BUILDING",
                    flush=True,
                )
        if len(split_payload[split]) != quota:
            raise ValueError(f"Insufficient {split} real pairs: {len(split_payload[split])}/{quota}")
    train_groups = {row["section_id"] for row in rows if row["split"] == "train"}
    validation_groups = {row["section_id"] for row in rows if row["split"] == "validation"}
    if train_groups & validation_groups:
        raise AssertionError("Generated real-pair sections overlap")
    np.savez_compressed(
        dataset_path,
        train=np.stack(split_payload["train"]),
        validation=np.stack(split_payload["validation"]),
    )
    core = {
        "status": "COMPLETE",
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "segmentation_threshold": evaluation["selected_threshold"],
        "train_matched_pairs": len(split_payload["train"]),
        "validation_matched_pairs": len(split_payload["validation"]),
        "train_section_ids": sorted(train_groups),
        "validation_section_ids": sorted(validation_groups),
        "section_disjoint": True,
        "balanced_50_50": True,
        "rows": rows,
        "expert_data_accessed": False,
    }
    core["dataset_sha256"] = _sha256(dataset_path)
    manifest_path.write_text(json.dumps(core, indent=2, sort_keys=True) + "\n")
    return {**core, "action": "RUN"}


def _scores(model: nn.Module, arrays: np.ndarray, device: torch.device) -> np.ndarray:
    flat = torch.from_numpy(arrays.reshape(-1, *arrays.shape[2:])).to(device)
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(flat), 64):
            outputs.append(torch.sigmoid(model(flat[start : start + 64])).cpu())
    return torch.cat(outputs).numpy().reshape(len(arrays), 2)


def _operating_point(positive: np.ndarray, negative: np.ndarray) -> dict[str, float | int]:
    candidates = np.unique(np.concatenate(([0.0], positive, negative, [1.0])))
    rows = []
    for threshold in candidates:
        tp = int(np.count_nonzero(positive >= threshold))
        fn = int(len(positive) - tp)
        fp = int(np.count_nonzero(negative >= threshold))
        tn = int(len(negative) - fp)
        rows.append({
            "threshold": float(threshold),
            "tpr": float(tp / len(positive)),
            "fpr": float(fp / len(negative)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    eligible = [row for row in rows if row["fpr"] <= float(PAIR_PROTOCOL["gates"]["fpr_max"])]
    return max(eligible, key=lambda row: (row["tpr"], row["threshold"]))


def train_real_pair_classifier(*, device: str = "cuda") -> dict[str, Any]:
    manifest = build_real_pair_dataset(device=device)
    result_path = PAIR_ROOT / "result.json"
    checkpoint_path = PAIR_ROOT / "checkpoint.pt"
    if result_path.exists() and checkpoint_path.exists():
        existing = json.loads(result_path.read_text())
        if (
            existing.get("status") in {"CRACKS_REAL_PAIR_CLASSIFIER_PASS", "CRACKS_REAL_PAIR_CLASSIFIER_GATE_FAIL"}
            and existing.get("dataset_sha256") == manifest["dataset_sha256"]
            and existing.get("expert_data_accessed") is False
        ):
            return {**existing, "action": "SKIP"}
        raise ValueError("Existing real-pair classifier provenance drift")
    with np.load(PAIR_ROOT / "pairs.npz") as payload:
        train = payload["train"].astype(np.float32)
        validation = payload["validation"].astype(np.float32)
    seed = int(PAIR_PROTOCOL["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch_device = torch.device(device)
    model = CRACKSRealPairClassifier().to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(PAIR_PROTOCOL["learning_rate"]),
        weight_decay=float(PAIR_PROTOCOL["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    final_losses = []
    for epoch in range(int(PAIR_PROTOCOL["epochs"])):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        final_losses = []
        for start in range(0, len(order), int(PAIR_PROTOCOL["batch_pairs"])):
            index = order[start : start + int(PAIR_PROTOCOL["batch_pairs"])].numpy()
            batch = torch.from_numpy(train[index]).to(torch_device)
            logits = model(batch.flatten(0, 1)).reshape(len(index), 2)
            targets = torch.tensor([1.0, 0.0], device=torch_device).expand_as(logits)
            bce = F.binary_cross_entropy_with_logits(logits, targets)
            ranking = F.softplus(
                -(logits[:, 0] - logits[:, 1]) / float(PAIR_PROTOCOL["ranking_temperature"])
            ).mean()
            loss = bce + float(PAIR_PROTOCOL["ranking_weight"]) * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_losses.append(float(loss.detach()))
        if (epoch + 1) % 20 == 0:
            print(f"phase=cracks_real_pair_train epoch={epoch + 1}/100 loss={np.mean(final_losses):.6f}", flush=True)
    train_scores = _scores(model, train, torch_device)
    validation_scores = _scores(model, validation, torch_device)
    operating = _operating_point(validation_scores[:, 0], validation_scores[:, 1])
    train_metrics = balanced_matched_pair_metrics(
        train_scores[:, 0], train_scores[:, 1], threshold=float(operating["threshold"])
    )
    validation_metrics = balanced_matched_pair_metrics(
        validation_scores[:, 0], validation_scores[:, 1], threshold=float(operating["threshold"])
    )
    gates = PAIR_PROTOCOL["gates"]
    checks = {
        "auroc": validation_metrics["auroc"] >= float(gates["auroc"]),
        "balanced_auprc": validation_metrics["balanced_auprc"] >= float(gates["balanced_auprc"]),
        "fpr": float(operating["fpr"]) <= float(gates["fpr_max"]),
        "tpr": float(operating["tpr"]) >= float(gates["tpr_min"]),
    }
    status = "CRACKS_REAL_PAIR_CLASSIFIER_PASS" if all(checks.values()) else "CRACKS_REAL_PAIR_CLASSIFIER_GATE_FAIL"
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "protocol_sha256": manifest["protocol_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "operating_threshold": operating["threshold"],
        "expert_data_accessed": False,
    }
    torch.save(checkpoint, checkpoint_path)
    rows = []
    for split, scores in (("train", train_scores), ("validation", validation_scores)):
        for index, (positive, negative) in enumerate(scores):
            rows.append({"split": split, "pair_id": index, "positive_score": float(positive), "negative_score": float(negative)})
    with (PAIR_ROOT / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": status,
        "action": "RUN",
        "checks": checks,
        "protocol_sha256": manifest["protocol_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "checkpoint_sha256": _sha256(checkpoint_path),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_operating_point": operating,
        "train_matched_pairs": len(train),
        "validation_matched_pairs": len(validation),
        "balanced_50_50": True,
        "section_disjoint": True,
        "expert_data_accessed": False,
        "expert_scores_used": False,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
