"""Frozen SBPP proposals and source-balanced P0 corridor caches."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from anza_tracegraph.frozen_source import infer_dense
from anza_tracegraph.ports_v3.metrics import branch_match
from anza_tracegraph.ports_v3.runner import _context as hard_context
from anza_tracegraph.ports_v3_b.candidates import propose_cluster_candidates
from anza_tracegraph.ports_v3_b.clustering import BranchCluster, cluster_branches
from anza_tracegraph.ports_v3_b.soft_branches import extract_soft_branches

from ..protocol import PROTOCOL
from ..split_data import generate_scene, scene_digest
from .corridor import branch_landing_corridor


STATUS_PRESENT = "CORRECT_CANDIDATE_PRESENT"
STATUS_NONE = "NO_VALID_CONTINUATION"
STATUS_MISS = "CANDIDATE_MISS"


def _cluster_matches(cluster: BranchCluster, target: np.ndarray) -> bool:
    return any(branch_match(member, target)[0] for member in cluster.members)


def propose_scene(scene: dict[str, Any], probability: np.ndarray) -> dict[str, Any]:
    """Run exactly the frozen V3-B proposal front-end at tau_s=0.20, K=12."""
    tau_s = float(PROTOCOL["sbpp"]["tau_s"])
    k = int(PROTOCOL["sbpp"]["candidate_k"])
    hard = hard_context(scene, probability, float(PROTOCOL["sbpp"]["tau_h"]))
    if hard["source"] is None:
        candidates: tuple[Any, ...] = ()
        clusters: tuple[BranchCluster, ...] = ()
    else:
        excluded = np.zeros_like(probability, dtype=bool)
        start, end = scene["input"]["relation_corridor_x"]
        excluded[:, start:end] = True
        soft = extract_soft_branches(
            probability,
            scene["input"]["model_input"][0],
            hard["mask"],
            hard["source"],
            tau_s=tau_s,
            excluded_mask=excluded,
        )
        clusters = cluster_branches(hard["branches"], soft)
        candidates = propose_cluster_candidates(hard["source"], clusters)[:k]
    positive = bool(scene["truth"]["has_valid_continuation"])
    target = scene["truth"]["destination_branch"]
    cluster_by_id = {cluster.cluster_id: cluster for cluster in clusters}
    correct = []
    if positive and target is not None:
        correct = [
            rank
            for rank, candidate in enumerate(candidates)
            if _cluster_matches(cluster_by_id[candidate.destination_branch_id], target)
        ]
    correct_rank = correct[0] if correct else -1
    status = STATUS_NONE if not positive else (STATUS_PRESENT if correct_rank >= 0 else STATUS_MISS)
    return {
        "source": hard["source"],
        "candidates": candidates,
        "status": status,
        "correct_rank": correct_rank,
        "positive": positive,
    }


def selected_training_ranks(status: str, correct_rank: int, candidates: tuple[Any, ...]) -> tuple[int, ...]:
    """One source contributes at most four pairs; candidate misses contribute none."""
    if status == STATUS_MISS or not candidates:
        return ()
    if status == STATUS_NONE:
        return tuple(range(min(4, len(candidates))))
    wrong = [rank for rank in range(len(candidates)) if rank != correct_rank]
    return (correct_rank, *wrong[:3])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0]) + sorted({key for row in rows for key in row}.difference(rows[0]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def materialize_split(
    split: str,
    *,
    model: torch.nn.Module,
    device: str,
    output_dir: Path,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Freeze source/candidate rows and a float16 corridor cache before P0 training."""
    settings = PROTOCOL["splits"][split]
    size = int(settings["size"])
    training = split == "relation_train"
    complete_manifest = output_dir / f"{split}_manifest.json"
    final_corridors = output_dir / f"{split}_corridors.npy"
    final_sources = output_dir / f"{split}_sources.csv"
    final_candidates = output_dir / f"{split}_candidates.csv"
    if all(path.exists() for path in (complete_manifest, final_corridors, final_sources, final_candidates)):
        existing = json.loads(complete_manifest.read_text())
        if existing.get("size") == size and existing.get("seed") == int(settings["seed"]):
            return {**existing, "action": "SKIP_COMPLETE"}
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_dir / f".{split}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, size, batch_size):
        chunk_path = chunk_dir / f"{start:06d}.npz"
        if chunk_path.exists():
            print(f"phase=ENDGAME-MATERIALIZE split={split} scenes={min(start + batch_size, size)}/{size} action=RESUME", flush=True)
            continue
        scenes = [generate_scene(split, index) for index in range(start, min(start + batch_size, size))]
        probabilities, _ = infer_dense(model, np.stack([scene["input"]["model_input"] for scene in scenes]), device=device)
        source_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        crops: list[np.ndarray] = []
        digest_payload = bytearray()
        for scene, probability in zip(scenes, probabilities):
            index = int(scene["input"]["index"])
            digest_payload.extend(index.to_bytes(4, "little"))
            digest_payload.extend(scene_digest(scene))
            proposal = propose_scene(scene, probability)
            candidates = proposal["candidates"]
            selected = selected_training_ranks(proposal["status"], proposal["correct_rank"], candidates) if training else tuple(range(len(candidates)))
            pair_start = len(crops)
            source = proposal["source"]
            for rank in selected:
                candidate = candidates[rank]
                if source is None:
                    raise AssertionError("candidate without source port")
                crop = branch_landing_corridor(
                    scene["input"]["model_input"],
                    probability,
                    source.point_yx,
                    candidate.landing_point_yx,
                    relation_corridor_x=scene["input"]["relation_corridor_x"],
                )
                crops.append(crop.astype(np.float16))
                candidate_rows.append({
                    "split": split,
                    "source_index": index,
                    "pair_index": len(crops) - 1,
                    "candidate_rank": rank,
                    "correct": int(rank == proposal["correct_rank"]),
                    "destination_branch_id": candidate.destination_branch_id,
                    "landing_y": candidate.landing_point_yx[0],
                    "landing_x": candidate.landing_point_yx[1],
                    "geometric_score": candidate.geometric_score,
                    "selected_for_training": int(training),
                })
            source_rows.append({
                "split": split,
                "index": index,
                "stratum": scene["input"]["stratum"],
                "positive": int(proposal["positive"]),
                "status": proposal["status"],
                "candidate_count": len(candidates),
                "correct_candidate_rank": proposal["correct_rank"],
                "pair_start": pair_start,
                "pair_count": len(selected),
                "relation_loss_included": int(bool(selected)),
                })
        crop_array = np.stack(crops).astype(np.float16) if crops else np.empty((0, 6, 32, 64), dtype=np.float16)
        temporary = chunk_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, corridors=crop_array, sources=np.asarray(json.dumps(source_rows)), candidates=np.asarray(json.dumps(candidate_rows)), digest=np.frombuffer(digest_payload, dtype=np.uint8))
        temporary.replace(chunk_path)
        print(f"phase=ENDGAME-MATERIALIZE split={split} scenes={min(start + batch_size, size)}/{size}", flush=True)
    chunks = [chunk_dir / f"{start:06d}.npz" for start in range(0, size, batch_size)]
    if not all(path.exists() for path in chunks):
        raise RuntimeError(f"incomplete materialization chunks for {split}")
    total_crops = sum(len(np.load(path, allow_pickle=False)["corridors"]) for path in chunks)
    crop_array = np.lib.format.open_memmap(final_corridors, mode="w+", dtype=np.float16, shape=(total_crops, 6, 32, 64))
    source_rows = []
    candidate_rows = []
    digest = hashlib.sha256()
    crop_offset = 0
    for path in chunks:
        with np.load(path, allow_pickle=False) as chunk:
            local_crops = chunk["corridors"]
            local_sources = json.loads(str(chunk["sources"]))
            local_candidates = json.loads(str(chunk["candidates"]))
            digest.update(chunk["digest"].tobytes())
            crop_array[crop_offset : crop_offset + len(local_crops)] = local_crops
            for row in local_sources:
                row["pair_start"] = int(row["pair_start"]) + crop_offset
                source_rows.append(row)
            for row in local_candidates:
                row["pair_index"] = int(row["pair_index"]) + crop_offset
                candidate_rows.append(row)
            crop_offset += len(local_crops)
    crop_array.flush()
    del crop_array
    _write_csv(final_sources, source_rows)
    _write_csv(final_candidates, candidate_rows)
    status_counts = {name: sum(row["status"] == name for row in source_rows) for name in (STATUS_PRESENT, STATUS_NONE, STATUS_MISS)}
    manifest = {
        "split": split,
        "size": size,
        "seed": int(settings["seed"]),
        "sha256": digest.hexdigest(),
        "corridors": total_crops,
        "corridor_shape": [total_crops, 6, 32, 64],
        "corridor_dtype": "float16",
        "status_counts": status_counts,
        "candidate_recall": status_counts[STATUS_PRESENT] / max(1, status_counts[STATUS_PRESENT] + status_counts[STATUS_MISS]),
        "relation_scores_opened": False,
    }
    complete_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    shutil.rmtree(chunk_dir)
    return manifest


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


class SourceCorridorDataset(Dataset[dict[str, Any]]):
    """Variable candidate sets with each source represented exactly once."""

    def __init__(self, source_csv: Path, corridor_npy: Path) -> None:
        self.rows = [row for row in read_csv(source_csv) if int(row["pair_count"]) > 0]
        self.corridors = np.load(corridor_npy, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.rows[item]
        start = int(row["pair_start"])
        count = int(row["pair_count"])
        corridors = np.asarray(self.corridors[start : start + count], dtype=np.float32)
        labels = np.zeros(count, dtype=np.float32)
        if row["status"] == STATUS_PRESENT:
            labels[0] = 1.0
        return {"corridors": corridors, "labels": labels, "source_index": int(row["index"]), "positive": row["status"] == STATUS_PRESENT}


def collate_sources(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    maximum = max(len(row["labels"]) for row in rows)
    corridors = np.zeros((len(rows), maximum, 6, 32, 64), dtype=np.float32)
    labels = np.zeros((len(rows), maximum), dtype=np.float32)
    mask = np.zeros((len(rows), maximum), dtype=bool)
    positive = np.zeros(len(rows), dtype=bool)
    for offset, row in enumerate(rows):
        count = len(row["labels"])
        corridors[offset, :count] = row["corridors"]
        labels[offset, :count] = row["labels"]
        mask[offset, :count] = True
        positive[offset] = bool(row["positive"])
    return {
        "corridors": torch.from_numpy(corridors),
        "labels": torch.from_numpy(labels),
        "mask": torch.from_numpy(mask),
        "positive": torch.from_numpy(positive),
    }
