"""Score cached corridors without regenerating or relabeling candidates."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..p0.dataset import read_csv


def score_split(model: torch.nn.Module, cache_dir: Path, split: str, *, device: str, batch_size: int = 256) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = [dict(row) for row in read_csv(cache_dir / f"{split}_sources.csv")]
    candidates: list[dict[str, Any]] = [dict(row) for row in read_csv(cache_dir / f"{split}_candidates.csv")]
    corridors = np.load(cache_dir / f"{split}_corridors.npy", mmap_mode="r")
    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(corridors), batch_size):
            batch = torch.from_numpy(np.asarray(corridors[start : start + batch_size], dtype=np.float32)).to(device)
            scores.extend(torch.sigmoid(model(batch)).cpu().tolist())
    if len(scores) != len(candidates):
        raise AssertionError("candidate/corridor score cardinality mismatch")
    for row, score in zip(candidates, scores):
        row["score"] = float(score)
    return sources, candidates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0]) + sorted({key for row in rows for key in row}.difference(rows[0]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
