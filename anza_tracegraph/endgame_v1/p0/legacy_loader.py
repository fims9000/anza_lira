"""Load, do not reimplement, the exact historical strongest corridor P0."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

from path_completion.pair_classifier import EndpointPairClassifier


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "path_completion/pair_classifier.py"
ORIGINAL_CHECKPOINT = ROOT / "results/path_completion/pair_classifier/checkpoint.pt"


def source_sha256() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def build_exact_p0() -> EndpointPairClassifier:
    return EndpointPairClassifier()


def architecture_receipt() -> dict[str, Any]:
    model = build_exact_p0()
    return {
        "class_name": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "source_file": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_sha256(),
        "class_source_sha256": hashlib.sha256(inspect.getsource(EndpointPairClassifier).encode()).hexdigest(),
        "input_channels": 6,
        "layers": [str(layer) for layer in model.encoder] + [str(layer) for layer in model.head],
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "original_checkpoint": str(ORIGINAL_CHECKPOINT.relative_to(ROOT)) if ORIGINAL_CHECKPOINT.exists() else None,
        "original_checkpoint_sha256": hashlib.sha256(ORIGINAL_CHECKPOINT.read_bytes()).hexdigest() if ORIGINAL_CHECKPOINT.exists() else None,
        "architecture_reimplemented": False,
    }
