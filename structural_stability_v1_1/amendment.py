"""Parent/amendment provenance and byte-integrity helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from structural_stability_v1_1.protocol import AMENDMENT_SOURCE, PARENT_RESULT_ROOT, PROTOCOL, ROOT, protocol_hash


OLD_STOP_PATHS = (
    ROOT / "results/lira_final/f1_gap_audit/validator.json",
    ROOT / "results/lira_intervention_final/i2_candidate/validator.json",
    ROOT / "results/lira_graph_cut_v2/benchmark/retention.json",
    ROOT / "results/lira_h1/final/ANZA_LIRA_H1_MASTER_RESULT.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def parent_integrity_snapshot() -> dict[str, Any]:
    parent = tree_manifest(PARENT_RESULT_ROOT)
    old_stops = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in OLD_STOP_PATHS}
    return {
        "parent_root": str(PARENT_RESULT_ROOT),
        "parent_files": parent,
        "parent_file_count": len(parent),
        "parent_tree_sha256": hashlib.sha256(
            json.dumps(parent, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "old_stop_files": old_stops,
    }


def amendment_payload() -> dict[str, Any]:
    if not AMENDMENT_SOURCE.is_file():
        raise FileNotFoundError(f"V1.1 amendment source missing: {AMENDMENT_SOURCE}")
    return {
        "status": "AMENDMENT_FROZEN_BEFORE_TRAINING",
        "protocol": PROTOCOL,
        "protocol_sha256": protocol_hash(),
        "source_path": str(AMENDMENT_SOURCE),
        "source_sha256": sha256_file(AMENDMENT_SOURCE),
        "parent_ss1_status": json.loads(
            (PARENT_RESULT_ROOT / "ANZA_LIRA_SS_V1_MASTER_RESULT.json").read_text()
        )["status"],
        "new_training_started_before_amendment": False,
        "B0_B1_B2_B3_opened_before_amendment": False,
    }
