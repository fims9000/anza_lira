"""Small provenance helpers shared by the deadline orchestrator and validator."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


ALLOWED_VERDICTS = (
    "DEADLINE_RESULT_READY",
    "DEADLINE_RESULT_READY_WITH_NEGATIVE_MECHANISM",
    "BLOCKED_EVALUATOR",
    "BLOCKED_THRESHOLD_FREEZE",
    "BLOCKED_EXPERT_EVALUATION",
)
READY_VERDICTS = ALLOWED_VERDICTS[:2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def finite_json(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_json(item) for item in value)
    return True


def file_records(paths: Iterable[Path], *, base: Path | None = None) -> list[dict[str, Any]]:
    records = []
    for raw_path in sorted({Path(path) for path in paths}):
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        rendered = str(raw_path.relative_to(base)) if base is not None else str(raw_path)
        records.append(
            {
                "path": rendered,
                "size_bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
            }
        )
    return records


def verify_file_records(records: Iterable[dict[str, Any]], *, base: Path | None = None) -> bool:
    for record in records:
        path = Path(record["path"])
        if base is not None:
            path = base / path
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            return False
    return True


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    if not finite_json(payload):
        raise ValueError(f"Non-finite JSON value: {path}")
    return payload
