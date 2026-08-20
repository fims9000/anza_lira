#!/usr/bin/env python3
"""Precompute reproducible CRACKS crowd-fused targets without expert labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.cracks import POLICIES, fuse_crowd_masks, load_rgb_mask
from scripts.audit_cracks_archives import DATA_ROOT, write_json
from scripts.prepare_cracks_protocol import V2_RESULT_ROOT, canonical_hash


DEFAULT_PROTOCOL = V2_RESULT_ROOT / "protocol.json"
ANNOTATION_ROOT = DATA_ROOT / "annotations"
TARGET_ROOT = DATA_ROOT / "crowd_targets"


def _mask_path(annotator: str, section_id: int) -> Path:
    return ANNOTATION_ROOT / annotator / f"section_{section_id:03d}.png"


def _write_target(path: Path, payload: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        target=payload["target"].astype(np.float16),
        valid=payload["valid"].astype(np.uint8),
        weight_sum=payload["weight_sum"].astype(np.float16),
        support=payload["support"].astype(np.uint8),
        human_entropy=payload["human_entropy"].astype(np.float16),
        human_entropy_valid=payload["human_entropy_valid"].astype(np.uint8),
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_existing_target(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path) as payload:
            required = {
                "target",
                "valid",
                "weight_sum",
                "support",
                "human_entropy",
                "human_entropy_valid",
            }
            if not required.issubset(payload.files) or payload["target"].shape != (255, 701):
                return None
            return {name: payload[name] for name in required}
    except (OSError, ValueError, KeyError):
        return None


def build_split_targets(
    section_ids: Sequence[int],
    annotators: Sequence[str],
    *,
    split_name: str,
    policy_name: str,
    output_root: Path = TARGET_ROOT,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    if "expert" in annotators:
        raise ValueError("Expert annotations are forbidden in crowd target construction")
    output_dir = output_root / policy_name / split_name
    section_records: list[dict[str, Any]] = []
    valid_pixels = 0
    total_pixels = 0
    disagreement_pixels = 0
    reused_sections = 0
    written_sections = 0
    for section_id in section_ids:
        available = [(name, _mask_path(name, int(section_id))) for name in annotators]
        available = [(name, path) for name, path in available if path.is_file()]
        if not available:
            continue
        output_path = output_dir / f"section_{int(section_id):03d}.npz"
        fused = _load_existing_target(output_path) if reuse_existing else None
        if fused is None:
            fused = fuse_crowd_masks(
                [load_rgb_mask(path) for _, path in available],
                [name for name, _ in available],
                policy_name,
            )
            digest = _write_target(output_path, fused)
            written_sections += 1
        else:
            digest = _target_digest(output_path)
            reused_sections += 1
        valid_pixels += int(fused["valid"].sum())
        total_pixels += int(fused["valid"].size)
        disagreement_pixels += int(fused["human_entropy_valid"].sum())
        section_records.append(
            {
                "section_id": int(section_id),
                "annotator_count": len(available),
                "sha256": digest,
            }
        )
    return {
        "split": split_name,
        "policy": policy_name,
        "annotators": list(annotators),
        "section_count": len(section_records),
        "sections": section_records,
        "reused_sections": reused_sections,
        "written_sections": written_sections,
        "valid_pixel_fraction": valid_pixels / total_pixels if total_pixels else 0.0,
        "disagreement_supported_fraction": disagreement_pixels / total_pixels if total_pixels else 0.0,
    }


def build_all(protocol_path: Path = DEFAULT_PROTOCOL, output_root: Path = TARGET_ROOT) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    setting = protocol["setting_a"]
    specs = {
        "train": (setting["training_section_ids"], setting["training_annotators"]),
        "heldout": (setting["held_out_validation_section_ids"], setting["held_out_annotators"]["all"]),
    }
    config = {
        "protocol_sha256": protocol["sha256"],
        "policies": sorted(POLICIES),
        "splits": {name: {"section_ids": ids, "annotators": annotators} for name, (ids, annotators) in specs.items()},
    }
    manifest_path = V2_RESULT_ROOT / "cracks" / "crowd_target" / "manifest.json"
    config_hash = canonical_hash(config)
    previous: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
    if previous.get("status") == "COMPLETE" and previous.get("config_sha256") == config_hash:
        expected = [
            output_root / policy / split / f"section_{int(section):03d}.npz"
            for policy in sorted(POLICIES)
            for split, (sections, _annotators) in specs.items()
            for section in sections
        ]
        if all(path.is_file() for path in expected):
            previous["resume_action"] = "SKIP_COMPLETE_SAME_HASH"
            return previous
    reuse_existing = previous.get("config_sha256") == config_hash
    write_json(
        manifest_path,
        {
            "status": "IN_PROGRESS",
            "scientific_result": False,
            "config_sha256": config_hash,
            "resume_action": "RESUME" if reuse_existing else "NEW_OR_CHANGED_CONFIG",
        },
    )
    records = []
    for policy_name in sorted(POLICIES):
        for split_name, (section_ids, annotators) in specs.items():
            records.append(
                build_split_targets(
                    section_ids,
                    annotators,
                    split_name=split_name,
                    policy_name=policy_name,
                    output_root=output_root,
                    reuse_existing=reuse_existing,
                )
            )
    manifest = {
        "status": "COMPLETE",
        "scientific_result": False,
        "config_sha256": config_hash,
        "records": records,
        "resume_action": "RESUMED" if reuse_existing else "BUILT",
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=TARGET_ROOT)
    args = parser.parse_args()
    manifest = build_all(args.protocol, args.output_root)
    print("CRACKS CROWD TARGETS: COMPLETE")
    for record in manifest["records"]:
        print(
            f"{record['policy']} {record['split']}: sections={record['section_count']} "
            f"valid={record['valid_pixel_fraction']:.4f} support5={record['disagreement_supported_fraction']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
