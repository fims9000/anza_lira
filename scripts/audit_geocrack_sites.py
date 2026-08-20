#!/usr/bin/env python3
"""Audit explicit official GeoCrack source-to-site metadata without filename inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def audit_site_mapping(
    source_image_ids: Iterable[str],
    official_mapping: Path | None,
    *,
    output_path: Path | None = None,
) -> dict:
    sources = sorted(set(source_image_ids))
    mapping: dict[str, str] = {}
    if official_mapping is not None:
        with official_mapping.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {"source_image_id", "geological_site"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"Official site mapping must contain exact columns {sorted(required)}")
        for row in rows:
            source, site = row["source_image_id"].strip(), row["geological_site"].strip()
            if not source or not site:
                raise ValueError("Official site mapping contains an empty source or site")
            if source in mapping and mapping[source] != site:
                raise ValueError(f"Ambiguous official geological_site for {source}")
            mapping[source] = site
    missing = sorted(set(sources) - set(mapping))
    established = bool(official_mapping is not None and not missing)
    payload = {
        "site_mapping_status": "ESTABLISHED" if established else "NOT_ESTABLISHED",
        "inference_from_filename": False,
        "official_mapping_path": str(official_mapping.resolve()) if official_mapping else None,
        "source_image_count": len(sources),
        "mapped_source_count": len(set(sources) & set(mapping)),
        "missing_source_image_ids": missing,
        "mapping": {source: mapping[source] for source in sources if source in mapping},
        "fallback_split_unit": "source_image_id" if not established else "geological_site",
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-list", type=Path, required=True, help="Text file with one source_image_id per line")
    parser.add_argument("--official-mapping", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/geocrack/splits/site_mapping_audit.json"))
    args = parser.parse_args()
    sources = [line.strip() for line in args.source_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = audit_site_mapping(sources, args.official_mapping, output_path=args.output)
    print(f"SITE MAPPING STATUS: {payload['site_mapping_status']}")
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
