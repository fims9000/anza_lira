#!/usr/bin/env python3
"""List local agent skill metadata without loading skill bodies."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


FIELD_RE = re.compile(r"^(name|description):\s*(.+?)\s*$")


def read_metadata(path: Path) -> tuple[str, str]:
    fields: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        if handle.readline().strip() != "---":
            raise ValueError(f"{path}: missing YAML frontmatter")
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.strip() == "---":
                break
            match = FIELD_RE.match(line)
            if match:
                fields[match.group(1)] = match.group(2).strip('"\'')
    if not fields.get("name") or not fields.get("description"):
        raise ValueError(f"{path}: name and description are required")
    return fields["name"], fields["description"]


def list_skills(root: Path) -> list[tuple[str, str, Path]]:
    rows = []
    for path in sorted(root.glob("*/SKILL.md")):
        name, description = read_metadata(path)
        rows.append((name, description, path))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".agents/skills"))
    args = parser.parse_args()
    rows = list_skills(args.root)
    if not rows:
        raise SystemExit(f"No skills found under {args.root}")
    for name, description, path in rows:
        print(f"{name}\t{description}\t{path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
