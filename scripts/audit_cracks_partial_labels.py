#!/usr/bin/env python3
"""Audit frozen T1 CRACKS semantics without opening expert annotations."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cracks_experiment.partial_labels import audit_nonexpert_annotations


def main() -> None:
    protocol = json.loads((PROJECT_ROOT / "results" / "anza_v2_study" / "protocol.json").read_text())
    setting = protocol["setting_a"]
    annotators = [*setting["training_annotators"], *setting["held_out_annotators"]["all"]]
    sections = sorted(set(setting["training_section_ids"]) | set(setting["held_out_validation_section_ids"]))
    result = audit_nonexpert_annotations(
        PROJECT_ROOT / "data" / "cracks" / "annotations",
        annotators,
        sections,
    )
    result["training_annotators"] = setting["training_annotators"]
    result["held_out_annotators"] = setting["held_out_annotators"]["all"]
    result["annotator_split_disjoint"] = not bool(
        set(setting["training_annotators"]) & set(setting["held_out_annotators"]["all"])
    )
    output = PROJECT_ROOT / "results" / "final_practical_cycle" / "cracks_t1"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "audit.json"
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("Frozen T1 palette audit drift")
    path.write_text(encoded)
    print(
        f"CRACKS_T1_AUDIT={result['status']} files={result['annotation_files']} "
        f"explicit_fraction={result['palette']['explicit_fraction']:.8f} expert=LOCKED"
    )


if __name__ == "__main__":
    main()
