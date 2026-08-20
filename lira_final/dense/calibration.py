"""Expert-blind dense threshold calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cracks_experiment.evaluation import hard_cldice
from cracks_experiment.partial_labels import map_partial_annotation
from datasets.cracks import load_rgb_mask
from lira_final.dense.ensemble import load_probability
from lira_final.protocol import HELDOUT_ANNOTATORS, PROTOCOL


def calibrate_dense_threshold(cache_root: Path, annotation_root: Path, section_ids: list[int]) -> dict[str, object]:
    sweep = []
    for threshold in PROTOCOL["dense"]["threshold_candidates"]:
        section_values = []
        counts = {"tp": 0, "fp": 0, "fn": 0}
        for section_id in section_ids:
            probability = load_probability(cache_root, section_id)
            for annotator in HELDOUT_ANNOTATORS:
                path = Path(annotation_root) / annotator / f"section_{section_id:03d}.png"
                if not path.is_file():
                    continue
                target, weight = map_partial_annotation(load_rgb_mask(path))
                valid = weight > 0
                prediction = probability >= float(threshold)
                truth = target >= 0.5
                counts["tp"] += int(np.count_nonzero(prediction & truth & valid))
                counts["fp"] += int(np.count_nonzero(prediction & ~truth & valid))
                counts["fn"] += int(np.count_nonzero(~prediction & truth & valid))
                section_values.append(hard_cldice(prediction & valid, truth & valid))
        precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
        sweep.append({"threshold": float(threshold), "precision": float(precision), "recall": float(recall), "mean_cldice": float(np.mean(section_values)), **counts})
    feasible = [row for row in sweep if row["precision"] >= 0.75]
    choice = max(feasible or sweep, key=lambda row: (row["mean_cldice"], -row["threshold"]))
    return {"selected_threshold": choice["threshold"], "precision_constraint_feasible": bool(feasible), "selection_split": "dense_calibration", "selection_rule": PROTOCOL["dense"]["threshold_selection"], "selected": choice, "sweep": sweep, "expert_accessed": False}

