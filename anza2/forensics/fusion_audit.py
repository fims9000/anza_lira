"""Raw ANZA, generic, and frozen fused-affinity diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from anza2.eval.low_fpr import low_fpr_metrics


def distribution_summary(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: None for key in ("min", "q10", "q25", "median", "q75", "q90", "max", "mean")}
    return {
        "min": float(array.min()), "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)), "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)), "q90": float(np.quantile(array, 0.90)),
        "max": float(array.max()), "mean": float(array.mean()),
    }


def summarize_fusion(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    pooled: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"positive": [], "negative": []} for name in ("raw_anza", "generic", "fused")
    }
    generic_for_corr, anza_for_corr, effective_terms, generic_logits = [], [], [], []
    changed = changed_correct = changed_incorrect = eligible = 0
    for record in records:
        positive = record["positive"]
        negative = record["negative"]
        for name in pooled:
            scores = record[name]
            pooled[name]["positive"].append(scores[positive])
            pooled[name]["negative"].append(scores[negative])
            for label, mask in (("positive", positive), ("negative", negative)):
                rows.append({
                    "seed": record["seed"], "sample_index": record["sample_index"],
                    "case": record["case"], "source": name, "label": label,
                    **distribution_summary(scores[mask]),
                })
        valid = positive | negative
        generic_for_corr.append(record["generic"][valid]); anza_for_corr.append(record["raw_anza"][valid])
        effective_terms.append(record["effective_term"][valid]); generic_logits.append(record["generic_logits"][valid])
        if positive.any() and negative.any():
            eligible += 1
            generic_correct = float(record["generic"][positive].mean()) > float(record["generic"][negative].mean())
            fused_correct = float(record["fused"][positive].mean()) > float(record["fused"][negative].mean())
            if generic_correct != fused_correct:
                changed += 1
                changed_correct += int(fused_correct)
                changed_incorrect += int(not fused_correct)

    summary: dict[str, Any] = {"sources": {}}
    for name, groups in pooled.items():
        positive = np.concatenate(groups["positive"]); negative = np.concatenate(groups["negative"])
        summary["sources"][name] = {
            "low_fpr": low_fpr_metrics(positive, negative),
            "positive_distribution": distribution_summary(positive),
            "negative_distribution": distribution_summary(negative),
            "distribution_overlap_fraction": float(np.mean(positive <= np.quantile(negative, 0.90))),
        }
    generic_array = np.concatenate(generic_for_corr); anza_array = np.concatenate(anza_for_corr)
    effective = np.concatenate(effective_terms); logits = np.concatenate(generic_logits)
    summary.update({
        "generic_anza_correlation": float(np.corrcoef(generic_array, anza_array)[0, 1]),
        "candidate_ordering_samples": eligible,
        "fraction_candidate_orderings_changed": changed / max(eligible, 1),
        "fraction_changed_correct_direction": changed_correct / max(changed, 1),
        "fraction_changed_incorrect_direction": changed_incorrect / max(changed, 1),
        "effective_term_abs_median": float(np.median(np.abs(effective))),
        "generic_logit_abs_median": float(np.median(np.abs(logits))),
        "effective_to_generic_abs_median_ratio": float(
            np.median(np.abs(effective)) / max(np.median(np.abs(logits)), 1e-12)
        ),
    })
    return rows, summary
