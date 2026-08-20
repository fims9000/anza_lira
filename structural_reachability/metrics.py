"""Low-FPR and section-clustered statistics for structural relations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def evaluate_low_fpr_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    pair_ids: np.ndarray | None = None,
    fpr_max: float = 0.05,
) -> dict[str, Any]:
    """Evaluate discrimination with the primary operating region fixed in advance."""

    truth = np.asarray(labels, dtype=np.int8)
    value = np.asarray(scores, dtype=np.float64)
    if truth.shape != value.shape or truth.ndim != 1 or len(truth) == 0:
        raise ValueError("labels and scores must be same-length nonempty vectors")
    if not np.isfinite(value).all() or set(np.unique(truth)) != {0, 1}:
        raise ValueError("finite scores and both binary classes are required")
    if not 0 < float(fpr_max) <= 1:
        raise ValueError("fpr_max must lie in (0, 1]")
    fpr, tpr, thresholds = roc_curve(truth, value, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= float(fpr_max) + 1e-12)
    best = int(eligible[np.argmax(tpr[eligible])])
    if fpr[best] < fpr_max and best + 1 < len(fpr):
        right = best + 1
        weight = (fpr_max - fpr[best]) / max(fpr[right] - fpr[best], 1e-12)
        boundary_tpr = float(tpr[best] + weight * (tpr[right] - tpr[best]))
    else:
        boundary_tpr = float(tpr[best])
    inside = fpr < fpr_max
    area_fpr = np.concatenate((fpr[inside], np.asarray([fpr_max])))
    area_tpr = np.concatenate((tpr[inside], np.asarray([boundary_tpr])))
    partial_auc = float(np.trapezoid(area_tpr, area_fpr) / fpr_max)
    ranking = float("nan")
    pair_count = 0
    if pair_ids is not None:
        pair_array = np.asarray(pair_ids)
        wins = []
        for pair_id in np.unique(pair_array):
            selected = pair_array == pair_id
            positives = value[selected & (truth == 1)]
            negatives = value[selected & (truth == 0)]
            if len(positives) != 1 or len(negatives) != 1:
                raise ValueError("each matched pair must contain one positive and one negative")
            wins.append(float(positives[0] > negatives[0]) + 0.5 * float(positives[0] == negatives[0]))
        pair_count = len(wins)
        ranking = float(np.mean(wins))
    return {
        "sample_count": int(len(truth)),
        "positive_count": int(np.count_nonzero(truth == 1)),
        "negative_count": int(np.count_nonzero(truth == 0)),
        "fpr_max": float(fpr_max),
        "tpr_at_fpr_max": float(tpr[best]),
        "achieved_fpr": float(fpr[best]),
        "threshold": float(thresholds[best]),
        "low_fpr_partial_auc_normalized": partial_auc,
        "auroc_secondary": float(roc_auc_score(truth, value)),
        "matched_pair_count": int(pair_count),
        "matched_ranking_probability": ranking,
        "curve": [
            {"fpr": float(x), "tpr": float(y), "threshold": float(z)}
            for x, y, z in zip(fpr, tpr, thresholds)
        ],
    }


def _metric_for_rows(rows: list[Mapping[str, Any]], relation: str, seed: int, fpr_max: float) -> dict[str, Any]:
    selected = [row for row in rows if row["relation"] == relation and int(row["seed"]) == int(seed)]
    return evaluate_low_fpr_curve(
        np.asarray([int(row["label"]) for row in selected]),
        np.asarray([float(row["score"]) for row in selected]),
        pair_ids=np.asarray([str(row["pair_id"]) for row in selected]),
        fpr_max=fpr_max,
    )


def section_paired_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    *,
    candidate_relation: str,
    baseline_relation: str,
    seeds: tuple[int, ...],
    metric: str,
    resamples: int = 10_000,
    random_seed: int = 20260818,
    fpr_max: float = 0.05,
) -> dict[str, Any]:
    """Paired bootstrap by section; candidate rows are never treated as independent."""

    materialized = [
        row for row in rows if row["relation"] in {candidate_relation, baseline_relation}
    ]
    sections = sorted({int(row["section_id"]) for row in materialized})
    if len(sections) < 2 or resamples < 1:
        raise ValueError("at least two sections and one resample are required")

    by_section = {
        section_id: [row for row in materialized if int(row["section_id"]) == section_id]
        for section_id in sections
    }

    def difference(sampled_sections: list[int]) -> float:
        sampled: list[Mapping[str, Any]] = []
        for draw, section_id in enumerate(sampled_sections):
            for row in by_section[section_id]:
                copy = dict(row)
                copy["pair_id"] = f"draw{draw}:{copy['pair_id']}"
                sampled.append(copy)
        candidate = np.mean([
            float(_metric_for_rows(sampled, candidate_relation, seed, fpr_max)[metric]) for seed in seeds
        ])
        baseline = np.mean([
            float(_metric_for_rows(sampled, baseline_relation, seed, fpr_max)[metric]) for seed in seeds
        ])
        return float(candidate - baseline)

    point = difference(sections)
    rng = np.random.default_rng(int(random_seed))
    samples = np.empty(int(resamples), dtype=np.float64)
    section_array = np.asarray(sections, dtype=np.int64)
    for index in range(int(resamples)):
        samples[index] = difference(rng.choice(section_array, size=len(section_array), replace=True).tolist())
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "metric": metric,
        "candidate_relation": candidate_relation,
        "baseline_relation": baseline_relation,
        "section_count": len(sections),
        "seed_count": len(seeds),
        "resamples": int(resamples),
        "random_seed": int(random_seed),
        "point_delta": point,
        "ci95": [float(low), float(high)],
        "resampling_unit": "section_id",
        "seed_aggregation": "metric_mean_within_bootstrap_draw",
    }
