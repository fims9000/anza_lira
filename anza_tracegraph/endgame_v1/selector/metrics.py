"""Source-level system metrics with explicit frozen denominators."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from ..p0.dataset import STATUS_MISS, STATUS_NONE, STATUS_PRESENT


def source_decisions(
    sources: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[int(candidate["source_index"])].append(candidate)
    output = []
    for source in sources:
        index = int(source["index"])
        local = grouped.get(index, [])
        top = max(local, key=lambda row: (float(row["score"]), -int(row["candidate_rank"]))) if local else None
        accepted = bool(top is not None and float(top["score"]) >= threshold)
        top_correct = bool(top is not None and int(top["correct"]) == 1)
        status = str(source["status"])
        output.append({
            "split": source["split"],
            "index": index,
            "stratum": source["stratum"],
            "status": status,
            "positive": int(source["positive"]),
            "candidate_count": int(source["candidate_count"]),
            "top_candidate_rank": -1 if top is None else int(top["candidate_rank"]),
            "top_score": "" if top is None else float(top["score"]),
            "accepted": int(accepted),
            "selected_none": int(not accepted),
            "top_correct": int(top_correct),
            "correct_accepted": int(status == STATUS_PRESENT and accepted and top_correct),
            "wrong_branch": int(status == STATUS_PRESENT and accepted and not top_correct),
            "false_bridge": int(status == STATUS_NONE and accepted),
            "candidate_miss_accepted": int(status == STATUS_MISS and accepted),
        })
    return output


def _ratio(rows: list[dict[str, Any]], numerator: str, statuses: tuple[str, ...]) -> float:
    local = [row for row in rows if row["status"] in statuses]
    return float(sum(int(row[numerator]) for row in local) / len(local)) if local else float("nan")


def relation_metrics(decisions: list[dict[str, Any]]) -> dict[str, float | int]:
    present = [row for row in decisions if row["status"] == STATUS_PRESENT]
    positives = [row for row in decisions if int(row["positive"]) == 1]
    none = [row for row in decisions if row["status"] == STATUS_NONE]
    return {
        "sources": len(decisions),
        "positive_sources": len(positives),
        "candidate_available_positives": len(present),
        "candidate_miss_positives": sum(row["status"] == STATUS_MISS for row in decisions),
        "none_sources": len(none),
        "correct_accepted": sum(int(row["correct_accepted"]) for row in decisions),
        "CCR": _ratio(decisions, "correct_accepted", (STATUS_PRESENT,)),
        "RelationRecovery": float(sum(int(row["correct_accepted"]) for row in positives) / len(positives)) if positives else float("nan"),
        "FalseBridge": _ratio(decisions, "false_bridge", (STATUS_NONE,)),
        "WrongBranch": _ratio(decisions, "wrong_branch", (STATUS_PRESENT,)),
        "NONERecall": float(sum(int(row["selected_none"]) for row in none) / len(none)) if none else float("nan"),
        "top1_candidate_accuracy": float(sum(int(row["top_correct"]) for row in present) / len(present)) if present else float("nan"),
        "candidate_miss_accept_rate": _ratio(decisions, "candidate_miss_accepted", (STATUS_MISS,)),
    }


def secondary_pair_metrics(candidates: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> dict[str, float]:
    labels = np.asarray([int(row["correct"]) for row in candidates], dtype=int)
    scores = np.asarray([float(row["score"]) for row in candidates], dtype=float)
    if len(np.unique(labels)) < 2:
        return {key: float("nan") for key in ("AUROC", "balanced_AUPRC", "TPR_at_FPR_0_05", "low_FPR_pAUC", "Brier", "ECE", "pair_ranking")}
    positives = max(1, int(labels.sum()))
    negatives = max(1, int((labels == 0).sum()))
    weights = np.where(labels == 1, 0.5 / positives, 0.5 / negatives)
    fpr, tpr, _ = roc_curve(labels, scores)
    allowed = tpr[fpr <= 0.05]
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        selected = (scores >= left) & (scores < right if right < 1.0 else scores <= right)
        if selected.any():
            ece += float(selected.mean()) * abs(float(scores[selected].mean()) - float(labels[selected].mean()))
    present = [row for row in decisions if row["status"] == STATUS_PRESENT]
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "balanced_AUPRC": float(average_precision_score(labels, scores, sample_weight=weights)),
        "TPR_at_FPR_0_05": float(allowed.max()) if len(allowed) else 0.0,
        "low_FPR_pAUC": float(roc_auc_score(labels, scores, max_fpr=0.05)),
        "Brier": float(brier_score_loss(labels, scores)),
        "ECE": float(ece),
        "pair_ranking": float(sum(int(row["top_correct"]) for row in present) / len(present)) if present else float("nan"),
    }


def bootstrap_source_metrics(decisions: list[dict[str, Any]], *, resamples: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    size = len(decisions)
    fields = ("CCR", "RelationRecovery", "FalseBridge", "WrongBranch")
    values = {field: np.empty(resamples, dtype=np.float64) for field in fields}
    for iteration in range(resamples):
        sample = [decisions[index] for index in rng.integers(0, size, size=size)]
        metrics = relation_metrics(sample)
        for field in fields:
            values[field][iteration] = float(metrics[field])
    return {
        "unit": "source_scene",
        "resamples": resamples,
        "seed": seed,
        "intervals": {
            field: {
                "estimate": float(relation_metrics(decisions)[field]),
                "low": float(np.nanquantile(array, 0.025)),
                "high": float(np.nanquantile(array, 0.975)),
            }
            for field, array in values.items()
        },
    }
