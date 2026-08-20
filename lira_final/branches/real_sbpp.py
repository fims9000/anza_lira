"""Real-image adaptation of frozen SBPP V3-B without truth-based ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from anza_tracegraph.ports_v3.branches import Branch, extract_branches
from anza_tracegraph.ports_v3.candidates import select_source_port, source_ports
from anza_tracegraph.ports_v3_b.candidates import propose_cluster_candidates
from anza_tracegraph.ports_v3_b.clustering import BranchCluster, cluster_branches
from anza_tracegraph.ports_v3_b.soft_branches import SoftBranch, extract_soft_branches
from datasets.cracks import load_section_image
from lira_final.data.natural_gaps import NaturalGap
from lira_final.dense.ensemble import load_probability


@dataclass(frozen=True)
class ProposedCandidate:
    rank: int
    destination_branch_id: int
    landing_yx: tuple[float, float]
    distance: float
    geometric_score: float
    mean_probability: float
    correct: bool


def _members_match(cluster: BranchCluster, target: np.ndarray, landing: tuple[float, float], landing_band: float) -> bool:
    target = np.asarray(target, dtype=np.float64)
    if len(target) == 0:
        return False
    if float(np.min(np.linalg.norm(target - np.asarray(landing), axis=1))) <= landing_band:
        return True
    for member in cluster.members:
        distances = np.min(np.linalg.norm(member.points_yx[:, None, :] - target[None, :, :], axis=2), axis=1)
        if float(np.mean(distances <= 3.0)) >= 0.35:
            return True
    return False


def _soft_bank(probability: np.ndarray, image_scalar: np.ndarray, hard_mask: np.ndarray, source, ratios: tuple[float, ...], hard_threshold: float) -> tuple[SoftBranch, ...]:
    output = []
    for ratio in ratios:
        for branch in extract_soft_branches(probability, image_scalar, hard_mask, source, tau_s=ratio * hard_threshold):
            output.append(SoftBranch(
                branch_id=len(output), points_yx=branch.points_yx, mean_probability=branch.mean_probability,
                mean_contrast=branch.mean_contrast, orientation_coherence=branch.orientation_coherence,
                hysteresis_rule=f"real_bank_{ratio:.1f}",
            ))
    return tuple(output)


def propose_for_gap(gap: NaturalGap, probability: np.ndarray, image: np.ndarray, hard_threshold: float, *, landing_band: float = 12.0, k: int = 12) -> dict[str, object]:
    image_scalar = np.asarray(image, dtype=np.float32).mean(axis=-1)
    hard_mask = probability >= hard_threshold
    hard = extract_branches(hard_mask, probability, image_scalar, tau_micro=0.4 * hard_threshold)
    ports = source_ports(hard, probability)
    source = select_source_port(ports, gap.source_yx, gap.source_tangent_yx)
    if source is None or float(np.linalg.norm(np.asarray(source.point_yx) - np.asarray(gap.source_yx))) > 8.0:
        return {"gap_id": gap.gap_id, "source_available": False, "candidate_count": 0, "candidate_recalled": False, "candidates": []}
    soft = _soft_bank(probability, image_scalar, hard_mask, source, (0.4, 0.6, 0.8), hard_threshold)
    clusters = cluster_branches(hard, soft)
    by_id = {cluster.cluster_id: cluster for cluster in clusters}
    proposed = propose_cluster_candidates(source, clusters)[:k]
    candidates = []
    target = np.asarray(gap.destination_context_yx, dtype=np.float64)
    for rank, candidate in enumerate(proposed, 1):
        correct = _members_match(by_id[candidate.destination_branch_id], target, candidate.landing_point_yx, landing_band)
        candidates.append(ProposedCandidate(rank, candidate.destination_branch_id, candidate.landing_point_yx, candidate.distance, candidate.geometric_score, candidate.mean_probability, correct))
    return {
        "gap_id": gap.gap_id,
        "source_available": True,
        "source_yx": list(source.point_yx),
        "candidate_count": len(candidates),
        "candidate_recalled": any(row.correct for row in candidates),
        "candidates": [asdict(row) for row in candidates],
    }


def evaluate_split(gaps: tuple[NaturalGap, ...], cache_root: Path, hard_threshold: float, *, landing_band: float = 12.0) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    loaded: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, gap in enumerate(gaps):
        if gap.section_id not in loaded:
            loaded[gap.section_id] = (
                load_probability(cache_root, gap.section_id),
                load_section_image(Path("data/cracks/images") / f"section_{gap.section_id:03d}.png"),
            )
        probability, image = loaded[gap.section_id]
        row = propose_for_gap(gap, probability, image, hard_threshold, landing_band=landing_band)
        rows.append({"section_id": gap.section_id, "annotator": gap.annotator, "trace_id": gap.trace_id, "gap_length": gap.length_px, **row})
        if (index + 1) % 100 == 0 or index + 1 == len(gaps):
            print(f"phase=F2_CANDIDATE gap={index + 1}/{len(gaps)}", flush=True)
    counts = [int(row["candidate_count"]) for row in rows]
    summary = {
        "positive_gaps": len(rows),
        "source_available": sum(bool(row["source_available"]) for row in rows),
        "candidate_recalled": sum(bool(row["candidate_recalled"]) for row in rows),
        "candidate_recall": float(np.mean([bool(row["candidate_recalled"]) for row in rows])) if rows else 0.0,
        "median_candidates": float(np.median(counts)) if counts else 0.0,
        "p95_candidates": float(np.quantile(counts, 0.95)) if counts else 0.0,
        "landing_band": float(landing_band),
        "k": 12,
    }
    return summary, rows

