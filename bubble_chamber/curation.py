"""Build ordered 2D track groups ONLY from validated human pair decisions.

No inferred path is added across a photographic gap. All 3D reconstruction,
physical start/stop, particle identity and ionization must remain unavailable.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any
import numpy as np

from .pilot import fit_quadratic
from .review import validate_labels, pair_key, result_digest


class ReviewConflictError(ValueError):
    pass


def curate_tracks(result: dict[str, Any], reviewed: dict[str, Any]) -> dict[str, Any]:
    counts = validate_labels(result, reviewed)
    segments = {int(s["segment_id"]): s for s in result["segments"]}
    if len(segments) != len(result["segments"]):
        raise ReviewConflictError("duplicate source segment IDs")
    candidates = {pair_key(p): p for p in result["join_proposals"]}
    approved = []
    rejected = []
    for label in reviewed["labels"]:
        if label["label"] == "same_track":
            approved.append(candidates[label["pair_key"]])
        if label["label"] == "different_track":
            rejected.append(candidates[label["pair_key"]])
    # Exactly one continuity relation per geometrical fragment endpoint.
    endpoint_links = {}
    parent = {sid: sid for sid in segments}

    def find(a: int) -> int:
        while parent[a] != a:
            a = parent[a]
        return a

    for edge in approved:
        a, sa, b, sb = int(edge["segment_A"]), edge["endpoint_A"], int(edge["segment_B"]), edge["endpoint_B"]
        if a not in segments or b not in segments or a == b:
            raise ReviewConflictError("invalid same-track segment endpoint")
        if (a, sa) in endpoint_links or (b, sb) in endpoint_links:
            raise ReviewConflictError("same endpoint assigned two continuations")
        if find(a) == find(b):
            raise ReviewConflictError("cyclic continuity relation requires separate expert topology treatment")
        endpoint_links[(a, sa)] = (b, sb)
        endpoint_links[(b, sb)] = (a, sa)
        parent[find(a)] = find(b)
    for edge in rejected:
        a, b = int(edge["segment_A"]), int(edge["segment_B"])
        if find(a) == find(b):
            raise ReviewConflictError("same-track chain contradicts different-track label")
    parts: dict[int, list[int]] = defaultdict(list)
    for sid in segments:
        parts[find(sid)].append(sid)

    tracks = []
    for component in sorted(parts.values(), key=min):
        # Select a free endpoint for a deterministic traversal (no implicit cycles).
        free = sorted((sid, side) for sid in component for side in ("A", "B")
                      if (sid, side) not in endpoint_links)
        if len(free) != 2:
            raise ReviewConflictError("component is not a simple open chain")
        sid, incoming = free[0]
        ordered = []
        bridges = []
        last_xy = None
        seen = set()
        while True:
            if sid in seen:
                raise ReviewConflictError("unexpected repeat segment in accepted chain")
            seen.add(sid)
            data = segments[sid]
            xy = np.asarray(data["measurement_xy_px"], dtype=float)
            if xy.ndim != 2 or xy.shape != (14, 2) or not np.isfinite(xy).all():
                raise ReviewConflictError("source fragment lacks 14 finite XY measurements")
            orientation = "original" if incoming == "A" else "reversed"
            observed = xy if incoming == "A" else xy[::-1]
            if last_xy is not None:
                gap = float(np.linalg.norm(observed[0] - last_xy))
                bridges.append({"from_xy_px": last_xy.tolist(), "to_xy_px": observed[0].tolist(),
                                "unobserved_gap_px": round(gap, 5),
                                "interpretation": "human_accepted_continuity_NO_OBSERVED_PIXELS"})
            ordered.append({"fragment_id": sid, "orientation": orientation,
                            "observed_measurements_xy_px": observed.tolist()})
            outgoing = "B" if incoming == "A" else "A"
            last_xy = observed[-1]
            link = endpoint_links.get((sid, outgoing))
            if link is None:
                break
            sid, incoming = link
        if len(seen) != len(component):
            raise ReviewConflictError("accepted links are not a single continuous chain")
        all_pts = np.concatenate([np.asarray(item["observed_measurements_xy_px"]) for item in ordered])
        geom = fit_quadratic(all_pts[:, ::-1]) if len(all_pts) >= 4 else None
        tracks.append({"track_group_index": len(tracks), "source_fragment_ids": [x["fragment_id"] for x in ordered],
                       "pieces": ordered, "accepted_gap_bridges": bridges,
                       "quadratic_fit_of_observed_points": geom,
                       "fit_includes_unobserved_intervals": bool(bridges),
                       "physical_track_identity_verified": False,
                       "physical_start_stop_known": False})
    return {"schema":"bubble_reviewed_2d_tracks_v1", "source_sha256_grayscale_pixels":result["source_sha256_grayscale_pixels"],
            "pipeline":result["pipeline"], "result_sha256":result_digest(result),
            "reviewed_by":reviewed.get("annotator",""), "label_counts":counts,
            "original_fragment_count":len(segments), "approved_gap_count":len(approved),
            "track_group_count":len(tracks), "tracks":tracks,
            "physical_result_status":"UNAVAILABLE_REQUIRES_STEREO_AND_RUN_CALIBRATION",
            "auto_merged_fragments":0,
            "no_inferred_pixels_across_gaps":True}
