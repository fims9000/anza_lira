"""Audited ingestion of a large visualMetro export without pretending clicks are masks.

When one track of a frame is corrupt, the whole frame is quarantined for
supervised scene-level relation learning (unknown candidates would otherwise
be mislabelled as negatives). Only explicitly valid groups are admitted.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import re
from typing import Any

from .visualmetro_import import (
    AnnotationFormatError, FRAME_NAME, SCHEMA, TRACK_ID,
    parse_fiducials, parse_track,
)


# Film 070_305 has source IDs e1to1/e1to2/e1to3 (literal letter 'o').
# Keep those identifiers unchanged rather than silently assuming 'o' means zero.
LEGACY_ID = re.compile(r"^e[0-9]+to[0-9]+$")


def audited_large_manifest(root: Path) -> dict[str, Any]:
    source = Path(root).resolve()
    if not source.is_dir():
        raise AnnotationFormatError(f"not a directory: {source}")
    folders = sorted(f for f in source.iterdir() if f.is_dir() and FRAME_NAME.fullmatch(f.name))
    if not folders:
        raise AnnotationFormatError("no visualMetro frame folders")

    frames: list[dict] = []
    excluded: list[dict] = []
    all_tracks = 0
    all_parsed = 0
    all_points_parsed = 0
    orders: Counter[str] = Counter()
    source_hashes: list[str] = []
    legacy_raw_track_ids: list[dict] = []
    unlisted_tracks: list[dict] = []
    for folder in folders:
        frame_id = folder.name
        listing_path = folder / f"{frame_id}_trkList.txt"
        fiducials_path = folder / f"{frame_id}_fids.txt"
        if not listing_path.is_file() or not fiducials_path.is_file():
            excluded.append({"frame_id": frame_id, "reason": "missing_track_list_or_fiducials"})
            continue
        listing = listing_path.read_text(encoding="utf-8-sig").split()
        all_tracks += len(listing)
        for candidate_file in folder.glob(f"{frame_id}_*.txt"):
            candidate_name = candidate_file.stem.removeprefix(frame_id + "_")
            if candidate_name not in listing and not candidate_name.endswith("_report") and candidate_name not in {"fids", "trkList"}:
                unlisted_tracks.append({"frame_id": frame_id, "filename": candidate_file.name, "reason": "not_listed_by_operator"})
        if not listing or len(set(listing)) != len(listing) or any(not (TRACK_ID.fullmatch(n) or LEGACY_ID.fullmatch(n)) for n in listing):
            excluded.append({"frame_id": frame_id, "reason": "invalid_or_empty_track_list"})
            continue
        try:
            fiducials = parse_fiducials(fiducials_path)
        except (OSError, AnnotationFormatError) as err:
            excluded.append({"frame_id": frame_id, "reason": "invalid_fiducials", "details": str(err)})
            continue
        tracks = []
        invalid = []
        for tid in listing:
            if LEGACY_ID.fullmatch(tid):
                legacy_raw_track_ids.append({"frame_id": frame_id, "raw_track_id": tid, "note": "original nonstandard operator ID preserved verbatim"})
            path = folder / f"{frame_id}_{tid}.txt"
            try:
                track = parse_track(path, frame_id, tid)
                all_parsed += 1
                all_points_parsed += track["n_points"]
                orders["/".join(map(str, track["raw_view_block_order"]))] += 1
                tracks.append(track)
            except (OSError, AnnotationFormatError) as err:
                invalid.append({"track_id": tid, "reason": str(err)})
        if invalid:
            excluded.append({"frame_id": frame_id, "reason": "incomplete_frame_tracks", "invalid_tracks": invalid,
                             "parsed_tracks_not_used_for_training": len(tracks)})
            continue
        source_hashes.extend(t["file_sha256"] for t in tracks)
        source_hashes.extend([fiducials["file_sha256"], sha256(listing_path.read_bytes()).hexdigest()])
        frames.append({"frame_id": frame_id, "film_id": frame_id[:3], "tracks": tracks,
                       "fiducials": fiducials, "track_list_sha256": sha256(listing_path.read_bytes()).hexdigest(),
                       "image_filenames_present": [p.name for p in folder.iterdir() if p.suffix.lower() in {".bmp", ".png", ".jpg"}]})

    used_tracks = sum(len(f["tracks"]) for f in frames)
    used_points = sum(t["n_points"] for f in frames for t in f["tracks"])
    return {
        "schema": SCHEMA,
        "audit_schema": "visualmetro_large_audit_v1",
        "frames": frames,
        "excluded_frames": excluded,
        "legacy_raw_track_ids": legacy_raw_track_ids,
        "unlisted_track_files": unlisted_tracks,
        "summary": {
            "frame_folders_found": len(folders), "frames_complete_used": len(frames),
            "frames_quarantined": len(excluded), "track_entries_in_lists": all_tracks,
            "tracks_parsed_overall": all_parsed, "spatial_points_parsed_overall": all_points_parsed,
            "tracks_used": used_tracks, "spatial_points_used": used_points,
            "stereo_pixel_measurements_used": used_points * 3,
            "tracks_of_incomplete_frames_not_used": all_parsed - used_tracks,
            "legacy_nonstandard_track_ids": len(legacy_raw_track_ids),
            "unlisted_track_files": len(unlisted_tracks),
            "track_view_block_orders": dict(sorted(orders.items())),
            "distinct_film_ids": sorted({f["film_id"] for f in frames}),
            "matched_bmp_count": sum(len(f["image_filenames_present"]) for f in frames),
            "split_limitation": "frame-disjoint is achievable; independent FILM/RUN-disjoint is not",
            "no_full_pixel_segmentation_masks": True,
        },
        "provenance_sha256": sha256("\n".join(sorted(source_hashes)).encode()).hexdigest(),
        "semantics": "expert discrete stereo clicks; artificial gaps are not real occlusions; other image pixels are UNKNOWN",
    }


def save_audit(root: Path, path: Path, *, save_tracks: bool = False) -> dict[str, Any]:
    manifest = audited_large_manifest(root)
    content = manifest if save_tracks else {k: v for k, v in manifest.items() if k != "frames"}
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
