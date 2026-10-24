"""Fixture-based checks; do not require proprietary source archive in CI."""

from pathlib import Path
import pytest

from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.visualmetro_import import parse_track, AnnotationFormatError


def make_fake(root: Path, frame: str = "070_004", order=(1, 3, 2)) -> Path:
    folder = root / frame
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{frame}_trkList.txt").write_text("e1t01\n")
    (folder / f"{frame}_fids.txt").write_text("\n".join(
        f"{v} -- view\n" + "\n".join(f"{1000+i} {3000+i} f{i}" for i in range(1, 9))
        for v in (1, 3, 2)
    ))
    p = folder / f"{frame}_e1t01.txt"
    head = "7 1 2 3 4 5 6 1"
    xyz = [f"{i} 0.1 {i+1} 0.1 {i+2} 0.1 " + " ".join(["0"] * 13) for i in range(7)]
    pixels = [f"{v} $000000FF {100+i} {200+i}" for v in order for i in range(7)]
    p.write_text("\n".join([head, *xyz, *pixels, "2026.06.24 11:40:06"]) + "\n")
    return p


@pytest.mark.parametrize("order", [(1,3,2), (3,1,2), (1,2,3)])
def test_accept_exact_block_permutations(tmp_path, order):
    track = make_fake(tmp_path, order=order)
    record = parse_track(track, "070_004", "e1t01")
    assert record["raw_view_block_order"] == list(order)
    assert set(record["measured_pixel_xy"]) == {"1", "2", "3"}
    result = audited_large_manifest(tmp_path)
    assert result["summary"]["frames_complete_used"] == 1
    assert result["summary"]["tracks_used"] == 1


def test_quarantine_whole_scene_on_corrupt_view_row(tmp_path):
    track = make_fake(tmp_path)
    rows = track.read_text().splitlines()
    rows[1 + 7] = rows[1 + 7].replace("1 $", "3 $")
    track.write_text("\n".join(rows) + "\n")
    with pytest.raises(AnnotationFormatError):
        parse_track(track, "070_004", "e1t01")
    audit = audited_large_manifest(tmp_path)
    assert audit["summary"]["tracks_used"] == 0
    assert audit["summary"]["frames_quarantined"] == 1
    assert audit["excluded_frames"][0]["reason"] == "incomplete_frame_tracks"


def test_missing_track_list_is_excluded(tmp_path):
    make_fake(tmp_path)
    second = tmp_path / "070_159"
    second.mkdir()
    (second / "070_159_fids.txt").write_text("unused")
    audit = audited_large_manifest(tmp_path)
    assert audit["summary"]["frame_folders_found"] == 2
    assert audit["summary"]["frames_complete_used"] == 1
    assert audit["summary"]["frames_quarantined"] == 1


def test_duplicate_film_frames_not_independent(tmp_path):
    make_fake(tmp_path)
    make_fake(tmp_path, "070_005", (3, 1, 2))
    audit = audited_large_manifest(tmp_path)
    assert len(audit["summary"]["distinct_film_ids"]) == 1
    assert audit["summary"]["frames_complete_used"] == 2
