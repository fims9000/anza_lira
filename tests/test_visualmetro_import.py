"""Portable tests: source user archive is never required to run in CI."""
from pathlib import Path
import pytest
from bubble_chamber.visualmetro_import import (
    read_measurement_tree, parse_track, AnnotationFormatError, write_manifest,
)


def _fixture(root: Path, *, suffix: str = " Stop pi-") -> Path:
    d = root / "070_008"
    d.mkdir()
    (d / "070_008_trkList.txt").write_text("e1t01\n")
    (d / "070_008_fids.txt").write_text("\n".join(
        f"{v} -- view\n" + "\n".join(f"{1000+i} {3000+i} f{i}" for i in range(1, 9))
        for v in (1, 3, 2)
    ) + "\n")
    p = d / "070_008_e1t01.txt"
    header = "4 1.0 2.0 3.0 4.0 5.0 6.0 -1" + suffix
    xyz = [f"{i} 0.1 {i+1} 0.1 {i+2} 0.2 " + " ".join(["0.0"]*13) for i in (1, 2, 3, 4)]
    points = [f"{view} $000000FF {1000+index*12} {3000+index*10}"
              for view in (1, 3, 2) for index in range(4)]
    p.write_text("\n".join([header, *xyz, *points, "2026.06.24 11:40:06"]) + "\n")
    return p


def test_complete_fixture_and_operator_note(tmp_path):
    source = _fixture(tmp_path)
    m = read_measurement_tree(tmp_path)
    assert m['summary']['frame_count'] == 1
    assert m['summary']['track_count'] == 1
    assert m['summary']['spatial_point_count'] == 4
    assert m['summary']['stereo_pixel_measurement_count'] == 12
    assert m['summary']['fiducial_point_count'] == 24
    assert m['summary']['image_files_present_in_extracted_root'] == 0
    assert m['split_status'] == 'INSUFFICIENT_INDEPENDENT_FILMS'
    t = m['frames'][0]['tracks'][0]
    assert t['operator_note_unverified'] == 'Stop pi-'
    assert t['xyz_cm'][0] == [1.0, 2.0, 3.0]
    assert t['measured_pixel_xy']['3'][1] == [1012, 3010]
    assert len(t['measured_pixel_xy']['2']) == t['n_points'] == 4


def test_bad_view_order_rejected(tmp_path):
    p = _fixture(tmp_path)
    ls = p.read_text().splitlines()
    idx = 1 + 4
    ls[idx] = ls[idx].replace('1 $', '2 $', 1)
    p.write_text('\n'.join(ls) + '\n')
    with pytest.raises(AnnotationFormatError, match='view order'):
        parse_track(p,'070_008','e1t01')


def test_missing_named_track_is_error(tmp_path):
    p = _fixture(tmp_path)
    p.unlink()
    with pytest.raises(AnnotationFormatError, match='missing named track'):
        read_measurement_tree(tmp_path)


def test_manifest_is_deterministic(tmp_path):
    _fixture(tmp_path)
    p = tmp_path/'manifest.json'
    a = write_manifest(tmp_path,p)
    b = write_manifest(tmp_path,p)
    assert a == b
    assert p.read_text().startswith('{')
