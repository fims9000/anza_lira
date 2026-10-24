import importlib.util
import io
from pathlib import Path

import cv2
import numpy as np
import pytest


def module():
    filename = Path(__file__).resolve().parents[1] / 'scripts/research/bubble_chamber/download_public_samples.py'
    spec = importlib.util.spec_from_file_location('bc_public_samples', filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_source_list_is_small_and_credible():
    r = module()
    assert len(r.SOURCES) == 2
    assert len({s['source_page'] for s in r.SOURCES}) == 2
    assert all(s['url'].startswith('https://upload.wikimedia.org/wikipedia/commons/') for s in r.SOURCES)
    assert all(s['medium'] == 'liquid_hydrogen' for s in r.SOURCES)


def test_download_adapter_records_provenance_without_inventing_annotations(tmp_path, monkeypatch):
    r = module()
    arr = np.full((640, 640), 175, dtype=np.uint8)
    ok, buf = cv2.imencode('.jpg', arr)
    assert ok
    class MockResponse(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
    monkeypatch.setattr(r, 'urlopen', lambda *args, **kwargs: MockResponse(buf.tobytes()))
    meta = r.fetch_subset(tmp_path)
    assert len(meta['entries']) == 2
    assert all(row['label_status'] == 'NO_TRACK_LABELS' for row in meta['entries'])
    assert all(row['calibration'] == 'not_provided' for row in meta['entries'])
    assert (tmp_path / 'public_manifest.json').exists()


def test_refuse_unexpected_non_image_or_overlarge_payload(tmp_path):
    r = module()
    with pytest.raises(ValueError, match='plausible'):
        r._save_sample(r.SOURCES[0], tmp_path, b'not-an-image')
    with pytest.raises(ValueError, match='large'):
        r._save_sample(r.SOURCES[0], tmp_path, b'0' * 15_000_001)
