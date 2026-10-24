from __future__ import annotations

import copy
import json
import re

import numpy as np
import pytest

from bubble_chamber.pilot import analyze_frame, PilotConfig
from bubble_chamber.review import SCHEMA, make_review_html, pair_key, result_digest, validate_labels
from bubble_chamber.synthetic import generate_case


def _case():
    img, _, _ = generate_case('gap')
    result, _, _ = analyze_frame(img, PilotConfig(threshold=18, min_segment_length_px=35))
    assert result['proposed_joins_count'] > 0
    return img, result


def test_review_html_is_self_contained_and_provenance_checked():
    img, result = _case()
    html = make_review_html(result, img)
    assert html.startswith('<!doctype html>')
    assert 'data:image/jpeg;base64,' in html
    assert 'Инструкция' not in html  # UI is not an external reference
    payload_json = re.search(r'<script id="source" type="application/json">(.*?)</script>', html, flags=re.S).group(1)
    data = json.loads(payload_json)
    assert data['source_sha256_grayscale_pixels'] == result['source_sha256_grayscale_pixels']
    assert data['result_sha256'] == result_digest(result)
    assert len(data['candidates']) == len(result['join_proposals'])
    assert all(c['pair_key'] for c in data['candidates'])
    assert 'Экспортировать JSON' in html


def test_image_mismatch_blocks_review_export():
    img, result = _case()
    changed = img.copy()
    changed[2, 2] ^= 1
    with pytest.raises(ValueError, match='SHA differ'):
        make_review_html(result, changed)


def test_strict_label_schema_validates_and_blocks_stale_annotations():
    img, result = _case()
    key = pair_key(result['join_proposals'][0])
    bundle = dict(schema=SCHEMA, source_sha256_grayscale_pixels=result['source_sha256_grayscale_pixels'],
                  pipeline=result['pipeline'], result_sha256=result_digest(result),
                  labels=[dict(pair_key=key, label='same_track', comment='expert annotated')])
    assert validate_labels(result, bundle)['same_track'] == 1
    with pytest.raises(ValueError, match='duplicate'):
        validate_labels(result, {**bundle, 'labels': bundle['labels'] * 2})
    with pytest.raises(ValueError, match='mismatch'):
        validate_labels(result, {**bundle, 'source_sha256_grayscale_pixels': 'wrong'})
    with pytest.raises(ValueError, match='result content mismatch'):
        validate_labels(result, {**bundle, 'result_sha256': '0' * 64})
    with pytest.raises(ValueError, match='invalid label'):
        validate_labels(result, {**bundle, 'labels': [{**bundle['labels'][0], 'label': 'proton'}]})
    with pytest.raises(ValueError, match='unknown candidate'):
        validate_labels(result, {**bundle, 'labels': [{**bundle['labels'][0], 'pair_key': '987:A|986:B'}]})


def test_pair_key_is_order_invariant():
    p = dict(segment_A=7, endpoint_A='B', segment_B=2, endpoint_B='A')
    q = dict(segment_A=2, endpoint_A='A', segment_B=7, endpoint_B='B')
    assert pair_key(p) == pair_key(q) == '2:A|7:B'
