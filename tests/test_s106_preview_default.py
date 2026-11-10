"""Synthetic operator-cross preview must remain unchanged unless explicitly requested."""
import json

import cv2
import numpy as np

from scripts.research.bubble_chamber.physicists_s106_fiducial_guard import run_audit


def test_s106_is_report_only_by_default(tmp_path):
    root = tmp_path / 's105'
    root.mkdir()
    image = np.full((450, 450), 206, np.uint8)
    cv2.line(image, (60, 390), (390, 60), 50, 6, cv2.LINE_AA)
    cv2.line(image, (60, 60), (390, 390), 50, 6, cv2.LINE_AA)
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay[:, :, 1] = np.clip(overlay[:, :, 1].astype(int) + 12, 0, 255).astype(np.uint8)
    original_mask = np.ones_like(image, dtype=np.uint8) * 255
    for filename, data in (
        ('01_grayscale_visual_only.png', image),
        ('13_S105_LONG_RANGE_CONTEXT_SAME_DISPLAY_ID.png', overlay),
        ('02_S88_observed_mask.png', original_mask),
        ('14_S105_CONTEXT_SOURCE_ONLY_MASK.png', original_mask),
    ):
        assert cv2.imwrite(str(root / filename), data)
    (root / 'report.json').write_text(json.dumps({
        'source_name': 'EXAMPLE.bmp', 'source_sha256': 'a' * 64,
        'crop_xyxy_source_px': [0, 0, 450, 450],
    }))
    labels = tmp_path / 'labels.json'
    labels.write_text(json.dumps({'EXAMPLE.bmp': {
        'original_width': 450, 'original_height': 450,
        'keypoints': [{'label': 'f4', 'x_px': 225, 'y_px': 225}],
    }}))
    out = tmp_path / 'no_filter'
    report = run_audit(s105_dir=root, labels=labels, out_dir=out)
    shown = cv2.imread(str(out/'02_S106_REVIEW_PREVIEW.png'))
    assert np.array_equal(shown, overlay)
    assert not report['s106_quarantine_actually_applied_to_preview']
    assert report['s105_display_px_inside_proposed_corridor'] > 0
    assert report['overlap_is_not_proven_false_positive']
    out2 = tmp_path / 'opt_in'
    report2 = run_audit(s105_dir=root, labels=labels, out_dir=out2,
                        apply_quarantine=True)
    shown2 = cv2.imread(str(out2/'02_S106_REVIEW_PREVIEW.png'))
    assert not np.array_equal(shown2, overlay)
    assert report2['s106_quarantine_actually_applied_to_preview']
    assert np.array_equal(cv2.imread(str(root/'02_S88_observed_mask.png'), 0), original_mask)
