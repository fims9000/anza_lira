"""S120 uniform-green review layer for S118 source-only geometry.

Keep distinct global trajectory IDs in JSON but draw all provisional track-like
groups in the SAME green color. Unknown and rejected observations stay in an
auditable separate layer. A review color is NOT a physical class label.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import cv2
import joblib
import numpy as np


def group_scores(groups, scores, lengths):
    result = {}
    for group in groups:
        keys = [k for k in group['source_curve_keys'] if k in scores]
        prob = np.asarray([scores[k] for k in keys])
        weight = np.asarray([min(1000., max(10., lengths[k])) for k in keys])
        value = float(np.average(prob, weights=weight)) if len(prob) else 0.
        if value >= .55:
            decision = 'AUTO_TRACK_LIKE_NOT_GT'
        elif value >= .35:
            decision = 'REVIEW_UNCERTAIN'
        else:
            decision = 'DEPRIORITIZED_NOT_DELETED'
        result[group['global_group_id']] = {
            'score': round(value, 4), 'decision': decision,
            'source_curve_count': len(group['source_curve_keys'])
        }
    return result


def _sha(path):
    h = sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(bmp: Path, run_dir: Path, features_file: Path, model_file: Path,
        out: Path, manual_json: Path | None = None):
    """Do not load a joblib model from an untrusted source (pickle semantics)."""
    bmp = bmp.resolve()
    cfg = json.loads((run_dir/'RUN_CONFIG.json').read_text())
    gray = cv2.imread(str(bmp), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(bmp)
    if list(gray.shape) != cfg['shape_hw'] or _sha(bmp) != cfg['source_image_sha256']:
        raise ValueError('Input BMP identity mismatch')
    z = np.load(features_file)
    if str(z['source_sha256']) != cfg['source_image_sha256']:
        raise ValueError('S120 feature file belongs to a different BMP')
    # The caller must trust the local model file; joblib is pickle-based.
    predictor = joblib.load(model_file)
    probabilities = predictor.predict_proba(z['features'])[:, 1]
    score = {str(k): float(p) for k, p in zip(z['keys'], probabilities)}
    length = {str(k): float(v) for k, v in zip(z['keys'], z['meta'][:, 0])}
    n = cfg['grid_count']
    manifest = json.loads((run_dir/f'S118_MANIFEST_{n:05d}.json').read_text())
    stitched = json.loads((run_dir/f'S118_STITCH_{n:05d}'/'S118_STITCH_NOT_GT.json').read_text())
    global_groups = group_scores(stitched['groups'], score, length)
    bykey = {k: g['global_group_id'] for g in stitched['groups'] for k in g['source_curve_keys']}
    if set(bykey) != set(score):
        raise ValueError('S118 global memberships and S120 curve features differ')
    width = 1350
    scale = width/gray.shape[1]
    small = cv2.resize(gray, (width, round(gray.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    original = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    green, diagnostic = original.copy(), original.copy()
    stats = Counter()
    for tile in manifest['tiles']:
        off = np.asarray(tile['xyxy'][:2], np.float32)
        data = json.loads((run_dir/tile['curves']).read_text())
        for curve in data['curves']:
            key = tile['tile_id']+':'+curve['candidate_id']
            decision = global_groups[bykey[key]]['decision']
            stats[decision] += 1
            pts = np.rint((np.asarray(curve['observed_pixel_path_xy'], np.float32)+off)*scale).astype(np.int32)
            if len(pts) < 2:
                continue
            if decision == 'AUTO_TRACK_LIKE_NOT_GT':
                cv2.polylines(green, [pts], False, (42, 178, 32), 2, cv2.LINE_AA)
                cv2.polylines(diagnostic, [pts], False, (42, 178, 32), 2, cv2.LINE_AA)
            elif decision == 'REVIEW_UNCERTAIN':
                cv2.polylines(diagnostic, [pts], False, (60, 184, 238), 1, cv2.LINE_AA)
            else:
                # Suppressed only in the display; observations remain in S115.
                pass
    panes = [original, green, diagnostic]
    headings = ['ORIGINAL BMP', 'S120 GREEN TRACK-LIKE (NOT GT)', 'GREEN / AMBER REVIEW']
    for panel, title in zip(panes, headings):
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 45), (25, 25, 25), -1)
        cv2.putText(panel, title, (7, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    .58, (255, 255, 255), 1, cv2.LINE_AA)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out/'FULL_FRAME_S120_GREEN.jpg'), green, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out/'FULL_FRAME_S120_REVIEW_3WAY.jpg'),
                np.concatenate(panes, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 93])
    report = {
        'source_sha256': cfg['source_image_sha256'], 'source_image_id': bmp.name,
        'curve_status_counts': dict(stats),
        'group_status_counts': dict(Counter(x['decision'] for x in global_groups.values())),
        'all_original_detected_curve_ids_preserved': True,
        'single_color_does_not_imply_single_particle_id': True,
        'no_manual_labels_used_for_inference': True,
        'model_training_may_have_used_this_same_frame': True,
        'true_physical_identity_accuracy_unknown': True,
        'model_score_not_probability_calibrated': True,
    }
    (out/'S120_GROUP_STATUS.json').write_text(json.dumps(global_groups, ensure_ascii=False, indent=2))
    (out/'S120_REVIEW_REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
