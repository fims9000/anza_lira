"""Extract frozen S120 photo-only features, with optional manual labels for audit.

Manual RLE is *never* passed to photo_features. It is used after features
have been frozen, to obtain training/QA labels. Private labeled NPZ files
must not be committed to public GitHub.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from bubble_chamber.s120_photo_features import photo_features, FEATURE_NAMES


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _manual_tree(data):
    coords, ids, classes = [], [], {}
    for k, obj in enumerate(data['tracks']):
        if obj.get('class') not in ('track', 'trash') or not obj.get('line_pixels'):
            continue
        r = np.asarray(obj['line_pixels'], np.int32)
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError('Invalid manual RLE triplets')
        coords.append(np.column_stack((r[:, 1]+r[:, 2]/2, r[:, 0])).astype(np.float32))
        ids.extend([k]*len(r))
        classes[k] = obj['class']
    if not coords:
        raise ValueError('No annotated strokes')
    return cKDTree(np.vstack(coords)), np.asarray(ids, np.int32), classes


def _manual_match(points, ref):
    tree, ids, classes = ref
    pts = np.asarray(points, dtype=np.float32)
    subset = pts[::max(1, len(pts)//140)]
    dist, at = tree.query(subset, distance_upper_bound=8.)
    at = at[np.isfinite(dist)]
    if not len(at):
        return -1, 'unknown', 0., 0.
    uniq, count = np.unique(ids[at], return_counts=True)
    order = np.argsort(-count)
    best, n = int(uniq[order[0]]), int(count[order[0]])
    other = int(count[order[1]]) if len(order) > 1 else 0
    frac = n/len(subset)
    if n >= 8 and frac >= .28 and other <= n*.5:
        return best, classes[best], round(frac, 3), round(other/max(1, n), 3)
    return -1, 'unknown', round(frac, 3), round(other/max(1, n), 3)


def run(bmp: Path, run_dir: Path, out: Path, manual_json: Path | None = None):
    bmp = bmp.resolve()
    raw = cv2.imread(str(bmp), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(bmp)
    cfg = json.loads((run_dir/'RUN_CONFIG.json').read_text())
    if list(raw.shape) != cfg['shape_hw'] or _sha(bmp) != cfg['source_image_sha256']:
        raise ValueError('S118 checkpoint does not match source image')
    manifest = json.loads((run_dir/f"S118_MANIFEST_{cfg['grid_count']:05d}.json").read_text())
    ref = None
    if manual_json:
        annot = json.loads(manual_json.read_text())
        if annot['image']['sha256'] != cfg['source_image_sha256']:
            raise ValueError('Manual annotation belongs to another BMP')
        ref = _manual_tree(annot)
    features, keys, labels, label_ids, meta = [], [], [], [], []
    for tile in manifest['tiles']:
        x0, y0, x1, y1 = tile['xyxy']
        crop = raw[y0:y1, x0:x1]
        curves = json.loads((run_dir/tile['curves']).read_text())['curves']
        for c in curves:
            pts = np.asarray(c['observed_pixel_path_xy'], dtype=np.float32)
            feature = photo_features(crop, pts)
            key = tile['tile_id']+':'+c['candidate_id']
            idx, label, frac, rival = (
                _manual_match(pts+[x0, y0], ref) if ref
                else (-1, 'unknown', 0., 0.)
            )
            features.append(feature)
            keys.append(key)
            labels.append(label)
            label_ids.append(idx)
            meta.append((float(c['observed_centerline_length_px']), frac, rival))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, features=np.asarray(features), keys=np.asarray(keys),
        labels=np.asarray(labels), label_ids=np.asarray(label_ids),
        meta=np.asarray(meta), names=np.asarray(FEATURE_NAMES),
        source_sha256=cfg['source_image_sha256'])
    return {
        'source_image_sha256': cfg['source_image_sha256'],
        'curve_features': len(keys),
        'posthoc_manual_label_counts': dict(Counter(labels)),
        'features_frozen_before_label_matching': True,
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bmp', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manual-for-training-audit', type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.bmp, a.run_dir, a.out, a.manual_for_training_audit), indent=2))
