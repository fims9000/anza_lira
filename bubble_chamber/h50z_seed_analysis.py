"""Auditable real-film *keypoint* analysis, with no fabricated dense masks.

The only valid annotations are ``start`` and f1...f8 landmark centers. A
Frangi/contrast comparison measures proximity to known starts, which is NOT
segmentation accuracy nor proof of particle-track identity.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from .bitmap_tiles import GrayBMP8, sha256_file
from .pilot import PilotConfig, analyze_frame, extract_skeleton_segments


def audit_landmarks(labels_path: Path, image_root: Path) -> dict[str, Any]:
    labels = json.loads(labels_path.read_text(encoding='utf-8-sig'))
    if not isinstance(labels, dict) or not labels:
        raise ValueError('labels must map image names to records')
    available = []
    classes = Counter()
    for name, obj in sorted(labels.items()):
        if Path(name).name != name or not name.lower().endswith('.bmp'):
            raise ValueError('invalid filename in annotations')
        if len(obj.get('keypoints', [])) != obj.get('keypoints_count'):
            raise ValueError(f'invalid keypoints count: {name}')
        for kp in obj['keypoints']:
            if kp['label'] not in {'start', *(f'f{i}' for i in range(1, 9))}:
                raise ValueError('unsupported landmark')
            if not (0 <= kp['x_px'] < obj['original_width'] and 0 <= kp['y_px'] < obj['original_height']):
                raise ValueError('out of image bounds')
            classes[kp['label']] += 1
        src = image_root / name
        if src.is_file():
            bitmap = GrayBMP8(src)
            if (bitmap.width, bitmap.height) != (obj['original_width'], obj['original_height']):
                raise ValueError(f'BMP/label dimensions differ: {name}')
            available.append(name)
    return {'annotated_frames_in_json': len(labels), 'available_original_bmps': len(available),
            'missing_original_bmps': len(labels) - len(available),
            'keypoints_by_class_all_frames': dict(sorted(classes.items())),
            'available_filenames': available,
            'available_start_points': sum(kp['label']=='start' for name in available for kp in labels[name]['keypoints']),
            'available_fiducial_points': sum(kp['label'].startswith('f') for name in available for kp in labels[name]['keypoints']),
            'segmentation_mask_available': False,
            'valid_labels': 'start or fiducial landmarks ONLY, not pixel mask or particle identity',
            'unknown_pixels': 'UNKNOWN - not background supervision'}


def _nearest_mask(mask: np.ndarray, xy_local: tuple[float, float], *, minimum_length: int = 28) -> float | None:
    skeleton = skeletonize(mask.astype(bool))
    surviving = extract_skeleton_segments(skeleton, min_length_px=minimum_length)
    if not surviving:
        return None
    candidate = np.zeros(mask.shape, dtype=bool)
    for path in surviving:
        pts = np.rint(path).astype(int)
        candidate[pts[:, 0], pts[:, 1]] = True
    distances = distance_transform_edt(~candidate)
    x,y=xy_local
    yi,xi=int(round(y)),int(round(x))
    return round(float(distances[yi,xi]),3)


def make_keypoint_pilot(labels_path: Path, image_root: Path, out_dir: Path, *, tile_size: int = 384,
                        proximity_px: float = 12.0) -> dict[str, Any]:
    """Extract actual-film annotated tiles and evaluate fixed classical signals.

    The Frangi/contrast hyperparameters were selected previously on a synthetic
    validation split, never chosen from these keypoint distances.
    """
    audit = audit_landmarks(labels_path, image_root)
    if tile_size < 128 or tile_size % 16:
        raise ValueError('tile_size must be a multiple of 16 >= 128')
    labels = json.loads(labels_path.read_text(encoding='utf-8-sig'))
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    frames = []
    for name in audit['available_filenames']:
        src = GrayBMP8(image_root / name)
        frame_sha = sha256_file(image_root / name)
        points = labels[name]['keypoints']
        frames.append({'image': name, 'sha256': frame_sha, 'shape_yx': [src.height,src.width],
                       'start_landmarks':sum(p['label']=='start' for p in points),
                       'fiducial_landmarks':sum(p['label'].startswith('f') for p in points)})
        for idx, point in enumerate(points):
            tile, loc = src.crop((point['x_px'], point['y_px']), tile_size)
            label = point['label']
            prefix = f"{Path(name).stem}_{idx:02d}_{label}"
            tile_file = out_dir / f'{prefix}_original.png'
            cv2.imwrite(str(tile_file), tile, [cv2.IMWRITE_PNG_COMPRESSION, 5])
            rec = {'parent': name, 'image_sha256': frame_sha, 'label': label, 'source_index': idx,
                   'roi_original_xyxy': list(loc.original_xyxy),
                   'center_original_xy': [point['x_px'],point['y_px']],
                   'center_tile_xy': list(loc.center_xy_local),
                   'tile_filename': tile_file.name, 'tile_sha256': sha256_file(tile_file),
                   'no_dense_target': True}
            if label == 'start':
                method_summaries = {}
                for method in ('contrast','frangi'):
                    cfg = PilotConfig(threshold=12, min_segment_length_px=28,
                                      segmentation_method=method, ridge_response_fraction=0.16)
                    result, mask, overlay = analyze_frame(tile, cfg)
                    distance = _nearest_mask(mask>0,loc.center_xy_local)
                    method_summaries[method] = {
                        'nearest_surviving_skeleton_px': distance,
                        'has_surviving_ridge_within_12px': bool(distance is not None and distance <= proximity_px),
                        'fragment_count': result['fragment_count'],
                    }
                    img = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    cx,cy=map(lambda v:int(round(v)),loc.center_xy_local)
                    cv2.circle(img,(cx,cy),int(round(proximity_px)),(20,20,220),2)
                    cv2.drawMarker(img,(cx,cy),(0,0,255),cv2.MARKER_CROSS,28,2)
                    cv2.imwrite(str(out_dir/f'{prefix}_{method}_overlay.png'),img)
                rec['fixed_baseline_diagnostics'] = method_summaries
            summaries.append(rec)
    starts=[r for r in summaries if r['label']=='start']
    diagnostics={method:{'start_landmark_count':len(starts),
                         'near_start_count':sum(r['fixed_baseline_diagnostics'][method]['has_surviving_ridge_within_12px'] for r in starts),
                         'mean_nearest_ridge_px':round(float(np.mean([r['fixed_baseline_diagnostics'][method]['nearest_surviving_skeleton_px'] for r in starts if r['fixed_baseline_diagnostics'][method]['nearest_surviving_skeleton_px'] is not None])),3)}
                for method in ('contrast','frangi')}
    result={'status':'KEYPOINT_ADJACENCY_ONLY_NO_PIXEL_SEGMENTATION_GROUND_TRUTH',
            'audit':audit, 'frames':frames, 'labelled_tiles':summaries, 'keypoint_proximity_diagnostics':diagnostics,
            'reference_threshold_origin':'P2 synthetic validation only, no tuning on H50Z',
            'real_segmentation_accuracy':'NOT_MEASURABLE', 'anZa_vs_unet_comparison':'NOT_TRAINABLE_WITH_ONLY_KEYPOINT_LABELS',
            'scientific_limit':'Ridge near a start location is not proof of the correct particle track, nor a precision/recall statistic'}
    (out_dir/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    return result
