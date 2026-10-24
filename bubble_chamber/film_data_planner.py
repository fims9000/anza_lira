"""Plan small traceable film crops and audit fiducial labels without altering originals.

Only observed labels are exported; a 'start' click is not a track pixel mask.
Inferred affine models compare positions of fiducials, not physical optics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import csv
import json
import math

import cv2
import numpy as np


def _group(filename: str) -> str:
    fields = filename.split('_')
    if len(fields) != 4 or Path(filename).name != filename or not filename.endswith('.bmp'):
        raise ValueError(f'Unexpected image name: {filename}')
    return '_'.join(fields[:3])


def _load(labels_path: Path) -> dict:
    source = json.loads(Path(labels_path).read_text(encoding='utf-8-sig'))
    if not isinstance(source, dict) or not source:
        raise ValueError('Expected non-empty H50Z frame-keyed JSON')
    for name, frame in source.items():
        _group(name)
        points = frame['keypoints']
        if len(points) != frame['keypoints_count']:
            raise ValueError(f'Keypoint count mismatch in {name}')
        if int(frame['original_width']) <= 0 or int(frame['original_height']) <= 0:
            raise ValueError(f'Invalid image dimensions for {name}')
        for point in points:
            if point['label'] not in {'start', *(f'f{i}' for i in range(1, 9))}:
                raise ValueError(f'Unknown landmark type in {name}')
            x, y = float(point['x_px']), float(point['y_px'])
            if not (0 <= x < frame['original_width'] and 0 <= y < frame['original_height']):
                raise ValueError(f'Out-of-frame annotation in {name}')
    return source


def _unique_fids(frame: dict) -> tuple[dict[str, tuple[float, float]], list[str]]:
    counter = Counter(p['label'] for p in frame['keypoints'] if p['label'].startswith('f'))
    repeated = sorted(k for k, n in counter.items() if n > 1)
    unique = {p['label']: (float(p['x_px']), float(p['y_px']))
              for p in frame['keypoints'] if p['label'].startswith('f') and counter[p['label']] == 1}
    return unique, repeated


def audit_fiducials(labels_path: Path, *, residual_limit_px: float = 60.0) -> dict:
    """RANSAC in each name-defined scan group; flag, never auto-relabel.

    A 2D affine transform can reveal outlier ID correspondences, but cannot
    prove which fiducial ID is wrong or replace optical camera calibration.
    """
    if residual_limit_px <= 0:
        raise ValueError('Positive residual threshold required')
    dataset = _load(labels_path)
    groups: dict[str, list[str]] = defaultdict(list)
    for name in dataset:
        groups[_group(name)].append(name)
    records = []
    for group, names in sorted(groups.items()):
        # Prefer a source without duplicated fiducials and with all 8 IDs.
        names = sorted(names)
        ref_name = max(names, key=lambda n: (len(_unique_fids(dataset[n])[0]), -len(_unique_fids(dataset[n])[1]), -names.index(n)))
        reference, _ = _unique_fids(dataset[ref_name])
        for name in names:
            fid, duplicates = _unique_fids(dataset[name])
            common = sorted(set(fid) & set(reference))
            residuals: dict[str, float] = {}
            outliers = []
            model_status = 'INSUFFICIENT_COMMON_FIDUCIALS'
            if len(common) >= 4:
                source_xy = np.asarray([fid[k] for k in common], dtype=np.float64)
                reference_xy = np.asarray([reference[k] for k in common], dtype=np.float64)
                cv2.setRNGSeed(42)
                transform, inliers = cv2.estimateAffine2D(
                    source_xy, reference_xy, method=cv2.RANSAC,
                    ransacReprojThreshold=residual_limit_px,
                    maxIters=5000, confidence=0.999, refineIters=10,
                )
                if transform is not None:
                    prediction = source_xy @ transform[:, :2].T + transform[:, 2]
                    delta = np.linalg.norm(prediction - reference_xy, axis=1)
                    residuals = {k: round(float(d), 3) for k, d in zip(common, delta)}
                    outliers = sorted(k for k, d in residuals.items() if d > residual_limit_px)
                    model_status = 'ROBUST_AFFINE_SCREEN_ONLY_NOT_OPTICAL_CALIBRATION'
                else:
                    model_status = 'AFFINE_RANSAC_FAILED'
            records.append({
                'image': name, 'scan_group': group, 'reference_frame': ref_name,
                'start_count': sum(p['label'] == 'start' for p in dataset[name]['keypoints']),
                'fiducial_count_raw': sum(p['label'].startswith('f') for p in dataset[name]['keypoints']),
                'duplicate_fiducial_ids': duplicates,
                'outlying_fiducial_ids': outliers, 'fitted_residual_px': residuals,
                'model_status': model_status,
                'note': 'Review flags manually. No inferred fiducial renaming.',
            })
    return {
        'source_sha256': sha256(Path(labels_path).read_bytes()).hexdigest(),
        'image_count': len(dataset), 'scan_group_counts': {k: len(v) for k, v in groups.items()},
        'start_count': sum(sum(p['label']=='start' for p in f['keypoints']) for f in dataset.values()),
        'duplicate_id_frames': [r['image'] for r in records if r['duplicate_fiducial_ids']],
        'outlier_id_frames': [r['image'] for r in records if r['outlying_fiducial_ids']],
        'frames': records,
        'interpretation': 'RANSAC is a data-quality screen, not an optical correction or physical 3D calibration',
    }


def _spread_subset(points: list[tuple[int, dict]], width: int, height: int, count: int) -> list[tuple[int, dict]]:
    """Deterministic farthest-first landmarks; never use unmarked regions as negatives."""
    if count < 1:
        return []
    if len(points) <= count:
        return points
    normalized = np.asarray([(p['x_px'] / width, p['y_px'] / height) for _, p in points], dtype=float)
    first = min(range(len(points)), key=lambda i: (sum((normalized[i] - [0.5, 0.5])**2), points[i][0]))
    selected = [first]
    while len(selected) < count:
        scores = np.asarray([min(np.sum((normalized[i] - normalized[j])**2) for j in selected)
                             if i not in selected else -1.0 for i in range(len(points))])
        selected.append(int(scores.argmax()))
    return sorted((points[i] for i in selected), key=lambda x: x[0])


def build_crop_request(labels_path: Path, visualmetro_root: Path, *,
                       available_root: Path | None = None,
                       per_new_frame: int = 6, tile_size: int = 1024) -> dict:
    """Prioritize existing measured 3-view 070 over unlabeled extra H50Z images."""
    from .visualmetro_import import read_measurement_tree
    if tile_size < 64 or tile_size % 16:
        raise ValueError('tile_size must be multiple of 16 and >=64')
    if per_new_frame < 1:
        raise ValueError('per_new_frame must be >=1')
    labels = _load(labels_path)
    scan_groups: dict[str, list[str]] = defaultdict(list)
    for file in labels:
        scan_groups[_group(file)].append(file)
    available_root = Path(available_root) if available_root is not None else Path('/not_available')
    available = {name for name in labels if (available_root / name).is_file()}
    # Request one separately named group plus a high-annotation-density second group.
    # Scan-group != independently verified physical run; do not infer that.
    selected = []
    for group, take in [('H50Z_030_02', 3), ('H50Z_030_01', 1)]:
        candidates = sorted((n for n in scan_groups[group] if n not in available),
                            key=lambda n: (-sum(p['label']=='start' for p in labels[n]['keypoints']), n))
        selected.extend(candidates[:take])
    tile_rows = []
    summary = []
    for priority, name in enumerate(selected, 1):
        frame=labels[name]
        w,h=int(frame['original_width']),int(frame['original_height'])
        if min(w,h)<tile_size:
            raise ValueError(f'tile size exceeds frame dimensions: {name}')
        starts = [(i,p) for i,p in enumerate(frame['keypoints']) if p['label']=='start']
        chosen = _spread_subset(starts,w,h,per_new_frame)
        for idx, point in chosen:
            cx,cy=float(point['x_px']),float(point['y_px'])
            x0=max(0,min(int(round(cx))-tile_size//2,w-tile_size))
            y0=max(0,min(int(round(cy))-tile_size//2,h-tile_size))
            tile_rows.append({
                'priority':priority,'source':'H50Z_LABELS_JSON', 'source_image':name,
                'frame_id':Path(name).stem,'view_id':'', 'point_id':f'start_{idx:03d}',
                'label_semantics':'VERTEX_START_KEYPOINT_NOT_TRACK_MASK',
                'x0':x0,'y0':y0,'x1':x0+tile_size,'y1':y0+tile_size,
                'source_x_px':round(cx,3),'source_y_px':round(cy,3),
                'local_x_px':round(cx-x0,3),'local_y_px':round(cy-y0,3),
                'status':'REQUESTED',
            })
        summary.append({'image':name,'priority':priority,'start_count':len(starts),
                        'requested_crops':len(chosen),'scan_group':_group(name),
                        'already_available':False})
    visual = read_measurement_tree(visualmetro_root)
    frame_by_name={f['frame_id']:f for f in visual['frames']}
    desired_frames=('070_008','070_004','070_009')
    for priority, frame_id in enumerate(desired_frames, 1):
        if frame_id not in frame_by_name:
            continue
        frame=frame_by_name[frame_id]
        track=max(frame['tracks'],key=lambda t:t['n_points'])
        indices=sorted({0,track['n_points']//2,track['n_points']-1})
        for idx in indices:
            for view in (1,2,3):
                x,y=map(float,track['measured_pixel_xy'][str(view)][idx])
                x0=int(round(x))-tile_size//2;y0=int(round(y))-tile_size//2
                tile_rows.append({
                    'priority':priority,'source':'VISUALMETRO_SPARSE_3VIEW',
                    'source_image':f'{frame_id} VIEW {view} -- EXACT BMP NAME MUST BE MATCHED',
                    'frame_id':frame_id,'view_id':str(view),'point_id':f'{track["track_id"]}_index_{idx:02d}',
                    'label_semantics':'EXPERT_TRACK_CLICK_SPARSE_NOT_DENSE_MASK',
                    'x0':x0,'y0':y0,'x1':x0+tile_size,'y1':y0+tile_size,
                    'source_x_px':x,'source_y_px':y,
                    'local_x_px':round(x-x0,3),'local_y_px':round(y-y0,3),
                    'status':'REQUESTED_VERIFY_VIEW_AND_BMP_BOUNDARIES',
                })
    return {
        'schema':'bubble_small_bmp_request_v2',
        'note':'No dense mask or source-film cross-matching invented',
        'h50z_source_sha256':sha256(Path(labels_path).read_bytes()).hexdigest(),
        'visualmetro_manifest_sha256':visual['manifest_digest_sha256'],
        'required_first':'3 exact BMP views matching 070_008 measurements; crop using rows whose source VISUALMETRO_SPARSE_3VIEW',
        'h50z_image_priority':summary,
        'tile_size':tile_size,
        'requests':tile_rows,
        'known_existing_images':sorted(available),
        'split_warning':'H50Z scan groups are NOT verified independent physics runs; all 070 frames same film; never random crop split',
        'physical_medium':'NOT_VERIFIED_FROM_FILENAME',
    }


def save_plan(plan: dict, qc: dict, out_dir: Path) -> None:
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    fieldnames=list(plan['requests'][0])
    with (out_dir/'requested_crops.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(plan['requests'])
    (out_dir/'requested_data_summary.json').write_text(json.dumps({k:v for k,v in plan.items() if k!='requests'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out_dir/'fiducial_qc.json').write_text(json.dumps(qc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out_dir/'fiducial_review_flags.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['image','scan_group','reference','duplicate_ids','outlier_ids','status'])
        for frame in qc['frames']:
            if frame['duplicate_fiducial_ids'] or frame['outlying_fiducial_ids']:
                writer.writerow([frame['image'],frame['scan_group'],frame['reference_frame'],
                                 '|'.join(frame['duplicate_fiducial_ids']),
                                 '|'.join(frame['outlying_fiducial_ids']),frame['model_status']])
