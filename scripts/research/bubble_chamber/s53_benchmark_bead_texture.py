#!/usr/bin/env python3
"""S53 held-out machine-only bead-renderer ablation, NO native particle GT.

Paired renderers use the same real H50Z background and unrelated measured
film-070 2D geometry. The scored object is SYNTHETIC RENDERER ORIGIN, never
whether a real H50Z mark is a particle, scratch, or calibration artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_simulated_gaps import extract_track_templates, _shape_curve
from bubble_chamber.s39_machine_auto import ridge_evidence
from bubble_chamber.s43_artifact_triage import FEATURE_NAMES as S43_FIELDS, path_features
from bubble_chamber.s53_bead_texture import FEATURE_NAMES as S53_FIELDS, SCHEMA, texture_features, merge_features
from scripts.research.bubble_chamber.s43_machine_artifact_trial import frozen_frame_folds, select_calibration_threshold

FITTING_NEGATIVES = ('continuous_hairline', 'periodic_dashes', 'scattered_grains')
TEST_ONLY_NEGATIVES = ('same_bead_renderer_scratch', 'unseen_bead_clusters')
RENDERERS = ('simulated_bead_track', *FITTING_NEGATIVES, *TEST_ONLY_NEGATIVES)
PROTOCOL = {
    'train_scenes_per_direction': 40,
    'calibration_scenes_per_direction': 20,
    'test_scenes_per_direction_per_seed': 40,
    'source_train_frame_count': 72,
    'source_calibration_frame_count': 37,
    'source_test_frame_count': 36,
    'train_seeds': {'005': 20265301, '006': 20265311},
    'calibration_seeds': {'005': 20265302, '006': 20265312},
    'test_seeds': {'005': [20265303, 20265304], '006': [20265313, 20265314]},
    'calibration_negative_quantile': .90,
    'same_background_path_and_nominal_ink_for_all_scene_variants': True,
}


def _line_mask(points: np.ndarray, shape: tuple[int, int], *, width: int = 1) -> np.ndarray:
    a = np.zeros(shape, dtype=np.uint8)
    cv2.polylines(a, [np.rint(points).astype(np.int32)], False, 255, width, cv2.LINE_8)
    return a.astype('float32') / 255.


def _blob_mask(points: np.ndarray, rng: np.random.Generator,
               shape: tuple[int, int], *, clustered: bool = False) -> np.ndarray:
    curve = np.asarray(points, dtype=float)
    steps = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    times = np.r_[0., np.cumsum(steps)]
    mask = np.zeros(shape, dtype='float32')
    along = float(rng.uniform(.5, 3.))
    while along < times[-1]:
        x = float(np.interp(along, times, curve[:, 0]))
        y = float(np.interp(along, times, curve[:, 1]))
        v = np.diff(curve, axis=0)
        k = min(len(v) - 1, int(np.searchsorted(times, along, side='right') - 1))
        tangent = v[k] / max(np.linalg.norm(v[k]), 1.e-6)
        normal = np.array([-tangent[1], tangent[0]])
        off = float(rng.normal(0, 1.65 if clustered else .55))
        pos = np.rint(np.array([x, y]) + off * normal).astype(int)
        radius = int(rng.choice([1, 2, 3] if clustered else [1, 2],
                                p=[.2, .5, .3] if clustered else [.55, .45]))
        if 0 <= pos[0] < shape[1] and 0 <= pos[1] < shape[0]:
            cv2.circle(mask, tuple(pos), radius, 1., -1, cv2.LINE_8)
        along += float(rng.uniform(2.5, 8.5) if clustered else rng.uniform(3.2, 6.0))
    return mask


def renderer_mask(kind: str, points: np.ndarray, rng: np.random.Generator,
                  shape: tuple[int, int]) -> np.ndarray:
    if kind not in RENDERERS:
        raise ValueError('Unknown generator family')
    if kind in ('simulated_bead_track', 'same_bead_renderer_scratch'):
        return _blob_mask(points, rng, shape)
    if kind == 'unseen_bead_clusters':
        return _blob_mask(points, rng, shape, clustered=True)
    if kind == 'continuous_hairline':
        return _line_mask(points, shape, width=1)
    if kind == 'periodic_dashes':
        result = np.zeros(shape, dtype='float32')
        for start in range(0, len(points), 16):
            part = points[start:start+10]
            if len(part) >= 2:
                result = np.maximum(result, _line_mask(part, shape))
        return result
    # Scattered grains on the identical curve, but with wider cross-line jitter.
    return _blob_mask_scattered(points, rng, shape)


def _blob_mask_scattered(points: np.ndarray, rng: np.random.Generator,
                         shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype='float32')
    for point in points[::3]:
        pos = np.rint(point + rng.normal(0., 3.8, size=2)).astype(int)
        if 0 <= pos[0] < shape[1] and 0 <= pos[1] < shape[0]:
            cv2.circle(m, tuple(pos), int(rng.integers(1, 3)), 1., -1)
    return m


def render_variants(background: np.ndarray, curve: np.ndarray, *, seed: int,
                    include_open_set: bool) -> dict[str, np.ndarray]:
    if background.shape != (192, 192) or background.dtype != np.uint8:
        raise ValueError('Source must be an untouched uint8 native crop')
    rng = np.random.default_rng(seed)
    ink = float(rng.uniform(22., 54.))
    kinds = RENDERERS if include_open_set else RENDERERS[:4]
    masks = {kind: renderer_mask(kind, curve, rng, background.shape) for kind in kinds}
    positive_ink_mass = float(np.sum(masks['simulated_bead_track']))
    if positive_ink_mass < 70:
        raise ValueError('Insufficient synthetic bead mass')
    output = {}
    for kind, mask in masks.items():
        mass = float(mask.sum())
        if mass < 10:
            raise ValueError('Degenerate synthetic scratch mask')
        # Same source image, same geometry and same NOMINAL injected darkness.
        # Per-pixel intensity is allowed to differ: that is the object of study.
        opacity = mask * (positive_ink_mass / mass)
        out = np.clip(background.astype('float32') - ink * opacity, 0., 255.)
        output[kind] = cv2.GaussianBlur(out.astype('uint8'), (0, 0), .35)
    return output


def prepare_examples(manifest: dict, frames: set[str], photo_tiles: Path,
                     *, photo: str, n: int, seed: int, test: bool) -> dict:
    if photo not in ('005', '006') or n <= 0:
        raise ValueError('Unknown acquisition or empty experiment')
    photos = sorted(photo_tiles.glob(f'H50Z_030_01_{photo}_*.png'))
    if len(photos) != (21 if photo == '005' else 18):
        raise ValueError('H50Z image count does not match private provenance')
    templates = extract_track_templates(manifest, chosen_frames=frames)
    rng = np.random.default_rng(seed)
    rows43, rows53, labels, families, scene_ids = [], [], [], [], []
    counts = Counter()
    previews = []
    for scene_id in range(n):
        background = cv2.imread(str(photos[int(rng.integers(0, len(photos)))]),
                                cv2.IMREAD_GRAYSCALE)
        if background is None or background.shape != (512, 512):
            raise ValueError('Private H50Z source crop corrupt')
        x0 = int(rng.integers(0, 321)); y0 = int(rng.integers(0, 321))
        clean = background[y0:y0+192, x0:x0+192]
        template = templates[int(rng.integers(len(templates)))]
        center = float(rng.uniform(78., 114.))
        curve = _shape_curve(template, length=float(rng.uniform(107., 146.)),
                             center_y=center, tilt=float(rng.uniform(-12., 12.)))
        if not (np.all(curve > 13.) and np.all(curve < 179.)):
            raise ValueError('A measured template projected out of crop bounds')
        # All renderer families see the SAME imperfect observed source path.
        estimated = curve.copy()
        estimated[:, 1] += float(rng.normal(0., .75))
        variants = render_variants(clean, estimated,
                                   seed=int(rng.integers(0, 2**31)), include_open_set=test)
        for family, img in variants.items():
            maps = ridge_evidence(img)
            rows43.append(path_features(img, estimated, maps=maps))
            rows53.append(texture_features(img, estimated))
            labels.append(int(family == 'simulated_bead_track'))
            families.append(family)
            scene_ids.append(scene_id)
            counts[family] += 1
        if scene_id < 3 and test:
            previews.append({'scene_id': scene_id, 'images': variants,
                             'photo': photo, 'label_synthetic_only': True})
    x43 = np.stack(rows43)
    x53 = np.stack(rows53)
    return {'X43': x43, 'X53': x53,
            'labels': np.asarray(labels, dtype=int),
            'families': np.asarray(families), 'scenes': np.asarray(scene_ids),
            'counts': dict(counts), 'source_frames': len(frames),
            'previews': previews, 'photo': photo}


def _fit(train: dict, *, extended: bool, seed: int):
    x = merge_features(train['X43'], train['X53']) if extended else train['X43']
    clf = make_pipeline(StandardScaler(), LogisticRegression(
        C=1., class_weight='balanced', random_state=seed, max_iter=700))
    clf.fit(x, train['labels'])
    return clf


def _scores(model, data: dict, *, extended: bool) -> np.ndarray:
    x = merge_features(data['X43'], data['X53']) if extended else data['X43']
    return model.predict_proba(x)[:, 1]


def _paired_bootstrap_delta(data: dict, old: np.ndarray, new: np.ndarray,
                            *, seed: int, iterations: int = 500) -> dict:
    y = data['labels']
    if len(y) != len(new) or len(old) != len(y):
        raise ValueError('Paired comparison must contain same examples')
    groups = np.unique(data['scenes'])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        scene_pick = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([np.flatnonzero(data['scenes'] == s) for s in scene_pick])
        samples.append(float(roc_auc_score(y[indices], new[indices]) -
                             roc_auc_score(y[indices], old[indices])))
    return {'delta_AUROC_S53_minus_S43_SYNTHETIC': float(roc_auc_score(y, new)-roc_auc_score(y, old)),
            'simulator_scene_bootstrap_CI_95pct':np.quantile(samples, [.025, .975]).tolist(),
            'physical_run_disjoint_CI': False}


def evaluate(train: dict, cal: dict, test: dict, *, seed: int, iterations: int = 500) -> tuple[dict, dict]:
    models = {'existing_S43': _fit(train, extended=False, seed=seed),
              'S43_plus_bead_texture_S53': _fit(train, extended=True, seed=seed)}
    summaries = {}
    score_out = {}
    for method in ('raw_flank_contrast', *models):
        if method == 'raw_flank_contrast':
            sc = cal['X43'][:, 3]
            st = test['X43'][:, 3]
        else:
            sc = _scores(models[method], cal, extended=('S53' in method))
            st = _scores(models[method], test, extended=('S53' in method))
        # No use of test labels or photo in threshold selection.
        threshold = select_calibration_threshold(sc, cal['labels'] == 0)
        labs = test['labels']
        clone = (test['families'] == 'same_bead_renderer_scratch')
        positive = labs == 1
        hard = positive | clone
        summaries[method] = {
            'overall_AUROC_SYNTHETIC_RENDERER_ONLY': float(roc_auc_score(labs, st)),
            'matched_bead_clone_only_AUROC_SYNTHETIC_RENDERER_ONLY':
                float(roc_auc_score(labs[hard], st[hard])),
            'calibration_only_cutoff_q90_negative': float(threshold),
            'test_synthetic_bead_acceptance': float(np.mean(st[positive] > threshold)),
            'false_acceptance_by_synthetic_negative_renderer': {
                family: float(np.mean(st[test['families'] == family] > threshold))
                for family in RENDERERS[1:] if family in test['families']},
        }
        score_out[method] = st
    summaries['S43_plus_bead_texture_S53']['paired_scene_bootstrap_delta_vs_S43'] = (
        _paired_bootstrap_delta(test, score_out['existing_S43'],
                                score_out['S43_plus_bead_texture_S53'],
                                seed=seed+111,iterations=iterations))
    return summaries, models


def run(visualmetro: Path, photos: Path, output: Path, *,
        train_n: int = 40, cal_n: int = 20, test_n: int = 40,
        bootstrap: int = 500, private_dir: Path | None = None):
    if (train_n < 8 or cal_n < 5 or test_n < 8):
        raise ValueError('Insufficient independent synthetic scenes')
    manifest = audited_large_manifest(visualmetro)
    frames = frozen_frame_folds(manifest)
    if len(manifest['frames']) != 145 or [len(frames[k]) for k in ('train','calibration','test')] != [72,37,36]:
        raise ValueError('Frozen S43 source split was altered')
    report = {
        'schema': SCHEMA, 'scored_truth': 'RENDERER_ORIGIN_SYNTHETIC_ONLY',
        'native_H50Z_expert_GT': 0, 'native_auto_joins':0,
        'same_physical_H50Z_run':True, 'unrelated_visualmetro_curve_origins':True,
        'all_negative_and_positive_curves_geometry_matched_within_scene': True,
        'no_open_set_negative_seen_during_fit_or_calibration':True,
        'protocol': dict(PROTOCOL, train_scenes_per_direction=train_n,
                         calibration_scenes_per_direction=cal_n,
                         test_scenes_per_direction_per_seed=test_n),
        'frame_disjoint_sources': {k:len(v) for k,v in frames.items()},
        'source_archive_sha256':manifest['provenance_sha256'],
        'directions':{},
    }
    if private_dir:
        private_dir.mkdir(parents=True, exist_ok=True)
    for train_photo,test_photo in (('005','006'),('006','005')):
        tr = prepare_examples(manifest, frames['train'], photos, photo=train_photo,
             n=train_n, seed=PROTOCOL['train_seeds'][train_photo], test=False)
        ca = prepare_examples(manifest, frames['calibration'], photos, photo=train_photo,
             n=cal_n, seed=PROTOCOL['calibration_seeds'][train_photo], test=False)
        record = {'train_photo':train_photo, 'test_photo':test_photo,
                  'train_counts':tr['counts'], 'calibration_counts':ca['counts'], 'tests':{}}
        for test_seed in PROTOCOL['test_seeds'][train_photo]:
            te = prepare_examples(manifest,frames['test'],photos,photo=test_photo,
                   n=test_n,seed=test_seed,test=True)
            stats,models = evaluate(tr,ca,te,seed=PROTOCOL['train_seeds'][train_photo],
                                    iterations=bootstrap)
            record['tests'][str(test_seed)] = {'test_counts':te['counts'], 'arms':stats}
            print(f'S53 {train_photo}->{test_photo} seed={test_seed}: '
                  f"S43 {stats['existing_S43']['overall_AUROC_SYNTHETIC_RENDERER_ONLY']:.4f}, "
                  f"S53 {stats['S43_plus_bead_texture_S53']['overall_AUROC_SYNTHETIC_RENDERER_ONLY']:.4f}, "
                  f"clone {stats['S43_plus_bead_texture_S53']['matched_bead_clone_only_AUROC_SYNTHETIC_RENDERER_ONLY']:.4f}",flush=True)
            if private_dir:
                p = private_dir / f'{train_photo}_to_{test_photo}_{test_seed}'
                p.mkdir(parents=True, exist_ok=True)
                for entry in te['previews']:
                    for kind,img in entry['images'].items():
                        cv2.imwrite(str(p / f"scene_{entry['scene_id']:02d}_{kind}.png"),img)
                # The model is a synthetic generator classifier, not native GT.
                from bubble_chamber.s43_artifact_triage import export_simulator_linear_model
                for arm,model in models.items():
                    # S53 has more feature dimensions than S43; keep sklearn
                    # model parameters in JSON without pickle deserialization.
                    sc = model.named_steps['standardscaler']
                    clf = model.named_steps['logisticregression']
                    weights = {'schema':SCHEMA,'synthetic_only':True,
                      'source_photo':train_photo,'test_photo':test_photo,
                      'arm':arm,'feature_names':list(S43_FIELDS)+ (list(S53_FIELDS) if 'S53' in arm else []),
                      'scaler_mean':sc.mean_.tolist(),'scaler_scale':sc.scale_.tolist(),
                      'coefficients':clf.coef_[0].tolist(), 'intercept':float(clf.intercept_[0]),
                      'native_physics_classes_known':False,'native_auto_join_allowed':False}
                    (p / (arm+'_simulator_only.json')).write_text(json.dumps(weights,indent=2)+'\n')
        report['directions'][f'{train_photo}_to_{test_photo}']=record
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--private-preview-dir',type=Path)
    p.add_argument('--train-scenes',type=int,default=40)
    p.add_argument('--cal-scenes',type=int,default=20)
    p.add_argument('--test-scenes',type=int,default=40)
    p.add_argument('--bootstrap',type=int,default=500)
    a=p.parse_args()
    run(a.visualmetro,a.photo_tiles,a.output,train_n=a.train_scenes,cal_n=a.cal_scenes,
        test_n=a.test_scenes,bootstrap=a.bootstrap,private_dir=a.private_preview_dir)
if __name__=='__main__':main()
