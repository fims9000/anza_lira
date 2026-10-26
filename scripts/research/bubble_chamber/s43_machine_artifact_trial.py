#!/usr/bin/env python3
"""S43 held-out synthetic-origin scratch triage + private native-only advisory.

All observed "truth" belongs to injected simulated strokes. No native particle
GT is claimed, no real H50Z pseudo labels or graph joins are modified.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
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
from bubble_chamber.s35_simulated_gaps import (
    frame_split, extract_track_templates, make_scene,
)
from bubble_chamber.s39_machine_auto import ridge_evidence
from bubble_chamber.s43_artifact_triage import (
    FEATURE_NAMES, SCHEMA, path_features, score_native_candidates,
    annotate_graph_advisory, export_simulator_linear_model,
)

SEEDS = {'005': 202610435, '006': 202610436}
SCENES = {'train': 24, 'calibration': 12, 'test': 24}
FAMILIES = ('matched_same_renderer', 'straight_hairline', 'double_rail',
            'broken_streak', 'wobbly_streak')
UNSEEN_FAMILY = 'unseen_broad_patchy_streak'


def frozen_frame_folds(manifest: dict) -> dict[str, set[str]]:
    development, test = frame_split(manifest)
    ordered = sorted(development, key=lambda fid: sha256(fid.encode()).hexdigest())
    folds = {'train': set(ordered[:72]), 'calibration': set(ordered[72:]), 'test': test}
    if ([len(folds[k]) for k in ('train', 'calibration', 'test')] != [72, 37, 36]
            or len(set.union(*folds.values())) != len(manifest['frames'])
            or any(folds[a] & folds[b] for a in folds for b in folds if a != b)):
        raise AssertionError('Measured source frame split leakage')
    return folds


def _stroke(canvas: np.ndarray, pts: np.ndarray, *, ink: float, width: int,
            broken: bool = False, patchy: bool = False) -> None:
    if patchy:
        cuts = [(i, min(i + 5, len(pts))) for i in range(0, len(pts), 11)]
    elif broken:
        cuts = [(i, min(i + 9, len(pts))) for i in range(0, len(pts), 15)]
    else:
        cuts = [(0, len(pts))]
    for a, b in cuts:
        if b - a < 2:
            continue
        pixels = np.zeros(canvas.shape, dtype=np.uint8)
        cv2.polylines(pixels, [np.rint(pts[a:b]).astype(np.int32)], False,
                      1, width, lineType=cv2.LINE_8)
        canvas -= ink * pixels


def add_synthetic_scratches(scene, rng: np.random.Generator, *, open_set: bool):
    """Same underlying grayscale and ink range as track injection.

    The test-only family is never present in fitting/calibration. Neither
    generator's origin is a claim about physical film-defect truth.
    """
    canvas = scene.gray.astype(np.float32)
    negatives = [('matched_same_renderer', scene.right[2])]
    for family in FAMILIES[1:] + ((UNSEEN_FAMILY,) if open_set else ()):
        u = np.linspace(0, 1, 81)
        y0 = rng.uniform(45, 145)
        slope = rng.uniform(-36, 36)
        y = y0 + slope * (u - .5)
        if family == 'wobbly_streak':
            y += rng.uniform(2., 5.) * np.sin(3 * np.pi * u + rng.uniform(-np.pi, np.pi))
        if family == UNSEEN_FAMILY:
            y += rng.uniform(0., 3.) * np.sin(5 * np.pi * u)
        path = np.column_stack((20 + 151 * u, np.clip(y, 9, 182)))
        ink = float(rng.uniform(27, 75))
        width = int(rng.integers(1, 4))
        if family == 'double_rail':
            # A rail surface can look like two particle tracks. Mark one stroke
            # centerline, not the blank geometric midpoint.
            one = path + np.array([0, 2.5])
            other = path - np.array([0, 2.5])
            _stroke(canvas, one, ink=ink, width=1)
            _stroke(canvas, other, ink=ink, width=1)
            path = one
        elif family == 'broken_streak':
            _stroke(canvas, path, ink=ink, width=width, broken=True)
        elif family == UNSEEN_FAMILY:
            _stroke(canvas, path, ink=ink, width=6, patchy=True)
        else:
            _stroke(canvas, path, ink=ink, width=width)
        negatives.append((family, path))
    rendered = cv2.GaussianBlur(np.clip(canvas, 0, 255).astype('uint8'),
                                (0, 0), .35)
    return rendered, negatives


def make_examples(visualmetro: dict, folders: dict[str, set[str]], tiles: Path,
                  *, source_photo: str, fold: str, count: int, seed: int) -> dict:
    if source_photo not in {'005', '006'} or fold not in SCENES or count < 2:
        raise ValueError('Invalid frozen train/test request')
    images = sorted(tiles.glob(f'H50Z_030_01_{source_photo}_*.png'))
    if len(images) != (21 if source_photo == '005' else 18):
        raise ValueError('Private H50Z crop count wrong')
    templates = extract_track_templates(visualmetro, chosen_frames=folders[fold])
    rng = np.random.default_rng(seed)
    feat, lab, family, scene_ids = [], [], [], []
    for i in range(count):
        path = images[int(rng.integers(len(images)))]
        s = make_scene(path, templates, seed=int(rng.integers(0, 2**31)),
                       scenario=('parallel', 'crossing', 'scratched', 'empty_gap')[i % 4])
        if not set(s.source_frame_ids) <= folders[fold]:
            raise AssertionError('Leaked visualMetro source track frame')
        gray, scratches = add_synthetic_scratches(s, rng, open_set=(fold == 'test'))
        hessian = ridge_evidence(gray)
        positives = [*s.left, *s.right[:2]]
        for points in positives:
            feat.append(path_features(gray, points, maps=hessian))
            lab.append(1)
            family.append('simulated_measured_curve')
            scene_ids.append(i)
        for name, points in scratches:
            feat.append(path_features(gray, points, maps=hessian))
            lab.append(0)
            family.append(name)
            scene_ids.append(i)
    x = np.asarray(feat)
    if not np.isfinite(x).all() or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError('Bad features')
    return {'X': x, 'labels': np.asarray(lab), 'families': np.asarray(family),
            'scenes': np.asarray(scene_ids), 'photo': source_photo,
            'fold': fold, 'source_frame_count': len(folders[fold])}


def select_calibration_threshold(score: np.ndarray, negative: np.ndarray) -> float:
    """Fixed q90; not optimized on test. Strict greater-than with ties."""
    if score.shape != negative.shape or score.ndim != 1 or not np.isfinite(score).all():
        raise ValueError('Invalid calibration data')
    bad = score[negative]
    if len(bad) < 10:
        raise ValueError('Need multiple scratch families and scenes')
    return float(np.quantile(bad, .90))


def _fpr(y, s, cutoff):
    return float(np.mean(s[y == 0] > cutoff))


def _tpr(y, s, cutoff):
    return float(np.mean(s[y == 1] > cutoff))


def scene_bootstrap_auc(test: dict, score: np.ndarray,
                        baseline: np.ndarray, *, seed: int) -> list[float]:
    ids = np.unique(test['scenes'])
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(1200):
        sample = rng.choice(ids, size=len(ids), replace=True)
        indices = np.concatenate([np.flatnonzero(test['scenes'] == k) for k in sample])
        yy = test['labels'][indices]
        deltas.append(roc_auc_score(yy, score[indices]) -
                      roc_auc_score(yy, baseline[indices]))
    return np.quantile(deltas, [.025, .975]).round(6).tolist()


def evaluate_arm(train, cal, test, *, seed: int):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.0, class_weight='balanced',
                                           max_iter=1000, random_state=seed))
    clf.fit(train['X'], train['labels'])
    variants = {
        'raw_contrast': lambda data: data['X'][:, 3],
        'raw_hessian_support': lambda data: data['X'][:, 12],
        'geometry_plus_profile_linear': lambda data: clf.predict_proba(data['X'])[:, 1],
    }
    summary = {}
    for name, predictor in variants.items():
        cal_scores = predictor(cal)
        cut = select_calibration_threshold(cal_scores, cal['labels'] == 0)
        pred = predictor(test)
        negatives = {kind: round(float(np.mean(pred[(test['labels'] == 0) &
                                         (test['families'] == kind)] > cut)), 5)
                     for kind in sorted(set(test['families'][test['labels'] == 0]))}
        summary[name] = {
            'synthetic_origin_pooled_AUROC': round(float(roc_auc_score(test['labels'], pred)), 6),
            'calibration_scratch_acceptance': round(_fpr(cal['labels'], cal_scores, cut), 6),
            'test_simulated_track_acceptance': round(_tpr(test['labels'], pred, cut), 6),
            'test_simulated_scratch_acceptance': round(_fpr(test['labels'], pred, cut), 6),
            'test_simulated_scratch_acceptance_by_origin': negatives,
            'calibrated_threshold_SIMULATOR_ONLY': round(cut, 6),
        }
    summary['geometry_plus_profile_linear']['paired_scene_bootstrap_AUROC_difference_vs_contrast_CI_SIMULATION_ONLY'] = (
        scene_bootstrap_auc(test, variants['geometry_plus_profile_linear'](test),
                            variants['raw_contrast'](test), seed=seed + 1400))
    summary['geometry_plus_profile_linear']['paired_scene_bootstrap_AUROC_difference_vs_hessian_CI_SIMULATION_ONLY'] = (
        scene_bootstrap_auc(test, variants['geometry_plus_profile_linear'](test),
                            variants['raw_hessian_support'](test), seed=seed + 1500))
    return summary, clf


def apply_to_native(s39_path: Path, graph_path: Path, fiducial_graph_path: Path,
                    images_path: Path, model_by_native_photo: dict, out: Path) -> dict:
    doc = json.loads(s39_path.read_text())
    graphs = json.loads(graph_path.read_text())
    guarded = json.loads(fiducial_graph_path.read_text())
    if len(doc.get('tiles', [])) != 39 or doc.get('native_expert_gt_created') != 0:
        raise ValueError('Wrong native H50Z provenance')
    by_graph = {x['crop_id']: x for x in graphs['graphs']}
    by_guard = {x['crop_id']: x['machine_graph_risk_annotation_PRIVATE'] for x in guarded['graphs']}
    if len(by_graph) != 39 or len(by_guard) != 39:
        raise ValueError('Wrong S39/S40 graph identity mapping')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    totals = Counter()
    for tile in doc['tiles']:
        crop_id = tile['crop_id']
        photo = tile['photo_id']
        path = images_path / f'{crop_id}.png'
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or sha256(gray.tobytes()).hexdigest() != tile['source_gray_sha256']:
            raise ValueError('Wrong/altered genuine H50Z pixels')
        ranked = score_native_candidates(gray, tile['proposals'], model_by_native_photo[photo])
        graph = annotate_graph_advisory(by_graph[crop_id], ranked, by_guard[crop_id])
        if len(ranked) != len(tile['proposals']):
            raise AssertionError('Native candidate accidentally lost')
        # Values were fitted on synthetic line-origins, not human annotated H50Z.
        sim = [p['SIMULATED_measured_curve_similarity_NOT_NATIVE_PROBABILITY'] for p in ranked]
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for proposal, prob in zip(tile['proposals'], sim):
            # Amber means low simulated affinity, not proven scratch.
            color = (0, 180, 240) if prob < .5 else (90, 205, 65)
            cv2.polylines(bgr, [np.rint(np.asarray(proposal['points_xy'])).astype('int32')],
                          False, color, 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f'{crop_id}_S43_ADVISORY.jpg'), bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append({'crop_id': crop_id, 'photo': photo,
                     'original_H50Z_source_sha256': tile['source_gray_sha256'],
                     'candidate_scores_SIMULATOR_ONLY': ranked,
                     'graph_advisory': graph})
        totals['tiles'] += 1
        totals['candidate_proposals_preserved'] += len(ranked)
        totals['synthetic_model_score_below_0p5_NOT_ARTIFACT_GT'] += sum(v < .5 for v in sim)
        totals['graph_proposals_preserved'] += len(graph['edges'])
        totals['known_fiducial_risk_graph_edges'] += graph['fiducial_risk_edges']
    manifest = {'schema': SCHEMA, 'photos': ['005', '006'], 'tiles': rows,
                'native_particle_labels': 0, 'native_confirmed_joins': 0}
    (out / 'S43_PRIVATE_MACHINE_RANKING.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return dict(totals)


def run(visualmetro: Path, tiles: Path, s39: Path, s39_graph: Path,
        s40_graph: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest = audited_large_manifest(visualmetro)
    folds = frozen_frame_folds(manifest)
    print('S43 source-frame split', {k: len(v) for k, v in folds.items()}, flush=True)
    results = {}
    native_models = {}
    for test_photo in ('006', '005'):
        train_photo = '005' if test_photo == '006' else '006'
        seed = SEEDS[train_photo]
        train = make_examples(manifest, folds, tiles, source_photo=train_photo,
                              fold='train', count=SCENES['train'], seed=seed + 11)
        cal = make_examples(manifest, folds, tiles, source_photo=train_photo,
                            fold='calibration', count=SCENES['calibration'], seed=seed + 22)
        test = make_examples(manifest, folds, tiles, source_photo=test_photo,
                             fold='test', count=SCENES['test'], seed=seed + 33)
        model_result, clf = evaluate_arm(train, cal, test, seed=seed)
        results[f'{train_photo}_train_{test_photo}_test'] = model_result
        native_models[test_photo] = clf
        portable = export_simulator_linear_model(clf, source_photo=train_photo)
        (output / f'PRIVATE_S43_SIMULATOR_LINEAR_FIT_FROM_{train_photo}.json').write_text(
            json.dumps(portable, indent=2) + '\n')
        print(f'S43 {train_photo} -> {test_photo} test complete', flush=True)
    hard_matched_failure = any(
        summary['geometry_plus_profile_linear']['test_simulated_scratch_acceptance_by_origin']['matched_same_renderer'] > .10
        for summary in results.values()
    )
    native = apply_to_native(s39, s39_graph, s40_graph, tiles, native_models,
                             output / 'PRIVATE_native_H50Z_advisory')
    report = {
        'schema': SCHEMA, 'experiment_date': '2026-10-25',
        'source_145_measured_frames': 145, 'measured_frame_disjoint_splits':
            {k: len(v) for k, v in folds.items()},
        'real_H50Z_photo_count': 2, 'independent_physical_runs': 1,
        'physical_track_ground_truth_labels': 0,
        'injection_scenes_per_direction': SCENES,
        'simulated_scratch_types_in_train': list(FAMILIES),
        'heldout_unseen_scratch_type': UNSEEN_FAMILY,
        'methods': ['raw_contrast', 'raw_hessian_support', 'geometry_plus_profile_linear'],
        'test': results,
        'native_machine_advisory_aggregate': native,
        'native_scratch_classification_proven': False,
        'native_particle_identity_proven': False,
        'native_auto_join_permitted': False,
        'automatic_native_candidates_removed': 0,
        'automatic_native_pseudo_labels_created': 0,
        'hard_matched_decoy_failure_both_or_either_direction': hard_matched_failure,
        'deployment_decision': 'ADVISORY_ONLY_NO_SCRATCH_GATE',
        'uncertainty_warning': 'Scene bootstrap is conditional on this simulator and 1 physical H50Z run, not native-domain confidence.',
    }
    (output / 'S43_PUBLIC_AGGREGATES.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--visualmetro', type=Path, required=True)
    p.add_argument('--photo-tiles', type=Path, required=True)
    p.add_argument('--s39-private', type=Path, required=True)
    p.add_argument('--s39-graph-private', type=Path, required=True)
    p.add_argument('--s40-graph-private', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r = run(a.visualmetro, a.photo_tiles, a.s39_private,
            a.s39_graph_private, a.s40_graph_private, a.output)
    print(json.dumps({
        'native': r['native_machine_advisory_aggregate'],
        'deploy': r['deployment_decision'],
        'test': {k: v['geometry_plus_profile_linear'] for k, v in r['test'].items()},
    }, indent=2))


if __name__ == '__main__':
    main()
