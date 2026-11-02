#!/usr/bin/env python3
"""Add train-only position robustness to PAIR geometry in the CT28 HGB pack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import build_ct28_relation_features as relation_features
import build_ct28_scene_plan as scene_builder
from ct28_perturbations import perturb_scene_endpoints
import train_ct28_relation_head as training


ROOT = training.DEFAULT_SCENE_DIR
GEOMETRY = list(training.PAIR_GEOMETRY_FEATURES)


def pair_geometry_rows(candidates: pd.DataFrame, endpoints: pd.DataFrame, *,
                       repetition: int, seed: int, position_noise_mm: float) -> pd.DataFrame:
    by_scene = {scene_id: group for scene_id, group in candidates.groupby('scene_id', sort=False)}
    rows = []
    for scene_id, scene_endpoints in endpoints.groupby('scene_id', sort=False):
        if scene_id not in by_scene:
            continue
        perturbed = perturb_scene_endpoints(
            scene_endpoints, seed=seed, repetition=repetition,
            position_noise_mm=position_noise_mm, tangent_noise_deg=0.0,
        ).set_index('endpoint_id')
        for candidate in by_scene[scene_id].itertuples():
            first_id, second_id = candidate.endpoint_ids.split('|')
            items = []
            for endpoint_id in (first_id, second_id):
                endpoint = perturbed.loc[endpoint_id]
                items.append({
                    'position': endpoint[['x_lps', 'y_lps', 'z_lps']].to_numpy(dtype=float),
                    'tangent': endpoint[['tangent_x', 'tangent_y', 'tangent_z']].to_numpy(dtype=float),
                })
            rows.append({
                'scene_id': scene_id, 'candidate_id': candidate.candidate_id,
                'repetition': repetition,
                **dict(zip(GEOMETRY, scene_builder.pair_geometry(*items))),
            })
    return pd.DataFrame(rows)


def replace_scores(output: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = output.merge(predictions, on=['scene_id', 'candidate_id'],
                          how='left', validate='one_to_one')
    mask = merged.new_geometry.notna()
    merged.loc[mask, 'geometry_score'] = merged.loc[mask, 'new_geometry']
    return merged.drop(columns='new_geometry')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, default=ROOT)
    parser.add_argument('--baseline-dir', type=Path,
                        default=ROOT/'ablations/junction_position_robustness')
    parser.add_argument('--out-dir', type=Path,
                        default=ROOT/'ablations/pair_junction_geometry_robust')
    parser.add_argument('--train-repetitions', type=int, default=5)
    parser.add_argument('--position-noise-mm', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=20260920)
    args = parser.parse_args()
    if (args.train_repetitions, args.position_noise_mm) != (5, 0.8):
        raise ValueError('frozen diagnostic: 5 train repeats at 0.8 mm position noise')

    plan = pd.read_csv(args.scene_dir/'scene_plan_train_val.csv')
    all_candidates = pd.read_csv(args.scene_dir/'scene_candidates_train_val.csv')
    training.validate_scene_inputs(plan, all_candidates)
    baseline_protocol = json.loads((args.baseline_dir/'relation_model_protocol.json').read_text())
    for filename, key in [('scene_plan_train_val.csv', 'scene_plan'),
                          ('scene_candidates_train_val.csv', 'scene_candidates'),
                          ('scene_candidate_features_train_val.csv', 'scene_candidate_features')]:
        if training.sha256(args.scene_dir/filename) != baseline_protocol['source_sha256'][key]:
            raise ValueError(f'frozen source changed: {filename}')
    baseline = joblib.load(args.baseline_dir/'relation_models.joblib')
    if baseline['local_model_family'] != 'hgb':
        raise ValueError('position-robust JUNCTION HGB reference required')
    feature_precision = (
        6 if 'precision' in baseline['local_models']['JUNCTION']['geometry'].named_steps
        else None
    )
    original_scores = pd.read_csv(args.baseline_dir/'relation_local_scores_train_val.csv')
    features = pd.read_csv(args.scene_dir/'scene_candidate_features_train_val.csv', low_memory=False)
    candidates = all_candidates[all_candidates.candidate_type.eq('PAIR')].merge(
        plan[['scene_id', 'scan_id', 'split']], on='scene_id', validate='many_to_one')
    endpoints = pd.read_csv(args.scene_dir/'scene_endpoints_train_val.csv').merge(
        plan[['scene_id', 'scan_id', 'split']], on='scene_id', validate='many_to_one')
    clean = candidates[['scene_id', 'candidate_id', 'scan_id', 'split', 'y']].merge(
        features[['scene_id', 'candidate_id', *GEOMETRY]],
        on=['scene_id', 'candidate_id'], validate='one_to_one')
    train = clean[clean.split.eq('train')].copy()
    val = clean[clean.split.eq('val')].copy()
    # Use the original shared local-score fold assignment for PAIR.
    fold_rows = original_scores[original_scores.split.eq('train')]
    fold_counts = fold_rows.groupby('scan_id').local_oof_fold.nunique()
    if not fold_counts.eq(1).all():
        raise ValueError('a train patient occupies multiple OOF folds')
    folds = fold_rows.groupby('scan_id').local_oof_fold.first().astype(int).to_dict()
    if set(folds) != training.EXPECTED_PATIENT_IDS['train'] or set(folds.values()) != set(range(5)):
        raise ValueError('unexpected patient OOF fold mapping')

    train_clean = train.copy()
    train_clean['weight'] = 0.5
    augmented = []
    for repetition in range(args.train_repetitions):
        part = pair_geometry_rows(
            candidates[candidates.split.eq('train')], endpoints[endpoints.split.eq('train')],
            repetition=repetition, seed=args.seed, position_noise_mm=args.position_noise_mm,
        ).merge(train[['scene_id', 'candidate_id', 'scan_id', 'y']],
                on=['scene_id', 'candidate_id'], validate='one_to_one')
        if len(part) != len(train):
            raise ValueError('incomplete train augmentation')
        part['weight'] = 0.5 / args.train_repetitions
        augmented.append(part)
        print(f'PAIR train position repeat {repetition + 1}/{args.train_repetitions}', flush=True)
    fit_data = pd.concat([train_clean, *augmented], ignore_index=True)
    output = original_scores.copy()
    local = {kind: dict(models) for kind, models in baseline['local_models'].items()}
    for fold in range(5):
        held = {patient for patient, assigned in folds.items() if assigned == fold}
        fit = fit_data[~fit_data.scan_id.isin(held)]
        target = train[train.scan_id.isin(held)]
        if fit.y.nunique() != 2 or target.y.nunique() != 2:
            raise ValueError(f'PAIR fold {fold} lacks a class')
        model = training.local_model('hgb', feature_precision)
        model.fit(fit[GEOMETRY], fit.y, hgb__sample_weight=fit.weight)
        predicted = target[['scene_id', 'candidate_id']].copy()
        predicted['new_geometry'] = model.predict_proba(target[GEOMETRY])[:, 1]
        output = replace_scores(output, predicted)
        print(f'PAIR OOF fold {fold + 1}/5', flush=True)
    model = training.local_model('hgb', feature_precision)
    model.fit(fit_data[GEOMETRY], fit_data.y, hgb__sample_weight=fit_data.weight)
    local['PAIR']['geometry'] = model
    predicted = val[['scene_id', 'candidate_id']].copy()
    predicted['new_geometry'] = model.predict_proba(val[GEOMETRY])[:, 1]
    output = replace_scores(output, predicted)
    if not np.isfinite(output[['geometry_score', 'ct_score', 'geometry_plus_ct_score']].to_numpy()).all():
        raise ValueError('incomplete local scores')
    before = original_scores[original_scores.candidate_type.eq('JUNCTION')].reset_index(drop=True)
    after = output[output.candidate_type.eq('JUNCTION')].reset_index(drop=True)
    if not before.equals(after):
        raise ValueError('JUNCTION local scores changed')
    pair_mask = original_scores.candidate_type.eq('PAIR')
    if not original_scores.loc[pair_mask, ['ct_score', 'geometry_plus_ct_score']].equals(
        output.loc[pair_mask, ['ct_score', 'geometry_plus_ct_score']]
    ):
        raise ValueError('PAIR CT/fusion scores changed')
    if len(output) != len(original_scores) or not output[['scene_id', 'candidate_id']].equals(
        original_scores[['scene_id', 'candidate_id']]
    ):
        raise ValueError('candidate identity/order changed')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.out_dir/'relation_local_scores_train_val.csv'
    output.to_csv(score_path, index=False)
    plan_rows = relation_features.read_csv(args.scene_dir/'scene_plan_train_val.csv')
    candidate_rows = relation_features.read_csv(args.scene_dir/'scene_candidates_train_val.csv')
    score_rows = relation_features.read_csv(score_path)
    heads, metrics, predictions, columns_by_head = {}, {}, [], {}
    for name, streams in [('geometry_only', ['geometry_score']),
                          ('geometry_plus_ct', ['geometry_plus_ct_score']),
                          ('ct_conditioned', ['geometry_score', 'ct_score'])]:
        rows, columns = training.build_relation_feature_rows(
            plan_rows, candidate_rows, score_rows, streams)
        metric, prediction, head = training.fit_relation_head(rows, columns, .85, name)
        heads[name], metrics[name], columns_by_head[name] = head, metric, columns
        predictions.extend(prediction)
    model_pack = dict(baseline)
    model_pack['local_models'] = local
    model_pack['relation_heads'] = heads
    for name, key in [('geometry_only', 'geometry_relation_features'),
                      ('geometry_plus_ct', 'geometry_plus_ct_relation_features'),
                      ('ct_conditioned', 'ct_conditioned_relation_features')]:
        model_pack[key] = columns_by_head[name]
    joblib.dump(model_pack, args.out_dir/'relation_models.joblib')
    pd.DataFrame(predictions).to_csv(args.out_dir/'relation_predictions_val.csv', index=False)
    (args.out_dir/'relation_metrics_val.json').write_text(json.dumps({'models': metrics}, indent=2)+'\n')
    protocol = {
        'status': 'validation_ablation_not_frozen', 'held_out_test_accessed': False,
        'source_pack': str(args.baseline_dir),
        'local_change': 'PAIR geometry HGB only; JUNCTION position-robust geometry and all CT/fusion local models unchanged',
        'local_feature_precision_decimals': feature_precision,
        'relation_heads': 'retrained on revised patient-OOF local geometry scores',
        'patient_oof_folds_pair_geometry': {str(k): v for k, v in sorted(folds.items())},
        'patient_oof_folds_junction_geometry': baseline_protocol['patient_oof_folds'],
        'train_perturbations': 5, 'position_noise_mm_sigma': 0.8,
        'tangent_noise_deg_sigma': 0.0,
        'clean_train_weight': 0.5, 'perturbed_train_weight_total': 0.5,
        'source_sha256': baseline_protocol['source_sha256'],
        'input_pack_sha256': {
            name: training.sha256(args.baseline_dir/name)
            for name in ('relation_models.joblib', 'relation_local_scores_train_val.csv')},
        'output_sha256': {
            name: training.sha256(args.out_dir/name)
            for name in ('relation_local_scores_train_val.csv', 'relation_models.joblib',
                         'relation_predictions_val.csv', 'relation_metrics_val.json')},
    }
    (args.out_dir/'relation_model_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps({name: {'raw_accuracy': result['raw']['accuracy'],
                             'tau_accuracy': result['tau_0_85']['accuracy']}
                      for name, result in metrics.items()}, indent=2))


if __name__ == '__main__':
    main()
