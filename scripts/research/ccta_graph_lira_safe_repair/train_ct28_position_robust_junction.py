#!/usr/bin/env python3
"""Train-only position augmentation of JUNCTION geometry evidence."""
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
GEOMETRY = list(training.JUNCTION_GEOMETRY_FEATURES)


def geometry_rows(candidates: pd.DataFrame, endpoints: pd.DataFrame, *, repetition: int,
                  seed: int, position_noise_mm: float) -> pd.DataFrame:
    rows = []
    candidates_by_scene = {scene_id: group for scene_id, group in candidates.groupby('scene_id', sort=False)}
    for scene_id, scene_endpoints in endpoints.groupby('scene_id', sort=False):
        if scene_id not in candidates_by_scene:
            continue
        perturbed = perturb_scene_endpoints(
            scene_endpoints, seed=seed, repetition=repetition,
            position_noise_mm=position_noise_mm, tangent_noise_deg=0.0)
        lookup = perturbed.set_index('endpoint_id')
        for candidate in candidates_by_scene[scene_id].itertuples():
            items = []
            for endpoint_id in candidate.endpoint_ids.split('|'):
                endpoint = lookup.loc[endpoint_id]
                items.append({'position': endpoint[['x_lps','y_lps','z_lps']].to_numpy(dtype=float),
                              'tangent': endpoint[['tangent_x','tangent_y','tangent_z']].to_numpy(dtype=float)})
            values = scene_builder.oriented_junction_geometry(items)
            full_geometry = dict(zip(training.JUNCTION_GEOMETRY_FEATURES, values))
            rows.append({'scene_id': scene_id, 'candidate_id': candidate.candidate_id,
                         'repetition': repetition, **full_geometry})
    return pd.DataFrame(rows)


def rank_summary(scores: pd.DataFrame) -> dict[str, object]:
    rows = []
    for (scene_id, repetition), group in scores.groupby(['scene_id','repetition']):
        if not group.y.eq(1).any():
            continue
        best = group.loc[group.geometry_score.idxmax()]
        rows.append({'scene_id': scene_id, 'repetition': int(repetition),
                     'true_top1': int(best.y == 1),
                     'true_score': float(group.loc[group.y.eq(1),'geometry_score'].max()),
                     'false_score': float(group.loc[group.y.eq(0),'geometry_score'].max())
                     if group.y.eq(0).any() else None})
    frame = pd.DataFrame(rows)
    return {'junction_scenes': int(frame.scene_id.nunique()),
            'scene_repeats': int(len(frame)),
            'top1_rate': float(frame.true_top1.mean()),
            'top1_by_scene_rate': float(frame.groupby('scene_id').true_top1.mean().mean()),
            'mean_true_score': float(frame.true_score.mean()),
            'mean_false_score': float(frame.false_score.mean())}


def main() -> None:
    global GEOMETRY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, default=ROOT)
    parser.add_argument('--baseline-dir', type=Path,
                        default=ROOT/'ablations/junction_numeric_robustness')
    parser.add_argument('--out-dir', type=Path,
                        default=ROOT/'ablations/junction_position_robustness')
    parser.add_argument('--train-repetitions', type=int, default=5)
    parser.add_argument('--validation-repetitions', type=int, default=15)
    parser.add_argument('--position-noise-mm', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--drop-junction-span', action='store_true',
                        help='Ablate junction_span_mm from the local geometry model; candidate generation is unchanged.')
    parser.add_argument('--drop-geometry-feature', choices=GEOMETRY,
                        help='Ablate one local JUNCTION geometry input; candidate generation is unchanged.')
    args = parser.parse_args()
    if args.drop_junction_span and args.drop_geometry_feature:
        raise ValueError('choose only one geometry feature to ablate')
    dropped_feature = ('junction_span_mm' if args.drop_junction_span
                       else args.drop_geometry_feature)
    if dropped_feature:
        GEOMETRY = [name for name in GEOMETRY if name != dropped_feature]
        if args.out_dir == ROOT/'ablations/junction_position_robustness':
            raise ValueError('geometry-feature ablation requires an explicit --out-dir')
    if (args.train_repetitions, args.validation_repetitions, args.position_noise_mm) != (5,15,0.8):
        raise ValueError('this ablation uses 5 train repeats, 15 validation repeats and 0.8 mm position noise')
    plan = pd.read_csv(args.scene_dir/'scene_plan_train_val.csv')
    all_candidates = pd.read_csv(args.scene_dir/'scene_candidates_train_val.csv')
    training.validate_scene_inputs(plan, all_candidates)
    candidates = all_candidates[all_candidates.candidate_type.eq('JUNCTION')].merge(
        plan[['scene_id','scan_id','split']], on='scene_id', validate='many_to_one')
    endpoints = pd.read_csv(args.scene_dir/'scene_endpoints_train_val.csv').merge(
        plan[['scene_id','scan_id','split']], on='scene_id', validate='many_to_one')
    scores = pd.read_csv(args.baseline_dir/'relation_local_scores_train_val.csv')
    baseline = joblib.load(args.baseline_dir/'relation_models.joblib')
    if baseline['local_model_family'] != 'hgb':
        raise ValueError('local HGB reference required')
    feature_precision = (
        6 if 'precision' in baseline['local_models']['JUNCTION']['geometry'].named_steps
        else None
    )
    baseline_protocol = json.loads((args.baseline_dir/'relation_model_protocol.json').read_text())
    for filename, key in [('scene_plan_train_val.csv','scene_plan'),
                          ('scene_candidates_train_val.csv','scene_candidates'),
                          ('scene_candidate_features_train_val.csv','scene_candidate_features')]:
        if training.sha256(args.scene_dir/filename) != baseline_protocol['source_sha256'][key]:
            raise ValueError(f'source changed: {filename}')
    features = pd.read_csv(args.scene_dir/'scene_candidate_features_train_val.csv', low_memory=False)
    clean = candidates[['scene_id','candidate_id','scan_id','split','y']].merge(
        features[['scene_id','candidate_id',*GEOMETRY]], on=['scene_id','candidate_id'], validate='one_to_one')
    train = clean[clean.split.eq('train')]
    val = clean[clean.split.eq('val')]
    folds = training.patient_folds(train, 5)
    augmented = []
    for repetition in range(args.train_repetitions):
        part = geometry_rows(candidates[candidates.split.eq('train')],
                             endpoints[endpoints.split.eq('train')],
                             repetition=repetition, seed=args.seed,
                             position_noise_mm=args.position_noise_mm)
        part = part.merge(train[['scene_id','candidate_id','scan_id','y']],
                          on=['scene_id','candidate_id'], validate='one_to_one')
        if len(part) != len(train):
            raise ValueError('incomplete train augmentation')
        augmented.append(part)
        print(f'train position repeat {repetition+1}/{args.train_repetitions}', flush=True)
    train_clean = train.copy()
    train_clean['weight'] = 0.5
    for part in augmented:
        part['weight'] = 0.5 / args.train_repetitions
    fit_data = pd.concat([train_clean, *augmented], ignore_index=True)
    output = scores.copy()
    local = dict(baseline['local_models'])
    local['JUNCTION'] = dict(local['JUNCTION'])
    for fold in sorted(set(folds.values())):
        held = {patient for patient, assigned in folds.items() if assigned == fold}
        fit = fit_data[~fit_data.scan_id.isin(held)]
        target = train[train.scan_id.isin(held)]
        model = training.local_model('hgb',feature_precision)
        model.fit(fit[GEOMETRY], fit.y, hgb__sample_weight=fit.weight)
        lookup = target[['scene_id','candidate_id']].copy()
        lookup['geometry_score'] = model.predict_proba(target[GEOMETRY])[:,1]
        output = output.merge(lookup.rename(columns={'geometry_score':'new_geometry'}),
                              on=['scene_id','candidate_id'],how='left',validate='one_to_one')
        mask = output.new_geometry.notna()
        output.loc[mask,'geometry_score'] = output.loc[mask,'new_geometry']
        output.drop(columns='new_geometry',inplace=True)
    model = training.local_model('hgb',feature_precision)
    model.fit(fit_data[GEOMETRY],fit_data.y,hgb__sample_weight=fit_data.weight)
    local['JUNCTION']['geometry'] = model
    lookup = val[['scene_id','candidate_id']].copy()
    lookup['geometry_score'] = model.predict_proba(val[GEOMETRY])[:,1]
    output = output.merge(lookup.rename(columns={'geometry_score':'new_geometry'}),
                          on=['scene_id','candidate_id'],how='left',validate='one_to_one')
    mask = output.new_geometry.notna()
    output.loc[mask,'geometry_score'] = output.loc[mask,'new_geometry']
    output.drop(columns='new_geometry',inplace=True)
    assert len(output)==len(scores) and output[['geometry_score','ct_score','geometry_plus_ct_score']].notna().all().all()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    score_path = args.out_dir/'relation_local_scores_train_val.csv'
    output.to_csv(score_path,index=False)

    plan_rows=relation_features.read_csv(args.scene_dir/'scene_plan_train_val.csv')
    candidate_rows=relation_features.read_csv(args.scene_dir/'scene_candidates_train_val.csv')
    score_rows=relation_features.read_csv(score_path)
    heads={}
    metrics={}
    predictions=[]
    columns_by_head={}
    for name, streams in [('geometry_only',['geometry_score']),
                          ('geometry_plus_ct',['geometry_plus_ct_score']),
                          ('ct_conditioned',['geometry_score','ct_score'])]:
        rows, columns=training.build_relation_feature_rows(plan_rows,candidate_rows,score_rows,streams)
        metric,prediction,head=training.fit_relation_head(rows,columns,.85,name)
        heads[name]=head
        metrics[name]=metric
        predictions.extend(prediction)
        columns_by_head[name]=columns
    model_pack=dict(baseline)
    model_pack['local_models']=local
    model_pack['relation_heads']=heads
    model_pack['geometry_relation_features']=columns_by_head['geometry_only']
    model_pack['geometry_plus_ct_relation_features']=columns_by_head['geometry_plus_ct']
    model_pack['ct_conditioned_relation_features']=columns_by_head['ct_conditioned']
    joblib.dump(model_pack,args.out_dir/'relation_models.joblib')
    pd.DataFrame(predictions).to_csv(args.out_dir/'relation_predictions_val.csv',index=False)
    (args.out_dir/'relation_metrics_val.json').write_text(json.dumps({'models':metrics},indent=2)+'\n')

    validation_candidates=candidates[candidates.split.eq('val')]
    validation_endpoints=endpoints[endpoints.split.eq('val')]
    rank_rows=[]
    for repetition in range(args.validation_repetitions):
        part=geometry_rows(validation_candidates,validation_endpoints,repetition=repetition,
                           seed=args.seed,position_noise_mm=args.position_noise_mm)
        part=part.merge(validation_candidates[['scene_id','candidate_id','y']],
                        on=['scene_id','candidate_id'],validate='one_to_one')
        for label, evaluated_model in [('baseline',baseline['local_models']['JUNCTION']['geometry']),
                                       ('position_robust',model)]:
            scored=part[['scene_id','candidate_id','y','repetition']].copy()
            model_columns = list(evaluated_model.feature_names_in_)
            scored['geometry_score']=evaluated_model.predict_proba(part[model_columns])[:,1]
            scored['model']=label
            rank_rows.append(scored)
        print(f'validation position repeat {repetition+1}/{args.validation_repetitions}',flush=True)
    rank_data=pd.concat(rank_rows,ignore_index=True)
    rank_data.to_csv(args.out_dir/'junction_position_rank_scores_val.csv.gz',index=False,
                     compression={'method':'gzip','mtime':0})
    ranking={label:rank_summary(part) for label,part in rank_data.groupby('model')}
    (args.out_dir/'junction_position_ranking_val.json').write_text(json.dumps(ranking,indent=2)+'\n')
    protocol={'status':'train_val_diagnostic','held_out_test_accessed':False,
              'local_feature_precision_decimals':feature_precision,
              'train_perturbations':5,'validation_perturbations':15,
              'position_noise_mm_sigma':0.8,'tangent_noise_deg_sigma':0.0,
              'clean_train_weight':0.5,'perturbed_train_weight_total':0.5,
              'patient_oof_folds':{str(k):v for k,v in sorted(folds.items())},
              'local_change':'JUNCTION geometry HGB only; all other local models frozen',
              'junction_geometry_features': GEOMETRY,
              'junction_span_removed': args.drop_junction_span,
              'dropped_geometry_feature': dropped_feature,
              'relation_heads':'retrained on revised patient-OOF geometry scores',
              'source_sha256':baseline_protocol['source_sha256'],
              'output_sha256':{
                  name: training.sha256(args.out_dir/name)
                  for name in ('relation_local_scores_train_val.csv',
                               'relation_models.joblib', 'relation_predictions_val.csv',
                               'relation_metrics_val.json',
                               'junction_position_rank_scores_val.csv.gz',
                               'junction_position_ranking_val.json')},
              'ranking':ranking}
    (args.out_dir/'relation_model_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'ranking':ranking,'relation_raw_accuracy':
                      {k:v['raw']['accuracy'] for k,v in metrics.items()}},indent=2))


if __name__=='__main__':
    main()
