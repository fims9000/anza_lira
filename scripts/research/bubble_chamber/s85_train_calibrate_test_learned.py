#!/usr/bin/env python3
"""S85: strictly isolated TRAIN -> CAL -> fresh TEST for learned pixel fragments.

Requires authorized PRIVATE H50Z tiles and visualMetro source data for real
scientific execution. Never claims genuine H50Z physical tracking accuracy.
Public aggregates exclude pixels, source point coordinates and model weights.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from bubble_chamber.s85_fragment_ranker import (
    ARMS, FEATURE_NAMES, FEATURE_SCHEMA, MODEL_SCHEMA, candidate_features,
    fit_train_only, model_sha256, select_extras,
)
from bubble_chamber.s85_selection_gate import choose_cal_only
from bubble_chamber.s85_image_disturbances import condition_pixels

EXPERIMENT_ID = 'ANZA-LIRA-BC-S85-20261027'
SCHEMA = 'S85_TRAIN_CAL_TEST_SYNTHETIC_ON_PRIVATE_PHOTO_V1'
PROTOCOL = ROOT / 'docs/bubble_chamber/S85_FROZEN_LEARNED_FRAGMENT_GATE_2026_10_27.md'
OPERATOR = ROOT / 'bubble_chamber/s85_fragment_ranker.py'
LABELS = ROOT / 'bubble_chamber/s85_train_reference_labels.py'
GATE = ROOT / 'bubble_chamber/s85_selection_gate.py'
NOISE = ROOT / 'bubble_chamber/s85_image_disturbances.py'
S80_SOURCE = ROOT / 'bubble_chamber/s80_multiscale_instances.py'
S84_SOURCE = ROOT / 'bubble_chamber/s84_graph_context_candidates.py'
FROZEN_S72_CAL = ROOT / 'docs/bubble_chamber/S72_PUBLIC_FROZEN_CAL_2026_10_26.json'
FROZEN_S80_CAL = ROOT / 'docs/bubble_chamber/S80_PUBLIC_FROZEN_CAL_2026_10_26.json'
FROZEN_S84_CAL = ROOT / 'docs/bubble_chamber/S84_PUBLIC_FROZEN_CAL_2026_10_26.json'
TRAIN_SEEDS = {'005':20268500, '006':20268510}
CAL_SEEDS = {'005':20268501, '006':20268511}
TEST_SEEDS = {'005':(20268503,20268504), '006':(20268513,20268514)}
N_TRAIN = 64
N_CAL = 24
N_TEST = 24
TRAIN_CONDITIONS = ('clean','blur_sigma_11','scratch_grain')
CAL_CONDITIONS = TRAIN_CONDITIONS
TEST_CONDITIONS = ('clean','blur_sigma_11','blur_sigma_16','scratch_grain')
EXTRA_METRICS = ('extra_considered','extra_accepted','extra_rejected')
BOOTSTRAP = 1200


def digest(path:Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _deps():
    """Older frozen repo modules are loaded only when real private runs start.

    Local source-only contract tests work without copying private frame data or
    altering the frozen S72-S84 source modules.
    """
    from bubble_chamber.s56_three_piece_scenes import SCENARIOS, make_three_piece_scene
    from bubble_chamber.s84_graph_context_candidates import _score_original_contacts
    from bubble_chamber.s85_train_reference_labels import label_extras_after_inference
    from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
    from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
    from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75
    from scripts.research.bubble_chamber import s80_calibrate_then_test_multiscale as s80
    from scripts.research.bubble_chamber import s84_calibrate_then_fresh_test_graph_context as s84
    return SCENARIOS, make_three_piece_scene, _score_original_contacts, label_extras_after_inference, s68,s72,s75,s80,s84


def code_locks():
    return {name:digest(path) for name,path in (
        ('protocol',PROTOCOL), ('operator',OPERATOR), ('labels',LABELS),
        ('gate',GATE),('noise',NOISE),('benchmark',Path(__file__)),
        ('S80_predictor',S80_SOURCE), ('S84_graph_contacts',S84_SOURCE),
        ('original_S72_CAL',FROZEN_S72_CAL))}


def _validate_private(visualmetro, photo_tiles):
    if not visualmetro.is_dir() or not photo_tiles.is_dir():
        raise FileNotFoundError('Private H50Z photo tiles and visualMetro film-070 frames are not mounted')
    SCENARIOS,make_scene,contacts,label_extras,s68,s72,s75,s80,s84 = _deps()
    locks = code_locks()
    if locks['original_S72_CAL'] != s80.LOCKED_S72_SHA256:
        raise ValueError('Original frozen S72 model/policy changed')
    frozen80 = json.loads(FROZEN_S80_CAL.read_text(encoding='utf8'))
    frozen84 = json.loads(FROZEN_S84_CAL.read_text(encoding='utf8'))
    if frozen80.get('operator_sha256') != locks['S80_predictor']:
        raise ValueError('S80 original candidate generator differs from its frozen CAL')
    if frozen84.get('operator_sha256') != locks['S84_graph_contacts']:
        raise ValueError('S84 original graph-contact scorer differs from its frozen CAL')
    frozen = s75.frozen_s72(FROZEN_S72_CAL)
    templates,split = s72._source_data(visualmetro)
    if split != s80.SPLIT_SHA:
        raise ValueError('Whole-frame visualMetro train/cal/test split changed')
    if not all(templates[p] for p in ('train','cal','test')):
        raise ValueError('One of the three independent geometry partitions is empty')
    photos = {p:sorted(photo_tiles.glob(f'H50Z_030_01_{p}_*.png')) for p in ('005','006')}
    if (len(photos['005']),len(photos['006'])) != (21,18):
        raise ValueError('Need the authenticated 21+18 private H50Z photograph tiles')
    context = {'locks':locks, 'source_split_sha256':split,
               's72_source_code_locks':s72.LOCKS,
               'H50Z_independent_physical_runs':1,
               'native_particle_GT_count':0,
               'H50Z_070_physical_event_correspondence_proven':False,
               'physical_auto_joins_authorized':False}
    return (SCENARIOS,make_scene,contacts,label_extras,s68,s72,s75,s80,s84),templates,photos,frozen,context


def extract_blind_features(gray, original, consensus, baseline_edges, scorer, contacts_fn):
    """Output features + duplicate-free candidate contact map without any GT."""
    if not isinstance(gray,np.ndarray) or gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('Pixels only: uint8, 192x192')
    if any(a is not b for a,b in zip(original,consensus)) or len(consensus)<len(original):
        raise ValueError('S80 must preserve original source line identities')
    extras = consensus[len(original):]
    mapping = contacts_fn(original, extras, scorer)
    occupied = {port for e in baseline_edges for port in e.ports()}
    X = [candidate_features(gray,curve,original,mapping.get(j,[]),occupied)
         for j,curve in enumerate(extras)]
    if X and (np.asarray(X).shape != (len(extras),len(FEATURE_NAMES)) or
              not np.isfinite(X).all()):
        raise ValueError('Invalid pixel-only S85 feature grid')
    return X,mapping,occupied


def predict_blind(gray, scorer, cutoff, graph_arm, frozen_model):
    """Image-only end-to-end S72/S80/S85 predictions, no scene or GT argument."""
    dependencies = _deps()
    s80, s84 = dependencies[-2:]
    old = s80.infer_from_pixels(gray,scorer,cutoff,graph_arm)
    base = old['baseline']; cons = old['s80_consensus_add6']
    source = base['polylines']; proposed = cons['polylines']
    contacts_fn = dependencies[2]
    _, contact_map, occupied = extract_blind_features(gray,source,proposed,base['chosen'],scorer,contacts_fn)
    result = {'baseline':{**base,'extra_considered':0,'extra_accepted':0,'extra_rejected':0}}
    for arm in ARMS[1:]:
        selection = select_extras(gray,source,proposed,contact_map,occupied,frozen_model,arm)
        count = selection['extra_accepted']
        if count == 0:
            pred = dict(base)
        elif count == len(proposed)-len(source):
            pred = dict(cons)
        else:
            chosen, pair_count, ok = s84._solve(selection['polylines'],scorer,cutoff)
            pred = {'polylines':selection['polylines'],'chosen':chosen,
                    'pair_count':pair_count,'solver_success':ok,
                    'physics_auto_joins_authorized':False,
                    'added_pixel_instances':count,
                    'original_polylines_retained':len(source),
                    'duplicate_suppressed':cons['duplicate_suppressed'],
                    'scale_consensus_rejected':cons['scale_consensus_rejected']}
        result[arm] = {**pred,**{k:selection[k] for k in EXTRA_METRICS}}
        if result[arm]['added_pixel_instances'] != count:
            raise AssertionError('S85 extra line count accounting drift')
        if any(a is not b for a,b in zip(source,result[arm]['polylines'])):
            raise AssertionError('S85 changed an original S72 line')
        if result[arm]['physics_auto_joins_authorized']:
            raise AssertionError('S85 is research advisory only')
    return result


def _scenario_stream(photo_paths, templates, seed, n, scenarios, make_scene):
    rng = np.random.default_rng(seed)
    for i in range(n):
        photo_path=photo_paths[int(rng.integers(len(photo_paths)))]
        injected_seed=int(rng.integers(1, 2**31-1))
        nuisance_seed=int(rng.integers(1, 2**31-1))
        scene=make_scene(photo_path,templates,seed=injected_seed,
                         scenario=scenarios[i % len(scenarios)])
        yield i,scene,nuisance_seed


def train_source_model(photos, templates, seed, n, scorer, cutoff, graph_arm,
                       scenarios, make_scene, s80, contacts_fn, label_fn):
    """TRAIN only. Rows/labels/groups never written to public results."""
    X=[];y=[];groups=[]
    label_counts={'positive':0,'negative':0,'abstain':0,'all_extra_proposals':0}
    for i,scene,nseed in _scenario_stream(photos,templates,seed,n,scenarios,make_scene):
        for ci,cond in enumerate(TRAIN_CONDITIONS):
            pixels=condition_pixels(scene.gray,cond,noise_seed=nseed+ci)
            blind=s80.infer_from_pixels(pixels,scorer,cutoff,graph_arm)
            original=blind['baseline']['polylines']
            pred=blind['s80_consensus_add6']['polylines']
            vectors,_,_=extract_blind_features(pixels,original,pred,blind['baseline']['chosen'],scorer,contacts_fn)
            # Critical barrier: simulator reference is accessed ONLY here, after features.
            labels=label_fn(pred,len(original),scene.nodes)
            if len(labels)!=len(vectors):raise AssertionError('TRAIN-only label grid mismatch')
            label_counts['all_extra_proposals']+=len(labels)
            for j,vec in enumerate(vectors):
                label=labels[j]
                if label is None:
                    label_counts['abstain']+=1
                    continue
                if label not in (0,1):raise ValueError('Unexpected synthetic TRAIN label')
                label_counts['positive' if label==1 else 'negative']+=1
                X.append(vec);y.append(label);groups.append(i) # same group for paired conditions
    matrix = np.asarray(X,dtype=float).reshape(-1,len(FEATURE_NAMES))
    model=fit_train_only(matrix, y, groups)
    if label_counts['positive']+label_counts['negative']+label_counts['abstain'] != label_counts['all_extra_proposals']:
        raise AssertionError('TRAIN-only label accounting mismatch')
    return model,label_counts


def _aggregate(rows, metrics):
    if not rows or any(set(metrics)-set(row) for row in rows):
        raise ValueError('Missing after-inference metrics')
    return {**{key:int(sum(r[key] for r in rows)) for key in metrics},'scenes':len(rows)}


def _paired(a,b,seed):
    if not a or len(a)!=len(b):raise ValueError('Paired scene samples required')
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(a),size=(BOOTSTRAP,len(a)))
    keys=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
          'complete_three_piece_paths_synthetic','shape_true_fragments_matched',
          'shape_true_one_or_both_endpoints_far','scene_any_noncorrect')
    out={}
    for k in keys:
        diff=np.asarray([new[k]-base[k] for base,new in zip(a,b)],dtype=float)
        out[k]={'delta_total':int(diff.sum()),'mean_delta_per_scene':float(diff.mean()),
                'paired_scene_bootstrap_95pct':[float(q) for q in np.quantile(diff[draws].mean(axis=1),(.025,.975))]}
    return out


def evaluate_scenes(photos,templates,seed,n,conditions,model,scorer,cutoff,graph_arm,
                    scenarios,make_scene,s80):
    metric_names=(*s80.MEASURES,*EXTRA_METRICS)
    outcomes={condition:{arm:[] for arm in ARMS} for condition in conditions}
    for i,scene,nseed in _scenario_stream(photos,templates,seed,n,scenarios,make_scene):
        for ci,cond in enumerate(conditions):
            pixels=condition_pixels(scene.gray,cond,noise_seed=nseed+ci)
            predictions=predict_blind(pixels,scorer,cutoff,graph_arm,model)
            # Artificial reference ONLY enters the evaluator after all predictions exist.
            scores=s80.score_after_inference(predictions,scene.nodes)
            for arm in ARMS:
                row={**scores[arm],**{k:int(predictions[arm][k]) for k in EXTRA_METRICS}}
                if set(metric_names)-set(row):raise AssertionError('Missing end-to-end S85 metric')
                if row['extra_accepted']!=row['added_pixel_instances']:
                    raise AssertionError('Extra S85 model output count drift')
                outcomes[cond][arm].append(row)
    return outcomes


def _model_receipt(raw,ctx):
    if (raw.get('schema')!=SCHEMA or raw.get('phase')!='train' or
        raw.get('smoke_NOT_CONFIRMATORY') or raw.get('locks')!=ctx['locks'] or
        raw.get('source_split_sha256')!=ctx['source_split_sha256'] or
        raw.get('s72_source_code_locks')!=ctx['s72_source_code_locks']):
        raise ValueError('S85 private TRAIN model / code source contract mismatch')
    if set(raw.get('models',{}))!={'005_to_006','006_to_005'}:
        raise ValueError('Both private source models required')
    for record in raw['models'].values():
        if model_sha256(record['model'])!=record['model_sha256']:
            raise ValueError('Private TRAIN model weights/candidate feature schema changed')
    return True


def run(phase: str,visualmetro:Path,tiles:Path,*,train_models:Path|None=None,
        calibration:Path|None=None,smoke=False):
    if phase not in ('train','cal','test'):
        raise ValueError('S85 supports train, cal, test phases only')
    deps,templates,photos,frozen_s72,context=_validate_private(visualmetro,tiles)
    scenarios,make_scene,contacts_fn,label_fn,s68,s72,s75,s80,s84=deps
    if smoke and phase=='test':raise ValueError('S85 smoke test is non-confirmatory')
    record={'schema':SCHEMA,'experiment_id':EXPERIMENT_ID,'phase':phase,
            'smoke_NOT_CONFIRMATORY':bool(smoke),**context,
            'feature_schema':FEATURE_SCHEMA,'train_seeds':TRAIN_SEEDS,
            'CAL_seeds':CAL_SEEDS,'TEST_seeds':TEST_SEEDS,
            'conditional_synthetic_only':True,'directions':{}}
    if phase=='train':record['models']={}
    if phase in ('cal','test'):
        if train_models is None or not train_models.is_file():
            raise FileNotFoundError('Private frozen TRAIN models are required before CAL/TEST')
        trained=json.loads(train_models.read_text(encoding='utf8'))
        _model_receipt(trained,context)
        record['frozen_TRAIN_models_sha256']=digest(train_models)
    if phase=='test':
        if calibration is None or not calibration.is_file():
            raise FileNotFoundError('Immutable non-smoke CAL must exist before held-out TEST')
        chosen=json.loads(calibration.read_text(encoding='utf8'))
        for k in ('schema','locks','source_split_sha256','s72_source_code_locks','frozen_TRAIN_models_sha256'):
            if chosen.get(k)!=record[k]:raise ValueError('TRAIN/CAL or sources changed before TEST: '+k)
        if chosen.get('phase')!='cal' or chosen.get('smoke_NOT_CONFIRMATORY'):
            raise ValueError('Confirmatory full CAL required before TEST')
        record['frozen_CAL_sha256']=digest(calibration)
    for di,source in enumerate(('005','006')):
        target='006' if source=='005' else '005';direction=f'{source}_to_{target}'
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        frozen_policy=frozen_s72['directions'][direction]
        graph_arm=frozen_policy['CAL_selected']
        if float(cutoff)!=float(frozen_policy['S56_frozen_CAL_cutoff']):
            raise ValueError('Original S56 source policy cutoff differs')
        entry={'source_photo':source,'opposite_photo':target,'original_graph_arm':graph_arm,
               'frozen_source_cutoff':float(cutoff)}
        if phase=='train':
            model,counts=train_source_model(photos[source],templates['train'],TRAIN_SEEDS[source],
                2 if smoke else N_TRAIN,scorer,cutoff,graph_arm,
                scenarios,make_scene,s80,contacts_fn,label_fn)
            record['models'][direction]={'model':model,'model_sha256':model_sha256(model),
                                         'label_counts':counts}
            entry={'TRAIN_only_label_counts':counts,'model_trained':bool(model['trained']),
                   'model_sha256':model_sha256(model),
                   'no_native_physics_GT':True}
        elif phase=='cal':
            model=trained['models'][direction]['model']
            entry['train_model_sha256']=model_sha256(model)
            measures=evaluate_scenes(photos[source],templates['cal'],CAL_SEEDS[source],
                                     2 if smoke else N_CAL,CAL_CONDITIONS,model,
                                     scorer,cutoff,graph_arm,scenarios,make_scene,s80)
            stats={arm:{cond:_aggregate(measures[cond][arm],(*s80.MEASURES,*EXTRA_METRICS))
                        for cond in CAL_CONDITIONS} for arm in ARMS}
            entry['CAL_arms']=stats
            entry['CAL_decision']=choose_cal_only(stats,bool(model['trained']))
        else:
            saved=chosen['directions'][direction]
            model=trained['models'][direction]['model']
            if (saved['original_graph_arm']!=graph_arm or
                float(saved['frozen_source_cutoff'])!=float(cutoff) or
                saved['train_model_sha256']!=model_sha256(model)):
                raise ValueError('Source model/graph arm or CAL policy modified after selection')
            entry['CAL_selected']=saved['CAL_decision']['CAL_selected']
            entry['train_model_sha256']=model_sha256(model)
            entry['TEST_seeds']={}
            for si,seed in enumerate(TEST_SEEDS[source]):
                observations=evaluate_scenes(photos[target],templates['test'],seed,
                                             N_TEST,TEST_CONDITIONS,model,scorer,cutoff,
                                             graph_arm,scenarios,make_scene,s80)
                entry['TEST_seeds'][str(seed)]={cond:{
                    'arms':{arm:_aggregate(observations[cond][arm],(*s80.MEASURES,*EXTRA_METRICS))
                            for arm in ARMS},
                    'paired_vs_baseline':{arm:_paired(observations[cond]['baseline'],
                                                observations[cond][arm],20268590+di*100+si*10+ci)
                                           for arm in ARMS[1:]}}
                    for ci,cond in enumerate(TEST_CONDITIONS)}
        record['directions'][direction]=entry
    if phase=='test':
        record['all_TEST_aggregates']={cond:{arm:{key:int(sum(
            seedrow[cond]['arms'][arm][key]
            for direction in record['directions'].values()
            for seedrow in direction['TEST_seeds'].values()))
            for key in (*s80.MEASURES,*EXTRA_METRICS,'scenes')}
            for arm in ARMS} for cond in TEST_CONDITIONS}
        record['CAL_selected_TEST']={cond:{key:int(sum(
            seedrow[cond]['arms'][direction['CAL_selected']][key]
            for direction in record['directions'].values()
            for seedrow in direction['TEST_seeds'].values()))
            for key in (*s80.MEASURES,*EXTRA_METRICS,'scenes')}
            for cond in TEST_CONDITIONS}
        record['physical_auto_joins_authorized']=False
    return record


def public_train_receipt(train_record,private_path):
    """No coefficient, scaler, pixel, candidate score, original coordinates."""
    if train_record.get('phase')!='train':raise ValueError('Expected actual TRAIN phase')
    return {k:v for k,v in train_record.items() if k != 'models' and k != 'directions'} | {
        'private_TRAIN_models_sha256':digest(private_path),
        'model_summary_by_direction':{source:{'model_sha256':record['model_sha256'],
             'trained':record['model']['trained'], 'TRAIN_label_counts':record['label_counts']}
             for source,record in train_record['models'].items()},
        'no_model_weights_or_private_pixels':True}


def main():
    p=argparse.ArgumentParser(description='S85 isolated TRAIN->CAL->TEST, private H50Z required')
    p.add_argument('phase',choices=('train','cal','test'))
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--train-models',type=Path,help='TRAIN private JSON for CAL/TEST')
    p.add_argument('--calibration',type=Path,help='Frozen CAL JSON for held-out TEST')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--public-train-receipt',type=Path)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Immutable S85 research results must not be overwritten')
    if args.phase == 'train' and (args.out.resolve() == ROOT or ROOT in args.out.resolve().parents):
        raise ValueError('Never write private learned model coefficients inside the public repository')
    if args.public_train_receipt and args.public_train_receipt.exists():
        raise FileExistsError('Public TRAIN receipt already exists')
    if args.public_train_receipt and args.phase!='train':
        raise ValueError('Public TRAIN receipts only available in TRAIN phase')
    record=run(args.phase,args.visualmetro,args.photo_tiles,
               train_models=args.train_models,calibration=args.calibration,smoke=args.smoke)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if args.public_train_receipt:
        pub=public_train_receipt(record,args.out)
        args.public_train_receipt.parent.mkdir(parents=True,exist_ok=True)
        args.public_train_receipt.write_text(json.dumps(pub,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('S85',args.phase,'file',args.out,'SHA-256',digest(args.out),
          'physical_GT',record['native_particle_GT_count'])

if __name__=='__main__':
    main()
