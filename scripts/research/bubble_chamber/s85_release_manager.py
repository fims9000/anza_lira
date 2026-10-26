#!/usr/bin/env python3
"""S85 practical release workflow: preflight, immutable PRIVATE experiments, receipts.

Run real H50Z/film070 experiments only with independently authorized data.
Public source-only checks never fake physics GT or promote S72 to a learned model.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber import s85_train_calibrate_test_learned as s85

SCHEMA='S85_RELEASE_PREFLIGHT_AND_PRIVATE_REPLAY_V1'
REQUIRED_PUBLIC=(
    'bubble_chamber/s56_three_piece_scenes.py',
    'bubble_chamber/s80_multiscale_instances.py',
    'bubble_chamber/s84_graph_context_candidates.py',
    'bubble_chamber/s85_fragment_ranker.py',
    'bubble_chamber/s85_train_reference_labels.py',
    'scripts/research/bubble_chamber/s72_fresh_corrected_branch_benchmark.py',
    'scripts/research/bubble_chamber/s80_calibrate_then_test_multiscale.py',
    'scripts/research/bubble_chamber/s84_calibrate_then_fresh_test_graph_context.py',
    'scripts/research/bubble_chamber/s85_train_calibrate_test_learned.py',
    'docs/bubble_chamber/S85_FROZEN_LEARNED_FRAGMENT_GATE_2026_10_27.md',
)
PRIVATE_OUTPUTS=(
    'S85_TRAIN_MODELS_PRIVATE.json',
    'S85_FROZEN_CAL_PRIVATE.json',
    'S85_FRESH_TEST_PRIVATE.json',
    'S85_REPEAT_TEST_PRIVATE.json',
    'S85_NATIVE_UNKNOWN_PRIVATE.json',
)
SUMMARY_NAME='S85_AGGREGATE_ONLY_RESEARCH_SUMMARY.json'


def digest(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def _json_bytes(value)->bytes:
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')


def _write_once(path:Path, payload, *,private=True):
    """Atomic, no-overwrite JSON receipt; private modes never leave model readable by others."""
    if path.exists():raise FileExistsError('Cannot overwrite frozen S85 artifact: '+path.name)
    path.parent.mkdir(parents=True,exist_ok=True)
    mode=0o600 if private else 0o644
    fd,temp=tempfile.mkstemp(prefix='.s85_',suffix='.tmp',dir=str(path.parent))
    try:
        os.fchmod(fd,mode)
        with os.fdopen(fd,'wb') as f:
            f.write(_json_bytes(payload));f.flush();os.fsync(f.fileno())
        # Hard-link, unlike rename, refuses overwriting a file created concurrently.
        os.link(temp,path)
    finally:
        if Path(temp).exists():os.unlink(temp)


def _source_checks():
    missing=[p for p in REQUIRED_PUBLIC if not (ROOT/p).is_file()]
    result={'required_public_source_files':len(REQUIRED_PUBLIC),
            'missing_public_source_files':missing,
            'source_bundle_complete':not missing,
            'source_locks_valid':False,
            'S72_S80_S84_source_unchanged':False}
    if missing:return result
    try:
        from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
        from scripts.research.bubble_chamber import s80_calibrate_then_test_multiscale as s80
        from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75
        import json as j
        locks=s85.code_locks()
        cal80=j.loads(s85.FROZEN_S80_CAL.read_text())
        cal84=j.loads(s85.FROZEN_S84_CAL.read_text())
        same=(locks['original_S72_CAL']==s80.LOCKED_S72_SHA256 and
              cal80.get('operator_sha256')==locks['S80_predictor'] and
              cal84.get('operator_sha256')==locks['S84_graph_contacts'])
        result['S72_S80_S84_source_unchanged']=bool(same)
        if same:
            s75.frozen_s72(s85.FROZEN_S72_CAL)
            result['source_locks_valid']=True
        else:
            result['source_problem']='Frozen S72/S80/S84 byte locks do not match'
    except (ImportError,FileNotFoundError,ValueError,KeyError) as exc:
        # Never misrepresent an unavailable scientific model as ready.
        result['source_problem']=type(exc).__name__+': '+str(exc)[:240]
    return result


def preflight(visualmetro:Path|None=None, tiles:Path|None=None,*,inspect_pixels=True):
    """No training, no GT, no private image export; source-only works without data."""
    result={'schema':SCHEMA,'experiment_id':s85.EXPERIMENT_ID,
            'source':_source_checks(),
            'private_inputs':{'visualmetro_available':False,'H50Z_tiles_available':False,
                              'H50Z_tile_counts':{'005':0,'006':0},
                              'film070_source_split_verified':False,
                              'photo_tiles_all_512_gray':False},
            'native_expert_GT_count':0,
            'H50Z_independent_physical_runs_known':1,
            'film070_is_NOT_native_H50Z_event_GT':True,
            'physical_auto_joins_authorized':False,
            'TRAIN_CAL_TEST_executed_by_preflight':False,
            'ready_for_private_TRAIN':False,
            'blockers':[]}
    inp=result['private_inputs'];blocks=result['blockers']
    if not result['source']['source_locks_valid']:
        blocks.append('MISSING_OR_CHANGED_FROZEN_PUBLIC_SOURCE')
    if visualmetro is None or not visualmetro.is_dir():
        blocks.append('VISUALMETRO_FILM070_NOT_MOUNTED')
    else:
        inp['visualmetro_available']=True
    if tiles is None or not tiles.is_dir():
        blocks.append('PRIVATE_H50Z_PHOTO_TILES_NOT_MOUNTED')
    else:
        inp['H50Z_tiles_available']=True
        images={p:sorted(tiles.glob(f'H50Z_030_01_{p}_*.png')) for p in ('005','006')}
        inp['H50Z_tile_counts']={p:len(images[p]) for p in ('005','006')}
        if (len(images['005']),len(images['006']))!=(21,18):
            blocks.append('H50Z_PHOTO_TILE_COUNTS_NOT_21_18')
        elif inspect_pixels:
            import cv2
            ok=all((pic:=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)) is not None and
                   pic.shape==(512,512) and pic.dtype.name=='uint8'
                   for paths in images.values() for path in paths)
            inp['photo_tiles_all_512_gray']=bool(ok)
            if not ok:blocks.append('H50Z_AUTHENTICATED_512_GRAY_TILES_INVALID')
        else:
            blocks.append('H50Z_IMAGE_VALIDATION_NOT_EXECUTED')
    if inp['visualmetro_available'] and result['source']['source_locks_valid']:
        try:
            from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
            from scripts.research.bubble_chamber import s80_calibrate_then_test_multiscale as s80
            templates,split=s72._source_data(visualmetro)
            if split!=s80.SPLIT_SHA or any(len(templates[k])<24 for k in ('train','cal','test')):
                blocks.append('FILM070_FRAME_SPLIT_OR_TEMPLATES_CHANGED')
            else:inp['film070_source_split_verified']=True
        except (ValueError,FileNotFoundError,KeyError,ImportError) as exc:
            blocks.append('FILM070_UNVERIFIED_'+type(exc).__name__)
    result['ready_for_private_TRAIN']=not blocks
    return result


def _safe_directory(path:Path):
    out=path.resolve()
    if out==ROOT or ROOT in out.parents:
        raise ValueError('Private source images, models and samples MUST NOT be written inside the public source repo')
    if out.exists() and out.is_file():raise ValueError('Private output path must be a directory')
    return out


def _resume_or_write(path:Path,make,phase,*,resume=False):
    if path.exists():
        if not resume:raise FileExistsError('Frozen '+phase+' exists; use --resume without overwriting')
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema')!=s85.SCHEMA or data.get('phase')!=phase or data.get('smoke_NOT_CONFIRMATORY'):
            raise ValueError('Cannot resume from incompatible/smoke '+phase+' artifact')
        return data
    record=make()
    if record.get('phase')!=phase or record.get('schema')!=s85.SCHEMA or record.get('smoke_NOT_CONFIRMATORY'):
        raise ValueError('Scientific runner did not produce a confirmatory phase '+phase)
    _write_once(path,record)
    return record


def _public_summary(train,cal,test,native,paths:dict):
    for source in (cal,test):
        for key in ('schema','locks','source_split_sha256','s72_source_code_locks'):
            if source.get(key)!=train.get(key):
                raise ValueError('Private S85 TRAIN/CAL/TEST source lineage changed: '+key)
    for direction in ('005_to_006','006_to_005'):
        model_sha=train['models'][direction]['model_sha256']
        if cal['directions'][direction].get('train_model_sha256')!=model_sha or \
           test['directions'][direction].get('train_model_sha256')!=model_sha:
            raise ValueError('S85 CAL or TEST source direction uses other TRAIN model')
        if test['directions'][direction].get('CAL_selected')!=cal['directions'][direction]['CAL_decision']['CAL_selected']:
            raise ValueError('S85 frozen CAL selection changed after TEST')
    if test.get('frozen_CAL_sha256')!=digest(paths['cal']):
        raise ValueError('Held-out TEST did not use saved frozen CAL')
    if cal.get('frozen_TRAIN_models_sha256')!=digest(paths['train']):
        raise ValueError('Frozen CAL did not use saved private TRAIN weights')
    if native.get('frozen_CAL_sha256')!=digest(paths['cal']) or native.get('frozen_TRAIN_sha256')!=digest(paths['train']):
        raise ValueError('Native audit not based on identical frozen source policy')
    if not (train.get('native_particle_GT_count')==cal.get('native_particle_GT_count')==
            test.get('native_particle_GT_count')==0 and native.get('independent_physics_GT_count')==0):
        raise ValueError('S85 result would falsely assert native physical GT')
    trained={k:{'trained':bool(v['model']['trained']),
                'candidate_train_counts':v['label_counts'],
                'model_sha256':v['model_sha256']}
             for k,v in train['models'].items()}
    cal_decisions={key:row['CAL_decision']['CAL_selected'] for key,row in cal['directions'].items()}
    test_delta={}
    for cond in s85.TEST_CONDITIONS:
        original=test['all_TEST_aggregates'][cond]['baseline']
        selected=test['CAL_selected_TEST'][cond]
        test_delta[cond]={key:int(selected[key]-original[key]) for key in (
            'selected_synthetic_correct_links','synthetic_noncorrect_selected',
            'complete_three_piece_paths_synthetic','shape_true_fragments_matched')}
    summary={
        'schema':'S85_AGGREGATE_ONLY_RESEARCH_SUMMARY_V1',
        'experiment_id':s85.EXPERIMENT_ID,
        'public_aggregate_only':True,
        'synthetic_on_photo_TEST_only':True,
        'native_particle_GT_count':0,
        'H50Z_independent_physical_runs_known':1,
        'visualmetro_070_NOT_real_H50Z_GT':True,
        'physical_auto_joins_authorized':False,
        'TEST_replay_byte_identical':True,
        'artifact_sha256':{key:digest(p) for key,p in paths.items() if key!='replay'},
        'TRAIN_direction_summary':trained,
        'CAL_frozen_selections':cal_decisions,
        'TEST_aggregate_metrics':test['all_TEST_aggregates'],
        'TEST_CAL_selected_vs_S72_delta':test_delta,
        'native_unknown_machine_aggregate':native['aggregates'],
        'fresh_TEST_no_more_noncorrect_all_conditions':all(
            v['synthetic_noncorrect_selected']<=0 for v in test_delta.values()),
        'scientific_conclusion':'Synthetic conditional evaluation only; no native physical validation',
        'full_physics_deployment_ready':False,
    }
    serialized=json.dumps(summary,sort_keys=True,ensure_ascii=False,allow_nan=False)
    if any(f'"{field}"' in serialized for field in (
        'coef','intercept','scaler_mean','scaler_scale','points_xy','image_base64','source_pixels')):
        raise AssertionError('Private model weights, source pixels or coordinates leaked into public summary')
    return summary


def run_all(visualmetro:Path,tiles:Path,out_dir:Path,*,resume=False,runner=s85.run,
            native_auditor=None):
    """One command reproduces PRIVATE TRAIN -> CAL -> TEST twice -> native UNKNOWN.

    Previously completed phases are never overwritten. Resume only with verified
    locks; private artifact checks in runner reject stale source/model hashes.
    """
    dst=_safe_directory(out_dir)
    inspection=preflight(visualmetro,tiles)
    if not inspection['ready_for_private_TRAIN']:
        raise RuntimeError('Preflight blocks S85 private run: '+', '.join(inspection['blockers']))
    paths={name:dst/filename for name,filename in zip(('train','cal','test','replay','native'),PRIVATE_OUTPUTS)}
    if SUMMARY_NAME in [p.name for p in paths.values()]:raise AssertionError('Private output names collide')
    train=_resume_or_write(paths['train'],lambda:runner('train',visualmetro,tiles), 'train',resume=resume)
    # Validate all already-on-disk TRAIN locks before proceeding.
    s85._model_receipt(train,inspection_private_context(visualmetro,tiles))
    cal=_resume_or_write(paths['cal'],lambda:runner('cal',visualmetro,tiles,train_models=paths['train']),
                         'cal',resume=resume)
    if cal.get('frozen_TRAIN_models_sha256')!=digest(paths['train']):
        raise ValueError('Private CAL train-model SHA-256 does not match immutable TRAIN file')
    test=_resume_or_write(paths['test'],lambda:runner('test',visualmetro,tiles,train_models=paths['train'],
                            calibration=paths['cal']),'test',resume=resume)
    replay=_resume_or_write(paths['replay'],lambda:runner('test',visualmetro,tiles,train_models=paths['train'],
                            calibration=paths['cal']),'test',resume=resume)
    if paths['test'].read_bytes()!=paths['replay'].read_bytes():
        raise ValueError('S85 TEST replay was NOT byte-identical; stop and investigate source/RNG')
    if native_auditor is None:
        from scripts.research.bubble_chamber.s85_native_unknown_audit_v2 import audit as native_auditor
    if paths['native'].exists():
        if not resume:raise FileExistsError('Frozen native H50Z audit exists')
        native=json.loads(paths['native'].read_text())
    else:
        native=native_auditor(visualmetro,tiles,paths['train'],paths['cal'])
        if native.get('schema')!='S85_REAL_39_H50Z_NO_NATIVE_PHYSICS_GT_V1':
            raise ValueError('Unknown native audit schema')
        _write_once(paths['native'],native)
    # Sanitize PUBLIC summary; no per-candidate features, pixels or private weights.
    summary=_public_summary(train,cal,test,native,paths)
    dest=dst/SUMMARY_NAME
    if dest.exists():
        if not resume:raise FileExistsError('Frozen aggregate-only report exists')
        if dest.read_bytes()!=_json_bytes(summary):raise ValueError('Existing aggregate differs from frozen sources')
    else:_write_once(dest,summary)
    return summary


def inspection_private_context(visualmetro,tiles):
    """Get canonical S72/source locks, not a fabricated test context."""
    # Dataset validation has already succeeded; do not duplicate synthetic labels.
    return s85._validate_private(visualmetro,tiles)[-1]


def main():
    p=argparse.ArgumentParser(description='ANZA-LIRA S85 private scientific release gate')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('preflight',help='Source/data readiness (does NOT train or make GT)')
    q.add_argument('--visualmetro',type=Path)
    q.add_argument('--photo-tiles',type=Path)
    q.add_argument('--out',type=Path)
    q.add_argument('--strict',action='store_true',help='Fail if inputs unavailable')
    r=sub.add_parser('run-all',help='Execute frozen PRIVATE TRAIN, CAL, TEST, repeat, native')
    r.add_argument('--visualmetro',type=Path,required=True)
    r.add_argument('--photo-tiles',type=Path,required=True)
    r.add_argument('--private-out-dir',type=Path,required=True)
    r.add_argument('--resume',action='store_true')
    x=p.parse_args()
    if x.command=='preflight':
        status=preflight(x.visualmetro,x.photo_tiles)
        if x.out:_write_once(x.out,status,private=False)
        print(json.dumps(status,ensure_ascii=False,indent=2))
        if x.strict and not status['ready_for_private_TRAIN']:
            raise SystemExit(2)
    elif x.command=='run-all':
        res=run_all(x.visualmetro,x.photo_tiles,x.private_out_dir,resume=x.resume)
        print(json.dumps({'experiment_id':res['experiment_id'],
              'TEST_replay_byte_identical':True,
              'native_physical_GT_count':0,
              'release_summary':str(x.private_out_dir/SUMMARY_NAME),
              'full_physics_deployment_ready':False},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
