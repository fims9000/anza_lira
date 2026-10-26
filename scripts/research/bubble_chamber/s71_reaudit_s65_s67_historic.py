#!/usr/bin/env python3
"""S71: retrospective dual-audit S65/S66/S67 on UNCHANGED predictions.

Nothing here trains, predicts or changes geometry based on oracle labels. It
wraps *only* the evaluation-only callable after pixel-derived graph selection.
All source scene geometry and camera-photo pixel arrays remain private.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np

from bubble_chamber.s64_pixel_to_path import audit_synthetic_scene as HISTORIC_AUDITOR
from bubble_chamber.s69_orientation_invariant_audit import audit_orientation_invariant_synthetic
from scripts.research.bubble_chamber import (
    s65_calibrate_endpoint_abstention as s65,
    s66_test_image_only_endpoint_extensions as s66,
    s67_benchmark_instance_continuity as s67,
)

EXPERIMENT_ID='ANZA-LIRA-BC-S71-20261026'
SCHEMA='S71_RETROSPECTIVE_DUAL_EVALUATOR_S65_S67_SYNTHETIC_ONLY_V1'
PROTOCOL=ROOT/'docs/bubble_chamber/S71_FROZEN_RETROSPECTIVE_S65_S67_ORIENTATION_AUDIT_2026_10_26.md'
LOCKS={
  'bubble_chamber/s64_pixel_to_path.py': 'c148e55dc9a94f2841e34d2f5e1f3f6b8c41a1ebd345152d2ca21ee7c3ff824c',
  'bubble_chamber/s69_orientation_invariant_audit.py': '3e4ad80c0fc544b7b231b6a7b7dcdc3501484830a98d6d19c133dc0eb1c5b495',
  'scripts/research/bubble_chamber/s65_calibrate_endpoint_abstention.py': '958f959e2235ee1a90011cc1535afe0d58661cb94e9e0136b47fe032f9eb6515',
  'scripts/research/bubble_chamber/s66_test_image_only_endpoint_extensions.py': 'ec9ff00cd12a306ff2df78c36b1ec66a1b107110d3c020e76eca1f22029d6adf',
  'scripts/research/bubble_chamber/s67_benchmark_instance_continuity.py': '5e5259639a9c99977dcec4a07bb46b07e82ae27e1285bb5b228143c0594c814c',
}
HIST={'S65':('S65_PUBLIC_FROZEN_CAL_2026_10_26.json','S65_PUBLIC_LOCKED_TEST_2026_10_26.json'),
      'S66':('S66_PUBLIC_FROZEN_CAL_2026_10_26.json','S66_PUBLIC_FRESH_TEST_2026_10_26.json'),
      'S67':('S67_PUBLIC_SYNTHETIC_RESULTS_2026_10_26.json',None)}
ALLOWED_DIFFERENT=frozenset({'selected_synthetic_correct_links','selected_synthetic_wrong_endpoint',
        'synthetic_noncorrect_selected','scene_any_noncorrect','complete_three_piece_paths_synthetic'})
ALL_PROBE_KEYS=frozenset({'candidate_polylines','candidate_pairs_after_strict_gate',
 'source_node_matches','synthetic_decoy_matched','unmatched_polylines',
 'ambiguous_polylines','selected_provisional_links','selected_synthetic_wrong_identified',
 'selected_synthetic_unmatched_or_ambiguous','solver_failure','physical_verified_links'})
DIRECTIONS=('005_to_006','006_to_005')


def sha(path):return sha256(Path(path).read_bytes()).hexdigest()


def verify_locks():
    for name, expected in LOCKS.items():
        observed=sha(ROOT/name)
        if observed!=expected:raise ValueError(f'Frozen source modified: {name}; {observed}')
    if not PROTOCOL.exists():raise FileNotFoundError('Frozen S71 preregistration')


def assert_stable(old,new):
    if any(old.get(key)!=new.get(key) for key in ALL_PROBE_KEYS):
        differences={k:(old.get(k),new.get(k)) for k in ALL_PROBE_KEYS if old.get(k)!=new.get(k)}
        raise AssertionError(f'Inference drift or changed matching: {differences}')
    other=set(old)&set(new)-ALLOWED_DIFFERENT-ALL_PROBE_KEYS
    for key in other:
        if old[key]!=new[key]:raise AssertionError(f'Unexpected changed metric: {key}')
    if old['selected_synthetic_correct_links']>new['selected_synthetic_correct_links']:
        raise AssertionError('Corrected evaluator lost a previously correct join')
    if (new['selected_synthetic_correct_links']+new['synthetic_noncorrect_selected']!=
        new['selected_provisional_links']):
        raise AssertionError('Incorrect revised partition of graph choices')


@contextmanager
def dual_evaluator(module):
    original=module.audit_synthetic_scene
    assert original is HISTORIC_AUDITOR
    pairs=[]
    def audit_both(prediction,source_proxy):
        historic=original(prediction,source_proxy)
        corrected=audit_orientation_invariant_synthetic(prediction,source_proxy)
        assert_stable(historic,corrected)
        pairs.append((historic.copy(),corrected.copy()))
        return historic
    module.audit_synthetic_scene=audit_both
    try:yield pairs
    finally:module.audit_synthetic_scene=original


def collect(records,index,arms,n):
    # Within each scene, source benchmarks iterate arms in deterministic order.
    count=len(arms)*n
    data=records[index:index+count]
    if len(data)!=count:raise AssertionError(f'Missing scenes {len(data)} / {count}')
    return ({arm:[row[a] for row in data[k::len(arms)]] for k,arm in enumerate(arms)
                for a in (0,1)},index+count)


def split_records(records,arms,source_seeds,n):
    cursor=0; batches={}
    for direction, seeds in source_seeds.items():
        batches[direction]={}
        for seed in seeds:
            size=len(arms)*n
            source=records[cursor:cursor+size]
            if len(source)!=size:raise AssertionError('Incomplete scene records')
            batches[direction][str(seed)]={arm:{'historic':[source[k][0] for k in range(i,size,len(arms))],
                                                  'corrected':[source[k][1] for k in range(i,size,len(arms))]}
                for i,arm in enumerate(arms)}
            cursor+=size
    if cursor!=len(records):raise AssertionError(f'Unused evaluations {len(records)-cursor}')
    return batches


def summarize(rows):
    if not rows:raise ValueError('Empty frozen run')
    keys=('candidate_polylines','candidate_pairs_after_strict_gate','source_node_matches',
          'synthetic_decoy_matched','selected_provisional_links',
          'selected_synthetic_correct_links','synthetic_noncorrect_selected',
          'selected_synthetic_wrong_endpoint','selected_synthetic_wrong_identified',
          'selected_synthetic_unmatched_or_ambiguous','scene_any_noncorrect',
          'complete_three_piece_paths_synthetic','solver_failure')
    return {**{k:int(sum(int(row[k]) for row in rows)) for k in keys},'scene_count':len(rows)}


def paired_delta(old,new,seed):
    if not old or len(old)!=len(new):raise ValueError('Not paired')
    n=len(old);rng=np.random.default_rng(seed)
    draw=rng.integers(0,n,size=(1200,n))
    fields=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
             'selected_synthetic_wrong_endpoint','complete_three_piece_paths_synthetic',
             'scene_any_noncorrect')
    out={}
    for key in fields:
        diff=np.array([b[key]-a[key] for a,b in zip(old,new)],dtype=float)
        out[key]={'corrected_minus_historic_total':int(diff.sum()),
           'paired_scene_CI95_per_scene':[float(q) for q in np.quantile(diff[draw].mean(axis=1),(.025,.975))]}
    return out


def ensure_historical_equal(observed,reference,study,phase):
    if observed!=reference:
        # Use short key diff, but never reduce evidence threshold to permit drift.
        def differences(a,b,prefix=''):
            if isinstance(a,dict) and isinstance(b,dict):
                ks=set(a)|set(b)
                for k in sorted(ks):
                    if k not in a or k not in b:yield prefix+str(k)
                    else:yield from differences(a[k],b[k],prefix+str(k)+'/')
            elif a!=b:yield prefix[:-1]
        raise AssertionError(f'{study} {phase} historic snapshot NOT identical: {list(differences(observed,reference))[:10]}')


def evaluate_s65(visualmetro,tiles):
    historic_cal=json.loads((ROOT/'docs/bubble_chamber'/HIST['S65'][0]).read_text())
    historical_test=json.loads((ROOT/'docs/bubble_chamber'/HIST['S65'][1]).read_text())
    arms=(None,*s65.THRESHOLD_GRID)
    with dual_evaluator(s65) as pairs:
        old_cal=s65.run(visualmetro,tiles,'cal')
    ensure_historical_equal(old_cal,historic_cal,'S65','CAL')
    records=split_records(pairs,arms,{
        '005_to_006':(s65.CAL_SEEDS['005'],),'006_to_005':(s65.CAL_SEEDS['006'],)},s65.CAL_SCENES)
    corrected_selections={}
    entries={}; corrected_cal_by_photo={}
    for direction, seeds in records.items():
        rows=next(iter(seeds.values()))
        corrected_counts={a:summarize(rows[a]['corrected']) for a in arms}
        calc=s65.choose_cal_only({a:{'correct':v['selected_synthetic_correct_links'],
            'noncorrect':v['synthetic_noncorrect_selected'],'scenes':v['scene_count']}
            for a,v in corrected_counts.items()})
        corrected_selections[direction]=calc
        original=historic_cal['directions'][direction]['CAL_decision']
        entries[direction]={'CAL_historic_selected':original['selected_CAL_only'],
            'CAL_corrected_selected':calc['selected_CAL_only'],
            'CAL_hist_correct':{('baseline' if a is None else str(a)):summarize(rows[a]['historic']) for a in arms},
            'CAL_corrected':{('baseline' if a is None else str(a)):v for a,v in corrected_counts.items()},
            'CAL_decision_changed':original['selected_CAL_only']!=calc['selected_CAL_only']}
    old_path=ROOT/'docs/bubble_chamber'/HIST['S65'][0]
    with dual_evaluator(s65) as pairs:
        old_test=s65.run(visualmetro,tiles,'test',calibration=old_path)
    ensure_historical_equal(old_test,historical_test,'S65','TEST')
    seeds={'005_to_006':s65.TEST_SEEDS['005'],'006_to_005':s65.TEST_SEEDS['006']}
    old_arms={direction:(None,old_test['directions'][direction]['frozen_CAL_decision']['selected_CAL_only'])
            for direction in seeds}
    # If a source direction historical CAL selected None, only baseline is run.
    cursor=0;test_records={}
    for direction,ss in seeds.items():
        test_records[direction]={}
        arms_for_direction=[None] if old_arms[direction][1] is None else list(old_arms[direction])
        for seed in ss:
            n=s65.TEST_SCENES*len(arms_for_direction)
            block=pairs[cursor:cursor+n]
            if len(block)!=n:raise AssertionError('Incomplete S65 TEST seed')
            test_records[direction][str(seed)]={('baseline' if arm is None else str(arm)):{
                'historic':[block[k][0] for k in range(i,n,len(arms_for_direction))],
                'corrected':[block[k][1] for k in range(i,n,len(arms_for_direction))]}
                for i,arm in enumerate(arms_for_direction)}
            cursor+=n
    if cursor!=len(pairs):raise AssertionError('Unexpected S65 TEST audit count')
    for direction,seed_data in test_records.items():
        entries[direction]['TEST_historical_selected']=old_arms[direction][1]
        entries[direction]['TEST_per_seed']={}
        for seed,arm_data in seed_data.items():
            result={arm:{'historic':summarize(z['historic']),
                'corrected':summarize(z['corrected']),
                'correction_paired_scene_bootstrap':paired_delta(z['historic'],z['corrected'],int(seed)+711)}
                for arm,z in arm_data.items()}
            for arm,z in arm_data.items():
                if arm!='baseline':
                    result[arm]['corrected_method_minus_baseline_paired_bootstrap']=paired_delta(
                        arm_data['baseline']['corrected'],z['corrected'],int(seed)+714)
            entries[direction]['TEST_per_seed'][seed]=result
    return entries, corrected_selections


def evaluate_s66(visualmetro,tiles):
    hist_cal=json.loads((ROOT/'docs/bubble_chamber'/HIST['S66'][0]).read_text())
    hist_test=json.loads((ROOT/'docs/bubble_chamber'/HIST['S66'][1]).read_text())
    arms=s66.ARMS
    with dual_evaluator(s66) as calls:cal=s66.run(visualmetro,tiles,'cal')
    ensure_historical_equal(cal,hist_cal,'S66','CAL')
    calrows=split_records(calls,arms,{'005_to_006':(s66.CAL_SEEDS['005'],),
                '006_to_005':(s66.CAL_SEEDS['006'],)},s66.CAL_SCENES)
    entries={}
    for direction in DIRECTIONS:
        sample=next(iter(calrows[direction].values()))
        corr={a:summarize(sample[a]['corrected']) for a in arms}
        selection=s66.choose_calibration_only({a:{'synthetic_correct':v['selected_synthetic_correct_links'],
            'synthetic_noncorrect':v['synthetic_noncorrect_selected'],
            'matched_source_fragments':v['source_node_matches'],
            'synthetic_decoy_matched':v['synthetic_decoy_matched'],'scenes':v['scene_count']}
            for a,v in corr.items()})
        old=cal['directions'][direction]['CAL_only_decision']['CAL_only_selected']
        entries[direction]={'CAL_historic_selected':old,'CAL_corrected_selected':selection['CAL_only_selected'],
          'CAL_historic':{a:summarize(sample[a]['historic']) for a in arms},
          'CAL_corrected':corr,'CAL_decision_changed':old!=selection['CAL_only_selected']}
    with dual_evaluator(s66) as calls:
        test=s66.run(visualmetro,tiles,'test',calibration=ROOT/'docs/bubble_chamber'/HIST['S66'][0])
    ensure_historical_equal(test,hist_test,'S66','TEST')
    rows=split_records(calls,arms,{'005_to_006':s66.TEST_SEEDS['005'],
                 '006_to_005':s66.TEST_SEEDS['006']},s66.TEST_SCENES)
    for direction in DIRECTIONS:
        entries[direction]['TEST_per_seed']={}
        for seed,row in rows[direction].items():
            out={arm:{'historic':summarize(z['historic']),
             'corrected':summarize(z['corrected']),
             'correction_paired_scene_bootstrap':paired_delta(z['historic'],z['corrected'],int(seed)+712)}
                for arm,z in row.items()}
            for arm,z in row.items():
                if arm!='s39_original':
                    out[arm]['corrected_method_minus_baseline_paired_bootstrap']=paired_delta(
                        row['s39_original']['corrected'],z['corrected'],int(seed)+715)
            entries[direction]['TEST_per_seed'][seed]=out
    return entries


def evaluate_s67(visualmetro,tiles):
    hist=json.loads((ROOT/'docs/bubble_chamber'/HIST['S67'][0]).read_text())
    arms=s67.ARMS
    with dual_evaluator(s67) as calls:
        old_cal=s67.run(visualmetro,tiles,'cal')
    old_rows=split_records(calls,arms,{'005_to_006':(s67.CAL_SEEDS['005'],),
           '006_to_005':(s67.CAL_SEEDS['006'],)},s67.CAL_SCENES)
    entries={}
    for direction in DIRECTIONS:
        sample=next(iter(old_rows[direction].values()))
        for arm in arms:
            historical=hist['directions'][direction]['calibration_aggregates'][arm]
            measured=summarize(sample[arm]['historic'])
            for key,value in historical.items():
                if measured[key]!=value:raise AssertionError(f'S67 historic CAL mismatch {direction} {arm} {key}')
        corr={arm:summarize(sample[arm]['corrected']) for arm in arms}
        selection=s67.choose_calibration_only(corr)['CAL_only_selected']
        old=hist['directions'][direction]['calibration_selected']
        entries[direction]={'CAL_historic_selected':old,'CAL_corrected_selected':selection,
            'CAL_historic':{a:summarize(sample[a]['historic']) for a in arms},
            'CAL_corrected':corr,'CAL_decision_changed':old!=selection}
    # The historic public S67 CAL JSON includes only aggregate metrics. Preserve
    # the original source-CAL decision and output for the original TEST replay.
    private_cal=ROOT.parent/'S71_S67_HISTORIC_CAL_REGENERATED_PRIVATE.json'
    if private_cal.exists():raise FileExistsError('Do not overwrite replay CAL')
    private_cal.write_text(json.dumps(old_cal,ensure_ascii=False,indent=2)+'\n')
    try:
        with dual_evaluator(s67) as calls:
            test=s67.run(visualmetro,tiles,'test',calibration=private_cal)
    finally:
        private_cal.unlink(missing_ok=True)
    rows=split_records(calls,arms,{'005_to_006':s67.TEST_SEEDS['005'],
                   '006_to_005':s67.TEST_SEEDS['006']},s67.TEST_SCENES)
    for direction in DIRECTIONS:
        entries[direction]['TEST_per_seed']={}
        for seed,batch in rows[direction].items():
            recorded=s67.paired(batch['s39_original']['historic'],
                batch['s67_crossing_orientation_v1']['historic'],int(seed)+67)
            expected=hist['directions'][direction]['per_seed_paired_CI95'][seed]
            if recorded!=expected:
                raise AssertionError(f'S67 historical paired CI changed: {direction} {seed}')
            out={arm:{'historic':summarize(z['historic']),
                'corrected':summarize(z['corrected']),
                'correction_paired_scene_bootstrap':paired_delta(z['historic'],z['corrected'],int(seed)+713)}
                for arm,z in batch.items()}
            out['s67_crossing_orientation_v1']['corrected_method_minus_baseline_paired_bootstrap']=paired_delta(
                batch['s39_original']['corrected'],
                batch['s67_crossing_orientation_v1']['corrected'],int(seed)+716)
            entries[direction]['TEST_per_seed'][seed]=out
        for arm in arms:
            aggregate={key:sum(z[arm]['historic'][key] for z in entries[direction]['TEST_per_seed'].values())
                       for key in summarize(next(iter(rows[direction].values()))[arm]['historic'])}
            historical=hist['directions'][direction]['test_aggregates'][arm]
            for key,value in historical.items():
                if aggregate[key]!=value:raise AssertionError(f'S67 historic TEST mismatch {direction} {arm} {key}')
    return entries


def rollup(entries):
    result={}
    for direction in entries:
        for seed,arms in entries[direction]['TEST_per_seed'].items():
            for name,row in arms.items():
                key='baseline' if name in (None,'s39_original') else str(name)
                dest=result.setdefault(key,{'historic':defaultdict(int),'corrected':defaultdict(int),'scenes':0})
                for kind in ('historic','corrected'):
                    for field,n in row[kind].items():
                        if field=='scene_count':continue
                        dest[kind][field]+=n
                dest['scenes']+=row['historic']['scene_count']
    return {k:{'historic':dict(v['historic']),'corrected':dict(v['corrected']),
                'scenes':v['scenes']} for k,v in result.items()}


def run(visualmetro,tiles):
    verify_locks()
    tiles=Path(tiles)
    assert len(list(tiles.glob('H50Z_030_01_005_*.png')))==21
    assert len(list(tiles.glob('H50Z_030_01_006_*.png')))==18
    a, choices=evaluate_s65(visualmetro,tiles)
    b=evaluate_s66(visualmetro,tiles)
    c=evaluate_s67(visualmetro,tiles)
    public={'schema':SCHEMA,'experiment_id':EXPERIMENT_ID,
        'protocol_sha256':sha(PROTOCOL),'source_file_locks':LOCKS,
        'historical_test_set_reused':True,'new_independent_test_scenes':0,
        'native_verified_H50Z_particle_tracks':0,'physical_auto_joins_authorized':False,
        'all_metrics_synthetic_injection_only':True,
        'source_photographs_same_run':True,'visualmetro_film_070_not_H50Z_physics_GT':True,
        'history_preserved_unmodified':True,
        'studies':{'S65':{'directions':a,'aggregate':rollup(a)},
                   'S66':{'directions':b,'aggregate':rollup(b)},
                   'S67':{'directions':c,'aggregate':rollup(c)}}}
    return public


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Never overwrite a completed scientific replay')
    result=run(a.visualmetro,a.photo_tiles)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8')
    print('S71 completed, studies',list(result['studies']),flush=True)
if __name__=='__main__':main()
