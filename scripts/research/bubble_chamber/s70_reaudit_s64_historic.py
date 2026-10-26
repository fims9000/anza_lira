#!/usr/bin/env python3
"""Retrospective S64 scoring correction. Never gives source truth to inference."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.research.bubble_chamber import s64_benchmark_pixel_to_path as old_benchmark
from bubble_chamber.s69_orientation_invariant_audit import audit_orientation_invariant_synthetic

LOCKS={
 'original_benchmark':'ad31fef03bc40fc70606830527300873ae013967e6ac19836a115e1b95ed55c7',
 'old_evaluator':'c148e55dc9a94f2841e34d2f5e1f3f6b8c41a1ebd345152d2ca21ee7c3ff824c',
 'new_evaluator':'3e4ad80c0fc544b7b231b6a7b7dcdc3501484830a98d6d19c133dc0eb1c5b495',
 'original_aggregate':'0cf514960cc47651a0b0b5579d0cd728d108f8d639eaf0ee6dc89565f33c3301',
}
STABLE=('candidate_polylines','candidate_pairs_after_strict_gate','source_node_matches',
        'synthetic_decoy_matched','selected_provisional_links','solver_failure',
        'selected_synthetic_wrong_identified','selected_synthetic_unmatched_or_ambiguous')
FIELDS=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
        'selected_synthetic_wrong_endpoint','complete_three_piece_paths_synthetic','scene_any_noncorrect')
SCHEMA='S70_2026_10_26_RETROSPECTIVE_S64_ORIENTATION_CORRECTION_SYNTHETIC_ONLY'

def digest(p):return sha256(Path(p).read_bytes()).hexdigest()


def paired_bootstrap(records,seed):
    rng=np.random.default_rng(seed)
    n=len(records)
    ix=rng.integers(0,n,size=(1200,n))
    out={}
    for key in FIELDS:
        diff=np.asarray([b[key]-a[key] for a,b in records],dtype=float)
        out[key]={'difference_corrected_minus_historic_total':int(diff.sum()),
                  'mean_difference_per_scene':float(diff.mean()),
                  'paired_scene_95pct':[float(x) for x in np.quantile(diff[ix].mean(axis=1),[.025,.975])]}
    return out


def run(visualmetro,tiles):
    src={
        'original_benchmark':ROOT/'scripts/research/bubble_chamber/s64_benchmark_pixel_to_path.py',
        'old_evaluator':ROOT/'bubble_chamber/s64_pixel_to_path.py',
        'new_evaluator':ROOT/'bubble_chamber/s69_orientation_invariant_audit.py',
        'original_aggregate':ROOT/'docs/bubble_chamber/S64_PUBLIC_PIXEL_TO_GRAPH_RESULTS_2026_10_26.json',
    }
    for name,path in src.items():
        actual=digest(path)
        if actual!=LOCKS[name]:raise ValueError(f'Frozen {name} source has changed: {actual}')
    orig=json.loads(src['original_aggregate'].read_text(encoding='utf8'))
    if orig['smoke_NOT_confirmatory'] or orig['schema']!=old_benchmark.SCHEMA:
        raise ValueError('Refusing non-historic S64 aggregate')
    old_audit=old_benchmark.audit_synthetic_scene
    raw_pairs=[]
    def both_on_same_prediction(pred,ref):
        historic=old_audit(pred,ref)
        corrected=audit_orientation_invariant_synthetic(pred,ref)
        for key in STABLE:
            if historic[key]!=corrected[key]:raise AssertionError('Inferred prediction drift')
        if historic['selected_synthetic_correct_links']>corrected['selected_synthetic_correct_links']:
            raise AssertionError('Evaluation correction regressed known old correct link')
        raw_pairs.append((historic,corrected))
        return corrected
    try:
        old_benchmark.audit_synthetic_scene=both_on_same_prediction
        corrected_output=old_benchmark.run(Path(visualmetro),Path(tiles),smoke=False,private_dir=None)
    finally:
        old_benchmark.audit_synthetic_scene=old_audit
    n=2*2*24*len(old_benchmark.ARMS)
    if len(raw_pairs)!=n:raise AssertionError(f'Unexpected scene count: {len(raw_pairs)} vs {n}')
    public={'schema':SCHEMA,'experiment_id':'ANZA-LIRA-BC-S70-20261026',
            'retrospective_used_known_S64_historic_scenes':True,
            'synthetic_injection_only':True,'native_H50Z_verified_physical_tracks':0,
            'new_heldout_evidence':False,'promoted_method':None,
            'original_source_locks_sha256':LOCKS.copy(),
            'protocol_sha256':digest(ROOT/'docs/bubble_chamber/S70_FROZEN_S64_RETROSPECTIVE_REEVALUATION_2026_10_26.md'),
            'directions':{},'historic_aggregates_exactly_reproduced':True}
    cursor=0
    for source in ('005','006'):
        target='006' if source=='005' else '005'
        direction=f'{source}_to_{target}'
        old=orig['directions'][direction]
        new=corrected_output['directions'][direction]
        public['directions'][direction]={}
        if old['threshold_from_SOURCE_CAL_negative_FPR_only']!=new['threshold_from_SOURCE_CAL_negative_FPR_only']:
            raise AssertionError('Source-only score threshold drift')
        for seed in old_benchmark.SEEDS[source]:
            sid=str(seed)
            o=old['test_seeds'][sid]
            c=new['test_seeds'][sid]
            if o['oracle_geometry_only']!=c['oracle_geometry_only']:
                raise AssertionError('Historical synthetic upper oracle changed')
            seed_out={}
            # S64 evaluates both arms inside every scene, so calls are
            # interleaved arm0,arm1,arm0,arm1,... and not arm-major.
            block=raw_pairs[cursor:cursor+24*len(old_benchmark.ARMS)]
            cursor+=24*len(old_benchmark.ARMS)
            for arm_idx,arm in enumerate(old_benchmark.ARMS):
                pairs=block[arm_idx::len(old_benchmark.ARMS)]
                if len(pairs)!=24:raise AssertionError('Incomplete seed block')
                # Proxy upper bounds are calculated after both scoring calls;
                # verify them from final S64 output below, not this hook.
                keys=[k for k in old_benchmark.METRICS if not k.startswith('union_')]
                # Verify the entire frozen historic per-scene bootstrap summary,
                # not just aggregate totals. Same seed schedule as old S64.
                rng=np.random.default_rng(seed+arm_idx*11+900)
                draw=rng.integers(0,24,size=(1200,24))
                for key in keys:
                    v=np.asarray([a[key] for a,b in pairs],dtype=float)
                    expected_ci=[float(x) for x in np.quantile(v[draw].mean(axis=1),[.025,.975])]
                    if o[arm][key] != {'total':int(v.sum()),'per_scene':float(v.mean()),
                                       'scene_bootstrap_95pct':expected_ci}:
                        raise AssertionError(f'Historic CI mismatch: {direction} {sid} {arm} {key}')
                    historic_total=int(v.sum())
                    corrected_total=sum(b[key] if key in b else a[key] for a,b in pairs)
                    historic_snapshot_total=o[arm][key]['total']
                    if historic_total!=historic_snapshot_total:
                        raise AssertionError(f'Historic aggregate mismatched: {direction} {sid} {arm} {key}')
                    if corrected_total!=c[arm][key]['total']:
                        raise AssertionError(f'Corrected aggregate mismatch: {direction} {sid} {arm} {key}')
                    if key in STABLE and corrected_total!=historic_total:
                        raise AssertionError(f'Inference-related metric changed: {key}')
                # Ensure older unmodified proxy metrics have EXACT same aggregates.
                for key in ('union_proxy_fragments','union_proxy_links_upper','union_proxy_paths_upper'):
                    if o[arm][key]!=c[arm][key]:
                        raise AssertionError('Proxy upper bound changed')
                seed_out[arm]={
                    'original_synthetic_correct':o[arm]['selected_synthetic_correct_links']['total'],
                    'corrected_synthetic_correct':c[arm]['selected_synthetic_correct_links']['total'],
                    'original_synthetic_noncorrect':o[arm]['synthetic_noncorrect_selected']['total'],
                    'corrected_synthetic_noncorrect':c[arm]['synthetic_noncorrect_selected']['total'],
                    'original_complete_chains':o[arm]['complete_three_piece_paths_synthetic']['total'],
                    'corrected_complete_chains':c[arm]['complete_three_piece_paths_synthetic']['total'],
                    'paired_differences':paired_bootstrap(pairs,seed+len(arm)),
                }
            public['directions'][direction][sid]=seed_out
    if cursor!=len(raw_pairs):raise AssertionError('Unconsumed S64 source predictions')
    return public


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Frozen outputs cannot be overwritten')
    x=run(a.visualmetro,a.photo_tiles)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('S70 historic S64 audit verified, seeds',sum(map(len,x['directions'].values())),flush=True)
if __name__=='__main__':main()
