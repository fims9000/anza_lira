#!/usr/bin/env python3
"""S72: new CAL/TEST scenes with ALREADY fixed orientation-invariant audit.

Reuses all S68 predictors, source geometry fit and frozen CAL gate unchanged.
The only selected difference is evaluation-only callback from S69. No native
physics label is assumed or inferred from a single H50Z photo run.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_simulated_gaps import extract_track_templates
from bubble_chamber.s69_orientation_invariant_audit import audit_orientation_invariant_synthetic
from scripts.research.bubble_chamber.s37_benchmark_review_risk import split_source_frames
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68

EXPERIMENT_ID='ANZA-LIRA-BC-S72-20261026'
SCHEMA='S72_FRESH_CORRECTED_PIXEL_GRAPH_SYNTHETIC_ONLY_V1'
PROTOCOL=ROOT/'docs/bubble_chamber/S72_FROZEN_FRESH_ORIENTATION_CORRECTED_BRANCH_TEST_2026_10_26.md'
LOCKS={
  'bubble_chamber/s68_branch_preserving_instances.py':'dc16718c9bcac613eacc38771b2fa9a840266d9f7da377cc7538ae136bc4a91a',
  'scripts/research/bubble_chamber/s68_benchmark_branch_preservation.py':'f8a841eb7c6fc8e5074e7d88c667891a36f95be67d5a7eb86d145cc9c3fc3e71',
  'bubble_chamber/s69_orientation_invariant_audit.py':'3e4ad80c0fc544b7b231b6a7b7dcdc3501484830a98d6d19c133dc0eb1c5b495',
  'scripts/research/bubble_chamber/s56_benchmark_global_paths.py':'088024077e5f86b5f4308c5d23e72643c8b261a355cfb12f2bbc8c1d3d87355b',
}
CAL_SEEDS={'005':20267201,'006':20267211}
TEST_SEEDS={'005':(20267203,20267204),'006':(20267213,20267214)}
CAL_SCENES=32
TEST_SCENES=24


def digest(p):return sha256(Path(p).read_bytes()).hexdigest()


@contextmanager
def corrected_audit_only():
    old=s68.audit_synthetic_scene
    s68.audit_synthetic_scene=audit_orientation_invariant_synthetic
    try:yield
    finally:s68.audit_synthetic_scene=old


def _source_data(visualmetro):
    m=audited_large_manifest(visualmetro)
    train,cal,test,source_sha=split_source_frames(m)
    if (len(train),len(cal),len(test))!=(79,30,36):raise ValueError('Source split changed')
    templates={name:extract_track_templates(m,chosen_frames=frames) for name,frames in
               [('train',train),('cal',cal),('test',test)]}
    return templates,source_sha


def run(visualmetro,tiles,phase,calibration=None):
    if phase not in ('cal','test'):raise ValueError('Use cal or test')
    for path,expected in LOCKS.items():
        if digest(ROOT/path)!=expected:raise ValueError('Frozen predictor/evaluator changed: '+path)
    templates,split_sha=_source_data(visualmetro)
    if len(list(tiles.glob('H50Z_030_01_005_*.png')))!=21 or len(list(tiles.glob('H50Z_030_01_006_*.png')))!=18:
        raise ValueError('Private H50Z photo crop count changed')
    public={'schema':SCHEMA,'phase':phase,'experiment_id':EXPERIMENT_ID,'protocol_sha256':digest(PROTOCOL),
            'frozen_source_sha256':LOCKS,'source_frame_split_sha256':split_sha,
            'physical_verified_H50Z_GT':0,'physical_auto_join_authorized':False,
            'all_metrics_are_SYNTHETIC_ONLY':True,
            'H50Z_photos_one_physical_run':True,'visualmetro_film070_NOT_H50Z_GT':True,
            'directions':{}}
    if phase=='test':
        if calibration is None:raise ValueError('TEST requires immutable CAL')
        frozen=json.loads(Path(calibration).read_text())
        for field in ('schema','protocol_sha256','frozen_source_sha256','source_frame_split_sha256'):
            if frozen[field]!=public[field]:raise ValueError('CAL frozen lock mismatch: '+field)
        if frozen['phase']!='cal':raise ValueError('CAL phase required')
        public['frozen_CAL_sha256']=digest(calibration)
    with corrected_audit_only():
        for source in ('005','006'):
            target='006' if source=='005' else '005'
            scorer,cutoff=s68._fit_source(tiles,source,templates)
            direction=f'{source}_to_{target}'
            row={'source_H50Z_background':source,'target_H50Z_background':target,
                 'S56_frozen_CAL_cutoff':float(cutoff)}
            if phase=='cal':
                sample=s68.sample(s68._photo_subfolder(tiles,source),templates['cal'],
                                  CAL_SEEDS[source],CAL_SCENES,scorer,cutoff)
                sums={arm:s68.sums(scores) for arm,scores in sample.items()}
                decision=s68.choose_calibration_only(sums)
                row['CAL_selected']=decision['CAL_only_selected']
                row['CAL_fixed_arms']=sums
            else:
                old=frozen['directions'][direction]
                if old['S56_frozen_CAL_cutoff']!=float(cutoff):raise ValueError('Source scorer drift')
                row['CAL_selected']=old['CAL_selected']
                row['TEST_seeds']={}
                for seed in TEST_SEEDS[source]:
                    sample=s68.sample(s68._photo_subfolder(tiles,target),templates['test'],
                                      seed,TEST_SCENES,scorer,cutoff)
                    row['TEST_seeds'][str(seed)]={'arms':{arm:s68.sums(scores) for arm,scores in sample.items()},
                      'method_minus_S39_paired_scene_bootstrap':{arm:s68.paired(sample[s68.ARMS[0]],sample[arm],seed+72+i)
                             for i,arm in enumerate(s68.ARMS[1:])}}
                print('S72 TEST',direction,'selected on CAL',row['CAL_selected'],flush=True)
            public['directions'][direction]=row
    if phase=='test':
        public['all_TEST_arms']={arm:{key:int(sum(r['arms'][arm][key]
                     for direction in public['directions'].values()
                     for r in direction['TEST_seeds'].values()))
                     for key in ('selected_synthetic_correct_links','synthetic_noncorrect_selected',
                                 'complete_three_piece_paths_synthetic','scene_any_noncorrect','selected_provisional_links')}
                 for arm in s68.ARMS}
    return public


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('cal','test'))
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--calibration',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Frozen research artefacts cannot be overwritten')
    j=run(a.visualmetro,Path(a.photo_tiles),a.phase,a.calibration)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('S72',a.phase,'saved',a.out,flush=True)
if __name__=='__main__':main()
