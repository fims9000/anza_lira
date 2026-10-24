#!/usr/bin/env python3
"""S11 registered, source-locked ablation of graph-friendly thin-line segmentation.

The authentic `AZConv2d` and `AZUNet` are imported unmodified. S11 changes
only the training objective: regular BCE+Dice vs clDice, plus the ORIGINAL
`AZUNet.axis_alignment_loss`. Both AZ and U-Net get the same clDice ablation.
The data are GENERATED pixels/masks (measured track geometries optional),
never expert-dense photograph labels or true particle IDs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))

from bubble_chamber.s11_axial_topology import (
    StructuralWeights, structural_training_loss, binary_cldice,
    per_frame_confusion, count_metrics, paired_frame_bootstrap,
)
from scripts.research.bubble_chamber import s7_original_retrain as s7


@dataclass(frozen=True)
class Arm:
    name: str
    architecture: str
    topology_weight: float
    axis_weight: float


ARMS = (
    Arm('unet_base','plain_unet',0.,0.),
    Arm('unet_cldice','plain_unet',0.2,0.),
    Arm('az_base','original_azunet',0.,0.),
    Arm('az_cldice','original_azunet',0.2,0.),
    Arm('az_cldice_axis','original_azunet',0.2,0.05),
)


def _bootstrap_frame_cldice(base: np.ndarray, variant: np.ndarray, groups: list[str],
                            *, seed: int, iterations: int = 1000) -> dict:
    if len(base)!=len(variant) or len(base)!=len(groups):
        raise ValueError('Nonpaired source frames')
    frames=sorted(set(groups))
    if len(frames)<5:
        raise ValueError('Too few independent source frames')
    a=np.array([float(np.mean(base[np.array(groups)==f])) for f in frames])
    b=np.array([float(np.mean(variant[np.array(groups)==f])) for f in frames])
    rng=np.random.default_rng(seed)
    picks=rng.integers(0,len(frames),size=(iterations,len(frames)))
    diffs=(b[picks]-a[picks]).mean(axis=1)
    return {'candidate_minus_baseline_macro_cldice':float((b-a).mean()),
            'ci95':list(map(float,np.quantile(diffs,[.025,.975]))),
            'unit':'whole_source_frame','frames':len(frames)}


def prepare_source(cfg: s7.RunConfig, measurements: Path | None):
    """Critical: identical sample identities and no repeated train/test frames."""
    sets,meta=s7.make_datasets(cfg,measurements)
    group_ids={split:{g for _,_,g in rows} for split,rows in sets.items()}
    if any(group_ids[a] & group_ids[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Data leakage between source groups')
    sig=s7.split_fingerprints(sets)
    return sets,{**meta,'split_fingerprints':sig,
                 'train_test_disjoint':True,'all_pixel_masks_simulated':True}


def _objective_from(model,weights:StructuralWeights,base_loss):
    def objective(logits,y):
        # Called by S7 train_one for BOTH training and validation; same definition.
        base=base_loss(logits,y)
        total,_=structural_training_loss(logits,y,model,base,weights)
        return total
    return objective


def train_arms(cfg: s7.RunConfig,sets: dict, out: Path, *, arm_ids: tuple[str,...],
               cldice_iterations:int = 4, axis_iterations:int = 4):
    s7.verify_sources(REPO_ROOT)
    out.mkdir(parents=True,exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)
    fingerprints=s7.split_fingerprints(sets)
    results={}
    truth_masks=None
    test_groups=s7.TrackDataset(sets['test']).groups()
    original_loss=s7.masked_loss
    try:
        for arm in ARMS:
            if arm.name not in arm_ids:continue
            torch.manual_seed(cfg.seed)
            models=s7.build_models(cfg)
            model=models[arm.architecture]
            weights=StructuralWeights(cldice=arm.topology_weight,axis=arm.axis_weight,
                                      skeleton_iterations=cldice_iterations,axis_iterations=axis_iterations)
            weights.validate()
            s7.masked_loss=_objective_from(model,weights,original_loss)
            # Each arm has its own checkpoints and frozen source hash.
            run,clean,stress=s7.train_one(arm.name,model,sets,cfg,out,
                                          data_fingerprints=fingerprints)
            if truth_masks is None: truth_masks=np.asarray(clean[1])
            if not np.array_equal(truth_masks,clean[1]) or not np.array_equal(truth_masks,stress[1]):
                raise ValueError('Target mismatch across ablations')
            rows={}
            for condition,(pred,labels) in [('clean',clean),('stress',stress)]:
                binary=np.asarray(pred,bool)
                label=np.asarray(labels,bool)
                groups=per_frame_confusion(binary,label,test_groups)
                clds=np.array([binary_cldice(p,y) for p,y in zip(binary,label)],dtype=float)
                rows[condition]={'counts':count_metrics(groups),'cldice_macro':float(clds.mean()),
                                 'cldice_per_image':clds.tolist(), 'per_frame_counts':{
                                     group:vec.tolist() for group,vec in groups.items()},
                                 'test_images':int(len(pred))}
            results[arm.name]={'architecture':arm.architecture,'loss':asdict(weights),
                               'optimization':run,'evaluation':rows}
            (out/arm.name/'s11_arm.json').write_text(json.dumps({
                'arm':asdict(arm),'weights':asdict(weights),
                'source_git_blobs':s7.SOURCE_BLOBS,'data_fingerprints':fingerprints,
                'synthetic_only':True,'test_threshold_selected_on_validation_only':True,
            },indent=2)+'\n')
    finally:
        s7.masked_loss=original_loss
    if not results:raise ValueError('Empty arm selection')
    compare={}
    for condition in ['clean','stress']:
        for contrast in [('az_cldice','az_base'),('az_cldice_axis','az_cldice'),
                         ('unet_cldice','unet_base'),('az_cldice_axis','unet_cldice'),
                         ('az_cldice_axis','unet_base'),('az_base','unet_base')]:
            candidate,baseline=contrast
            if candidate not in results or baseline not in results:continue
            a=results[baseline]['evaluation'][condition]
            b=results[candidate]['evaluation'][condition]
            to_counts=lambda r:{f:np.array(c,dtype=np.int64) for f,c in r['per_frame_counts'].items()}
            global_delta=paired_frame_bootstrap(to_counts(a),to_counts(b),seed=s7.effective_data_seed(cfg)+177)
            local_delta=_bootstrap_frame_cldice(np.array(a['cldice_per_image']),
                         np.array(b['cldice_per_image']),test_groups,
                         seed=s7.effective_data_seed(cfg)+178)
            compare[f'{condition}_{candidate}_minus_{baseline}']={
                'dice_paired_frame':global_delta,'cldice_paired_frame':local_delta}
    return results,compare


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',choices=['procedural','visualmetro'],default='procedural')
    p.add_argument('--measurements',type=Path)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261024,help='Model initialization seed')
    p.add_argument('--data-seed',type=int,default=20261024,help='Fixed source frames, renderings and noise')
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--size',type=int,default=64)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--widths',nargs=4,type=int,default=[4,6,8,12])
    p.add_argument('--rules',type=int,default=2)
    p.add_argument('--train-limit',type=int,default=120)
    p.add_argument('--val-limit',type=int,default=26)
    p.add_argument('--test-limit',type=int,default=25)
    p.add_argument('--steps-per-epoch',type=int,default=0)
    p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--arms',type=str,default=','.join(a.name for a in ARMS))
    p.add_argument('--cldice-iterations',type=int,default=4)
    p.add_argument('--axis-iterations',type=int,default=4)
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.size<32 or args.size%16:
        p.error('Invalid training configuration')
    arm_ids=tuple(args.arms.split(','))
    if len(set(arm_ids))!=len(arm_ids) or any(a not in {x.name for x in ARMS} for a in arm_ids):
        p.error('Invalid arm identifiers')
    cfg=s7.RunConfig(source=args.source,seed=args.seed,data_seed=args.data_seed,
                     size=args.size,epochs=args.epochs,batch_size=args.batch_size,
                     widths=tuple(args.widths),num_rules=args.rules,
                     train_limit=args.train_limit,val_limit=args.val_limit,
                     test_limit=args.test_limit,steps_per_epoch=args.steps_per_epoch,
                     cpu_threads=args.cpu_threads,az_residual=True,robust_train=True)
    sets,meta=prepare_source(cfg,args.measurements)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    results,compare=train_arms(cfg,sets,args.out_dir,arm_ids=arm_ids,
                               cldice_iterations=args.cldice_iterations,
                               axis_iterations=args.axis_iterations)
    report={'experiment':'S11_AZ_AXIS_AND_CLDICE_ABLATIONS_V1',
            'status':'SYNTHETIC_OR_MEASURED_GEOMETRY_RENDERED_PIXELS_ONLY',
            'model_git_blobs':s7.SOURCE_BLOBS,'source_meta':meta,'config':asdict(cfg),
            'predeclared_arms':[asdict(a) for a in ARMS],
            'run_arms':list(results),'results':results,'comparisons':compare,
            'test_labels_are_real':False,'physical_track_ID_labels_are_real':False,
            'auto_merge_allowed':False,
            'limitations':['All training image pixels and masks are generated, never authentic film pixels',
                           'No independent physical film/run or hydrogen-propane generalization',
                           'ClDice assesses foreground connectivity, not distinct particle identities',
                           'Hyperparameters chosen before test, single model-seed run; multi-seed replication required']}
    (args.out_dir/'s11_metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    for a,v in results.items():
        print(a,'clean Dice',round(v['evaluation']['clean']['counts']['dice'],4),
               'stress Dice',round(v['evaluation']['stress']['counts']['dice'],4),
               'stress clDice',round(v['evaluation']['stress']['cldice_macro'],4),flush=True)
    print('S11_FINAL_METRICS',args.out_dir/'s11_metrics.json')

if __name__=='__main__':main()
