#!/usr/bin/env python3
"""S42 CPU pilot: original AZUNet vs U-Net on source-pure machine pseudo labels.

Strictly no S39 AZ-generated pseudo labels in training, no test-photo previews
for train-time decisions. Synthetic overlay metrics only, not real accuracy.
"""
from __future__ import annotations
import argparse
import json
from hashlib import sha256
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s33_photometric_domain import image_input
from bubble_chamber.s37_source_model import build_model,verify_model_source
from bubble_chamber.s42_source_pure_teacher import geometry_only_pseudo
from bubble_chamber.s39_machine_auto import fixed_budget_mask,ridge_evidence,contrast_baseline
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_simulated_gaps import frame_split,extract_track_templates,make_scene
from scripts.research.bubble_chamber.s39_run_machine_only import predict_cross_photo
from scripts.research.bubble_chamber.s39_benchmark_machine_segmentation import tolerant_fraction

SEED=20261026
TEST_SEEDS={'005':20261129,'006':20261130}
STEPS=32
EPOCHS=3
BATCH=8
N_SCENES=24


def _load_train_images(source:dict,tiles:Path,zones:list,photo:str,out:Path)->tuple[list[tuple],dict]:
    group=[];stats={'source_photo':photo,'source_crops':0,'teacher_positive':0,
                    'teacher_assumed_negative':0,'teacher_unknown':0,'teacher_hessian_paths':0}
    for row in source['tiles']:
        if row['photo_id']!=photo:continue
        path=tiles/(row['crop_id']+'.png')
        im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape!=(512,512) or sha256(im.tobytes()).hexdigest()!=row['source_gray_sha256']:
            raise ValueError('Source photo pixel hash / dimensions mismatch')
        t=geometry_only_pseudo(im,zones,row['native_origin_xy_PRIVATE'],photo)
        if t['AZ_model_used_in_teacher'] or t['S31_machine_draft_used_in_teacher']:
            raise AssertionError('Cross-photo teacher leakage')
        m=t['mask']; fname=f'{row["crop_id"]}_S42_SOURCE_PURE_PSEUDO.png'
        cv2.imwrite(str(out/fname),m)
        stats['source_crops']+=1;stats['teacher_positive']+=int((m==1).sum())
        stats['teacher_assumed_negative']+=int((m==0).sum())
        stats['teacher_unknown']+=int((m==255).sum())
        stats['teacher_hessian_paths']+=t['ridge_count']
        if int((m==1).sum())>8 and int((m==0).sum())>8:
            group.append((im,m))
    if stats['source_crops']!=(21 if photo=='005' else 18) or not group:
        raise ValueError('Bad independent source photo group')
    return group,stats


def _draw_batches(group:list[tuple],photo:str)->list[tuple[torch.Tensor,torch.Tensor]]:
    rng=np.random.default_rng(SEED+int(photo))
    patches=[]
    for epoch in range(EPOCHS):
        for step in range(STEPS):
            x=[];y=[]
            for k in range(BATCH):
                im,mask=group[int(rng.integers(len(group)))]
                ys,xs=np.nonzero(mask==1)
                idx=int(rng.integers(len(xs)))
                px=int(xs[idx]);py=int(ys[idx])
                left=int(np.clip(px-int(rng.integers(16,49)),0,512-64))
                top=int(np.clip(py-int(rng.integers(16,49)),0,512-64))
                rgb=im[top:top+64,left:left+64]
                cm=mask[top:top+64,left:left+64]
                x.append(image_input(rgb,'quantile'))
                y.append(cm)
            patches.append((torch.from_numpy(np.stack(x)[:,None].copy()),
                            torch.from_numpy(np.stack(y)[:,None].copy())))
    return patches


def train_original_model(arm:str, batches:list, out:Path, photo:str)->tuple[torch.nn.Module,dict]:
    file=out/f'S42_{arm}_train_photo_{photo}_seed_{SEED}.pt'
    if file.exists():
        ck=torch.load(file,map_location='cpu',weights_only=True)
        if (ck.get('architecture')!=arm or ck.get('source_photo')!=photo or
                ck.get('photo_teacher_type')!='geometry-only, no AZUNet' or
                ck.get('seed')!=SEED or ck.get('epochs')!=EPOCHS or
                ck.get('batch')!=BATCH or ck.get('steps')!=STEPS or
                ck.get('physics_native_gt') is not False):
            raise ValueError('Refusing incompatible checkpoint on resume')
        model=build_model(arm,widths=(4,6,8,12),rules=2)
        model.load_state_dict(ck['weights']);model.eval()
        print(f'S42 resumed validated checkpoint {photo} {arm}',flush=True)
        return model,{'train_photo':photo,'model':arm,'epochs':EPOCHS,
                      'steps_per_epoch':STEPS,'batch':BATCH,'checkpoint_resumed':True,
                      'model_checkpoint_private_sha256':sha256(file.read_bytes()).hexdigest()}
    torch.manual_seed(SEED)
    model=build_model(arm,widths=(4,6,8,12),rules=2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4)
    losses=[]
    for epoch in range(EPOCHS):
        model.train();epoch_losses=[]
        for x,mask in batches[epoch*STEPS:(epoch+1)*STEPS]:
            optimizer.zero_grad(set_to_none=True)
            pred=model(x)
            if pred.shape!=mask.shape:raise AssertionError('Source architecture shape changed')
            positive=mask==1; negative=mask==0
            if not positive.any() or not negative.any():
                raise ValueError('Sparse machine teacher did not create both labels in batch')
            loss=(F.binary_cross_entropy_with_logits(pred[positive],torch.ones_like(pred[positive]))+
                  F.binary_cross_entropy_with_logits(pred[negative],torch.zeros_like(pred[negative])))/2
            if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(round(float(np.mean(epoch_losses)),5))
        print(f'S42 train {photo} {arm} epoch {epoch+1}/{EPOCHS}: {losses[-1]:.5f}',flush=True)
    model.eval()
    torch.save({'architecture':arm,'original_architecture_source_verified':True,
                'source_photo':photo,'photo_teacher_type':'geometry-only, no AZUNet',
                'physics_native_gt':False,'epochs':EPOCHS,'batch':BATCH,'steps':STEPS,
                'seed':SEED,'input_norm':'quantile',
                'training_pseudo_balanced_bce_loss_by_epoch':losses,
                'weights':model.state_dict()},file)
    return model,{'train_photo':photo,'model':arm,'epochs':EPOCHS,'steps_per_epoch':STEPS,
                  'batch':BATCH,'training_pseudo_balanced_bce_loss_by_epoch':losses,
                  'model_checkpoint_private_sha256':sha256(file.read_bytes()).hexdigest()}


def eval_models(photo:str,tiles:Path,templates:list,models:dict[str,torch.nn.Module],
                private_out:Path,n:int=N_SCENES)->dict:
    all_tiles=sorted(tiles.glob(f'H50Z_030_01_{photo}_*.png'))
    if len(all_tiles)!=(21 if photo=='005' else 18):raise ValueError('Incomplete external photo')
    rng=np.random.default_rng(TEST_SEEDS[photo]);scenarios=('parallel','crossing','scratched','empty_gap')
    rows=[]
    for i in range(n):
        scene=make_scene(all_tiles[int(rng.integers(len(all_tiles)))],templates,
                         seed=int(rng.integers(1,2**31)),scenario=scenarios[i%4])
        truth=scene.extra_masks['injected_visible_strokes'].astype(bool)
        decoy=scene.extra_masks['injected_false_decoy'].astype(bool)
        scores={'local_darkness':contrast_baseline(scene.gray),
                'multiscale_hessian':ridge_evidence(scene.gray)['score']}
        for arm,model in models.items():
            scores[arm]=predict_cross_photo(scene.gray,model,step=64,batch=16)
        for arm,score in scores.items():
            selected=fixed_budget_mask(score,fraction=.035)
            rows.append({'scene':i,'arm':arm,'source_photo':photo,
                         'synthetic_track_covered_at_2px':tolerant_fraction(selected,truth),
                         'synthetic_decoy_covered_at_2px':tolerant_fraction(selected,decoy)})
            if i==0:
                rgb=cv2.cvtColor(scene.gray,cv2.COLOR_GRAY2BGR)
                rgb[selected]=(60,185,65)
                rgb[truth]=(0,0,245)
                cv2.imwrite(str(private_out/f'S42_{photo}_{arm}_INJECTED_PREVIEW.png'),rgb)
        if (i+1)%8==0:print('S42 heldout photo',photo,'scenes',i+1,'/',n,flush=True)
    out={};B=np.random.default_rng(SEED+int(photo))
    for arm in sorted({r['arm'] for r in rows}):
        r=[x for x in rows if x['arm']==arm]
        true=np.asarray([x['synthetic_track_covered_at_2px'] for x in r]);false=np.asarray([x['synthetic_decoy_covered_at_2px'] for x in r])
        inds=B.integers(0,len(true),(2000,len(true)))
        out[arm]={'injected_track_coverage_tolerance_2px':round(float(true.mean()),5),
                  'injected_decoy_scratch_activation_tolerance_2px':round(float(false.mean()),5),
                  'track_coverage_minus_scratch_activation':round(float((true-false).mean()),5),
                  'scene_bootstrap_95pct_CI_TRACK_COVERAGE_SYNTHETIC':np.quantile(true[inds].mean(1),[.025,.975]).round(5).tolist(),
                  'scenes':n}
    return out


def run(s39:Path,s40:Path,photo_tiles:Path,visualmetro:Path,output:Path,*,n=N_SCENES):
    if n!=N_SCENES:raise ValueError('S42 protocol freezes exactly 24 scenes per test photo')
    verify_model_source()
    torch.set_num_threads(2)
    source=json.loads(s39.read_text())
    if source.get('native_expert_gt_created')!=0 or len(source['tiles'])!=39:
        raise ValueError('Cannot use a different private photo source')
    fid=json.loads(s40.read_text())['fiducials_PRIVATE']
    if len(fid)!=15:raise ValueError('Missing 15 source fiducials')
    output.mkdir(parents=True,exist_ok=True)
    private_masks=output/'PRIVATE_geometry_only_teacher_masks';private_masks.mkdir(exist_ok=True)
    checkpoints=output/'PRIVATE_checkpoints';checkpoints.mkdir(exist_ok=True)
    preview=output/'PRIVATE_preview';preview.mkdir(exist_ok=True)
    data=audited_large_manifest(visualmetro)
    train,test=frame_split(data)
    if len(train)!=109 or len(test)!=36 or train&test:raise AssertionError('visualMetro split leakage')
    templates=extract_track_templates(data,chosen_frames=test)
    all_results={};training={};teacher={}
    for photo_train in ('005','006'):
        photo_test='006' if photo_train=='005' else '005'
        group,provenance=_load_train_images(source,photo_tiles,fid,photo_train,private_masks)
        teacher[photo_train]=provenance
        batches=_draw_batches(group,photo_train)
        models={}
        for arm in ('unet','azunet'):
            models[arm],meta=train_original_model(arm,batches,checkpoints,photo_train)
            training[f'{photo_train}_{arm}']=meta
        direction=f'{photo_train}_to_{photo_test}'
        partial=output/f'PRIVATE_resume_receipt_{direction}.json'
        if partial.exists():
            j=json.loads(partial.read_text())
            if j.get('test_photo')!=photo_test or j.get('scenes')!=n:
                raise ValueError('Resume receipt not matching S42 protocol')
            all_results[direction]=j['results']
            print('S42 reusing validated direction receipt',direction,flush=True)
        else:
            all_results[direction]=eval_models(photo_test,photo_tiles,templates,models,preview,n=n)
            partial.write_text(json.dumps({'test_photo':photo_test,'scenes':n,
                                          'results':all_results[direction]},indent=2)+'\n')
            print('S42 saved direction receipt',direction,flush=True)
    report={'schema':'s42_original_AZ_vs_U_source_pure_teacher_ablation_v1',
        'methods':'Original source-locked AZUNet/UNet, equal settings; pure Hessian teacher only',
        'source_photo_count':2,'independent_physical_runs':1,'training_photo_crops':teacher,
        'excluded_cross_photo_AZ_pseudo_teacher':True,
        'no_native_physics_GT':True,'no_human_labels_required':True,
        'source_visualMetro_frames':145,'heldout_geometry_frames':36,
        'training':training,'test_seeds':TEST_SEEDS,'tested_synthetic_scenes_per_direction':n,
        'track_and_decoy_injections_are_synthetic':True,
        'activation_budget':0.035,'comparators':['unet','azunet','multiscale_hessian','local_darkness'],
        'results':all_results,'uncertainty':'Scene bootstrap only, not independent run uncertainty',
        'no_real_physics_accuracy_or_particle_ID_claim':True}
    (output/'S42_PUBLIC_AGGREGATES.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report


def cli():
    p=argparse.ArgumentParser()
    for arg in ('s39','s40','photo-tiles','visualmetro','output'):
        p.add_argument('--'+arg,required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(run(a.s39,a.s40,a.photo_tiles,a.visualmetro,a.output),ensure_ascii=False,indent=2))
if __name__=='__main__':cli()
