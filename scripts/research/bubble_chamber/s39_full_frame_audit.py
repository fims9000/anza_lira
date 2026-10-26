#!/usr/bin/env python3
"""S39 unbiased spatial audit on the TWO full authentic H50Z BMP frames.

One physical run; neither randomly sampled crops nor machine predictions are GT.
Every random coordinate and native image is private. Public report has aggregates.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from collections import Counter
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s39_machine_auto import annotate_tile
from scripts.research.bubble_chamber.s39_run_machine_only import predict_cross_photo,overlay
from scripts.research.bubble_chamber.s35_benchmark_fragment_links import load_model

SHAPES={'005':(9020,26151),'006':(9268,25324)}


def _image(path:Path,photo:str):
    import struct
    with path.open('rb') as stream:
        header=stream.read(54)
    wh=struct.unpack_from('<ii',header,18)
    bits=struct.unpack_from('<H',header,28)[0]
    if header[:2]!=b'BM' or wh!=SHAPES[photo] or bits!=8:
        raise ValueError('Full BMP archive source does not match audited H50Z dimensions')
    # Only after strict source shape/format verification disable the generic
    # Pillow >179M pixel safeguard; originals are ~235M single-channel pixels.
    Image.MAX_IMAGE_PIXELS=None
    image=Image.open(path)
    if image.mode!='L' or image.size!=wh:raise ValueError('Source BMP must be genuine 8-bit gray')
    return image


def fixed_sample(w:int,h:int,photo:str,*,n=96,seed=20261025):
    if n<12 or n>240:raise ValueError('Unsafe sample size')
    # Floor-grid leaves partial end strips excluded. Separate vertical octiles
    # avoid sampling only a narrow photometric region of the film scan.
    ys=list(range(0,h-512+1,512));xs=list(range(0,w-512+1,512))
    grid=[(x,y) for y in ys for x in xs]
    bands={k:[] for k in range(8)}
    for x,y in grid:bands[min(7,int((y+256)/h*8))].append((x,y))
    rng=np.random.default_rng(seed+int(photo))
    size=[n//8]*8
    for k in range(n%8):size[k]+=1
    result=[]
    for k in range(8):
        if size[k]>len(bands[k]):raise ValueError('Grid too small for stratified sampling')
        ids=rng.choice(len(bands[k]),size=size[k],replace=False)
        result.extend(bands[k][int(i)] for i in ids)
    assert len(set(result))==n
    return result


def run(originals:Path,checkpoints:Path,out:Path,*,n=96,az_per_photo=24):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    all_rows=[];counts=Counter()
    for photo in ('005','006'):
        print('Starting full photo',photo,flush=True)
        bmp=originals/f'H50Z_030_01_{photo}.bmp'
        photo_sha=hashlib.sha256(bmp.read_bytes()).hexdigest()
        origin=_image(bmp,photo)
        w,h=origin.size
        positions=fixed_sample(w,h,photo,n=n)
        # Independent of image intensities, first N hashed sample indices get
        # AZ evidence from checkpoint trained only on the OTHER photo.
        model_source='006' if photo=='005' else '005'
        model,receipt=load_model(checkpoints,model_source)
        rows=[];time0=time.time()
        for index,(x,y) in enumerate(positions):
            if index%12==0:print('Beginning',photo,index,'/',n,flush=True)
            gray=np.asarray(origin.crop((x,y,x+512,y+512)),dtype='uint8')
            az=predict_cross_photo(gray,model) if index<az_per_photo else None
            proposals,extra=annotate_tile(gray,[],az_map=az,max_new=24)
            summary=extra['stats'];counts['tiles']+=1
            counts['novel_proposals']+=summary['new_ridge_proposals']
            counts['tiles_with_any_proposal']+=int(bool(proposals))
            if az is not None:counts['tiles_with_cross_photo_AZ']+=1
            rows.append({'photo':photo,'x_private':x,'y_private':y,'index':index,
                  'photometric_stratum_vertical_octile':min(7,int((y+256)/h*8)),
                  'gray_mean':round(float(gray.mean()),3),
                  'gray_saturation_fraction':round(float(np.mean(gray>=250)),5),
                  'cross_photo_AZ_available':az is not None,
                  'candidate_count':len(proposals),'candidates_PRIVATE':proposals,
                  'manual_GT_labels':0})
            # fixed first 6 and a separated set: no cherry-picking on scores.
            if index in {0,1,2,3,4,5,32,33,64,65,90,91}:
                private_dir=out/'private_spatial_samples'
                private_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(private_dir/f'{photo}_stratified_sample_{index:03d}_PRIVATE.png'),
                            overlay(gray,[],proposals))
            if index%24==23:
                print('H50Z',photo,index+1,'/',n,'sampled',round(time.time()-time0,2),'seconds',flush=True)
        origin.close()
        all_rows+=rows
        counts[f'photo_{photo}_unbiased_tiles']=len(rows)
        counts[f'photo_{photo}_with_candidates']=sum(r['candidate_count']>0 for r in rows)
        counts[f'photo_{photo}_proposals']=sum(r['candidate_count'] for r in rows)
        counts[f'photo_{photo}_saturation_rate_mean']=round(float(np.mean([r['gray_saturation_fraction'] for r in rows])),5)
    report={'schema':'s39_full_bmp_spatial_machine_only_audit_v1','date':'2026-10-25',
        'original_BMP_photos':2,'real_physical_runs':1,
        'original_BMP_dimensions':{k:list(v) for k,v in SHAPES.items()},
        'sample_protocol':'Precommitted uniformly chosen non-overlapping 512px grid tiles per photo, balanced over 8 vertical octiles; no model-score-based tile selection',
        'sample_seed':20261025,'sampled_crops_per_photo':n,
        'cross_photo_AZ_evidence_sampled_crops_per_photo':az_per_photo,
        'results':dict(counts),
        'native_expert_masks':0,'native_particle_identity_labels':0,
        'estimated_physical_track_recall_or_precision':None,
        'machine_artefacts_not_excluded':True,
        'source_private_coordinates_not_published':True,
        'source_full_bmp_sha256_private_only':True}
    (out/'S39_FULL_BMP_AUDIT_PUBLIC.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (out/'S39_FULL_BMP_CANDIDATES_PRIVATE.json').write_text(json.dumps({'schema':'s39_unbiased_full_bmp_machine_drafts_v1','no_truth':True,'frames':all_rows},indent=2,ensure_ascii=False)+'\n')
    return report


def cli():
    p=argparse.ArgumentParser()
    p.add_argument('--private-extracted-bmp-dir',type=Path,required=True)
    p.add_argument('--model-checkpoints',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--tiles-per-photo',type=int,default=96)
    p.add_argument('--AZ-tiles-per-photo',type=int,default=24)
    a=p.parse_args();print(json.dumps(run(a.private_extracted_bmp_dir,a.model_checkpoints,a.out_dir,n=a.tiles_per_photo,az_per_photo=a.AZ_tiles_per_photo),indent=2))
if __name__=='__main__':cli()
