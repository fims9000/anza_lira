#!/usr/bin/env python3
"""Train-label-independent image-input audit on both original H50Z BMP crops.

Measures a photometric invariance property, NOT human physical-track accuracy.
All pixels are from the same 39 source-verified H50Z 512px crops as S31.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import argparse,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s33_photometric_domain import image_input
from bubble_chamber.s35_photometric_consistency import paired_exposure_views
from scripts.research.bubble_chamber.s33_train_photometric import load_source_weak_masks


def source_audit(manifest,tiles,n_patches_per_crop=8,seed=20261025):
    if n_patches_per_crop != 8:raise ValueError('Preregistered audit coverage changed')
    groups,provenance=load_source_weak_masks(manifest,tiles)
    report={'schema':'s35_authentic_H50Z_photo_view_invariance_diagnostic_v1',
        'source':provenance,'natural_photo_track_pixel_truth':False,
        'both_photos_from_one_physical_run':True,'diagnostic_only_no_model_selection':True,
        'per_photo':{}}
    for photo,items in sorted(groups.items()):
        values={'raw':[],'quantile':[],'local':[]}
        input_variances=[];saturation_before=[];saturation_after=[];quantile_spans=[]
        for photoim,_,_,record in sorted(items,key=lambda x:x[3]['crop_id']):
            name=record['crop_id'];rng=np.random.default_rng(int(sha256(f'{seed}|{name}'.encode()).hexdigest()[:16],16))
            for i in range(n_patches_per_crop):
                y=int(rng.integers(0,449));x=int(rng.integers(0,449))
                patch=photoim[y:y+64,x:x+64]
                exposure_seed=seed*100+i*104729+int(sha256(name.encode()).hexdigest()[:8],16)
                # Derive the same synthetic affine view for ALL modes and report
                # raw-distance as a control. Never fit to target/test statistics.
                pr,qr=paired_exposure_views(patch,'quantile',seed=exposure_seed)
                pl,ql=paired_exposure_views(patch,'local',seed=exposure_seed)
                # Recover identical brightness transform without relying on mode output.
                ng=np.random.default_rng(exposure_seed)
                gain=float(ng.uniform(.74,1.26));offset=float(ng.uniform(-28,28))
                observed=np.clip(gain*patch.astype(np.float32)+offset,0,255).astype(np.uint8)
                raw=image_input(patch,'raw');raw2=image_input(observed,'raw')
                values['raw'].append(float(np.mean(np.abs(raw-raw2))))
                values['quantile'].append(float(np.mean(np.abs(pr-qr))))
                values['local'].append(float(np.mean(np.abs(pl-ql))))
                input_variances.append(float(patch.std()))
                quantile_spans.append(float(np.quantile(patch,.9)-np.quantile(patch,.1)))
                saturation_before.append(float(np.mean((patch==0)|(patch==255))))
                saturation_after.append(float(np.mean((observed==0)|(observed==255))))
        report['per_photo'][photo]={
            'crop_count':len(items),'patches':len(input_variances),
            'input_patch_std_gray_mean':float(np.mean(input_variances)),
            'mean_quantile_90_minus_10_gray_span':float(np.mean(quantile_spans)),
            'original_sensor_zero_or_255_fraction':float(np.mean(saturation_before)),
            'after_seeded_affine_zero_or_255_fraction':float(np.mean(saturation_after)),
            'mean_abs_normalized_input_change_under_seeded_affine_exposure':{k:float(np.mean(v)) for k,v in values.items()},
            'fraction_of_patches_with_lower_shift_than_raw':{k:float(np.mean(np.asarray(v)<np.asarray(values['raw']))) for k,v in values.items() if k!='raw'},
            'source_native_track_accuracy':'NOT_MEASURED'}
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--tiles',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=source_audit(a.manifest,a.tiles)
    a.out.parent.mkdir(exist_ok=True,parents=True)
    a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(r['per_photo'],ensure_ascii=False,indent=2))
