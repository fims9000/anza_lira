#!/usr/bin/env python3
"""Source-only AZConv fuzzy gate sensitivity on two authentic H50Z photos.

Diagnostics are descriptive; not expert-validated segmentation accuracy.
"""
from __future__ import annotations
import argparse, json,sys
from pathlib import Path
import numpy as np
import cv2
import torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s33_photometric_domain import image_input
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model,verify_model_source


def measure_model(model,photo_tiles,mode:str):
    layer=model.enc1.block[0]
    measures=[]
    for tile in photo_tiles:
        orig=cv2.imread(str(tile),cv2.IMREAD_GRAYSCALE)
        for j in (0,1,2):
            y0=(80,224,352)[j];x0=(144,224,320)[j]
            gray=orig[y0:y0+64,x0:x0+64]
            edited=np.clip(gray.astype(np.float32)*.8+30.,0,255).astype(np.uint8)
            def memberships(image):
                x=torch.from_numpy(image_input(image,mode)[None,None])
                return torch.softmax(layer.gate_conv(x),dim=1).numpy()[0]
            a=memberships(gray);b=memberships(edited)
            midpoint=.5*(a+b)
            js=.5*(a*np.log(np.maximum(a,1e-12)/np.maximum(midpoint,1e-12))).sum(0)+.5*(b*np.log(np.maximum(b,1e-12)/np.maximum(midpoint,1e-12))).sum(0)
            entropy=(-(a*np.log(np.maximum(a,1e-12))).sum(0)/np.log(2)).mean()
            measures.append({'gate_entropy':float(entropy),'gate_jsd_photometric':float(js.mean()),
                             'gate_mean_max_rule_probability':float(a.max(axis=0).mean()),
                             'gate_rule_1_mean':float(a[0].mean())})
    return {k:float(np.mean([r[k] for r in measures])) for k in measures[0]}


def run(tiles,models,output):
    torch.set_num_threads(2);verify_model_source()
    out={'schema':'s33_azconv_fuzzy_gate_photometric_diagnostic_v1',
         'true_real_particle_labels':0,'independent_physical_run_test':False,
         'method':'first AZConv layer fuzzy memberships of 64px photo patches',
         'results':[]}
    with torch.no_grad():
        for seed in (20261025,20261026):
            for mode in ('raw','quantile','local'):
                path=Path(models)/f'azunet_{mode}_seed{seed}.pt'
                ck=torch.load(path,weights_only=True,map_location='cpu')
                if ck['mode']!=mode or ck['architecture']!='azunet' or ck['seed']!=seed:
                    raise ValueError('incompatible AZ checkpoint')
                model=build_model('azunet',widths=(4,6,8,12),rules=2)
                model.load_state_dict(ck['model_state']);model.eval()
                record={'train_seed':seed,'input_mode':mode}
                for photo in ('005','006'):
                    paths=sorted(Path(tiles).glob(f'H50Z_030_01_{photo}_*.png'))
                    record[photo]=measure_model(model,paths,mode)
                out['results'].append(record)
                print('GATE',seed,mode,'JSD 005',round(record['005']['gate_jsd_photometric'],8),
                      '006',round(record['006']['gate_jsd_photometric'],8),'entropy 006',round(record['006']['gate_entropy'],4),flush=True)
    Path(output).write_text(json.dumps(out,indent=2)+'\n')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--tiles',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.tiles,a.models,a.out)
