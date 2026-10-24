#!/usr/bin/env python3
"""Qualitative transfer probe only; no fake labels or real-photo accuracy claims."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model,verify_model_source


def probe(index:Path,crops:Path,out:Path,*,seed:int=20261024):
    verify_model_source()
    idx=json.loads(index.read_text())
    fns=sorted(crops.glob('*.png'))
    if not fns: raise ValueError('no real unlabeled crops')
    images=[];names=[]
    for f in fns:
        im=cv2.imread(str(f),cv2.IMREAD_GRAYSCALE)
        if im is None:continue
        names.append(f.name)
        images.append(cv2.resize(im,(64,64),interpolation=cv2.INTER_AREA))
    im=np.stack(images)
    inp=torch.from_numpy(((255-im).astype('float32')/255)[:,None])
    records=[];pred_example={};torch.set_num_threads(2)
    for e in idx['entries']:
        if e['seed']!=seed:continue
        path=index.parent/e['relative_dir']
        saved=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
        model=build_model(e['architecture'],widths=(4,6,8,12),rules=2)
        model.load_state_dict(saved['weights']);model.eval()
        with torch.no_grad():prob=model(inp).sigmoid().numpy()[:,0]
        threshold=float(json.loads((path/'metrics.json').read_text())['threshold_selected_on_synthetic_dense_validation'])
        pred=(prob>=threshold)
        pred_example[e['architecture']+'_'+e['arm']]=np.uint8(np.clip(prob[0]*255,0,255))
        for j,name in enumerate(names):
            p=prob[j];q=np.clip(p,1e-6,1-1e-6)
            entropy=float((-q*np.log2(q)-(1-q)*np.log2(1-q)).mean())
            records.append({'architecture':e['architecture'],'arm':e['arm'],'seed':seed,
                            'image_name':name,'threshold_from_synthetic_validation':threshold,
                            'predicted_positive_fraction':float(pred[j].mean()),
                            'mean_probability':float(p.mean()),'mean_probability_entropy_bits':entropy})
    out.mkdir(parents=True,exist_ok=True)
    result={'schema':'s18_unlabeled_real_probe_v1','real_crops_count':len(names),
            'size_model_input_64x64_from_real_512px_crop':True,
            'statement':'No expert track masks, nothing here is accuracy. Severe downsample destroys narrow tracks; predictions are not physical reconstruction.',
            'source_files':[{'filename':p.name,'size_bytes':p.stat().st_size} for p in fns],
            'record':records,'real_accuracy':None,'real_pixel_predictions_are_ground_truth':False}
    (out/'unlabeled_real_probe.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    # One contact sheet: grayscale probabilities from first real input, no labels.
    names_out=['azunet_A_measured_only','azunet_B_measured_plus_book','azunet_C_measured_book_photo_style',
               'azunet_D_measured_book_synthetic_style','azunet_E_book_generic_robust_augmentation']
    from PIL import Image,ImageDraw
    scale=3;width=192;height=228
    canvas=Image.new('RGB',(width*3,height*2),(240,240,240));d=ImageDraw.Draw(canvas)
    blocks=[(im[0],'Unlabeled real crop (downsampled)')]+[(pred_example[key],key.replace('azunet_','')[:23]) for key in names_out]
    for i,(pic,name) in enumerate(blocks):
        x=(i%3)*width;y=(i//3)*height
        canvas.paste(Image.fromarray(pic).convert('RGB').resize((width,192),Image.Resampling.NEAREST),(x,y+24))
        d.text((x+4,y+4),name,fill=(0,0,0))
    canvas.save(out/'real_H50Z_unlabeled_qualitative.png')
    print(json.dumps({'real_crops':len(names),'pred_positive_fractions_by_az_arm':{arm:round(float(np.mean([r['predicted_positive_fraction'] for r in records if r['architecture']=='azunet' and r['arm']==arm])),3) for arm in idx_armset(idx)},'warning':result['statement']},ensure_ascii=False))
    return result


def idx_armset(idx):return sorted({e['arm'] for e in idx['entries']})


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--real-crops',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();probe(a.index,a.real_crops,a.out_dir)

if __name__=='__main__':main()
