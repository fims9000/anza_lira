#!/usr/bin/env python3
"""Quantify original AZConv learned tangent alignment on frozen test images.

Synthetic target mask is used ONLY in evaluation, never to re-fit the model.
No real-film validation, physical charge, or magnetic calibration is claimed.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber import s7_original_retrain as s7
from bubble_chamber.s12_orientation_field import reference_orientation
from models.azconv import AZConv2d


@torch.no_grad()
def evaluate_model_alignment(model, samples:s7.TrackDataset, cfg:s7.RunConfig, stage:int=0, stress:bool=False):
    layers=[m for m in model.modules() if isinstance(m,AZConv2d)]
    if not layers or not (0<=stage<len(layers)):
        raise ValueError('Missing requested original AZConv layer')
    layer=layers[stage]
    model.eval()
    per_frame={}
    per_image=[]
    for i in range(len(samples)):
        x,y,_=samples[i]
        inp=x[None]
        if stress:inp=s7.apply_stress(inp,torch.tensor([i]),cfg)
        model(inp)
        live=layer._last_live_interpretation
        mu=live.get('mu');theta=live.get('theta_map')
        if mu is None or theta is None:
            raise RuntimeError('Original AZ layer did not publish live rule orientation')
        c=(mu*torch.cos(2*theta)).sum(1,keepdim=True)
        s=(mu*torch.sin(2*theta)).sum(1,keepdim=True)
        target=F.interpolate(y[None],size=theta.shape[-2:],mode='area')
        cc,ss,weight=reference_orientation(target)
        weight=weight.clamp(0,1)
        weight_sum=float(weight.sum())
        if weight_sum<=1e-7:continue
        dot=((c*cc+s*ss)*weight).sum()/weight.sum()
        amplitude=(torch.sqrt(c*c+s*s+1e-8)*weight).sum()/weight.sum()
        angle_good=((c*cc+s*ss)/(torch.sqrt(c*c+s*s+1e-8))).clamp(-1,1)
        vector=float(dot)
        cosine=float((angle_good*weight).sum()/weight.sum())
        amp=float(amplitude)
        frame=samples.groups()[i]
        per_image.append({'frame':frame,'dot':vector,'cos':cosine,'amplitude':amp,'reference_support':weight_sum})
        d=per_frame.setdefault(frame,[])
        d.append((vector,cosine,amp))
    if not per_image:raise ValueError('No reliable measured-shaped target direction pixels')
    return {'stage':stage,'stress':stress,'n_images':len(per_image),'n_frames':len(per_frame),
        'mean_vector_alignment':float(np.mean([x['dot'] for x in per_image])),
        'mean_direction_cosine':float(np.mean([x['cos'] for x in per_image])),
        'mean_orientation_magnitude':float(np.mean([x['amplitude'] for x in per_image])),
        'per_frame':{f:{'vector_alignment':float(np.mean([v[0] for v in records])),
                         'direction_cosine':float(np.mean([v[1] for v in records])),
                         'orientation_magnitude':float(np.mean([v[2] for v in records]))}
                     for f,records in per_frame.items()},
        'interpretation':'+1 parallel axis, -1 perpendicular axis; target axes from SYNTHETIC masks'}


def run(measurements:Path,train_dir:Path,out:Path,arms=('az_base','az_direct05','az_direct_cldice05')):
    s7.verify_sources(ROOT)
    protocol=json.loads((train_dir/'s12_metrics.json').read_text())
    cfg=s7.RunConfig(**protocol['config'])
    torch.set_num_threads(cfg.cpu_threads)
    source,_=s7.make_datasets(cfg,measurements)
    if s7.split_fingerprints(source)!=protocol['data_fingerprints']:
        raise ValueError('Data mismatch')
    test=s7.TrackDataset(source['test'])
    result={'schema':'S12_AUTHENTIC_AZ_AXIS_ALIGNMENT_DIAGNOSTIC','source_is_real_image':False,
        'data_seed':s7.effective_data_seed(cfg),'model_seed':cfg.seed,'arms':{}}
    for arm in arms:
        model=s7.build_models(cfg)[protocol['results'][arm]['setting']['architecture']]
        saved=torch.load(train_dir/arm/'best_final.pt',map_location='cpu',weights_only=False)
        if saved['source_blobs']!=s7.SOURCE_BLOBS:raise RuntimeError('Model source mismatch')
        model.load_state_dict(saved['state_dict'])
        result['arms'][arm]={}
        for condition,stress in [('clean',False),('stress',True)]:
            result['arms'][arm][condition]=evaluate_model_alignment(model,test,cfg,stage=0,stress=stress)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--training-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    r=run(args.measurements,args.training_dir,args.out)
    for arm,x in r['arms'].items():
        print(arm,'clean axis',x['clean']['mean_vector_alignment'],'stress',x['stress']['mean_vector_alignment'])
if __name__=='__main__':main()
