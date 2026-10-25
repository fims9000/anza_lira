#!/usr/bin/env python3
"""Resume evaluation from trusted local S26 guided checkpoint.

Partial training artifacts must be labeled as such; photo006 only used for
final evaluation after immutable choice of checkpoint by photo005 validation.
"""
from __future__ import annotations
import argparse,json,torch
from pathlib import Path
from scripts.research.bubble_chamber.s26_two_photo_baselines import sample_two_photos,reviewed_scores,select_threshold,score_measures
from scripts.research.bubble_chamber.s26_guided_anza import build,two_channel

def evaluate(bmps,measurements,ck_path):
    torch.set_num_threads(2)
    ck=torch.load(ck_path,weights_only=True,map_location='cpu')
    data,meta=sample_two_photos(bmps,measurements,ck['seed'],train_n=144,val_n=48,test_n=96)
    model=build(ck['architecture']);model.load_state_dict(ck['state']);model.eval()
    def predict(im):
        with torch.no_grad():return model(torch.from_numpy(two_channel(im))[None]).sigmoid()[0,0].cpu().numpy()
    vs,y,_,_=reviewed_scores(data['val'],predict);cal=select_threshold(vs,y)
    ts,t,grp,_=reviewed_scores(data['test'],predict)
    return {'architecture':ck['architecture'],'method':'local+steerable_2ch', 'checkpoint_best_epoch':ck['epoch'],
            'training_completed_all_10_epochs':False,
            'val_005_calibration':cal,'test_006':score_measures(t,ts,cal['threshold'],grp),
            'trained_on':'synthetic added pixel labels only','true_real_photo_track_pixels':0}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bmps',type=Path,required=True);p.add_argument('--measurements',type=Path,required=True)
 p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 x=evaluate(a.bmps,a.measurements,a.checkpoint);a.out.write_text(json.dumps(x,indent=2)+'\n');print(x['test_006'])
