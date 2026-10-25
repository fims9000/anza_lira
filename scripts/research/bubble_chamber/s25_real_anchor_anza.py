#!/usr/bin/env python3
"""Real H50Z start-vs-fiducial local anchor classification using original AZUNet.

This is TRUE human marked *point category* supervision, NOT true track mask
segmentation, 3D particle identity or physical trajectory joining.
"""
from __future__ import annotations
import argparse,hashlib,json,random,sys
from pathlib import Path
import numpy as np,cv2,torch
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model


def dataset(bmp_dir,labels_path,photo,patch=64):
    meta=json.loads(Path(labels_path).read_text())[f'H50Z_030_01_{photo}.bmp']
    img_path=Path(bmp_dir)/f'H50Z_030_01_{photo}.bmp'
    img=cv2.imread(str(img_path),cv2.IMREAD_GRAYSCALE)
    if img is None or tuple(img.shape[::-1])!=(meta['original_width'],meta['original_height']):raise ValueError('source photo dims mismatch')
    data=[]
    for k in meta['keypoints']:
        label=k['label'];x=int(round(k['x_px']));y=int(round(k['y_px']))
        if label!='start' and not label.startswith('f'):continue
        if x-patch//2<0 or x+patch//2>=img.shape[1] or y-patch//2<0 or y+patch//2>=img.shape[0]:raise ValueError('marker on image border')
        crop=img[y-patch//2:y+patch//2,x-patch//2:x+patch//2].copy()
        data.append((crop,int(label=='start'),label))
    return data,{'photo':photo,'bmp_sha256':hashlib.sha256(img_path.read_bytes()).hexdigest(),'labels_sha256':hashlib.sha256(Path(labels_path).read_bytes()).hexdigest(),
                 'markers_total':len(data),'human_start_points':sum(y for _,y,_ in data),'human_fiducials':sum(1-y for _,y,_ in data)}


def model_score(model,x):
    logits=model(x)
    # A scalar discriminative patch score, not pixel-classification ground truth.
    return logits.mean(dim=(1,2,3))


def run(base,labels,out,seed=20261025,epochs=12):
    train,trainmeta=dataset(base,labels,'005');test,testmeta=dataset(base,labels,'006')
    out.mkdir(parents=True,exist_ok=True);report={'train':trainmeta,'heldout_photo_test':testmeta,'methods':{},
        'ground_truth_type':'human START point versus human fiducial category, not track/particle masks',
        'train_and_test_from_same_physical_run':True,'pixel_segmentation_accuracy':'NOT_MEASURED',
        'training_real_photographic_supervision':True,'independent_physical_run_validation':False}
    for name in ['unet','azunet']:
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.set_num_threads(2)
        model=build_model(name,widths=(4,6,8,12),rules=2)
        opt=torch.optim.AdamW(model.parameters(),lr=.0006)
        aug=[]
        for im,y,label in train:
            for k in range(4):aug.append((np.rot90(im,k).copy(),y,label))
        for epoch in range(epochs):
            model.train()
            for i in np.random.default_rng(seed+epoch).permutation(len(aug)):
                im,y,_=aug[i]
                x=torch.from_numpy((255-im).astype('float32')[None,None]/255)
                label=torch.tensor([float(y)])
                pred=model_score(model,x)
                loss=F.binary_cross_entropy_with_logits(pred,label)
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        model.eval();scores=[];labels_true=[]
        with torch.no_grad():
            for im,y,_ in test:
                x=torch.from_numpy((255-im).astype('float32')[None,None]/255)
                p=float(torch.sigmoid(model_score(model,x))[0]);scores.append(p);labels_true.append(y)
        auc=float(roc_auc_score(labels_true,scores))
        result={'auc_heldout_photo_006':auc,'heldout_labels':labels_true,
                'heldout_scores':[round(x,5) for x in scores],'epochs':epochs,
                'interpretation':'Only start-vs-fiducial anchor discrimination; N=11, not track segmentation'}
        report['methods'][name]=result
        torch.save({'model':model.state_dict(),'kind':name,'source':'real user H50Z anchors, NOT track segmentation checkpoint','seed':seed},out/f'{name}_anchors.pt')
        (out/'anchor_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(name,auc,flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bmps',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=12);args=p.parse_args();run(args.bmps,args.labels,args.out,epochs=args.epochs)
