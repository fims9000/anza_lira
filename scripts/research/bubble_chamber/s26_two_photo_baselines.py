#!/usr/bin/env python3
"""S26: actual H50Z pixel backgrounds; *synthetic* sparse trace vs scratch benchmark.

Only 005 is used for fitting and threshold/calibration selection; 006 is
held-out photo test. Neither has human pixel-level track masks. The 070
visualMetro tracks supply shape templates but these are NOT registered to H50Z.
Do not treat metrics as native real trace Dice, particle ID, or physical joins.
"""
from __future__ import annotations
import argparse,hashlib,json,time,sys,math
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score,average_precision_score
from skimage.filters import frangi, sato, meijering
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.synthetic_from_clicks import SyntheticFilmConfig,render_measured_track
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model, verify_model_source

@dataclass
class Example:
    gray:np.ndarray
    sparse:np.ndarray
    synthetic_track_mask:np.ndarray
    synthetic_scratch_mask:np.ndarray
    photo:str
    region:str
    center_xy:tuple[int,int]
    source_template_id:str


def render_one(background:np.ndarray,track:dict,seed:int,index:int,photo:str,region:str,coord:tuple[int,int]):
    size=background.shape[0]
    cfg=SyntheticFilmConfig(size=size,sample_seed=seed+103)
    _,fg,_=render_measured_track(track,(index%3)+1,index,cfg)
    rng=np.random.default_rng(seed*991+index*127+int(photo))
    gray=background.astype(np.float32)
    strength=rng.uniform(32.,93.)
    gray-=strength*fg
    scratches=np.zeros_like(fg,np.uint8)
    for _ in range(4):
        start=rng.integers(4,size-4,size=2).astype(int)
        shift=rng.integers(-18,19,size=2)
        end=np.clip(start+shift,2,size-3)
        cv2.line(scratches,tuple(start),tuple(end),1,int(rng.integers(1,3)),lineType=cv2.LINE_8)
    scratches[fg>0]=0
    gray-=rng.uniform(32.,87.)*scratches
    gray=cv2.GaussianBlur(np.clip(gray,0,255).astype('uint8'),(0,0),float(rng.uniform(.2,.7)))
    pos=np.column_stack(np.where(fg>0));neg=np.column_stack(np.where(scratches>0))
    if len(pos)<22 or len(neg)<22:raise ValueError('insufficient positive or distractor sampled strokes')
    sparse=np.full((size,size),255,np.uint8)
    for pts,label in ((pos,1),(neg,0)):
        selected=pts[rng.choice(len(pts),min(48,len(pts)),replace=False)]
        sparse[selected[:,0],selected[:,1]]=label
    if not (np.any(sparse==1) and np.any(sparse==0) and np.any(sparse==255)):
        raise AssertionError('all 3 target states required')
    return Example(gray,sparse,fg,scratches,photo,region,coord,f"{track['frame_id']}:{track['track_id']}")


def sample_two_photos(bmp_dir:Path,measurements:Path,seed:int,*,train_n=144,val_n=48,test_n=96,size=64):
    if size!=64:raise ValueError('S26 preregistered protocol uses 64px patches')
    manifest=audited_large_manifest(measurements)
    template=[t for f in manifest['frames'] for t in f['tracks'] if t['n_points']>=7]
    if len(template)<200:raise ValueError('insufficient audited templates')
    result={};source=[]
    rng=np.random.default_rng(seed)
    for photo,counts in [('005',{'train':train_n,'val':val_n}),('006',{'test':test_n})]:
        p=bmp_dir/f'H50Z_030_01_{photo}.bmp'
        im=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        if im is None or im.dtype!=np.uint8 or min(im.shape)<size*12:raise ValueError('invalid full original H50Z BMP')
        h,w=im.shape
        source.append({'photo':photo,'bmp_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'shape_yx':[h,w]})
        for split,n in counts.items():
            ys={'train':(.15,.69),'val':(.74,.90),'test':(.15,.9)}[split]
            group=[]
            for j in range(n):
                x=int(rng.uniform(.15*w,.85*w));y=int(rng.uniform(ys[0]*h,ys[1]*h))
                raw=im[y-size//2:y+size//2,x-size//2:x+size//2].copy()
                if raw.shape!=(size,size):raise ValueError('invalid crop extraction')
                idx={'train':0,'val':train_n+3000,'test':train_n+val_n+6000}[split]+j
                track=template[(idx*131+seed*7+int(photo))%len(template)]
                group.append(render_one(raw,track,seed,idx,photo,split,(x,y)))
            result[split]=group
        del im
    assert len(set((x.photo for x in result['train']+result['val'])))==1
    assert len(set(x.photo for x in result['test']))==1
    assert not {tuple(x.center_xy) for x in result['train']} & {tuple(x.center_xy) for x in result['val']}
    return result,{'source_bmps':source,'template_manifest_sha256':manifest['provenance_sha256'],
      'train_and_validation_photo':'005','heldout_test_photo':'006',
      'train_val_spatial_y_zones_disjoint':True,'independent_physical_run':False,
      'real_native_track_labels_available':False,
      'known_training_targets':'synthetic overlaid trace(1) versus artificial scratches(0), unknown native photograph(255)'}


def pixel_input(gray:np.ndarray,mode:str):
    if gray.dtype!=np.uint8 or gray.ndim!=2:raise ValueError('expected uint8 grayscale')
    if mode=='raw':return (255-gray).astype(np.float32)/255.
    if mode=='local':
        b=cv2.GaussianBlur(gray.astype(np.float32),(0,0),5)
        return np.clip((b-gray.astype(np.float32))/45.,0.,1.).astype(np.float32)
    raise ValueError('unknown normalization')


def classical_score(gray:np.ndarray,method:str):
    g=gray.astype(np.float32)/255.
    if method=='darkness':return (1.-g).astype(np.float32)
    if method=='local_contrast':return pixel_input(gray,'local')
    if method=='blackhat':return (cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))).astype(np.float32)/255.)
    if method=='clahe_blackhat':
        eq=cv2.createCLAHE(clipLimit=2.,tileGridSize=(4,4)).apply(gray)
        return classical_score(eq,'blackhat')
    x=1.-g
    if method=='frangi':return frangi(x,sigmas=(1,2,3),black_ridges=False).astype(np.float32)
    if method=='sato':return sato(x,sigmas=(1,2,3),black_ridges=False).astype(np.float32)
    if method=='meijering':return meijering(x,sigmas=(1,2,3),black_ridges=False).astype(np.float32)
    if method=='steerable_ridge':
        # Directional center-surround matched lines, not Steger subpixel detector.
        kernels=[]
        for deg in range(0,180,30):
            theta=np.deg2rad(deg)
            yy,xx=np.mgrid[-7:8,-7:8].astype(np.float32)
            along=xx*np.cos(theta)+yy*np.sin(theta)
            across=-xx*np.sin(theta)+yy*np.cos(theta)
            center=np.exp(-.5*(across/.9)**2)*np.exp(-.5*(along/5.)**4)
            surround=np.exp(-.5*(across/2.8)**2)*np.exp(-.5*(along/5.)**4)
            k=center/center.sum()-surround/surround.sum()
            kernels.append(k.astype(np.float32))
        return np.maximum.reduce([cv2.filter2D(x,-1,k,borderType=cv2.BORDER_REFLECT) for k in kernels])
    raise ValueError('unknown classical filter '+method)


def reviewed_scores(examples,fn):
    scores=[];truth=[];group=[];all_scores=[]
    for i,e in enumerate(examples):
        p=fn(e.gray)
        if p.shape!=e.gray.shape or not np.isfinite(p).all():raise ValueError('prediction invalid')
        known=e.sparse!=255
        scores.extend(p[known].tolist())
        truth.extend(e.sparse[known].astype(int).tolist())
        group.extend([i]*int(known.sum()))
        all_scores.append(p)
    return np.asarray(scores,float),np.asarray(truth,int),np.asarray(group,int),all_scores


def select_threshold(scores,targets,max_fpr=.10):
    if not (0<max_fpr<1):raise ValueError('FPR cap invalid')
    if set(np.unique(targets))!={0,1}:raise ValueError('both labels needed')
    neg=np.asarray(scores)[targets==0];pos=np.asarray(scores)[targets==1]
    # Candidate thresholds selected entirely on PHOTO 005 validation labels.
    values=np.unique(scores)
    if not np.isfinite(values).all():raise ValueError('nonfinite score')
    thresholds=np.r_[np.nextafter(values.max(),np.inf),values]
    best=None
    for t in thresholds:
        fpr=float((neg>=t).mean());tpr=float((pos>=t).mean())
        if fpr>max_fpr+1e-12:continue
        key=(tpr,-fpr,float(t))
        if best is None or key>best[0]:best=(key,float(t),tpr,fpr)
    if best is None:raise AssertionError('all-abstain always available')
    return {'threshold':best[1],'val_tpr':best[2],'val_fpr':best[3],
            'validation_selected_only_on_photo_005':True}


def score_measures(y,scores,threshold,group=None):
    y=np.asarray(y,int);p=np.asarray(scores,float);known=np.asarray(p>=threshold,bool)
    if set(np.unique(y))!={0,1}:raise ValueError('both marks required')
    out={'roc_auc_synthetic_marks':float(roc_auc_score(y,p)),
         'average_precision_synthetic_marks':float(average_precision_score(y,p)),
         'tpr_at_frozen_threshold':float(known[y==1].mean()),
         'fpr_on_synthetic_artifact_marks':float(known[y==0].mean()),
         'fp_synthetic_artifact_marks':int(known[y==0].sum()),
         'tp_synthetic_track_marks':int(known[y==1].sum()),
         'n_known_positive':int(sum(y==1)),'n_known_negative':int(sum(y==0))}
    if group is not None:
        ids=np.unique(group)
        out['macro_patch_balanced_accuracy']=float(np.mean([.5*((known[(group==i)&(y==1)]).mean()+(~known[(group==i)&(y==0)]).mean()) for i in ids]))
    return out


def train_model(name,mode,examples,seed,epochs,output:Path):
    torch.manual_seed(seed);torch.set_num_threads(2)
    model=build_model(name,widths=(4,6,8,12),rules=2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=6e-4)
    train=examples['train'];val=examples['val']
    xx=torch.tensor(np.stack([pixel_input(e.gray,mode) for e in train])[:,None],dtype=torch.float32)
    yy=torch.tensor(np.stack([e.sparse for e in train])[:,None],dtype=torch.long)
    vx=torch.tensor(np.stack([pixel_input(e.gray,mode) for e in val])[:,None],dtype=torch.float32)
    vy=torch.tensor(np.stack([e.sparse for e in val])[:,None],dtype=torch.long)
    best=float('inf');history=[];started=time.monotonic();batch_size=8
    output.mkdir(parents=True,exist_ok=True)
    for epoch in range(epochs):
        model.train();losses=[]
        for ids in np.array_split(np.random.default_rng(seed+epoch).permutation(len(train)),math.ceil(len(train)/batch_size)):
            batch_x=xx[ids].clone();batch_y=yy[ids]
            # Photo 005-only photometric jitter. All labels remain in same coordinates.
            rng=np.random.default_rng(seed+epoch*177+int(ids[0]))
            gain=torch.tensor(rng.uniform(.8,1.2,size=(len(ids),1,1,1)),dtype=torch.float32)
            shift=torch.tensor(rng.uniform(-.10,.10,size=(len(ids),1,1,1)),dtype=torch.float32)
            batch_x=(batch_x*gain+shift).clamp(0,1)
            logits=model(batch_x)
            p=logits[batch_y==1];n=logits[batch_y==0]
            if not len(p) or not len(n):raise RuntimeError('each training batch needs 2 injected classes')
            loss=.5*(F.binary_cross_entropy_with_logits(p,torch.ones_like(p))+
                      F.binary_cross_entropy_with_logits(n,torch.zeros_like(n)))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            optimizer.step();losses.append(float(loss.detach()))
        model.eval();vals=[]
        with torch.no_grad():
            for ids in np.array_split(np.arange(len(val)),math.ceil(len(val)/8)):
                logits=model(vx[ids]);labels=vy[ids]
                vals.append(float(.5*(F.binary_cross_entropy_with_logits(logits[labels==1],torch.ones_like(logits[labels==1]))+
                                    F.binary_cross_entropy_with_logits(logits[labels==0],torch.zeros_like(logits[labels==0])))))
        metric=float(np.mean(vals));history.append({'epoch':epoch+1,'train_bce':float(np.mean(losses)),'photo005_val_bce':metric})
        if metric<best:
            best=metric
            torch.save({'architecture':name,'mode':mode,'state':model.state_dict(),
              'seed':seed,'epoch':epoch+1,'truth_is_only_synthetic_injections':True,
              'test_006_never_used_for_checkpoint_selection':True},output/'best.pt')
    ck=torch.load(output/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state']);model.eval()
    def infer(gray):
        x=torch.from_numpy(pixel_input(gray,mode))[None,None]
        with torch.no_grad():return model(x).sigmoid()[0,0].cpu().numpy()
    return infer,{'epoch':ck['epoch'],'history':history,'seconds':round(time.monotonic()-started,3),'photo006_test_access_during_training':False}


def study(args):
    example,meta=sample_two_photos(args.bmps,args.measurements,args.seed,train_n=args.train_n,val_n=args.val_n,test_n=args.test_n)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    sources={'schema':'s26_real_photo_synthetic_mark_benchmark_v1','input':meta,'seed':args.seed,
      'number_images':{k:len(v) for k,v in example.items()},'architecture_source_locked':verify_model_source(),
      'validation_metric':'AUC or photo005 empirical TPR at photo005 FPR<=10%, NOT real native trace identification',
      'test_photo_006_never_used_to_choose_checkpoint_or_threshold':True,
      'methods':{},'native_photo_track_pixel_truth':'UNKNOWN','physical_auto_join_allowed':False}
    prior=out/'experiment.json'
    if prior.is_file():
        old=json.loads(prior.read_text())
        if old['seed']!=args.seed or old['number_images']!={k:len(v) for k,v in example.items()}:
            raise ValueError('incompatible preexisting S26 results')
        sources['methods']=old['methods']
    only=set(args.only.split(',')) if args.only else None
    classical=['darkness','local_contrast','blackhat','clahe_blackhat','frangi','sato','meijering','steerable_ridge']
    for method in classical:
        if only is not None and method not in only:continue
        if method in sources['methods']:continue
        start=time.monotonic();fn=lambda im,m=method:classical_score(im,m)
        vs,vy,vg,_=reviewed_scores(example['val'],fn)
        ts,ty,tg,pred=reviewed_scores(example['test'],fn)
        cal=select_threshold(vs,vy)
        sources['methods'][method]={'method_type':'no-trained-classical','validation':cal,
            'heldout_006':score_measures(ty,ts,cal['threshold'],tg),'processing_seconds':round(time.monotonic()-start,2)}
        print('CLASSICAL',method,round(sources['methods'][method]['heldout_006']['roc_auc_synthetic_marks'],4),flush=True)
        (out/'experiment.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False))
    for name in ('unet','azunet'):
        for mode in ('raw','local'):
            label=f'{name}_{mode}'
            if only is not None and label not in only:continue
            if label in sources['methods']:continue
            predictor,trainmeta=train_model(name,mode,example,args.seed,args.epochs,out/label)
            vs,vy,vg,_=reviewed_scores(example['val'],predictor)
            cal=select_threshold(vs,vy)
            ts,ty,tg,pred=reviewed_scores(example['test'],predictor)
            sources['methods'][label]={'method_type':'original_source_unchanged_neural','training':trainmeta,
                'validation':cal,'heldout_006':score_measures(ty,ts,cal['threshold'],tg)}
            print('NEURAL',label,'AUC',round(sources['methods'][label]['heldout_006']['roc_auc_synthetic_marks'],4),'testTPR',round(sources['methods'][label]['heldout_006']['tpr_at_frozen_threshold'],4),flush=True)
            (out/'experiment.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False))
    # raw real 512x512 H50Z tiles for visual proposals, never used as true pixel labels
    tiles=[] if args.skip_preview else sorted(Path(args.tiles).glob('H50Z_030_01_*_start*.png'))
    for photo in ('005','006'):
        source_tiles=[p for p in tiles if p.name.startswith(f'H50Z_030_01_{photo}_')]
        if not source_tiles:continue
        img=cv2.imread(str(source_tiles[len(source_tiles)//2]),cv2.IMREAD_GRAYSCALE)
        roi=cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
        scores={k:classical_score(roi,k) for k in ('frangi','sato','blackhat','steerable_ridge')}
        def visualize(z):
            high=np.quantile(z,.995)
            if high<=0:high=1.
            return np.clip(255*z/high,0,255).astype('uint8')
        original=cv2.cvtColor(roi,cv2.COLOR_GRAY2BGR)
        cells=[original]
        for method,z in scores.items():
            normalized=visualize(z)
            overlay=original.copy()
            active=z>=sources['methods'].get(method,{'validation':{'threshold':np.quantile(z,.98)}})['validation']['threshold']
            overlay[active]=[80,210,100]
            cells.append(overlay)
        montage=np.hstack(cells)
        cv2.imwrite(str(out/f'{photo}_classical_unlabeled_proposals.png'),montage)
    (out/'experiment.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False)+'\n')
    return sources

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--bmps',type=Path,required=True);p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--tiles',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261027);p.add_argument('--train-n',type=int,default=144)
    p.add_argument('--val-n',type=int,default=48);p.add_argument('--test-n',type=int,default=96)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--only',default='',help='comma-separated method IDs, only selected methods evaluated')
    p.add_argument('--skip-preview',action='store_true')
    args=p.parse_args();study(args)
