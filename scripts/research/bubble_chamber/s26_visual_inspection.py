#!/usr/bin/env python3
"""Honest local review overlays on actual H50Z 512px source crops.

No model predictions are written to expert labels. Reviewers must draw genuine
track/artifact strokes; real source pixels remain unknown until certified.
"""
from __future__ import annotations
import argparse,base64,hashlib,html,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from skimage.morphology import skeletonize
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s26_two_photo_baselines import pixel_input,classical_score
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model
from scripts.research.bubble_chamber.s26_guided_anza import build as build_guided,two_channel


def model_for_checkpoint(path):
    ck=torch.load(path,weights_only=True,map_location='cpu')
    nch=ck.get('input_channels',1)
    if nch==1:
        model=build_model(ck['architecture'],widths=(4,6,8,12),rules=2)
        mode=ck['mode']
        prep=lambda x:pixel_input(x,mode)[None]
    elif nch==2:
        model=build_guided(ck['architecture'])
        prep=two_channel
    else:raise ValueError('Unsupported input count')
    model.load_state_dict(ck['state']);model.eval()
    return model,prep


def tiled_predict(gray,model,prep,size=64):
    if gray.ndim!=2 or gray.shape[0]%size or gray.shape[1]%size:raise ValueError('unaligned grayscale tiles')
    h,w=gray.shape;sub=[]
    for y in range(0,h,size):
        for x in range(0,w,size):sub.append((y,x,prep(gray[y:y+size,x:x+size])))
    pred=np.zeros((h,w),np.float32)
    for a in range(0,len(sub),8):
        b=sub[a:a+8]
        t=torch.from_numpy(np.stack([x[2] for x in b]).astype('float32'))
        with torch.no_grad():scores=model(t).sigmoid()[:,0].cpu().numpy()
        for (y,x,_),q in zip(b,scores):pred[y:y+size,x:x+size]=q
    return pred


def overlay(gray,mask,color):
    rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    m=np.asarray(mask,bool)
    if np.any(m):
        clr=np.array(color,float)
        rgb[m]=np.clip(.38*rgb[m].astype(float)+.62*clr,0,255).astype('uint8')
    return rgb


def imagepng(im,*,jpeg=False):
    if im.ndim==3:im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
    ok,enc=cv2.imencode('.jpg' if jpeg else '.png',im, [cv2.IMWRITE_JPEG_QUALITY,88] if jpeg else [])
    if not ok:raise ValueError('image encode failed')
    return enc.tobytes()


def build(args):
    torch.set_num_threads(2)
    sources={}
    for path in [args.unet,args.azunet,args.guided_az]:
        if not path.is_file():raise FileNotFoundError(path)
    models={key:model_for_checkpoint(path) for key,path in [('unet',args.unet),('azunet',args.azunet),('guided_az',args.guided_az)]}
    cal={}
    for name in ('unet','azunet'):
        report=json.loads(args.report.read_text())
        cal[name]=float(report['methods'][name+'_local']['validation']['threshold'])
    guided_report=json.loads(args.guided_report.read_text())
    cal['guided_az']=guided_report['val_005_calibration']['threshold']
    cal['sato']=float(report['methods']['sato']['validation']['threshold'])
    cal['consensus']='not calibrated independently'
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    image_references=[];manifest=[];collage=[]
    methods=['original','sato','unet','azunet','guided_az','consensus','disagreement']
    for path in sorted(args.crops.glob('H50Z_030_01_*_start*.png')):
        gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape!=(512,512):raise ValueError('expected original 512 crop')
        soft={name:tiled_predict(gray,*model) for name,model in models.items()}
        # Classical Sato runs at full pixel resolution, no downsample for inference.
        sato=classical_score(gray,'sato')
        hard={name:score>=cal[name] for name,score in soft.items()}
        hard['sato']=sato>=cal['sato']
        hard['consensus']=hard['unet'] & hard['guided_az']
        hard['disagreement']=hard['unet'] ^ hard['guided_az']
        images={'original':cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)}
        for name in methods[1:]:
            color=[235,120,90] if name=='disagreement' else ([70,170,245] if name=='sato' else [80,235,120])
            images[name]=overlay(gray,hard[name],color)
        file_id=path.stem
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        source_photo='005' if '_005_' in path.name else '006'
        record={'crop_id':file_id,'source_photo':source_photo,'source_crop_sha256':sha,
                'image_width':512,'image_height':512,
                'no_human_track_labels':True,
                'methods':{name:{'selected_pixel_fraction_UNLABELED':float(mask.mean()),
                                 'connected_components_UNVERIFIED':int(cv2.connectedComponents(mask.astype('uint8'),8)[0]-1)} for name,mask in hard.items()}}
        manifest.append(record)
        image_references.append({'crop_id':file_id,'photo':source_photo,'sha256':sha,
             'image_modes':{name:('data:image/png;base64,' if name=='original' else 'data:image/jpeg;base64,')+base64.b64encode(imagepng(image,jpeg=name!='original')).decode('ascii') for name,image in images.items()}})
        sample=[cv2.resize(images[k],(192,192),interpolation=cv2.INTER_AREA) for k in ('original','sato','unet','azunet','guided_az','consensus')]
        collage.append(np.hstack(sample))
        print('RENDER',file_id,'accepted fractions',[round(hard[k].mean(),4) for k in ('sato','unet','azunet','guided_az','consensus')],flush=True)
    (out/'s26_unverified_candidate_stats.json').write_text(json.dumps({'schema':'s26_photo_candidate_visual_qc_v1',
           'source_input_kind':'real H50Z pixels, artificially supervised trained feature maps',
           'human_true_trace_masks':0,'source_crops':manifest,'models_selected_on_photo005_synthetic_marks':True},indent=2)+'\n')
    collage=np.vstack(collage)
    cv2.imwrite(str(out/'s26_nine_roi_comparison.png'),cv2.cvtColor(collage,cv2.COLOR_RGB2BGR))
    template=(ROOT/'scripts/research/bubble_chamber/s26_review_template.html').read_text()
    payload={'schema':'s26_review_visual_payload_v1','photos':image_references,
         'methods':methods,'no_expert_labels_in_predictions':True}
    safe=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    full=template.replace('___S26_PAYLOAD___',safe)
    (out/'S26_manual_sparse_review.html').write_text(full,encoding='utf-8')
    (out/'S26_manual_sparse_review_manifest.json').write_text(json.dumps({'photos':[{'crop_id':z['crop_id'],'source_crop_sha256':z['sha256']} for z in image_references],
        'human_review_marks_present':False,'source_bmp_embedded_in_private_html':True},indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--crops',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--guided-report',type=Path,required=True)
    p.add_argument('--unet',type=Path,required=True);p.add_argument('--azunet',type=Path,required=True)
    p.add_argument('--guided-az',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a)
