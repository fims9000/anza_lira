"""S120 review rendering: global identity stored separately from a uniform
manual-like green display. Classification labels are NOT physical GT.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from collections import Counter, defaultdict
import cv2,numpy as np
from scipy.spatial import cKDTree
import joblib

COL_GREEN=(42,178,32)
COL_REVIEW=(60,184,238)
COL_JUNK=(174,170,164)

def group_scores(groups,scores,lengths):
    decisions={}
    for g in groups:
        keys=g['source_curve_keys']
        prob=np.array([scores[k] for k in keys if k in scores])
        weight=np.array([min(1000.,max(10.,lengths[k])) for k in keys if k in scores])
        score=float(np.average(prob,weights=weight)) if len(prob) else 0.
        if score>=.55:decision='AUTO_TRACK_LIKE_NOT_GT'
        elif score>=.35:decision='REVIEW_UNCERTAIN'
        else:decision='DEPRIORITIZED_NOT_DELETED'
        decisions[g['global_group_id']]={'score':round(score,4),'decision':decision,'source_curve_count':len(keys)}
    return decisions

def build_kdtree(curves):
    coords=[]
    for pts in curves:
        pts=np.asarray(pts,dtype=np.float32)
        coords.append(pts[::max(1,len(pts)//800)])
    return cKDTree(np.concatenate(coords,axis=0)) if coords else None

def qa_strokes(annotation,tree,thresholds=(3.,5.,10.)):
    summary=defaultdict(lambda:defaultdict(lambda:[0,0]))
    for item in annotation['tracks']:
        if not item.get('line_pixels'):continue
        r=np.asarray(item['line_pixels'],np.int32)
        p=np.column_stack((r[:,1]+r[:,2]//2,r[:,0])).astype(np.float32)
        if tree is None:d=np.full(len(p),np.inf)
        else:d,_=tree.query(p,distance_upper_bound=max(thresholds)+1,workers=1)
        for t in thresholds:
            match=int(np.sum(d<=t))
            s=summary[item['class']][str(int(t))]
            s[0]+=match;s[1]+=len(p)
    return {typ:{k:round(v[0]/max(1,v[1]),4) for k,v in info.items()} for typ,info in summary.items()}

def main(bmp,run,features_npz,model_path,out,manual=None):
    out.mkdir(parents=True,exist_ok=True)
    gray=cv2.imread(str(bmp),cv2.IMREAD_GRAYSCALE)
    config=json.loads((run/'RUN_CONFIG.json').read_text())
    if gray is None or gray.shape!=tuple(config['shape_hw']):raise ValueError('Source frame mismatch')
    source_sha=hashlib.sha256(bmp.read_bytes()).hexdigest()
    if source_sha!=config['source_image_sha256']:raise ValueError('BMP hash mismatch')
    zcheck=np.load(features_npz)
    if str(zcheck['source_sha256'])!=source_sha:raise ValueError('Feature file does not match source BMP')
    manifest=json.load(open(run/'S118_MANIFEST_00090.json'))
    stitch=json.load(open(run/'S118_STITCH_00090'/'S118_STITCH_NOT_GT.json'))
    model=joblib.load(model_path)  # Caller must trust this locally trained model file.
    z=np.load(features_npz)
    keys=z['keys'];x=z['features']; lengths=z['meta'][:,0]
    probs=model.predict_proba(x)[:,1]
    score={str(k):float(p) for k,p in zip(keys,probs)}
    length={str(k):float(l) for k,l in zip(keys,lengths)}
    groups=group_scores(stitch['groups'],score,length)
    group_id={k:g['global_group_id'] for g in stitch['groups'] for k in g['source_curve_keys']}
    classes={k:groups[g]['decision'] for k,g in group_id.items()}
    rng=Counter(classes.values())
    global_counts=Counter(q['decision'] for q in groups.values())
    raw=gray
    # Review visual is a display layer; never changes detections or rewrites group IDs.
    width=1350;scale=width/gray.shape[1];height=int(round(gray.shape[0]*scale))
    low=cv2.resize(gray,(width,height),interpolation=cv2.INTER_AREA)
    src=cv2.cvtColor(low,cv2.COLOR_GRAY2BGR)
    confident=src.copy();uncertain=src.copy()
    dark_mask=np.zeros((height,width),np.uint8)
    preview_paths=[];manual_groups=[]
    total=0;by_class={x:[] for x in ('AUTO_TRACK_LIKE_NOT_GT','REVIEW_UNCERTAIN','DEPRIORITIZED_NOT_DELETED')}
    full_curves=[]
    for i,tile in enumerate(manifest['tiles']):
        rect=tile['xyxy'];cc=json.load(open(run/tile['curves']))['curves'];off=np.array(rect[:2],np.float32)
        for c in cc:
            key=tile['tile_id']+':'+c['candidate_id']
            pts=np.asarray(c['observed_pixel_path_xy'],np.float32)+off
            cclass=classes.get(key,'REVIEW_UNCERTAIN')
            full_curves.append((key,pts,cclass))
            by_class[cclass].append(pts)
            p=np.rint(pts*scale).astype(np.int32)
            if len(p)<2:continue
            if cclass=='AUTO_TRACK_LIKE_NOT_GT':
                cv2.polylines(confident,[p],False,COL_GREEN,2,cv2.LINE_AA)
                cv2.polylines(uncertain,[p],False,COL_GREEN,2,cv2.LINE_AA)
            elif cclass=='REVIEW_UNCERTAIN':
                cv2.polylines(uncertain,[p],False,COL_REVIEW,1,cv2.LINE_AA)
        if (i+1)%30==0:print('render',bmp.name,i+1,'/',len(manifest['tiles']),flush=True)
    for im,txt in [(src,'ORIGINAL BMP'),(confident,'S120 TRACK-LIKE ONLY (UNIFORM GREEN)'),(uncertain,'S120 GREEN + REVIEW AMBER')]:
        cv2.rectangle(im,(0,0),(im.shape[1],47),(30,30,30),-1)
        cv2.putText(im,txt,(9,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA)
    contact=np.concatenate((src,confident,uncertain),axis=1)
    cv2.imwrite(str(out/'FULL_FRAME_S120_UNIFORM_GREEN.jpg'),confident,[cv2.IMWRITE_JPEG_QUALITY,93])
    cv2.imwrite(str(out/'FULL_FRAME_S120_3WAY.jpg'),contact,[cv2.IMWRITE_JPEG_QUALITY,93])
    # Confirmed-only full-resolution centerline cloud without extrapolating gaps.
    manualqa=None
    if manual:
        gt=json.load(open(manual))
        if gt['image']['sha256']!=source_sha: raise ValueError('Manual JSON/BMP sha mismatch')
        manualqa=qa_strokes(gt,build_kdtree(by_class['AUTO_TRACK_LIKE_NOT_GT']))
        manualqa_review=qa_strokes(gt,build_kdtree(by_class['AUTO_TRACK_LIKE_NOT_GT']+by_class['REVIEW_UNCERTAIN']))
        manualqa_raw=qa_strokes(gt,build_kdtree([x[1] for x in full_curves]))
        manualqa={'auto_green':manualqa,'green_or_review':manualqa_review,'baseline_all':manualqa_raw}
    report={'source_image':bmp.name,'sha256':source_sha,'fragment_counts':dict(rng),
            'group_counts':dict(global_counts),'manual_stroke_coverage':manualqa,
            'classifier_role':'PRIVATE_SOURCE_PHOTO_CURATION_DEVELOPMENT_NOT_GT',
            'identity_group_ids_unchanged':True,'original_observations_unchanged':True,
            'no_manual_pixels_consulted_by_feature_extraction_or_group_inference':True,
            'model_may_have_been_trained_on_same_007_manual_labels':True,
            'cross_frame_true_accuracy_unknown':True,
            'probability_calibration_unverified':True}
    (out/'S120_REVIEW_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'S120_GROUP_STATUS.json').write_text(json.dumps({'groups':groups,'image_sha256':source_sha},ensure_ascii=False))
    print('QA',json.dumps(report,ensure_ascii=False)[:5000])
    # Render 1400x1150 native crop triptych. Grid in full-source pixels.
    if manual: locations=[(2750,14100),(3100,9500),(2400,16000),(2400,6500),(2900,21000),(2400,1000)]
    else:locations=[(2500,7000),(3500,13000),(3800,19000),(2900,3500),(4000,9000),(3300,21000)]
    zoom=[]
    for idx,(x0,y0) in enumerate(locations,1):
        w=1400;h=1150;x0=max(0,min(x0,raw.shape[1]-w));y0=max(0,min(y0,raw.shape[0]-h))
        plain=cv2.cvtColor(raw[y0:y0+h,x0:x0+w],cv2.COLOR_GRAY2BGR)
        pix=plain.copy();pixel_compare=plain.copy()
        for key,pts,cls in full_curves:
            if not (np.max(pts[:,0])>=x0 and np.min(pts[:,0])<x0+w and np.max(pts[:,1])>=y0 and np.min(pts[:,1])<y0+h):continue
            v=np.rint(pts-[x0,y0]).astype(np.int32)
            if cls=='AUTO_TRACK_LIKE_NOT_GT':
                cv2.polylines(pix,[v],False,COL_GREEN,2,cv2.LINE_AA)
                cv2.polylines(pixel_compare,[v],False,COL_GREEN,2,cv2.LINE_AA)
            elif cls=='REVIEW_UNCERTAIN':
                cv2.polylines(pixel_compare,[v],False,COL_REVIEW,2,cv2.LINE_AA)
            else:
                cv2.polylines(pixel_compare,[v],False,COL_JUNK,1,cv2.LINE_AA)
        for frame,header in ((plain,'SOURCE BMP'),(pix,'S120 GREEN (ACCEPTED)'),(pixel_compare,'GREEN / AMBER / GRAY')):
            cv2.rectangle(frame,(0,0),(frame.shape[1],41),(25,25,25),-1)
            cv2.putText(frame,header,(8,29),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2,cv2.LINE_AA)
        big=np.concatenate((plain,pix,pixel_compare),axis=1)
        dest=out/f'DETAIL_{idx:02d}_X{x0}_Y{y0}.jpg';cv2.imwrite(str(dest),big,[cv2.IMWRITE_JPEG_QUALITY,90]);zoom.append(dest)
    sheet=np.full((math.ceil(len(zoom)/2)*360,2*1450,3),245,np.uint8)
    for j,file in enumerate(zoom):
        im=cv2.resize(cv2.imread(str(file)),(1450,360),interpolation=cv2.INTER_AREA)
        sheet[(j//2)*360:(j//2+1)*360,(j%2)*1450:(j%2+1)*1450]=im
    cv2.imwrite(str(out/'DETAIL_CONTACT_SHEET.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,90])
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frame',choices=['007','006'],required=True)
    p.add_argument('--model',type=Path,default=Path('/mnt/data/s120_experiments/model_weighted.joblib'))
    a=p.parse_args();root=Path('/mnt/data');f=a.frame
    main(root/('H50Z_030_01_007.bmp' if f=='007' else 'source_unlabelled/extracted/H50Z_030_01_006.bmp'),
         root/('H50Z007_full_S119_safe' if f=='007' else 'H50Z006_full_S119_safe'),
         root/'s120_experiments'/f'features_{f}.npz',a.model,root/'s120_experiments'/f'review_{f}',
         root/'H50Z_030_01_007(1).json' if f=='007' else None)
