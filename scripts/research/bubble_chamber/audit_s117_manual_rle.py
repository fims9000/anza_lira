"""Independent, incomplete manual-stroke identity audit for S117 outputs.

No annotation is fed to the association model. The RLE annotation convention
uses [source_image_row, starting_column, length]; never exchange image axes.

Example:
  python scripts/research/bubble_chamber/audit_s117_manual_rle.py \
    --annotation H50Z_030_01_007.json --bmp H50Z_030_01_007.bmp \
    --results /tmp/s117_007 --roi A_top:3600:6000:1200
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def _sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as inp:
        for chunk in iter(lambda: inp.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit(annotation, bmp, result_root, regions):
    labels=json.loads(annotation.read_text(encoding='utf-8-sig'))
    image=labels['image']
    if bmp.name!=image['file_name'] or _sha(bmp)!=image['sha256']:
        raise ValueError('Image name or SHA256 does not match operator reference')
    W=int(image['width']);H=int(image['height'])
    report=[]
    for tag,x0,y0,size in regions:
        if not (0<=x0<W and 0<=y0<H and x0+size<=W and y0+size<=H):
            raise ValueError('ROI outside source photo')
        root=result_root/tag
        base=json.loads((root/'S115_CURVES.json').read_text())
        result=json.loads((root/'S117_GROUPS.json').read_text())
        if base.get('roi')!=tag or base.get('xyxy')!=[x0,y0,x0+size,y0+size]:
            raise ValueError('ROI coordinates disagree with frozen S115 result')
        curves=base['curves']
        if set(c['candidate_id'] for c in curves)!=set(n for g in result['groups'] for n in g['source_curve_ids']):
            raise ValueError('S117 and S115 candidate membership disagree')
        points=[]
        for c in curves:
            p=np.rint(np.asarray(c['observed_pixel_path_xy'])).astype(int)
            if len(p)==0 or np.any(p<0) or np.any(p[:,0]>=size) or np.any(p[:,1]>=size):
                raise ValueError('Curve outside expected S115 local source ROI')
            points.append(p)
        counts={i:{} for i in range(len(curves))}
        mask=np.zeros((size,size),np.uint8)
        manual_in_roi=0
        for j,t in enumerate(labels['tracks']):
            if t.get('class')!='track' or not t.get('line_pixels'):
                continue
            pix=np.asarray(t['line_pixels'],dtype=np.int32)
            pix=pix[(pix[:,0]>=y0)&(pix[:,0]<y0+size)&
                    (pix[:,1]<x0+size)&((pix[:,1]+pix[:,2])>x0)]
            if len(pix)==0:continue
            mask.fill(0)
            for row, col, width in pix:
                begin=max(0,int(col-x0));end=min(size,int(col+width-x0))
                if begin<end:mask[int(row-y0),begin:end]=1
            if int(mask.sum())<60:continue
            manual_in_roi+=1
            dil=cv2.dilate(mask,np.ones((11,11),np.uint8))
            for i,p in enumerate(points):
                n=int(dil[p[:,1],p[:,0]].sum())
                if n>=12:counts[i][j]=n
        inferred={}
        for i,c in enumerate(curves):
            ranked=sorted(counts[i].items(),key=lambda x:-x[1])
            if not ranked:continue
            j,n=ranked[0]
            second=ranked[1][1] if len(ranked)>1 else 0
            if n>=25 and n>=.2*len(points[i]) and n>=1.8*second:
                inferred[c['candidate_id']]=j
        by_group={cid:g['group_id'] for g in result['groups'] for cid in g['source_curve_ids']}
        before={};after={}
        for cid,mid in inferred.items():
            before.setdefault(mid,set()).add(cid)
            after.setdefault(mid,set()).add(by_group[cid])
        two_plus=[mid for mid in before if len(before[mid])>=2]
        status=[]
        for e in result['links']:
            if e['decision'] not in ('AUTO_PROVISIONAL_GEOMETRY_GROUP_NOT_GT','EXPERT_CONFIRMED_SAME_LINK'):
                continue
            a,b=(inferred.get(cid) for cid in e['curve_ids'])
            state='UNKNOWN' if a is None or b is None else 'SAME_MANUAL_ID' if a==b else 'CONFLICTING_MANUAL_IDS'
            status.append({'candidate_id':e['candidate_id'],'status':state,
                           'manual_id_a':a,'manual_id_b':b,'gap_px':e['gap_px']})
        row={'roi':tag,'manual_ids_in_roi':manual_in_roi,
             'confidently_assigned_curves':len(inferred),
             'manual_ids_with_2plus_source_curves':len(two_plus),
             'sum_source_groups_before':sum(len(before[mid]) for mid in two_plus),
             'sum_source_groups_after':sum(len(after[mid]) for mid in two_plus),
             'accepted_links':len(status),
             'same_manual_id':sum(x['status']=='SAME_MANUAL_ID' for x in status),
             'conflicting_manual_ids':sum(x['status']=='CONFLICTING_MANUAL_IDS' for x in status),
             'unresolved':sum(x['status']=='UNKNOWN' for x in status),
             'details':status,
             'limit':'Only existing human strokes on this frame; partial labels and intersections are ambiguous, not particle-level recall'}
        report.append(row)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--annotation',type=Path,required=True)
    p.add_argument('--bmp',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--roi',action='append',required=True,help='name:x0:y0:size, repeat')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    regions=[]
    for item in a.roi:
        tag,x0,y0,size=item.split(':')
        regions.append((tag,int(x0),int(y0),int(size)))
    result=audit(a.annotation,a.bmp,a.results,regions)
    report=json.dumps(result,ensure_ascii=False,indent=2)
    if a.output:a.output.write_text(report,encoding='utf8')
    print(report)
