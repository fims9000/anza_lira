#!/usr/bin/env python3
"""Private offline quicklook of arbitrary BMP ROIs: source vs Ridge vs LSD.

Only classical pixel-derived line hypotheses. NOT original AZUNet, S39 or S56;
no physical labels, joining, measured-track transfer or expert GT inference.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from bubble_chamber.bitmap_tiles import GrayBMP8
from bubble_chamber.physicists_intake import inventory_bmps,make_mark_preview
from bubble_chamber.interop_api import validate_lines
from bubble_chamber.interop_detectors import HessianRidgeDetector
from bubble_chamber.interop_detectors_cv5 import OpenCVLSDDetectorCompat


def _safe_output(out:Path):
    dest=out.resolve();root=ROOT.resolve()
    if dest==root or root in dest.parents:
        raise ValueError('PRIVATE source pixels must not be written inside the source repository')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('Private preview directory must be new/empty')
    return dest


def _centers(width:int,height:int,count:int):
    if not 1<=count<=12:raise ValueError('ROI count must be 1..12')
    cols=int(np.ceil(np.sqrt(count*width/height)))
    cols=max(1,min(count,cols))
    rows=int(np.ceil(count/cols))
    result=[]
    for y in (np.arange(rows)+.5)*height/rows:
        for x in (np.arange(cols)+.5)*width/cols:
            if len(result)<count:result.append((float(x),float(y)))
    return result


def overlay(gray,marks,offset,detector,kind):
    """Source operator crosses never mixed with model candidate identity."""
    result=make_mark_preview(gray,marks,origin_xy=offset,
                             title='ORIGINAL CROSSES / MACHINE HYPOTHESES UNKNOWN')
    if detector is None:return result,0
    lines=validate_lines(list(detector.detect(gray.copy())),max_lines=12)
    color=(80,240,90) if kind=='RIDGE' else (240,170,110)
    for line in lines:
        cv2.polylines(result,[np.rint(line).astype(np.int32)],False,color,1,cv2.LINE_AA)
    cv2.putText(result,kind+' CANDIDATES / NOT GT',
                (4,gray.shape[0]-8),cv2.FONT_HERSHEY_SIMPLEX,.35,(30,210,230),1,cv2.LINE_AA)
    return result,len(lines)


def run(*,bmp_root:Path,crosses:Path|None,out_dir:Path,
        max_images:int=4,per_image:int=4,roi_mode:str='grid'):
    _safe_output(out_dir)
    if roi_mode not in {'grid','start','both'}:
        raise ValueError('ROI mode must be grid/start/both')
    if not 1<=max_images<=100 or not 1<=per_image<=12:
        raise ValueError('Bounded max-images and per-image counts required')
    public,private=inventory_bmps(bmp_root,labels_json=crosses)
    detectors={'RIDGE':HessianRidgeDetector(max_lines=12),
               'LSD':OpenCVLSDDetectorCompat(max_lines=12)}
    scenes=[]
    for row in private['bmp_records'][:max_images]:
        bitmap=GrayBMP8(bmp_root/row['relative_path'])
        marks=row['marks']
        starts=[(float(p['xy_source_px'][0]),float(p['xy_source_px'][1])) for p in marks if p['label']=='start']
        centers=[]
        if roi_mode in {'start','both'}:centers.extend(starts[:per_image])
        if roi_mode in {'grid','both'}:
            for point in _centers(bitmap.width,bitmap.height,per_image):
                if len(centers)>=per_image:break
                if not any(np.linalg.norm(np.asarray(point)-np.asarray(q))<96 for q in centers):
                    centers.append(point)
        for center in centers:
            pixels,location=bitmap.crop(center,192)
            x0,y0,_,_=location.original_xyxy
            panels=[]
            for kind,detector in (('ORIGINAL',None),('RIDGE',detectors['RIDGE']),('LSD',detectors['LSD'])):
                img,n=overlay(pixels,marks,(x0,y0),detector,kind)
                filename=f'private_{len(scenes)+1:05}_{kind.lower()}.png'
                dest=out_dir/'images'/filename
                dest.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(dest),img):raise OSError('Could not create PRIVATE preview image')
                panels.append({'method':kind,'image':'images/'+filename,
                               'machine_line_candidates_UNKNOWN':n})
            scenes.append({'source_BMP_PRIVATE':row['relative_path'],
                           'source_roi_xyxy_PRIVATE':list(location.original_xyxy),
                           'source_pixels_sha256_PRIVATE':sha256(pixels.tobytes()).hexdigest(),
                           'views':panels})
    cards=[]
    for i,scene in enumerate(scenes,1):
        blocks=[]
        for view in scene['views']:
            blocks.append('<article><h3>'+html.escape(view['method'])+'</h3>'+
                '<img src="'+html.escape(view['image'])+'" alt="Machine-only 2D layer"/>'+
                '<p>2D line hypotheses: '+str(view['machine_line_candidates_UNKNOWN'])+'</p></article>')
        cards.append('<section><h2>ROI '+str(i)+' · '+html.escape(scene['source_BMP_PRIVATE'])+'</h2>'+
                     ''.join(blocks)+'</section>')
    page='''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ANZA-LIRA PRIVATE BMP review / classical algorithms only</title>
<style>body{font:14px/1.5 system-ui;background:#101a2a;color:#e6edf6;margin:0}
main{max-width:1400px;margin:auto;padding:25px}
section{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:12px;margin:35px 0;border-top:1px solid #566}
section h2{grid-column:1/-1;font-size:17px}article{padding:12px;background:#1a293d;border-radius:8px}
img{display:block;width:100%;height:auto}p{margin:7px 0;color:#becbdc}
@media(max-width:600px){section{grid-template-columns:1fr}}</style></head><body><main>
<h1>PRIVATE 2D candidate viewer</h1><p><strong>Настоящие пиксели ваших BMP; только классические Hessian/ridge и OpenCV LSD.</strong>
Оригинальные операторские кресты показываются отдельно от зелёных и сине-оранжевых машинных линий.
Это не предсказание обученной AZUNet, не полный S39/S56, не физическая разметка.
Все кандидаты UNKNOWN, NO native physics GT и NO automatic joining.</p>'''+''.join(cards)+'''
<p>НЕ ПУБЛИКОВАТЬ эти реальные изображения, координаты крестов или приватную галерею в GitHub.</p>
</main></body></html>'''
    (out_dir/'index.html').write_text(page,encoding='utf8')
    meta={'schema':'ANZA_LIRA_CLASSICAL_BMP_PREVIEW_PRIVATE_V1',
          'models':'Hessian ridge / OpenCV LSD classical, NOT trained ANZA',
          'sources':scenes,'native_particle_GT_count':0,
          'physical_auto_joins_authorized':False,'machine_outputs_all_UNKNOWN':True,
          'data_access':'PRIVATE_NEVER_GITHUB'}
    (out_dir/'PRIVATE_MANIFEST.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    (out_dir/'README_PRIVATE.txt').write_text(
        'Open index.html locally. Original crosses != machine proposals. All UNKNOWN. NEVER upload H50Z pixels to public Git.\n')
    return {'image_files_inspected':len({r['source_BMP_PRIVATE'] for r in scenes}),
            'crops_previewed':len(scenes),'models':['Hessian ridge','OpenCV LSD'],
            'native_particle_GT_count':0,'physical_auto_joins_authorized':False,
            'outputs_PRIVATE':True,'source_BMP_aggregate':public['valid_bmps']}


def main(argv=None):
    parser=argparse.ArgumentParser(description='PRIVATE BMP quicklook without labels or trained checkpoints')
    parser.add_argument('--bmp-root',type=Path,required=True)
    parser.add_argument('--crosses',type=Path)
    parser.add_argument('--private-out-dir',type=Path,required=True)
    parser.add_argument('--max-images',type=int,default=4)
    parser.add_argument('--per-image',type=int,default=4)
    parser.add_argument('--roi-mode',choices=['grid','start','both'],default='grid')
    args=parser.parse_args(argv)
    report=run(bmp_root=args.bmp_root,crosses=args.crosses,out_dir=args.private_out_dir,
               max_images=args.max_images,per_image=args.per_image,roi_mode=args.roi_mode)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
