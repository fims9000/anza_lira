#!/usr/bin/env python3
"""Full-resolution, ALL-TILE BMP inspection. Baseline LSD, NOT trained ANZA/S112.

Run on permitted original 8-bit H50Z BMPs. All outputs contain PRIVATE pixels
and must stay outside the Git repository. Endpoints A/B are geometrical only.
"""
from __future__ import annotations
import argparse,hashlib,html,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.bitmap_tiles import GrayBMP8,sha256_file


def tile_starts(length:int,size:int)->list[int]:
    if length<size:raise ValueError('Source is smaller than tile')
    out=list(range(0,length-size+1,size))
    if not out or out[-1]!=length-size:out.append(length-size)
    if out[0]!=0 or out[-1]+size!=length:raise AssertionError('Incomplete axis coverage')
    return out


def load_keypoints(path:Path|None,filename:str,width:int,height:int)->list[dict]:
    if path is None:return []
    j=json.loads(path.read_text(encoding='utf-8-sig'))
    record=j.get(filename)
    if record is None:return []
    if [int(record['original_width']),int(record['original_height'])]!=[width,height]:
        raise ValueError('Operator labels dimensions do not match this BMP')
    marks=record['keypoints']
    if len(marks)!=record['keypoints_count']:raise ValueError('Malformed keypoint count')
    for k in marks:
        if k['label'] not in ({'start'}|{f'f{i}' for i in range(1,9)}):
            raise ValueError('Unexpected operator keypoint name')
        if not(0<=float(k['x_px'])<width and 0<=float(k['y_px'])<height):
            raise ValueError('Out-of-bounds operator mark')
    return marks


def inspect_tile(gray:np.ndarray,marks:list[dict],original_xyxy,lsd=None,max_lines:int=24,
                 preview_side:int=200)->tuple[np.ndarray,int]:
    if gray.dtype!=np.uint8 or gray.ndim!=2:raise ValueError('Expect true uint8 grayscale ROI')
    if lsd is None:lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    found=lsd.detect(gray)[0]
    segments=[] if found is None else found.reshape(-1,4).tolist()
    segments.sort(key=lambda a:-float(np.hypot(a[0]-a[2],a[1]-a[3])))
    h,w=gray.shape;over=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    for idx,(x0,y0,x1,y1) in enumerate(segments[:max_lines]):
        p0=(int(round(x0)),int(round(y0)));p1=(int(round(x1)),int(round(y1)))
        cv2.line(over,p0,p1,(215,175,25),2,cv2.LINE_AA)
        if idx<4:
            cv2.circle(over,p0,5,(45,210,50),-1,cv2.LINE_AA)
            cv2.rectangle(over,(p1[0]-4,p1[1]-4),(p1[0]+4,p1[1]+4),(45,40,220),-1)
    left,top,right,bottom=original_xyxy
    for mark in marks:
        x=float(mark['x_px'])-left;y=float(mark['y_px'])-top
        if not(0<=x<w and 0<=y<h):continue
        at=(int(round(x)),int(round(y)))
        if mark['label'].startswith('f'):
            cv2.line(over,(at[0]-11,at[1]-11),(at[0]+11,at[1]+11),(220,40,230),2,cv2.LINE_AA)
            cv2.line(over,(at[0]-11,at[1]+11),(at[0]+11,at[1]-11),(220,40,230),2,cv2.LINE_AA)
            cv2.putText(over,mark['label'],(at[0]+13,at[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,35,230),1)
        else:
            cv2.circle(over,at,10,(240,220,30),2,cv2.LINE_AA)
            cv2.putText(over,'operator start',(at[0]+13,at[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,220,30),1)
    small=cv2.resize(over,(preview_side,preview_side),interpolation=cv2.INTER_AREA)
    return small,len(segments)


def run(bmp_path:Path,out_dir:Path,tile_size:int=1024,labels_json:Path|None=None,
        preview_side:int=180,max_lines:int=24,save_tiles:bool=False)->dict:
    if out_dir.resolve()==ROOT.resolve() or ROOT.resolve() in out_dir.resolve().parents:
        raise ValueError('PRIVATE previews must not be saved inside the Git repository')
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError('Use an empty new private output directory; never overwrite earlier evidence')
    out_dir.mkdir(parents=True,exist_ok=True)
    bmp=GrayBMP8(bmp_path)
    if tile_size<64 or tile_size%16 or tile_size>min(bmp.width,bmp.height):raise ValueError('Tile size invalid')
    marks=load_keypoints(labels_json,bmp_path.name,bmp.width,bmp.height)
    xx=tile_starts(bmp.width,tile_size);yy=tile_starts(bmp.height,tile_size)
    lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    margin=26;cell=preview_side+margin
    overview=np.full((len(yy)*cell,len(xx)*cell,3),245,np.uint8)
    cases=[]
    for row,y0 in enumerate(yy):
        for col,x0 in enumerate(xx):
            gray,loc=bmp.crop((x0+tile_size/2,y0+tile_size/2),size=tile_size)
            preview,nlines=inspect_tile(gray,marks,loc.original_xyxy,lsd,max_lines,preview_side)
            py=row*cell;px=col*cell
            overview[py:py+preview_side,px:px+preview_side]=preview
            cv2.putText(overview,f'R{row:02d} C{col:02d}',(px+2,py+cell-7),cv2.FONT_HERSHEY_SIMPLEX,.39,(0,0,0),1,cv2.LINE_AA)
            if save_tiles:
                full,_=inspect_tile(gray,marks,loc.original_xyxy,lsd,max_lines,preview_side=tile_size)
                cv2.imwrite(str(out_dir/f'R{row:02d}_C{col:02d}_ANNOTATION.png'),full)
            cases.append({'row':row,'column':col,'crop_original_xyxy':list(loc.original_xyxy),
                          'LSD_linelet_candidates_not_physical_tracks':nlines,
                          'expert_crosses_in_tile':sum(m['label'].startswith('f') and
                              x0<=m['x_px']<x0+tile_size and y0<=m['y_px']<y0+tile_size for m in marks)})
    cv2.imwrite(str(out_dir/'ALL_TILES_LSD_BASELINE_NOT_ANZA.png'),overview)
    report={'schema':'S113_FULL_FRAME_TILE_BASELINE_NOT_TRAINED_ANZA',
            'source_filename_private':bmp_path.name,'source_sha256':sha256_file(bmp_path),
            'frame_size_wh':[bmp.width,bmp.height], 'tile_size':tile_size,
            'grid_shape_cols_rows':[len(xx),len(yy)],'total_tiles':len(cases),
            'full_frame_pixel_coverage':True,
            'AZUNet_or_S112_used_on_full_frame':False,
            'track_particle_direction_inferred':False,'physical_track_ground_truth_available':False,
            'crosses_from_operator_labels_only':True,'cross_label_count':sum(m['label'].startswith('f') for m in marks),
            'model':'OpenCV_LSD_classical_linelet_baseline',
            'privacy':'This manifest and its images contain private image information; keep outside public repository',
            'tiles':cases}
    (out_dir/'audit_PRIVATE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out_dir/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Private H50Z all-tile QA</title>'
       '<body style="font:16px system-ui;background:#181a20;color:white">'
       '<h2>Вся фотография / каждая ячейка — LSD, НЕ обученная ANZA и НЕ физический GT</h2>'
       '<p>Зелёный круг A / красный квадрат B: концы геометрического отрезка, не направление частицы. '
       'Розовые f1..f8: только реально известные операторские метки.</p>'
       '<img src="ALL_TILES_LSD_BASELINE_NOT_ANZA.png" style="max-width:100%"></body>',encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bmp',type=Path,required=True)
    p.add_argument('--private-out',type=Path,required=True)
    p.add_argument('--crosses',type=Path)
    p.add_argument('--tile-size',type=int,default=1024)
    p.add_argument('--preview-side',type=int,default=180)
    p.add_argument('--max-lines',type=int,default=24)
    p.add_argument('--save-tiles',action='store_true')
    a=p.parse_args()
    print(json.dumps(run(a.bmp,a.private_out,a.tile_size,a.crosses,a.preview_side,a.max_lines,a.save_tiles),ensure_ascii=False,indent=2))