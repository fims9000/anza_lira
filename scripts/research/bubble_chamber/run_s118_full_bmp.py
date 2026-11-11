"""S114 -> S115 -> S117 -> S118 source-only, tiled BMP pipeline.

The full source BMP is never committed or copied into output. Geometric results
are provisional and NOT particle IDs. All run artifacts belong in private storage.

Example (3-tile pilot on a marked 1800x1600 region):
 PYTHONPATH=. python scripts/research/bubble_chamber/run_s118_full_bmp.py \
  --bmp /private/H50Z_030_01_007.bmp --out /private/replay \
  --bbox 3600:6000:5400:7600 --max-tiles 3

Increase --max-tiles with --resume to extend coverage, preserving old outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2

from bubble_chamber.s114_source_geometry import reconstruct
from bubble_chamber.s115_junction_graph import associate_observed_junctions
from bubble_chamber.s117_gap_identity import associate_long_gaps
from scripts.research.bubble_chamber.run_s118_tile_stitch import run as run_stitch


def _sha(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def _starts(lo,hi,size,overlap):
    width=hi-lo
    if width<=0:raise ValueError('Empty bbox extent')
    if width<=size:return [lo]
    stride=size-overlap
    origins=list(range(lo,hi-size+1,stride))
    if origins[-1]!=hi-size:origins.append(hi-size)
    return sorted(set(origins))


def _tiles(box,size,overlap):
    if size<64 or size*size>4194304 or overlap<0 or overlap>=size:
        raise ValueError('Tile size must be >=64 and <=4Mi pixels, overlap < size')
    x0,y0,x1,y1=box
    if min(x0,y0)<0 or x1<=x0 or y1<=y0:raise ValueError('Invalid bbox')
    out=[]
    for y in _starts(y0,y1,size,overlap):
        for x in _starts(x0,x1,size,overlap):
            out.append({'tile_id':f'T{len(out):05d}',
                        'xyxy':[x,y,min(x+size,x1),min(y+size,y1)]})
    return out


def _atomic_json(path, data):
    if path.exists():raise FileExistsError('Refusing to overwrite existing stage: '+str(path))
    tmp=path.with_name(path.name+'.partial')
    try:
        tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def run(bmp:Path, out:Path, bbox=None, tile_size=1200, overlap=300,
        max_tiles=None, resume=False):
    bmp=bmp.resolve()
    if max_tiles is not None and max_tiles<1:raise ValueError('max_tiles must be positive')
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError('Output directory exists; pass --resume to continue')
    # Retain original pixel values; do not substitute a contrast-stretched PNG.
    gray=cv2.imread(str(bmp),cv2.IMREAD_GRAYSCALE)
    if gray is None:raise FileNotFoundError(bmp)
    height,width=gray.shape
    box=list(bbox) if bbox is not None else [0,0,width,height]
    if box[2]>width or box[3]>height:raise ValueError('BBox exceeds BMP dimensions')
    grid=_tiles(box,tile_size,overlap)
    digest=_sha(bmp)
    config={'source_image_id':bmp.name,'source_image_sha256':digest,
            'shape_hw':[height,width],'bbox_xyxy':box,'tile_size':tile_size,
            'overlap':overlap,'grid_count':len(grid),'pipeline':'S114_S115_S117_S118_SOURCE_ONLY'}
    out.mkdir(parents=True,exist_ok=True)
    settings=out/'RUN_CONFIG.json'
    if settings.exists():
        previous=json.loads(settings.read_text(encoding='utf8'))
        if previous!=config:raise ValueError('Resuming a different source image, bbox or tile layout is forbidden')
    else:_atomic_json(settings,config)
    selected=grid[:max_tiles] if max_tiles is not None else grid
    tiled_root=out/'tiles';tiled_root.mkdir(exist_ok=True)
    manifest={'source_image_id':bmp.name,'source_image_sha256':digest,'tiles':[]}
    stage_log=[]
    for rec in selected:
        tag=rec['tile_id'];x0,y0,x1,y1=rec['xyxy']
        folder=tiled_root/tag;folder.mkdir(exist_ok=True)
        curves_file=folder/'S115_CURVES.json'
        links_file=folder/'S117_GROUPS.json'
        if curves_file.exists() and links_file.exists():
            original=json.loads(curves_file.read_text())
            local=json.loads(links_file.read_text())
            if original.get('xyxy')!=rec['xyxy'] or local.get('provenance',{}).get('source_curves_sha256')!=_sha(curves_file):
                raise ValueError('Checkpoint identity or provenance mismatch: '+tag)
            status='REUSED_VERIFIED_TILE'
        elif curves_file.exists() or links_file.exists():
            raise ValueError('Incomplete checkpoint detected: '+tag+' (remove partial tile manually after diagnosis)')
        else:
            crop=gray[y0:y1,x0:x1]
            s114=reconstruct(crop)
            s115=associate_observed_junctions(s114['tracks'],s114['skeleton'])
            curves=s115['curves']
            local=associate_long_gaps(curves,crop,response=s114['response'])
            payload={'roi':tag,'xyxy':rec['xyxy'],'curves':curves,
                     'stage':'S115_OBSERVED_GEOMETRY_NOT_GT'}
            _atomic_json(curves_file,payload)
            local['provenance']={'source_curves_sha256':_sha(curves_file),
                                 'source_image_sha256':digest,
                                 'source_tile_xyxy':rec['xyxy'],
                                 'manual_GT_used_for_inference':False}
            _atomic_json(links_file,local)
            status='PROCESSED_FROM_BMP'
        manifest['tiles'].append({**rec, 'curves':f'tiles/{tag}/S115_CURVES.json',
                                      's117':f'tiles/{tag}/S117_GROUPS.json'})
        stage_log.append({'tile_id':tag,'xyxy':rec['xyxy'],'status':status,
                          'observed_curve_count':len(json.loads(curves_file.read_text())['curves']),
                          'local_s117_group_count':len(json.loads(links_file.read_text())['groups'])})
    manifest_path=out/f'S118_MANIFEST_{len(selected):05d}.json'
    if not manifest_path.exists():_atomic_json(manifest_path,manifest)
    elif json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Existing manifest conflicts with this run')
    stitched_dir=out/f'S118_STITCH_{len(selected):05d}'
    if stitched_dir.exists() and (stitched_dir/'S118_STITCH_NOT_GT.json').is_file():
        stitch=json.loads((stitched_dir/'S118_STITCH_NOT_GT.json').read_text())
        if stitch['provenance']['source_manifest_sha256']!=_sha(manifest_path):
            raise ValueError('Stitch checkpoint manifest hash differs')
    else:
        if stitched_dir.exists() and any(stitched_dir.iterdir()):
            raise ValueError('Incomplete final stitch checkpoint; preserve and inspect before retry')
        stitch=run_stitch(manifest_path,stitched_dir,verify_bmp=bmp)
    report={'source_image_id':bmp.name,'bbox_xyxy':box,
            'processed_tile_count':len(selected),'planned_tile_count':len(grid),
            'processed_full_requested_bbox':len(selected)==len(grid),
            'input_bmp_SHA256_verified':True,
            'source_only_not_physical_GT':True,
            'stitch_report':stitch['report'],'tiles':stage_log}
    report_path=out/f'S118_RUN_REPORT_{len(selected):05d}.json'
    if not report_path.exists():_atomic_json(report_path,report)
    print(json.dumps({k:v for k,v in report.items() if k!='tiles'},ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bmp',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bbox',help='x0:y0:x1:y1, original BMP pixels')
    parser.add_argument('--tile-size',type=int,default=1200)
    parser.add_argument('--overlap',type=int,default=300)
    parser.add_argument('--max-tiles',type=int)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    box=tuple(map(int,args.bbox.split(':'))) if args.bbox else None
    run(args.bmp,args.out,box,args.tile_size,args.overlap,args.max_tiles,args.resume)
