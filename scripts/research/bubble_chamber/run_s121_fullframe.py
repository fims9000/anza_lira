"""S121 full-frame source-only continuation after S118 stitching."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import cv2
from bubble_chamber.s121_fullframe_continuation import associate_fullframe


def run(bmp:Path,run_dir:Path,out:Path):
    config=json.load(open(run_dir/'RUN_CONFIG.json'))
    gray=cv2.imread(str(bmp),cv2.IMREAD_GRAYSCALE)
    if gray is None:raise FileNotFoundError(bmp)
    h=hashlib.sha256()
    with bmp.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=config['source_image_sha256'] or list(gray.shape)!=config['shape_hw']:
        raise ValueError('S121 source BMP SHA/size mismatch')
    n=config['grid_count']
    manifest_path=run_dir/f'S118_MANIFEST_{n:05d}.json'
    manifest=json.load(open(manifest_path))
    s118=json.load(open(run_dir/f'S118_STITCH_{n:05d}'/'S118_STITCH_NOT_GT.json'))
    if len(manifest['tiles'])!=n or len(s118['tiles'])!=n:
        raise ValueError('Full-frame S118 coverage is incomplete')
    tiles=[];local={}
    for t in manifest['tiles']:
        tid=t['tile_id'];p=json.load(open(run_dir/t['curves']));q=json.load(open(run_dir/t['s117']))
        tiles.append({'tile_id':tid,'xyxy':t['xyxy'],'curves':p['curves']});local[tid]=q
    result=associate_fullframe(tiles,local,s118['groups'],gray)
    result['provenance']={'source_sha256':h.hexdigest(),
       's118_stitch_sha256':hashlib.sha256((run_dir/f'S118_STITCH_{n:05d}'/'S118_STITCH_NOT_GT.json').read_bytes()).hexdigest(),
       'source_manual_annotations_used':False,'is_GT':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Refusing to overwrite S121 output')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result['report']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bmp',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.bmp,a.run_dir,a.out),indent=2))
