"""CLI S118: cross-tile identity stitching without exposing private source BMPs.

Manifest, relative paths resolved against its own directory:
{
  "source_image_id": "H50Z_030_01_007.bmp",
  "source_image_sha256": "<64 hexadecimal characters>",
  "tiles": [{"tile_id":"A", "xyxy":[0,0,1200,1200],
             "curves":"A/S115_CURVES.json", "s117":"A/S117_GROUPS.json"}]
}
Run: PYTHONPATH=. python scripts/research/bubble_chamber/run_s118_tile_stitch.py \
  --manifest /private/tiles.json --out /private/output
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import re
from pathlib import Path
from bubble_chamber.s118_tile_stitch import stitch_tiles


def _load(path, key):
    result=json.loads(path.read_text(encoding='utf8'))
    if not isinstance(result,dict) or not isinstance(result.get(key),list):
        raise ValueError(f'Missing {key} in {path}')
    return result


def _paths(points, maximum=450):
    step=max(1,len(points)//maximum)
    return ' '.join(f'{p[0]:.2f},{p[1]:.2f}' for p in (points[::step]+points[-1:] if step>1 else points))


def svg_preview(tiles, result):
    # Vector-only preview: the real source pixels are neither copied nor stored.
    rect=[min(t['xyxy'][0] for t in tiles),min(t['xyxy'][1] for t in tiles),
          max(t['xyxy'][2] for t in tiles),max(t['xyxy'][3] for t in tiles)]
    w,h=rect[2]-rect[0],rect[3]-rect[1]
    group_by={k:g['global_group_id'] for g in result['groups'] for k in g['source_curve_keys']}
    color={g['global_group_id']: '#'+''.join(f'{int(c*255):02x}' for c in colorsys.hsv_to_rgb((i*.61803398875)%1,.69,.76))
           for i,g in enumerate(result['groups'])}
    contents=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{rect[0]} {rect[1]} {w} {h}" width="1400">',
              f'<rect x="{rect[0]}" y="{rect[1]}" width="{w}" height="{h}" fill="#f7f7f4"/>']
    for t in tiles:
        x0,y0,x1,y1=t['xyxy']
        contents.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#8c8c8c" stroke-width="1" stroke-dasharray="6 5"/>')
        for curve in t['curves']:
            cid=curve.get('candidate_id') or curve.get('curve_id') or curve.get('tracklet_id')
            full=[[p[0]+x0,p[1]+y0] for p in curve.get('observed_pixel_path_xy',curve.get('sampled_xy'))]
            points=_paths(full)
            key=t['tile_id']+':'+cid
            contents.append(f'<polyline points="{points}" fill="none" stroke="{color[group_by[key]]}" stroke-width="1.9" stroke-linecap="round"/>')
    for seam in result['seam_candidates']:
        if seam['decision'] in ('AUTO_TINY_SEAM_GAP_NOT_GT','EXPERT_SAME_INFERRED_GAP'):
            line=seam['inferred_connector_full_xy']
            contents.append(f'<path d="M {line[0][0]} {line[0][1]} L {line[1][0]} {line[1][1]}" stroke="#ec710b" stroke-width="2" stroke-dasharray="5 4" fill="none"/>')
    contents.append('</svg>')
    return '\n'.join(contents)+'\n'


def run(manifest:Path,out:Path,decisions:Path|None=None,verify_bmp:Path|None=None):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory is nonempty')
    payload=_load(manifest,'tiles')
    source_id=payload.get('source_image_id')
    sha=payload.get('source_image_sha256')
    if not source_id or not isinstance(sha,str) or re.fullmatch(r'[0-9a-f]{64}',sha) is None:
        raise ValueError('Source BMP identity and valid SHA256 required')
    source_verified=False
    if verify_bmp is not None:
        if verify_bmp.name != source_id:
            raise ValueError('Verification BMP name disagrees with manifest')
        digest=hashlib.sha256()
        with verify_bmp.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                digest.update(chunk)
        if digest.hexdigest()!=sha:
            raise ValueError('Verification BMP SHA256 disagrees with manifest')
        source_verified=True
    tiles=[];provenance=[]
    for entry in payload['tiles']:
        curve_path=(manifest.parent / entry['curves']).resolve()
        s117_path=(manifest.parent / entry['s117']).resolve()
        curves=_load(curve_path,'curves')
        local=_load(s117_path,'groups')
        if 'xyxy' in curves and curves['xyxy']!=entry['xyxy']:
            raise ValueError('Source curve coordinates disagree with tile manifest')
        curve_digest=hashlib.sha256(curve_path.read_bytes()).hexdigest()
        if local.get('provenance',{}).get('source_curves_sha256',curve_digest)!=curve_digest:
            raise ValueError('S117 groups were computed from a different curve file')
        tiles.append({'tile_id':entry['tile_id'],'xyxy':entry['xyxy'],
                      'source_image_id':source_id,'source_image_sha256':sha,
                      'curves':curves['curves'],'s117':local})
        provenance.append({'tile_id':entry['tile_id'],
                           'curves_sha256':curve_digest,
                           's117_sha256':hashlib.sha256(s117_path.read_bytes()).hexdigest()})
    expert=json.loads(decisions.read_text(encoding='utf8')) if decisions else None
    result=stitch_tiles(tiles, decisions=expert)
    result['provenance']={'tile_inputs':provenance,'source_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                          'manual_annotation_used':False,'all_connectors_are_inferred':True,
                          'source_image_sha256_verified':source_verified}
    out.mkdir(parents=True,exist_ok=True)
    (out/'S118_STITCH_NOT_GT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    (out/'S118_VECTOR_REVIEW.svg').write_text(svg_preview(tiles,result),encoding='utf8')
    print(json.dumps(result['report'],ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--decisions',type=Path)
    parser.add_argument('--verify-source-bmp',type=Path,help='Optional private BMP SHA256 check (never copied to output)')
    args=parser.parse_args()
    run(args.manifest,args.out,args.decisions,args.verify_source_bmp)
