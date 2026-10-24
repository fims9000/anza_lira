"""Produce source-verified PNG tiles from a request CSV; never infer BMP view IDs."""
from __future__ import annotations

import csv
from hashlib import sha256
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2

from .bitmap_tiles import GrayBMP8, sha256_file


def _within(root: Path, name: str) -> Path:
    """Only explicit relative mapped BMP paths inside the source directory."""
    rel=Path(name)
    if rel.is_absolute() or rel.suffix.lower() != '.bmp':
        raise ValueError('expected relative BMP filename')
    root=root.resolve()
    path=(root/rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError('BMP filename must remain within source directory')
    return path


def _read_070_map(mapping_csv: Path | None, root: Path | None) -> dict[tuple[str, int], Path]:
    if mapping_csv is None:
        return {}
    if root is None:
        raise ValueError('visualmetro BMP root required when view map is provided')
    mapping: dict[tuple[str,int],Path]={}
    with Path(mapping_csv).open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not {'frame_id','view_id','file'}.issubset(reader.fieldnames or []):
            raise ValueError('map requires frame_id,view_id,file')
        for row in reader:
            key=(row['frame_id'],int(row['view_id']))
            if key in mapping or key[1] not in {1,2,3}:
                raise ValueError('duplicate or invalid camera view in map')
            mapping[key]=_within(root,row['file'])
    return mapping


def create_small_bundle(requests_csv: Path, output_zip: Path, *,
                        h50z_root: Path | None = None,
                        visualmetro_root: Path | None = None,
                        visualmetro_view_map: Path | None = None,
                        strict: bool = False) -> dict:
    """Extract pixel-exact subimages without resampling or pseudo segmentation.

    Unknown/missing BMPs are recorded unless strict=True. Refuse to put duplicate
    requests or a wrong view map silently into the same output filename.
    """
    viewmap=_read_070_map(visualmetro_view_map,visualmetro_root)
    rows=list(csv.DictReader(Path(requests_csv).open(newline='',encoding='utf-8-sig')))
    if not rows:
        raise ValueError('empty crop request CSV')
    out=Path(output_zip);out.parent.mkdir(parents=True,exist_ok=True)
    records=[];missing=[];seen=set();readers={};source_hashes={}
    for i,r in enumerate(rows):
        source=r['source']
        if source=='H50Z_LABELS_JSON':
            path=None if h50z_root is None else _within(h50z_root,r['source_image'])
        elif source=='VISUALMETRO_SPARSE_3VIEW':
            path=viewmap.get((r['frame_id'],int(r['view_id'])))
        else:
            raise ValueError(f'unknown source {source!r}')
        x0,y0,x1,y1=[int(r[c]) for c in ('x0','y0','x1','y1')]
        w,h=x1-x0,y1-y0
        if w!=h or w<64 or w%16:
            raise ValueError('request tile size invalid')
        requested=(x0,y0,x1,y1)
        if not all(re.fullmatch(r'[A-Za-z0-9_-]+',r[key]) for key in ('frame_id','point_id')):
            raise ValueError('unsafe request identifier')
        if r['view_id'] not in ('','1','2','3'):
            raise ValueError('invalid requested view ID')
        identity=(source,r['frame_id'],r['view_id'],r['point_id'])
        if identity in seen:
            raise ValueError('duplicate source point in request')
        seen.add(identity)
        if path is None or not path.is_file():
            missing.append({'source':source,'frame_id':r['frame_id'],'view_id':r['view_id'],
                            'point_id':r['point_id'],'reason':'NO_EXPLICIT_MATCHED_BMP'})
            continue
        if path not in readers:
            readers[path]=GrayBMP8(path)
            source_hashes[path]=sha256_file(path)
        img,loc=readers[path].crop((float(r['source_x_px']),float(r['source_y_px'])),size=w)
        if loc.original_xyxy!=requested:
            raise ValueError('ROI inconsistent with actual BMP dimensions (wrong file or view?)')
        ok,buf=cv2.imencode('.png',img,[cv2.IMWRITE_PNG_COMPRESSION,6])
        if not ok:
            raise RuntimeError('failed to encode PNG')
        filename=f'tiles/{source.lower()}_{r["frame_id"]}_v{r["view_id"] or "x"}_{r["point_id"]}.png'
        data=buf.tobytes()
        records.append({'zip_name':filename,'payload':data,'record':{
            'zip_name':filename,'source':source,'frame_id':r['frame_id'],'view_id':r['view_id'],
            'point_id':r['point_id'],'label_semantics':r['label_semantics'],
            'crop_xyxy_in_full_bmp':list(loc.original_xyxy),
            'point_xy_local_px':list(loc.center_xy_local),
            'source_file_name':path.name,'source_sha256':source_hashes[path],
            'png_sha256':sha256(data),
        }})
    if strict and missing:
        raise FileNotFoundError(f'{len(missing)} source BMPs not supplied; no partial archive written')
    manifest={
        'schema':'bubble_verified_small_tile_bundle_v1',
        'tile_count':len(records),'missing_count':len(missing),
        'tiles':[r['record'] | {'png_sha256':r['record']['png_sha256'].hexdigest()} for r in records],
        'missing':missing,
        'warning':'Coordinates only: no true full masks, no stereo correction, no film-independent split',
    }
    with ZipFile(out,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for record in records:
            z.writestr(record['zip_name'],record['payload'])
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest
