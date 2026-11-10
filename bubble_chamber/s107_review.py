"""S107 source-locked independent review and partial-annotation evaluation.

IMPORTANT: filenames and sparse start/fiducial marks are not dense labels. An
unannotated pixel is UNKNOWN, not background. Every annotated stroke comes from
an actual reviewer and is never silently generated from S105 hypotheses.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import math
import shutil

import cv2
import numpy as np

try:
    # Canonical source reader on the existing ANZA-LIRA GitHub S101 branch.
    from .bitmap_tiles import GrayBMP8 as H50ZBitmap, sha256_file as digest_file
except ImportError:
    # Standalone S105 experiment package has the equivalent verified adapter.
    from .s105_bmp_adapter import H50ZBitmap, digest_file

SCHEMA = 'ANZA_S107_BLIND_H50Z_POLYLINE_REVIEW_V1'
EXPORT_SCHEMA = 'ANZA_S107_REVIEWED_POLYLINES_V1'
SOURCE_SHA = {
    'H50Z_030_01_005.bmp': 'e78a90e72b80e4b166be2c0b2a53a9e70558c265c2a932fb1efc52783d5352e6',
    'H50Z_030_01_006.bmp': 'd1251c30b460e31f9b8b3e8fb4b9849b920d5210dcf94eb133fd814b1d8cbea3',
    'labels.json': 'df3fd9034d44ea4c27e79c50b5bc3c2de873071403740ab2c8d9f62004165ad7',
}
KNOWN_CASES = (
    ('005_start3', 'H50Z_030_01_005.bmp', '005_start3_1200', 'OPERATOR_START_ROI_NOT_GT'),
    ('005_fiducial', 'H50Z_030_01_005.bmp', '005_f4_control_640', 'FIDUCIAL_CONTROL_NOT_NEGATIVE_GT'),
    ('006_start2', 'H50Z_030_01_006.bmp', '006_start2_1200', 'OPERATOR_START_ROI_NOT_GT'),
    ('006_fiducial', 'H50Z_030_01_006.bmp', '006_f4_control_640', 'FIDUCIAL_CONTROL_NOT_NEGATIVE_GT'),
)
KNOWN_RECTS = {
    '005_start3': (4108,15354,5308,16554),
    '005_fiducial': (2966,4664,3606,5304),
    '006_start2': (3733,8407,4933,9607),
    '006_fiducial': (2789,4052,3429,4692),
}
TILE = 1024


def _write_png(path: Path, array: np.ndarray) -> str:
    if not cv2.imwrite(str(path), array, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise OSError(f'Unable to save {path}')
    return digest_file(path)


def _gray_rect(bmp: H50ZBitmap, rect: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    if not (0 <= x0 < x1 <= bmp.width and 0 <= y0 < y1 <= bmp.height):
        raise ValueError('Out-of-frame crop')
    if getattr(bmp,'bit_depth',8) != 8:
        raise ValueError('S107 requires original 8-bit H50Z, not preannotated color composites')
    reader = bmp._data if hasattr(bmp, '_data') else bmp._raw
    return np.ascontiguousarray(reader[y0:y1, x0:x1])


def _blind_rects(image_sha: str, w: int, h: int, seed: int, tile_size: int = TILE):
    """One deterministic source-only crop per each of four disjoint quadrants.

    Never use an S105 score or operator start point to select blind tiles.
    """
    if w < 2*tile_size or h < 2*tile_size:
        raise ValueError('Source too small for four nonoverlapping blind strata')
    digest = sha256(f'S107_BLIND_GRID|{seed}|{image_sha}|{tile_size}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    for row in range(2):
        for col in range(2):
            left = col*(w//2)
            top = row*(h//2)
            right = (col+1)*(w//2) if col == 0 else w
            bottom = (row+1)*(h//2) if row == 0 else h
            x0 = int(rng.integers(left, right-tile_size+1))
            y0 = int(rng.integers(top, bottom-tile_size+1))
            yield f'blind_{row+1}{col+1}', (x0, y0, x0+tile_size, y0+tile_size)


def _manifest_identity(m: dict) -> str:
    # Hash manifest content excluding its self-reference to bind every image
    # and source coordinate used in an annotation export.
    base = {k: v for k, v in m.items() if k != 'manifest_sha256'}
    return sha256(json.dumps(base, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':')).encode()).hexdigest()


def build_pack(source_dir: Path, s105_root: Path | None, output: Path, *, seed: int = 20261110,
               blind_per_image: int = 4) -> dict:
    """Create original-resolution PNG review pack with optional hidden S105 layers."""
    source_dir, output = map(Path, (source_dir, output))
    s105_root = Path(s105_root) if s105_root is not None else None
    if output.exists():
        raise FileExistsError(f'Will not overwrite {output}')
    if blind_per_image not in (0, 4):
        raise ValueError('S107 frozen blind scheme uses 0 or 4 strata per image')
    readers = {}
    for filename, digest in SOURCE_SHA.items():
        path = source_dir / filename
        if not path.is_file() or digest_file(path) != digest:
            raise ValueError(f'Source missing or SHA-256 mismatch: {filename}')
        if filename.endswith('.bmp'):
            readers[filename] = H50ZBitmap(path)
    labels = json.loads((source_dir/'labels.json').read_text(encoding='utf-8'))
    for name, bmp in readers.items():
        row = labels[name]
        if (bmp.width, bmp.height) != (row['original_width'], row['original_height']):
            raise ValueError(f'BMP dimensions not equal original labels: {name}')
    tiles = []
    cases = []
    for short, photo, sub, policy in KNOWN_CASES:
        rect=KNOWN_RECTS[short]
        baseline=None
        if s105_root is not None and (s105_root/sub/'report.json').is_file():
            report=json.loads((s105_root/sub/'report.json').read_text(encoding='utf-8'))
            if (report['source_name'] != photo or report['source_sha256'] != SOURCE_SHA[photo]
                or tuple(report['crop_xyxy_source_px']) != rect):
                raise ValueError(f'Existing S105 run source/crop mismatch: {sub}')
            baseline=sub
        cases.append((short, photo, rect, policy, baseline))
    for photo, bmp in readers.items():
        if blind_per_image:
            for short, rect in _blind_rects(SOURCE_SHA[photo], bmp.width, bmp.height, seed):
                cases.append((f'{photo[-7:-4]}_{short}', photo, rect,
                              'BLIND_FIXED_GRID_NEVER_SCORED_FOR_SELECTION', None))
    output.mkdir(parents=True, mode=0o700)
    image_dir = output/'images'
    image_dir.mkdir(mode=0o700)
    for name, photo, rect, policy, baseline in cases:
        raw = _gray_rect(readers[photo], rect)
        normal_name = f'{name}_source.png'
        contrast_name = f'{name}_contrast_DISPLAY_ONLY.png'
        raw_digest = _write_png(image_dir/normal_name, raw)
        contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(raw)
        contrast_digest = _write_png(image_dir/contrast_name, contrast)
        item = {'tile_id': name, 'source_name': photo,
                'source_sha256': SOURCE_SHA[photo],
                'source_dimensions_xy': [readers[photo].width, readers[photo].height],
                'source_rect_xyxy': list(rect), 'width': int(raw.shape[1]),
                'height': int(raw.shape[0]), 'selection': policy,
                'original_png': f'images/{normal_name}',
                'original_png_sha256': raw_digest,
                'contrast_png': f'images/{contrast_name}',
                'contrast_png_sha256': contrast_digest,
                'contrast_is_display_only': True,
                's105_display_png': None, 's105_mask_png': None,
                'native_gt_exists': False}
        if baseline:
            src = s105_root/baseline
            for source, key, dest in [
                ('13_S105_LONG_RANGE_CONTEXT_SAME_DISPLAY_ID.png','s105_display_png',f'{name}_s105_display.png'),
                ('14_S105_CONTEXT_SOURCE_ONLY_MASK.png','s105_mask_png',f'{name}_s105_mask.png'),
            ]:
                arr = cv2.imread(str(src/source), cv2.IMREAD_UNCHANGED)
                if arr is None or arr.shape[:2] != raw.shape:
                    raise ValueError(f'S105 preview is missing or different size: {src/source}')
                digest = _write_png(image_dir/dest, arr)
                item[key] = f'images/{dest}'
                item[key + '_sha256'] = digest
            item['s105_is_machine_only_not_expert_label'] = True
        tiles.append(item)
    data = {'schema': SCHEMA, 'version': 1,
            'selection_seed': seed, 'stratification': 'deterministic 2-by-2, four blind ROIs per frame',
            'source_images': [{'name': name, 'sha256': SOURCE_SHA[name],
                               'width': bmp.width, 'height': bmp.height,
                               'film_run_id': 'H50Z_030_01 (same run, not independent)'}
                              for name, bmp in readers.items()],
            'labels_sha256': SOURCE_SHA['labels.json'],
            'tiles': tiles, 'unmarked_pixels_mean': 'UNKNOWN',
            's105_overlays_are_gt': False, 'physical_particle_identity_verified': False,
            'blind_selection_has_seen_s105': False}
    data['manifest_sha256'] = _manifest_identity(data)
    (output/'manifest.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return data


def verify_pack(pack: Path) -> dict:
    pack = Path(pack)
    m = json.loads((pack/'manifest.json').read_text(encoding='utf-8'))
    if m.get('schema') != SCHEMA or m.get('manifest_sha256') != _manifest_identity(m):
        raise ValueError('S107 manifest identity failure')
    for tile in m['tiles']:
        w,h = tile['width'],tile['height']
        x0,y0,x1,y1 = tile['source_rect_xyxy']
        if (x1-x0,y1-y0)!=(w,h) or w < 128 or h < 128:
            raise ValueError('Bad original coordinate contract')
        for key in ('original_png','contrast_png','s105_display_png','s105_mask_png'):
            relative=tile.get(key)
            if relative is None:
                continue
            path = pack/relative
            if not path.is_file() or digest_file(path) != tile[key + '_sha256']:
                raise ValueError(f'Missing or modified review pixel evidence: {relative}')
            img=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
            if img is None or img.shape[:2]!=(h,w):
                raise ValueError(f'Pixel dimension mismatch: {relative}')
    return m


def _finite_coords(row, rect):
    coords=row.get('points_source_xy')
    if not isinstance(coords,list) or not 2<=len(coords)<=10000:
        raise ValueError('Each stroke needs between 2 and 10,000 points')
    x0,y0,x1,y1=rect
    a=np.asarray(coords,dtype=float)
    if a.shape!=(len(coords),2) or not np.isfinite(a).all():
        raise ValueError('Malformed / nonfinite source points')
    if not ((a[:,0]>=x0)&(a[:,0]<x1)&(a[:,1]>=y0)&(a[:,1]<y1)).all():
        raise ValueError('Stroke coordinates outside source crop')
    return a


def validate_annotation(export: dict, manifest: dict) -> dict:
    if export.get('schema') != EXPORT_SCHEMA or export.get('manifest_sha256')!=manifest['manifest_sha256']:
        raise ValueError('S107 annotation version or source-manifest mismatch')
    lookup={tile['tile_id']:tile for tile in manifest['tiles']}
    if len(lookup)!=len(manifest['tiles']):
        raise ValueError('Duplicate review tile IDs')
    rows=export.get('strokes')
    if not isinstance(rows,list) or len(rows)>50000:
        raise ValueError('Missing/oversized strokes array')
    seen=set()
    count=defaultdict(int)
    for row in rows:
        if not isinstance(row,dict):
            raise ValueError('Malformed stroke')
        tid=row.get('tile_id')
        if tid not in lookup:
            raise ValueError('Stroke tile not part of source-bound pack')
        rid=row.get('stroke_id')
        if not isinstance(rid,str) or not rid or (tid,rid) in seen:
            raise ValueError('Duplicate/missing stroke ID')
        seen.add((tid,rid))
        kind=row.get('kind')
        if kind not in ('track','artifact','unknown'):
            raise ValueError('Invalid kind')
        if row.get('certainty') not in ('clear','uncertain'):
            raise ValueError('Invalid certainty')
        instance=row.get('visual_instance_id')
        if kind=='track' and (not isinstance(instance,str) or not instance.strip()):
            raise ValueError('Track strokes require a visual instance ID')
        if kind!='track' and instance not in (None,''):
            raise ValueError('Only tracks can have a visual instance ID')
        if kind=='artifact' and row.get('artifact_type') not in (
                'scratch','v_mark','fiducial','hair','dot','other'):
            raise ValueError('Artifact requires type')
        a = _finite_coords(row, lookup[tid]['source_rect_xyxy'])
        # A clear human centerline is only traced through observed pixels. A
        # large unconstrained straight hop could fabricate an invisible path.
        # Separate visible fragments should use multiple strokes with one ID.
        if kind == 'track' and row.get('certainty') == 'clear':
            lengths=np.linalg.norm(np.diff(a,axis=0),axis=1)
            if np.any(lengths > 32.01):
                raise ValueError('Track stroke vertex gap >32 px: split visible segments or add observed vertices')
        count[kind]+=1
    return {'strokes':len(rows),'count_by_kind':dict(count),
            'reviewer_confirmed':bool(export.get('reviewer_confirmed')),
            'independent_physical_events':1,
            'source_is_film_run_disjoint_test':False}


def _draw_polyline(tile: dict, row: dict, *, radius: int=0) -> np.ndarray:
    x0,y0,_,_=tile['source_rect_xyxy']
    pts = np.asarray(row['points_source_xy'],dtype=np.float32)
    pts -= np.array([x0,y0],np.float32)
    p=np.rint(pts).astype(np.int32)
    result=np.zeros((tile['height'],tile['width']),np.uint8)
    cv2.polylines(result,[p],False,1,thickness=1,lineType=cv2.LINE_8)
    if radius:
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
        result=cv2.dilate(result,kernel)
    return result.astype(bool)


def _polyline_sampling(tile: dict,row: dict) -> np.ndarray:
    x0,y0,_,_=tile['source_rect_xyxy']
    coords=np.asarray(row['points_source_xy'],dtype=float)-np.array([x0,y0])
    segments=[]
    for a,b in zip(coords[:-1],coords[1:]):
        n=max(2,int(math.ceil(np.linalg.norm(b-a)))+1)
        segments.append(np.linspace(a,b,n,endpoint=True)[:-1])
    segments.append(coords[-1:])
    pts=np.vstack(segments)
    q=np.rint(pts).astype(np.int32)
    q[:,0]=np.clip(q[:,0],0,tile["width"]-1)
    q[:,1]=np.clip(q[:,1],0,tile["height"]-1)
    return q


def evaluate(export: dict, pack: Path, out: Path, *, tolerance_px: int=3) -> dict:
    manifest=verify_pack(pack)
    summary=validate_annotation(export,manifest)
    if not 1<=tolerance_px<=12:
        raise ValueError('Unsupported matching tolerance')
    out=Path(out)
    if out.exists():
        raise FileExistsError('Will not overwrite any prior evaluation')
    out.mkdir(parents=True,exist_ok=False)
    by_tile=defaultdict(list)
    for row in export['strokes']:
        by_tile[row['tile_id']].append(row)
    evaluated=[]
    total_positive=total_covered=0
    total_artifact=total_artifact_hits=0
    export_label_dir=out/'sparse_labels'
    export_label_dir.mkdir()
    for tile in manifest['tiles']:
        tid=tile['tile_id'];rows=by_tile[tid]
        labels=np.zeros((tile['height'],tile['width']),np.uint8)  # 0=UNKNOWN
        positives=np.zeros_like(labels,bool)
        negatives=np.zeros_like(labels,bool)
        uncertain=np.zeros_like(labels,bool)
        for row in rows:
            region=_draw_polyline(tile,row,radius=2)
            if row['certainty']=='uncertain' or row['kind']=='unknown':
                uncertain|=region
            elif row['kind']=='track':
                positives|=region
            else:
                negatives|=region
        labels[positives]=1
        labels[negatives]=2
        labels[positives&negatives]=3  # overlap unknown for training/eval
        labels[uncertain]=3
        assert cv2.imwrite(str(export_label_dir/f'{tid}_partial_stroke_targets.png'),labels)
        baseline=tile.get('s105_mask_png')
        tile_stat={'tile_id':tid,'source_name':tile['source_name'],
                   'selection':tile['selection'],
                   's105_preview_exposed_during_review':bool(export.get('tile_state',{}).get(tid,{}).get('s105_preview_exposed',False)),
                   'clear_track_strokes':sum(x['kind']=='track' and x['certainty']=='clear' for x in rows),
                   'clear_artifact_strokes':sum(x['kind']=='artifact' and x['certainty']=='clear' for x in rows),
                   'unlabelled_pixels_not_assumed_background':True,
                   'train_target_counts':{str(i):int(np.sum(labels==i)) for i in range(4)},
                   'metric_status':'NO_BASELINE_IMAGE_OR_NO_CLEAR_STROKES'}
        if baseline:
            pred=cv2.imread(str(Path(pack)/baseline),cv2.IMREAD_GRAYSCALE)
            if pred is None or pred.shape!=labels.shape:
                raise ValueError('Baseline mismatched image in source pack')
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (2*tolerance_px+1,2*tolerance_px+1))
            near=cv2.dilate((pred>0).astype(np.uint8),kernel)>0
            pos_samples=[];art_samples=[];tracks=defaultdict(list)
            for row in rows:
                if row['certainty']!='clear' or row['kind']=='unknown':
                    continue
                pts=_polyline_sampling(tile,row)
                if row['kind']=='track':
                    pos_samples.append(near[pts[:,1],pts[:,0]])
                    tracks[row['visual_instance_id']].append(pts)
                elif row['kind']=='artifact':
                    art_samples.append(near[pts[:,1],pts[:,0]])
            pnum=sum(len(p) for p in pos_samples)
            anum=sum(len(a) for a in art_samples)
            covered=sum(int(p.sum()) for p in pos_samples)
            hits=sum(int(a.sum()) for a in art_samples)
            total_positive+=pnum; total_covered+=covered
            total_artifact+=anum; total_artifact_hits+=hits
            tr=[]
            for ident,parts in sorted(tracks.items()):
                vals=[near[p[:,1],p[:,0]] for p in parts]
                longest_uncovered=0
                for arr in vals:
                    runs=np.diff(np.r_[0,(~arr).astype(np.int8),0])
                    begins=np.flatnonzero(runs==1)
                    ends=np.flatnonzero(runs==-1)
                    longest_uncovered=max(longest_uncovered,int(max((e-b for b,e in zip(begins,ends)),default=0)))
                tr.append({'visual_instance_id':ident,
                           'annotated_observed_strokes':len(parts),
                           'sampled_centerline_px':sum(len(p) for p in parts),
                           'covered_fraction_with_tolerance':round(sum(int(v.sum()) for v in vals)/sum(len(v) for v in vals),4),
                           'longest_uncovered_sample_run_px':longest_uncovered,
                           'not_verified_physical_particle_identity':True})
            tile_stat.update({'metric_status':'PARTIALLY_ANNOTATED_VISUAL_CENTERLINES',
                  'annotated_track_coverage':round(covered/pnum,4) if pnum else None,
                  'artifact_line_candidate_overlap':round(hits/anum,4) if anum else None,
                  'track_points_evaluated':pnum, 'artifact_points_evaluated':anum,
                  'per_visual_instance':tr})
        evaluated.append(tile_stat)
    report={'schema':'ANZA_S107_SOURCE_LOCKED_PARTIAL_STROKE_EVALUATION_V1',
            'manifest_sha256':manifest['manifest_sha256'],
            'reviewer_confirmed':summary['reviewer_confirmed'],
            'model_preview_exposed_on_any_tile':any(x['s105_preview_exposed_during_review'] for x in evaluated),
            'independent_blind_evaluation_supported':bool(summary['reviewer_confirmed']) and not any(x['s105_preview_exposed_during_review'] for x in evaluated),
            'source_frames':len(manifest['source_images']),
            'independent_physical_events':1,
            'verified_same_particle_identity':False,
            'global_precision_or_recall_supported':False,
            'unlabelled_pixels_mean':'UNKNOWN not negative',
            's105_centerline_coverage_on_explicitly_marked_strokes':round(total_covered/total_positive,4) if total_positive else None,
            's105_candidate_overlap_on_explicitly_marked_artifacts':round(total_artifact_hits/total_artifact,4) if total_artifact else None,
            'evaluated_positive_stroke_pixels':total_positive,
            'evaluated_artifact_stroke_pixels':total_artifact,
            'tiles':evaluated}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report