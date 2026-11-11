"""Build a reproducible, photo-first, private 20-ROI review pack for S118/S120/S121.

Selection is systematic 4 columns x 5 rows, with seeded small jitter, over
THE COMPLETE source frame. No manual labels are read or used to select regions.
Only the derived small review JPEGs belong in private review storage by default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def grid20(width: int, height: int, roi: int = 1024, seed: int = 20261111):
    """Cover entire source photo in systematic rows and columns, not cherry-picked detections."""
    if roi < 64 or roi > min(width, height):
        raise ValueError('ROI must fit source frame, at least 64 pixels')
    rng = np.random.default_rng(seed)
    regions = []
    for row in range(5):
        for col in range(4):
            cx = (col + .5) * width / 4.
            cy = (row + .5) * height / 5.
            jitter_x = rng.uniform(-.10, .10) * width / 4.
            jitter_y = rng.uniform(-.10, .10) * height / 5.
            x = int(np.clip(round(cx + jitter_x - roi/2), 0, width-roi))
            y = int(np.clip(round(cy + jitter_y - roi/2), 0, height-roi))
            regions.append((f'R{row*4+col+1:02d}', x, y, x+roi, y+roi))
    assert len(regions) == 20
    return regions


def _dotted(image, pts, color=(0, 155, 255), thickness=2, dash=9, skip=7):
    # Draw sampled cubic connector as *inferred*, never source-observed.
    if len(pts) < 2:
        return
    for i in range(0, len(pts)-1, dash+skip):
        cv2.polylines(image, [pts[i:min(i+dash+1,len(pts))]], False, color,
                      thickness, cv2.LINE_AA)


def create_pack(bmp: Path, run_dir: Path, out: Path,
                scores: Path | None = None, s121: Path | None = None,
                roi_size: int = 1024, seed: int = 20261111):
    bmp = bmp.resolve()
    conf = json.loads((run_dir/'RUN_CONFIG.json').read_text(encoding='utf8'))
    raw = cv2.imread(str(bmp), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(bmp)
    digest = _sha(bmp)
    if digest != conf['source_image_sha256'] or list(raw.shape) != conf['shape_hw']:
        raise ValueError('BMP differs from S118 frozen source (SHA or dimensions)')
    if conf['bbox_xyxy'] != [0,0,raw.shape[1],raw.shape[0]]:
        raise ValueError('This protocol requires a full-frame S118 run, not a selected ROI')
    count = conf['grid_count']
    manifest = json.loads((run_dir/f'S118_MANIFEST_{count:05d}.json').read_text())
    if len(manifest['tiles']) != count:
        raise ValueError('Incomplete tile manifest')
    groups_path = run_dir/f'S118_STITCH_{count:05d}'/'S118_STITCH_NOT_GT.json'
    groups = json.loads(groups_path.read_text())['groups']
    lookup = {name:g['global_group_id'] for g in groups for name in g['source_curve_keys']}
    selected = None
    if scores:
        status = json.loads(scores.read_text(encoding='utf8'))
        selected = status.get('groups',status)
        if set(selected) != {g['global_group_id'] for g in groups}:
            raise ValueError('S120 review status does not match S118 groups')
    bridges=[]
    if s121:
        assoc=json.loads(s121.read_text(encoding='utf8'))
        if assoc['provenance']['source_sha256'] != digest:
            raise ValueError('S121 proposals refer to a different original BMP')
        bridges=[x['inferred_xy'] for x in assoc['proposals']
                 if x.get('decision') == 'AUTO_PROVISIONAL_CONTINUATION_NOT_GT']
    entries=[]
    for item in manifest['tiles']:
        x0,y0,x1,y1=item['xyxy']
        payload=json.loads((run_dir/item['curves']).read_text(encoding='utf8'))
        entries.append((x0,y0,x1,y1,item['tile_id'],payload['curves']))
    out.mkdir(parents=True,exist_ok=True)
    review_rows=[]
    for rid,x0,y0,x1,y1 in grid20(raw.shape[1],raw.shape[0],roi_size,seed):
        orig=cv2.cvtColor(raw[y0:y1,x0:x1],cv2.COLOR_GRAY2BGR)
        overlay=orig.copy()
        drawn=0
        uncertain=0
        for tx0,ty0,tx1,ty1,tid,curves in entries:
            if tx0>=x1 or ty0>=y1 or tx1<=x0 or ty1<=y0:
                continue
            for curve in curves:
                key=f'{tid}:{curve["candidate_id"]}'
                gid=lookup.get(key)
                if gid is None:
                    raise ValueError(f'Unknown S118 membership: {key}')
                status=(selected[gid]['decision'] if selected else 'AUTO_TRACK_LIKE_NOT_GT')
                if status == 'DEPRIORITIZED_NOT_DELETED':
                    continue
                points=np.asarray(curve['observed_pixel_path_xy'],np.float32)
                if len(points)<2:
                    continue
                points[:,0]+=tx0-x0
                points[:,1]+=ty0-y0
                if ((points[:,0].max()<0 or points[:,0].min()>=roi_size) or
                    (points[:,1].max()<0 or points[:,1].min()>=roi_size)):
                    continue
                rgba=(42,178,32) if status == 'AUTO_TRACK_LIKE_NOT_GT' else (28,196,252)
                cv2.polylines(overlay,[np.rint(points).astype(np.int32)],False,rgba,2,cv2.LINE_AA)
                drawn+=1
                uncertain+=int(status == 'REVIEW_UNCERTAIN')
        dashed=0
        for xy in bridges:
            pts=np.asarray(xy,np.float32)
            if len(pts)<2 or pts.shape[1]!=2:continue
            if pts[:,0].max()<x0 or pts[:,0].min()>=x1 or pts[:,1].max()<y0 or pts[:,1].min()>=y1:
                continue
            _dotted(overlay,np.rint(pts-[x0,y0]).astype(np.int32))
            dashed+=1
        # Put labels above image region; original pixel array remains unchanged on disk.
        side=np.concatenate([orig,overlay],axis=1)
        cv2.rectangle(side,(0,0),(side.shape[1],36),(20,20,20),-1)
        cv2.putText(side,f'{rid} SOURCE  |  PROPOSALS (NOT GT) / xy={x0},{y0}',
                    (12,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
        filename=f'{rid}_SOURCE_VS_PROPOSALS.jpg'
        cv2.imwrite(str(out/filename),side,[cv2.IMWRITE_JPEG_QUALITY,88])
        review_rows.append({'roi':rid,'frame':bmp.stem,'source_sha256':digest,
                            'x0':x0,'y0':y0,'x1':x1,'y1':y1,'image':filename,
                            'overlaid_observed_fragments':drawn,
                            'uncertain_fragments':uncertain,'inferred_connectors':dashed,
                            'reviewer':'','confirmed_tracks':'','missed_tracks':'',
                            'spurious_tracks':'','false_merges':'','false_splits':'',
                            'bad_crosses':'','missing_spirals':'','decision':'',
                            'notes':''})
    with (out/'REVIEW_20_REGIONS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(review_rows[0]))
        writer.writeheader();writer.writerows(review_rows)
    (out/'SOURCE_AND_REVIEW_INFO.json').write_text(json.dumps({
        'frame':bmp.name,'source_sha256':digest,'run_dir':str(run_dir),
        'tile_count':len(entries),'roi_selection':'systematic 4x5 grid with deterministic seeded jitter',
        'roi_seed':seed,'roi_size':roi_size,'regions':review_rows,
        'manual_labels_used_for_selection':False,
        'all_predicted_identities_unverified':True,
        'color_key':{'green':'provisional track-like observed fragments',
                     'yellow':'uncertain observed fragments',
                     'dotted_orange':'inferred continuation; not observed'},
    },ensure_ascii=False,indent=2),encoding='utf-8')
    return {'source':bmp.name,'n_regions':len(review_rows), 'tile_count':len(entries),
            'images_created':len(review_rows),'sha256':digest}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bmp',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--s120-status',type=Path,
                        help='S120_GROUP_STATUS.json, scored from separately trained local model')
    parser.add_argument('--s121',type=Path,help='Optional S121 full-frame JSON links')
    parser.add_argument('--roi-size',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=20261111)
    args=parser.parse_args()
    print(json.dumps(create_pack(args.bmp,args.run_dir,args.out,
                                args.s120_status,args.s121,args.roi_size,args.seed),indent=2))
