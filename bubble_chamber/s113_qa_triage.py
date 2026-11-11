"""Rank full-frame tiled model candidate-density anomalies for human inspection.

Neither a correctness score nor a particle classification; high/low-density
regions are relative priorities for QA. All tile coordinates remain private.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,cv2


def summarize(tiles:list[dict],rows:int,cols:int)->dict:
    if rows<1 or cols<1:raise ValueError('Missing full frame grid')
    if len(tiles)!=rows*cols:raise ValueError('Missing tiles, no full-frame claim allowed')
    density=np.full((rows,cols),np.nan,float)
    strong=np.full_like(density,np.nan)
    for item in tiles:
        row=int(item['row']);col=int(item['column'])
        if row<0 or col<0 or row>=rows or col>=cols:raise ValueError('Bad tile coordinate')
        if np.isfinite(density[row,col]):raise ValueError('Duplicate tile')
        density[row,col]=int(item['all_candidate_peaks_not_GT'])
        strong[row,col]=int(item['strong_peak_proposals'])
        if density[row,col]<strong[row,col] or strong[row,col]<0:raise ValueError('Inconsistent peak counts')
    if np.isnan(density).any():raise ValueError('Incomplete tile grid')
    med=float(np.median(density));mad=float(np.median(np.abs(density-med)))
    scale=max(mad*1.4826,1.0)
    z=(density-med)/scale
    ranked=sorted([{'row':r,'column':c,'all_bead_peak_proposals':int(density[r,c]),
                    'strong_candidate_peaks':int(strong[r,c]),
                    'relative_robust_z':round(float(z[r,c]),3)}
             for r in range(rows) for c in range(cols)],key=lambda p:abs(p['relative_robust_z']),reverse=True)
    return {'schema':'S113_PEAK_DENSITY_QA_RANKING_NOT_QUALITY_SCORE',
            'grid_rows_cols':[rows,cols],'tiles_verified':len(ranked),
            'median_candidates':med,'mad_candidates':mad,
            'high_priority_for_manual_inspection':[v for v in ranked if abs(v['relative_robust_z'])>=3][:20],
            'highest_count_tiles':sorted(ranked,key=lambda v:-v['all_bead_peak_proposals'])[:15],
            'lowest_count_tiles':sorted(ranked,key=lambda v:v['all_bead_peak_proposals'])[:15],
            'not_ground_truth':True,'no_physical_track_precision_measured':True}


def render(summary:dict,tiles:list[dict],out:Path):
    rows,cols=summary['grid_rows_cols']
    array=np.zeros((rows,cols),float)
    for t in tiles:array[t['row'],t['column']]=int(t['all_candidate_peaks_not_GT'])
    lo,hi=np.percentile(array,[5,95]);denom=max(hi-lo,1.)
    vals=np.clip((array-lo)/denom,0,1)
    heat=cv2.applyColorMap(np.uint8(vals*255),cv2.COLORMAP_VIRIDIS)
    tile=95;outimg=cv2.resize(heat,(cols*tile,rows*tile),interpolation=cv2.INTER_NEAREST)
    for row in range(rows):
        for col in range(cols):
            cv2.putText(outimg,f'R{row:02d}C{col:02d}',(col*tile+4,row*tile+18),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1,cv2.LINE_AA)
    cv2.imwrite(str(out),outimg)
    return out


def analyze_saved_summary(path:Path,out:Path)->dict:
    d=json.loads(path.read_text(encoding='utf-8'))
    rows,cols=d['grid_rows_cols']
    report=summarize(d['tiles'],rows,cols)
    out.mkdir(parents=True,exist_ok=True)
    (out/'density_priority_NOT_MODEL_ACCURACY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    render(report,d['tiles'],out/'density_priority_heatmap.png')
    return report