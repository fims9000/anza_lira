"""Hold out real visualMetro clicks to evaluate geometry reconstruction (not BMPs)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np


@dataclass(frozen=True)
class InterpolationProtocol:
    hidden_clicks: tuple[int,...] = (1, 3, 5)
    cut_fractions: tuple[float,...] = (0.35, 0.5, 0.65)


def interpolate_gap(observed_left: np.ndarray, observed_right: np.ndarray, hidden_count: int, method: str) -> np.ndarray:
    """Interpolate hidden clicks as a function of click index (not arc-length)."""
    left = np.asarray(observed_left, dtype=float)
    right = np.asarray(observed_right, dtype=float)
    if left.ndim != 2 or left.shape[1] not in (2,3) or right.ndim != 2 or right.shape[1] != left.shape[1] or min(len(left),len(right))<3 or hidden_count<1:
        raise ValueError('expected >=3 observed points on each side, dimension 2 or 3')
    p0=left[-1];p1=right[0]
    alpha=np.arange(1,hidden_count+1,dtype=float)[:,None]/(hidden_count+1)
    if method == 'linear':
        return p0[None,:]*(1-alpha)+p1[None,:]*alpha
    if method == 'hermite':
        # central-looking derivative using the neighboring TWO original clicks
        m0=(3*left[-1]-4*left[-2]+left[-3])/2
        m1=(-3*right[0]+4*right[1]-right[2])/2
        length=hidden_count+1
        h00=2*alpha**3-3*alpha**2+1
        h10=alpha**3-2*alpha**2+alpha
        h01=-2*alpha**3+3*alpha**2
        h11=alpha**3-alpha**2
        return h00*p0+h10*length*m0+h01*p1+h11*length*m1
    if method == 'quadratic':
        # Last and first 3 measured points, fit each image axis as a polynomial
        t=np.concatenate((np.arange(-2,1,dtype=float), np.arange(hidden_count+1,hidden_count+4,dtype=float)))
        observed=np.concatenate((left[-3:],right[:3]),axis=0)
        z=np.polyfit(t,observed,2)
        return np.polyval(z[:,0],np.arange(1,hidden_count+1))[:,None] if observed.shape[1]==1 else np.column_stack([np.polyval(z[:,k],np.arange(1,hidden_count+1)) for k in range(observed.shape[1])])
    raise ValueError(method)


def evaluate_interpolation(manifest: dict, cfg: InterpolationProtocol = InterpolationProtocol()) -> dict:
    if manifest.get('audit_schema') != 'visualmetro_large_audit_v1':
        raise ValueError('audited input required')
    methods=('linear','quadratic','hermite')
    per_gap:dict[str,dict] = {}
    errors_raw={}
    case_records=[]
    for k in cfg.hidden_clicks:
        errors={field:{method:[] for method in methods} for field in ('xy_px','xyz_cm')}
        by_frame={field:defaultdict(lambda: {method:[] for method in methods}) for field in errors}
        eligible_tracks=0
        for frame in manifest['frames']:
            for track in frame['tracks']:
                n=track['n_points']
                if n<k+6:continue
                eligible_tracks+=1
                sources=[('xyz_cm', np.asarray(track['xyz_cm'],float))]+[(f'v{v}_px', np.asarray(track['measured_pixel_xy'][str(v)],float)) for v in (1,2,3)]
                for frac in cfg.cut_fractions:
                    left_end=min(max(2,int(round((n-1)*frac))),n-k-4)
                    right_start=left_end+k+1
                    for srcname, points in sources:
                        field='xyz_cm' if srcname=='xyz_cm' else 'xy_px'
                        truth=points[left_end+1:right_start]
                        for method in methods:
                            predicted=interpolate_gap(points[:left_end+1],points[right_start:],k,method)
                            assert predicted.shape==truth.shape
                            difference=np.linalg.norm(predicted-truth,axis=1)
                            errors[field][method].extend(difference.tolist())
                            by_frame[field][frame['frame_id']][method].extend(difference.tolist())
                            if method=='hermite' and field=='xy_px' and srcname=='v1_px':
                                case_records.append({'frame_id':frame['frame_id'],'track_id':track['track_id'],
                                                     'hidden_clicks':k,'cut':frac,'rms_hermite_px':round(float(np.sqrt(np.mean(difference**2))),3),
                                                     'left_xy':points[left_end].tolist(),'right_xy':points[right_start].tolist(),
                                                     'true_hidden_xy':truth.tolist(),'pred_hidden_hermite_xy':predicted.tolist()})
        per_gap[str(k)]={'eligible_tracks':eligible_tracks,'errors':{}}
        for field in errors:
            per_gap[str(k)]['errors'][field]={}
            for method in methods:
                values=np.asarray(errors[field][method],float)
                per_gap[str(k)]['errors'][field][method]={
                    'count_hidden_positions':len(values),
                    'mean':round(float(np.mean(values)),5) if len(values) else None,
                    'median':round(float(np.median(values)),5) if len(values) else None,
                    'p90':round(float(np.quantile(values,0.9)),5) if len(values) else None,
                    'p95':round(float(np.quantile(values,0.95)),5) if len(values) else None,
                }
        errors_raw[str(k)]=errors
    return {'study':'S5_INTERPOLATION_REAL_EXPERT_CLICKS_SYNTHETIC_HOLDOUT','params':asdict(cfg),
            'source_digest':manifest['provenance_sha256'], 'results':per_gap,
            'examples_worst_hermite':sorted(case_records,key=lambda x:x['rms_hermite_px'],reverse=True)[:12],
            'limits':['Holdout is artificial; no actual missing track image pixel is reconstructed',
                      'Click-index spacing is not physical time or arc-length',
                      'No image features used; no segmentation/PID claim',
                      'All frames share film 070']}


def save_interpolation(manifest:dict, path:Path) -> dict:
    payload=evaluate_interpolation(manifest)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return payload
