"""SOURCE-ONLY generated scenes for S88. NO H50Z, film070 or particle GT.

Fixed, disjoint CAL and TEST seeds. Thresholds are chosen on CAL only,
subject to a matched mean candidate-pixel budget. A generated centerline is
synthetic injection truth, *not* an actual physical track annotation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label as component_label

from .full_bmp_segmentation import FullBMPConfig, predict_mask
from .ridges import frangi_dark_ridges
from .s88_weak_ridges import WeakRidgeConfig, dark_ridge_candidates, ridge_mask_from_score

SCHEMA = 'S88_GENERATED_ONLY_CAL_TEST_MATCHED_BUDGET_V1'
VARIANTS = ('faint_grain', 'blurred_scratches')
METHODS = ('contrast', 'frangi', 's88-steerable')
TARGET_BUDGET = 0.015
GRIDS = {
    'contrast': (5.5, 6.0, 6.1, 7.0, 7.5, 9.0, 12.0),
    'frangi': (0.10, 0.115, 0.125, 0.135, 0.16, 0.24, 0.35),
    's88-steerable': (1.6, 1.7, 1.8, 2.0, 2.5, 3.4, 4.5),
}


@dataclass(frozen=True)
class Scene:
    gray: np.ndarray
    injected_centerline: np.ndarray
    seed: int
    variant: str


def scene(seed: int, variant: str) -> Scene:
    """Two curving dark traces plus unrelated dark artifacts, noise and gradients."""
    if variant not in VARIANTS:
        raise ValueError('unknown generated variant')
    rng = np.random.default_rng(int(seed))
    height, width = 224, 256
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    background = (211.0 + 11.0*np.sin(xx/64 + seed*.04)
                  + 9.0*np.cos(yy/77 + seed*.06) + 0.02*(xx-128))
    std = 1.9 if variant == 'faint_grain' else 2.7
    gray = background + rng.normal(0, std, (height, width)).astype(np.float32)
    # Spatially localized bubbles and optical defects are *NOT* line GT.
    for _ in range(42 if variant == 'faint_grain' else 76):
        cx = int(rng.integers(8, width-8))
        cy = int(rng.integers(8, height-8))
        r = int(rng.integers(1, 3))
        cv2.circle(gray, (cx,cy), r, float(gray[cy,cx])-float(rng.uniform(5,18)), -1)
    line_truth = np.zeros((height,width), np.uint8)
    trace_layer = np.zeros((height,width), np.uint8)
    for line in range(2):
        xs = np.arange(12, width-12, dtype=np.float32)
        base = 60 + 103*line + rng.uniform(-10,10)
        ys = (base + (8+line*4)*np.sin(xs/(49+line*16)+seed*.031)
              + (xs-width/2)**2 * (0.00018 if line == 0 else -0.00022))
        pts = np.stack([xs, np.clip(ys,4,height-5)],axis=1).astype(np.int32)
        cv2.polylines(line_truth,[pts],False,255,1,cv2.LINE_8)
        cv2.polylines(trace_layer,[pts],False,255,2,cv2.LINE_8)
    trace = gaussian_filter(trace_layer.astype(np.float32)/255.0,
                            0.95 if variant=='faint_grain' else 1.55)
    amplitude = 10.0 if variant=='faint_grain' else 14.0
    gray -= amplitude*trace
    # Scratch/fiducial decoys must never be included as injected centerlines.
    dark_decoys = 2 if variant=='faint_grain' else 5
    for i in range(dark_decoys):
        x0=int(rng.integers(9,width-9));y0=int(rng.integers(9,height-9))
        x1=int(np.clip(x0+rng.integers(-100,101),3,width-4))
        y1=int(np.clip(y0+rng.integers(-100,101),3,height-4))
        decoy=np.zeros((height,width),np.uint8)
        cv2.line(decoy,(x0,y0),(x1,y1),255,1)
        gray -= float(rng.uniform(7,20))*gaussian_filter(decoy.astype(np.float32)/255,0.9)
    if variant=='blurred_scratches':
        gray=gaussian_filter(gray,0.55)
    return Scene(np.uint8(np.clip(gray,0,255).round()),line_truth.astype(bool),seed,variant)


def metrics(mask: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Fuzzy centerline matching at <=2px and off-GT proposal burden >3px."""
    if mask.shape!=gt.shape or not np.any(gt):
        raise ValueError('mask and nonempty generated centerline required')
    d_pred=distance_transform_edt(~mask)
    d_truth=distance_transform_edt(~gt)
    matched=float(np.mean(d_pred[gt] <= 2))
    area=float(np.mean(mask))
    off_truth=float(np.mean(mask & (d_truth > 3)))
    on_truth=float(np.count_nonzero(mask & (d_truth<=3)))
    return {
        'injected_centerline_tolerant_recall':round(matched,6),
        'candidate_pixel_fraction':round(area,6),
        'off_injected_centerline_pixel_fraction':round(off_truth,6),
        'on_injected_centerline_candidate_fraction':round(on_truth / max(1,int(mask.sum())),6),
    }


def _predict(gray:np.ndarray, method:str, threshold:float)->np.ndarray:
    k={'method':method,'core_size':256,'halo':32,'min_component_area':15}
    if method=='contrast':k['contrast_threshold']=threshold
    if method=='frangi':k['ridge_response_fraction']=threshold
    if method=='s88-steerable':
        k['s88_high_snr']=threshold
        k['s88_low_snr']=round(0.48*threshold,5)
    return predict_mask(gray,FullBMPConfig(**k))


def _prepare_response(gray:np.ndarray, method:str)->np.ndarray:
    """Cache threshold-independent photometric/Hessian computations on CAL."""
    if method == 'contrast':
        smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.7)
        background = cv2.GaussianBlur(smooth, (0, 0), sigmaX=14.0)
        return cv2.subtract(background, smooth)
    if method == 'frangi':
        _, response = frangi_dark_ridges(gray, response_fraction=0.12, min_area=15)
        return response
    if method == 's88-steerable':
        _, evidence = dark_ridge_candidates(gray, WeakRidgeConfig(min_component_pixels=15))
        return evidence['score_snr_proxy']
    raise ValueError('unknown method')


def _threshold_response(response:np.ndarray,method:str,threshold:float)->np.ndarray:
    """Identical mask rules to FullBMPConfig prediction, avoiding repeated filtering."""
    if method == 's88-steerable':
        return ridge_mask_from_score(response, WeakRidgeConfig(
            high_snr=threshold,low_snr=round(.48*threshold,5),min_component_pixels=15))
    if method == 'contrast':
        raw = response >= threshold
    elif method == 'frangi':
        raw = response >= threshold*float(response.max(initial=0.0)) if response.max(initial=0.0)>0 else np.zeros_like(response,dtype=bool)
    else:
        raise ValueError('unknown detector')
    components,_=component_label(raw,structure=np.ones((3,3),dtype=np.uint8))
    counts=np.bincount(components.ravel())
    return raw & (components>0) & (counts[components]>=15)


def _mean(rows:list[dict[str,float]])->dict[str,float]:
    if not rows:
        raise ValueError('metrics require at least one scene')
    return {key:round(float(np.mean([r[key] for r in rows])),6) for key in rows[0]}


def _ci_by_scene(values:np.ndarray, bootstrap_seed:int)->list[float]:
    # Resample *independent generated scenes*, not pixels or line segments.
    if len(values)<2:
        return [round(float(values[0]),6),round(float(values[0]),6)]
    rng=np.random.default_rng(bootstrap_seed)
    means=values[rng.integers(0,len(values),(4000,len(values)))].mean(axis=1)
    return [round(float(x),6) for x in np.quantile(means,[0.025,0.975])]


def benchmark(cal_count:int=8,test_count:int=12)->dict[str,Any]:
    if not 2<=cal_count<=40 or not 2<=test_count<=40:
        raise ValueError('2..40 independent generated scenes per condition per split required')
    # Distinct by construction; never optimize on TEST. The generator is purely
    # artificial. No native film crops, source profiles or marks are consulted.
    cal=[scene(seed,variant) for variant in VARIANTS for seed in range(1000,1000+cal_count)]
    test=[scene(seed,variant) for variant in VARIANTS for seed in range(5000,5000+test_count)]
    assert {(x.seed,x.variant) for x in cal}.isdisjoint({(x.seed,x.variant) for x in test})
    selected={};cal_results={};test_results={};test_scene_rows={};warnings=[]
    for method in METHODS:
        # Cached response computation makes CAL threshold search reproducible
        # and efficient. It never accesses TEST when selecting the threshold.
        cal_response=[_prepare_response(s.gray,method) for s in cal]
        attempts=[]
        for threshold in GRIDS[method]:
            scores=[metrics(_threshold_response(raw,method,threshold),s.injected_centerline)
                    for s,raw in zip(cal,cal_response)]
            avg=_mean(scores)
            attempts.append({'threshold':threshold,'cal':avg})
        eligible=[row for row in attempts if row['cal']['candidate_pixel_fraction'] <= TARGET_BUDGET]
        if eligible:
            best=max(eligible,key=lambda a:(a['cal']['injected_centerline_tolerant_recall'],
                                            -a['cal']['candidate_pixel_fraction'],a['threshold']))
            breached=False
        else:
            # Conservatively minimize candidate area when the method cannot
            # satisfy the budget, rather than relaxing budget using TEST.
            best=min(attempts,key=lambda a:(a['cal']['candidate_pixel_fraction'], -a['threshold']))
            breached=True
            warnings.append(method+': CAL budget unavailable; report budget exceedance')
        selected[method]={'high_or_threshold':best['threshold'],
                          's88_low_if_applicable':round(.48*best['threshold'],5) if method=='s88-steerable' else None,
                          'cal_pixel_budget_exceeded':breached}
        cal_results[method]={'selected':best['cal'],'grid':attempts}
        test_rows=[metrics(_threshold_response(_prepare_response(s.gray,method),method,best['threshold']),
                           s.injected_centerline) for s in test]
        test_scene_rows[method]=test_rows
        test_results[method]={'summary':_mean(test_rows),
                              'per_variant':{variant:_mean([row for s,row in zip(test,test_rows) if s.variant==variant])
                                             for variant in VARIANTS},
                              '95pct_bootstrap_centerline_recall_by_scene':_ci_by_scene(
                                  np.array([r['injected_centerline_tolerant_recall'] for r in test_rows]),
                                  bootstrap_seed=401 + METHODS.index(method))}
    paired={}
    for baseline in ('contrast','frangi'):
        diffs={}
        for metric in ('injected_centerline_tolerant_recall',
                       'off_injected_centerline_pixel_fraction',
                       'candidate_pixel_fraction'):
            values=np.array([a[metric]-b[metric] for a,b in
                             zip(test_scene_rows['s88-steerable'],test_scene_rows[baseline])],dtype=np.float64)
            diffs[metric]={'mean_S88_minus_'+baseline:round(float(values.mean()),6),
                           '95pct_paired_bootstrap_CI_by_generated_scene':_ci_by_scene(values,
                                                                                      bootstrap_seed=700+len(metric)+len(baseline))}
        paired['s88_vs_'+baseline]=diffs
    scene_details={method:[{'seed':s.seed,'variant':s.variant,'metrics':row}
                            for s,row in zip(test,rows)]
                   for method,rows in test_scene_rows.items()}
    # Frozen meaning: no physical source pixels and no domain-valid accuracy.
    return {
      'schema':SCHEMA,
      'research_code_sha256': {
         'benchmark': sha256(Path(__file__).read_bytes()).hexdigest(),
         'ridge_detector': sha256(Path(__file__).with_name('s88_weak_ridges.py').read_bytes()).hexdigest(),
         'full_bmp_pipeline': sha256(Path(__file__).with_name('full_bmp_segmentation.py').read_bytes()).hexdigest(),
      },
      'generator':'ARTIFICIAL_PHOTOMETRIC_CURVES_AND_DECOYS_ONLY',
      'ground_truth_semantics':'INJECTED_SYNTHETIC_CENTERLINES_NOT_NATIVE_H50Z_OR_PARTICLE_ID',
      'config':{'cal_seeds_per_variant':cal_count,'test_seeds_per_variant':test_count,
                'cal_seed_start':1000,'test_seed_start':5000,
                'variants':list(VARIANTS),'candidate_pixel_budget_mean_CAL':TARGET_BUDGET,
                'distance_tolerance_px':2,'off_trace_distance_px':3,
                'split_unit':'GENERATED_SCENE'},
      'calibration':cal_results,'selected_using_CAL_only':selected,'independent_generated_TEST':test_results,
      'paired_generated_TEST_differences':paired,'generated_TEST_scene_rows':scene_details,
      'warnings':warnings,'native_physics_ground_truth_count':0,
      'native_physical_track_recall_measured':False,
      'physical_auto_joins_authorized':False,
    }


def write_benchmark(out, cal_count:int=8, test_count:int=12):
    from pathlib import Path
    path=Path(out)
    data=benchmark(cal_count,test_count)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    return data
