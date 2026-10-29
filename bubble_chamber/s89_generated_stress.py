"""Hard-negative, generated-only S89 stress protocol (never native H50Z GT).

Frozen S88 CAL thresholds; a post-hoc perturbation-consistency diagnostic is
compared on fresh synthetic scenes, with real false-negative/false-positive
trade-offs reported. This diagnostic is NOT a physical artifact classifier.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .s88_weak_ridges import WeakRidgeConfig, dark_ridge_candidates
from .s89_private_candidate_audit import AuditConfig, persistence_under_perturbations

SCHEMA = 'S89_GENERATED_ONLY_ARTIFACT_STRESS_LOCKED_S88_V1'
VARIANTS = ('faint_curves', 'blurred_faint_curves', 'scratches_only',
            'fiducials_only', 'crossing_scratches_and_curves')
S88_LOCKED_LOW = .816
S88_LOCKED_HIGH = 1.7


def generate(seed: int, variant: str):
    """Return ONLY simulated film pixels and synthetic injected line centers."""
    if variant not in VARIANTS:
        raise ValueError('unsupported synthetic variant')
    rng = np.random.default_rng(seed)
    h, w = 224, 256
    yy, xx = np.indices((h,w),dtype=np.float32)
    gray = 194 + 14*np.sin(xx/71 + seed*.002) + 8*np.cos(yy/60 + seed*.005)
    gray = gray + rng.normal(0, 3.1, (h,w)).astype(np.float32)
    gt = np.zeros((h,w),np.uint8)
    contains_traces = variant not in ('scratches_only','fiducials_only')
    if contains_traces:
        for track in range(2):
            xs=np.arange(10,w-10).astype(float)
            ys=69 + 92*track + (9+3*track)*np.sin(xs/(37+14*track) + seed*.003)
            ys+= np.cos(xs/19+track)*3
            pts = np.column_stack((xs,ys)).astype(np.int32)
            layer=np.zeros((h,w),np.uint8)
            cv2.polylines(layer,[pts],False,255,2)
            cv2.polylines(gt,[pts],False,255,1)
            strength = 7.0 if variant=='faint_curves' else (8.0 if variant=='blurred_faint_curves' else 10.0)
            gray -= strength*gaussian_filter(layer.astype(np.float32)/255,1.0)
    # Artifacts are explicitly NOT physical tracks; even a perfectly stable
    # photo scratch is a synthetic false-positive if detected.
    num_scratches = (24 if variant=='scratches_only' else
                     12 if variant=='crossing_scratches_and_curves' else 5)
    for _ in range(num_scratches):
        x0=int(rng.integers(5,w-5));y0=int(rng.integers(5,h-5))
        dx=int(rng.integers(-100,101));dy=int(rng.integers(-90,91))
        x1=int(np.clip(x0+dx,2,w-3));y1=int(np.clip(y0+dy,2,h-3))
        scratch=np.zeros((h,w),np.uint8)
        cv2.line(scratch,(x0,y0),(x1,y1),255,int(rng.choice([1,1,2])))
        gray-=float(rng.uniform(8,23))*gaussian_filter(scratch.astype(np.float32)/255, .7)
    for _ in range(76):
        x=int(rng.integers(4,w-4));y=int(rng.integers(4,h-4))
        scratch=np.zeros((h,w),np.uint8)
        cv2.circle(scratch,(x,y),int(rng.choice([1,2,2,3])),255,-1)
        gray-=float(rng.uniform(5,21))*gaussian_filter(scratch.astype(np.float32)/255, .75)
    # Camera/film fiducials and plate scratches can be as dark or darker than
    # the desired image traces. They are disjoint from the injected line mask.
    if variant in ('fiducials_only','crossing_scratches_and_curves'):
        for x,y in [(32,32),(120,190),(210,61),(140,130)]:
            layer=np.zeros((h,w),np.uint8)
            cv2.line(layer,(x-12,y),(x+12,y),255,1)
            cv2.line(layer,(x,y-12),(x,y+12),255,1)
            gray-=14*gaussian_filter(layer.astype(np.float32)/255,.9)
    if variant=='blurred_faint_curves':
        gray=gaussian_filter(gray,1.4)
    return np.uint8(np.clip(np.rint(gray),0,255)),gt.astype(bool),contains_traces


def _metrics(mask: np.ndarray, truth: np.ndarray, has_lines: bool):
    area=float(np.mean(mask))
    if not has_lines:
        return {'candidate_pixel_fraction':round(area,6),
                'synthetic_injected_line_coverage_tolerance_2px':None,
                'off_injected_line_candidate_pixel_fraction':round(area,6)}
    d_detect=distance_transform_edt(~mask)
    d_gt=distance_transform_edt(~truth)
    return {'candidate_pixel_fraction':round(area,6),
            'synthetic_injected_line_coverage_tolerance_2px':round(float(np.mean(d_detect[truth]<=2)),6),
            'off_injected_line_candidate_pixel_fraction':round(float(np.mean(mask & (d_gt>3))),6)}


def _mean(scores:list[dict]):
    # Artifact-only scenes have no injected positive line; omit their missing
    # coverage from that particular mean, without treating missing as zero.
    result = {}
    for name in scores[0]:
        valid = [row[name] for row in scores if row[name] is not None]
        result[name] = round(float(np.mean(valid)), 6) if valid else None
    return result


def _paired_scene_ci(before,after,metric,seed):
    pairs=[(a[metric],b[metric]) for a,b in zip(before,after) if a[metric] is not None]
    if not pairs:
        return None
    diffs=np.array([y-x for x,y in pairs],dtype=float)
    r=np.random.default_rng(seed)
    samples=diffs[r.integers(0,len(diffs),(2000,len(diffs)))].mean(axis=1)
    return {'post_minus_original_mean':round(float(diffs.mean()),6),
            '95pct_paired_bootstrap_generated_scene_CI':[
                round(float(v),6) for v in np.quantile(samples,[.025,.975])],
            'independent_generated_scene_count':len(diffs)}


def benchmark(n_per_variant:int=4):
    if not 2 <= n_per_variant <= 20:
        raise ValueError('require 2..20 new independent generated scenes per condition')
    cfg=WeakRidgeConfig(low_snr=S88_LOCKED_LOW,high_snr=S88_LOCKED_HIGH,
                        min_component_pixels=15)
    audit=AuditConfig()
    out={}
    all_original=[];all_stable=[]
    for vindex,variant in enumerate(VARIANTS):
        originals=[];stables=[]
        for j in range(n_per_variant):
            seed=9000 + 300*vindex+j
            gray,truth,has_lines=generate(seed,variant)
            candidate,_=dark_ridge_candidates(gray,cfg)
            p1,p2=persistence_under_perturbations(gray,cfg,audit,seed+10000)
            diagnostic_consensus=candidate & p1 & p2
            originals.append(_metrics(candidate,truth,has_lines))
            stables.append(_metrics(diagnostic_consensus,truth,has_lines))
        all_original.extend(originals);all_stable.extend(stables)
        out[variant]={
            'generated_has_injected_lines':has_lines,
            'original_s88_candidates':_mean(originals),
            'posthoc_consensus_diagnostic_only':_mean(stables),
            'paired_generated_scene_coverage_delta':_paired_scene_ci(
                originals,stables,'synthetic_injected_line_coverage_tolerance_2px',400+vindex),
            'paired_generated_scene_offline_candidate_delta':_paired_scene_ci(
                originals,stables,'off_injected_line_candidate_pixel_fraction',500+vindex),
        }
    sources={p:sha256((Path(__file__).parent/p).read_bytes()).hexdigest()
             for p in ('s89_private_candidate_audit.py','s89_generated_stress.py',
                       's88_weak_ridges.py')}
    return {'schema':SCHEMA,
            'label_semantics':'SYNTHETIC_INJECTED_LINES_ONLY_NOT_PHYSICAL_PARTICLE_TRACKS',
            'source_code_sha256':sources,
            'protocol':{'seed_start_by_variant':{v:9000+300*i for i,v in enumerate(VARIANTS)},
                        'independent_scenes_per_variant':n_per_variant,
                        's88_frozen_CAL_thresholds':{'low':S88_LOCKED_LOW,'high':S88_LOCKED_HIGH},
                        's89_reuses_thresholds_without_TEST_tuning':True,
                        'diagnostic_consensus_is_NOT_deployed_filter':True,
                        'positive_tolerance_px':2,'off_centerline_distance_px':3,
                        'unit_of_bootstrap':'INDEPENDENT_GENERATED_SCENE'},
            'per_variant':out,
            'all_generated_original':_mean(all_original) if all_original else {},
            'all_generated_posthoc_consensus':_mean(all_stable) if all_stable else {},
            'native_physics_ground_truth_count':0,
            'physical_track_recall_or_precision_measured':False,
            's89_changes_original_S86_detector_masks':False,
            'physical_auto_accept_or_reject_authorized':False,
            'physical_auto_joins_authorized':False,
            'scientific_limitation':'Stable camera scratches and fiducials can resemble stable lines. Image consistency alone is NOT particle identity.'}


def write_benchmark(path:Path,n_per_variant:int=4):
    path=Path(path)
    result=benchmark(n_per_variant)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    return result
