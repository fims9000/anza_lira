#!/usr/bin/env python3
"""S44 exact BMP cross-frame co-visibility: same-position artifact *hypotheses*."""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
import argparse,json,sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s44_crossframe_persistence import (
    ExactBMP8, fit_common_fiducials, warp_corresponding_window,
    compare_registered_ridges, ridge_near_path,
)
from bubble_chamber.s39_machine_auto import ridge_evidence
from bubble_chamber.s40_fiducial_safety import fiducial_guard


def run(bmps:Path,private_s39:Path,s40_fiducials:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    images={k:ExactBMP8(bmps/f'H50Z_030_01_{k}.bmp') for k in ('005','006')}
    labels=json.loads((bmps/'labels.json').read_text())
    reg=fit_common_fiducials(labels)
    s40=json.loads(s40_fiducials.read_text())
    forward={'005':reg['forward_005_to_006_PRIVATE'],
             '006':reg['inverse_006_to_005_PRIVATE']}
    source=json.loads(private_s39.read_text())
    if len(source.get('tiles',[]))!=39 or source.get('native_expert_gt_created')!=0:
        raise ValueError('Wrong private source S39 H50Z manifest')
    private=[];stats=Counter();per_photo={'005':Counter(),'006':Counter()}
    examples=[]; scene_advantages=[]
    for idx,tile in enumerate(source['tiles']):
        key=tile['photo_id'];other='006' if key=='005' else '005'
        crop=images[key].crop(*tile['native_origin_xy_PRIVATE'],512,512)
        if sha256(crop.tobytes()).hexdigest()!=tile['source_gray_sha256']:
            raise ValueError('Source pixel SHA mismatch: no wrong crossframe images')
        stats['crop_sha_match']+=1
        try:
            paired=warp_corresponding_window(images[other],tile['native_origin_xy_PRIVATE'],forward[key])
        except ValueError:
            stats['no_counterpart_coverage']+=1
            private.append({'crop_id':tile['crop_id'],'photo':key,'overlap_status':'NO_OTHER_IMAGE_COVERAGE',
                            'candidate_count':len(tile['proposals'])})
            continue
        scores=compare_registered_ridges(crop,paired,tile['proposals'])
        paired_diff=[]
        for s,p in zip(scores,tile['proposals']):
            s['known_fiducial_uncertainty_NOT_NATIVE_TRUTH'] = fiducial_guard(
                p['points_xy'], tile['native_origin_xy_PRIVATE'], key,
                s40['fiducials_PRIVATE'])['fiducial_uncertainty_any']
            if s['known_fiducial_uncertainty_NOT_NATIVE_TRUTH']:
                stats['source_known_fiducial_risk'] += 1
            if s['displaced_same_shape_counterpart_control_coverage'] is not None:
                stats['candidate_shift_control_eligible'] += 1
                direct = s['both_path_coverage_advisory'] >= .45
                shifted = s['displaced_same_shape_counterpart_control_coverage'] >= .45
                stats['candidate_shift_control_high_support'] += int(shifted)
                stats['direct_high_given_eligible'] += int(direct)
                stats['nonfiducial_direct_high_given_eligible'] += int(
                    direct and not s['known_fiducial_uncertainty_NOT_NATIVE_TRUTH'])
                paired_diff.append(float(direct) - float(shifted))
        if paired_diff:
            scene_advantages.append(float(np.mean(paired_diff)))
        for s in scores:
            per_photo[key]['total']+=1
            stats['candidate_comparison']+=1
            if s['source_multiscale_ridge_coverage']>=.45:
                stats['source_strong_multiscale']+=1
            if s['counterpart_same_registered_path_coverage']>=.45:
                stats['counterpart_strong_multiscale']+=1
            if s['both_path_coverage_advisory']>=.45:
                stats['both_strong_multiscale_UNVERIFIED']+=1
                per_photo[key]['high_corresponding_structure']+=1
        stats['crop_both_covered']+=1
        private.append({'crop_id':tile['crop_id'],'photo':key,'source_sha256':tile['source_gray_sha256'],
                        'overlap_status':'BOTH_IMAGES_VALID',
                        'candidate_count':len(scores), 'private_candidate_evidence':scores})
        if idx in (0,3,11,23,31,37):
            original=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)
            counterpart=cv2.cvtColor(paired,cv2.COLOR_GRAY2BGR)
            for r,p in zip(scores,tile['proposals']):
                xy=np.rint(np.asarray(p['points_xy'])).astype(np.int32)
                color=(30,140,255) if r['both_path_coverage_advisory']>=.45 else (70,215,95)
                cv2.polylines(original,[xy],False,color,2,lineType=cv2.LINE_AA)
                cv2.polylines(counterpart,[xy],False,color,2,lineType=cv2.LINE_AA)
            panel=cv2.hconcat([original,counterpart])
            name=f'S44_PRIVATE_COMPARE_{idx:02d}.jpg'
            cv2.imwrite(str(out/name),panel,[cv2.IMWRITE_JPEG_QUALITY,86])
            examples.append(name)
    # A positive *image structure* control: native calibration X-arms.
    fid_support=[]
    fids={(f['photo'],f['label']):f for f in s40['fiducials_PRIVATE']}
    for label in ('f1','f2','f3','f4','f5','f6'):
        mark=fids[('005',label)]
        center=np.asarray(mark['center_xy_PRIVATE'])
        origin=np.rint(center).astype(int)-256
        src=images['005'].crop(int(origin[0]),int(origin[1]),512,512)
        dest=warp_corresponding_window(images['006'],origin,forward['005'])
        maps=(ridge_evidence(src)['score'],ridge_evidence(dest)['score'])
        for j,theta in enumerate(mark['arms_angles_rad_PRIVATE']):
            v=np.array([np.cos(theta),np.sin(theta)])
            path=(center-origin)[None,:]+np.linspace(-150,150,80)[:,None]*v[None,:]
            a=ridge_near_path(maps[0],path)
            b=ridge_near_path(maps[1],path)
            fid_support.append({'label':label,'arm_index':j,'source_support':a,
                                'counterpart_registered_support':b})
    stats['known_fiducial_arm_controls']=len(fid_support)
    rng=np.random.default_rng(20261044)
    if len(scene_advantages)>3:
        bs=np.asarray(scene_advantages)
        idx=rng.integers(0,len(bs),(2500,len(bs)))
        delta_ci=np.quantile(bs[idx].mean(axis=1),[.025,.975]).round(5).tolist()
    else:
        delta_ci=None
    # Persist full source markers & affine in PRIVATE only, never public Git.
    private_file={'schema':'s44_crossframe_registered_ridge_advisory_v1',
      'forward_affine_PRIVATE':reg['forward_005_to_006_PRIVATE'].tolist(),
      'fiducial_stability_PRIVATE':fid_support,
      'registered_crops_PRIVATE':private,
      'physical_track_GT_count':0,'automatic_native_tracks_removed':0}
    (out/'S44_PRIVATE_CROSSFRAME_EVIDENCE.json').write_text(json.dumps(private_file,indent=2)+'\n')
    summary={
      'schema':'s44_crossframe_2d_registration_machine_advisory_v1',
      'source_photographs':2,'independent_physical_runs':1,
      'common_identifiers_f1_through_f6':reg['common_fiducial_count'],
      'mean_calibration_residual_px':round(reg['mean_common_residual_px'],4),
      'max_calibration_residual_px':round(reg['max_common_residual_px'],4),
      'label_f8_in_005_vs_f7_in_006_residual_px_NOT_PROVEN_SAME_MARK':
        round(reg['additional_f8_005_to_f7_006_residual_px'],3),
      'known_fiducial_control_arms':len(fid_support),
      'known_fiducial_crossimage_support_mean':round(float(np.mean([v['counterpart_registered_support'] for v in fid_support])),4),
      'native_machine_candidates_total':sum(len(t['proposals']) for t in source['tiles']),
      'candidate_counters':dict(stats),
      'by_source_photo':{k:dict(v) for k,v in per_photo.items()},
      'registered_minus_shifted_high_support_fraction_scene_bootstrap_95pct_CI_EXPLORATORY':delta_ci,
      'high_correspondence_definition':'Both source and registered other-photo ridge maps: >=0.45 path support (95th percentile or nonzero fallback) with 2px tolerance. Uncalibrated exploratory statistic.',
      'known_native_track_GT':0,'real_physical_auto_joins_performed':0,
      'native_candidates_erased':0,'no_run_independence_claim':True,
      'repeating_structure_not_proof_of_scratch':True,
    }
    (out/'S44_PUBLIC_AGGREGATES.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bmp-directory',required=True,type=Path)
    p.add_argument('--s39-private',required=True,type=Path)
    p.add_argument('--s40-fiducials',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    run(a.bmp_directory,a.s39_private,a.s40_fiducials,a.out)
if __name__=='__main__':main()
