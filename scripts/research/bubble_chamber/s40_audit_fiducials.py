#!/usr/bin/env python3
"""S40: no-manual-label fiducial negative-control audit on genuine H50Z BMPs.

Sourced f1..f8 are calibration marks, not particle GT. Export original
pixel-level data only to a PRIVATE destination; Git gets aggregates only.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s40_fiducial_safety import (registered_fiducials,fiducial_guard,guard_proposals,guard_graph,infer_fiducial_cross_arms)
from bubble_chamber.s39_machine_auto import (ridge_evidence,raw_detector_mask,extract_ridges)


def run(labels_json:Path,bmp_dir:Path,selected_manifest:Path,fullgrid_manifest:Path,out_dir:Path,graph_manifest:Path|None=None):
    label_data=json.loads(labels_json.read_text())
    selected=json.loads(selected_manifest.read_text())
    fullgrid=json.loads(fullgrid_manifest.read_text())
    out_dir.mkdir(parents=True,exist_ok=True)
    all_zones=[]; rows=[]; region_records=[]; selected_hits=0;full_hits=0
    selected_total=0;full_total=0
    selected_arm=0;full_arm=0;diag_arm=0
    Image.MAX_IMAGE_PIXELS=None
    for photo in ('005','006'):
        zones=registered_fiducials(label_data,photo)
        all_zones.extend(zones)
        src=label_data[f'H50Z_030_01_{photo}.bmp']
        bmp=bmp_dir/f'H50Z_030_01_{photo}.bmp'
        if bmp.suffix.lower()!='.bmp' or not bmp.exists():raise FileNotFoundError(str(bmp))
        with Image.open(bmp) as im:
            if im.size!=(src['original_width'],src['original_height']) or im.mode!='L':
                raise ValueError('Wrong original bitmap dimensions/mode')
            for z in zones:
                x,y=z['center_xy_PRIVATE'];r=z['machine_ambiguity_radius_px']
                x0=max(0,min(round(x)-256,im.width-512)); y0=max(0,min(round(y)-256,im.height-512))
                patch=np.asarray(im.crop((x0,y0,x0+512,y0+512)),dtype=np.uint8)
                estimated_arms=infer_fiducial_cross_arms(patch,(x-x0,y-y0))
                z['arms_angles_rad_PRIVATE']=estimated_arms
                z['arm_half_span_px']=220
                z['arm_tolerance_px']=9
                ev=ridge_evidence(patch);raw=raw_detector_mask(ev)
                candidates=extract_ridges(ev,max_candidates=32)
                risks=[fiducial_guard(p,(x0,y0),photo,zones) for p in candidates]
                touched=sum(p['fiducial_mark_zone_overlap'] for p in risks)
                diag_arm+=sum(p['fiducial_arm_lookalike'] for p in risks)
                cx=x-x0;cy=y-y0
                yy,xx=np.ogrid[:512,:512]
                inside=(xx-cx)**2+(yy-cy)**2<=r*r
                annulus=((xx-cx)**2+(yy-cy)**2>1.8*r*1.8)&((xx-cx)**2+(yy-cy)**2<2.7*r*2.7)
                # This is DETECTOR ACTIVITY, not particle pixel precision.
                inner_coverage=float(np.mean(raw[inside]))
                annulus_coverage=float(np.mean(raw[annulus])) if np.any(annulus) else 0.
                rows.append({'photo':photo,'fiducial':z['label'],
                    'candidate_count':len(candidates),
                    'candidate_touches_known_fiducial_zone':touched,
                    'estimated_fiducial_X_arms':len(estimated_arms),
                    'candidate_fiducial_arm_lookalikes':sum(p['fiducial_arm_lookalike'] for p in risks),
                    'candidate_any_fiducial_risk':sum(p['fiducial_uncertainty_any'] for p in risks),
                    'ridge_activation_in_marker_zone':round(inner_coverage,5),
                    'ridge_activation_in_outer_reference_annulus':round(annulus_coverage,5)})
                # Private image: source pixels, source fiducial coordinate, and
                # generated curve pixels all remain outside public Git.
                rgb=cv2.cvtColor(patch,cv2.COLOR_GRAY2RGB)
                for p,is_risky in zip(candidates,risks):
                    pts=np.rint(p).astype(np.int32)
                    cv2.polylines(rgb,[pts],False,(237,82,57) if is_risky['fiducial_uncertainty_any'] else (25,218,198),2)
                for angle in estimated_arms:
                    dx=220*np.cos(angle);dy=220*np.sin(angle)
                    cv2.line(rgb,(round(cx-dx),round(cy-dy)),(round(cx+dx),round(cy+dy)),(232,162,239),1)
                cv2.circle(rgb,(round(cx),round(cy)),round(r),(245,212,15),2)
                cv2.circle(rgb,(round(cx),round(cy)),5,(251,205,3),-1)
                cv2.imwrite(str(out_dir/f'S40_PRIVATE_FID_{photo}_{z["label"]}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                region_records.append({'photo':photo,'fiducial':z['label'],
                    'source_origin_xy_PRIVATE':[x0,y0], 'source_xy_PRIVATE':[x,y],
                    'machine_candidates_PRIVATE':[np.round(p,2).tolist() for p in candidates],
                    'candidate_fiducial_proximity_flags':risks})
        for tile in selected['tiles']:
            if tile['photo_id']!=photo:continue
            guards=guard_proposals(tile['proposals'],tile['native_origin_xy_PRIVATE'],photo,zones)
            selected_total+=len(guards);selected_hits+=sum(x['fiducial_mark_zone_overlap'] for x in guards)
            selected_arm+=sum(x['fiducial_arm_lookalike'] for x in guards)
        for frame in fullgrid['frames']:
            if frame['photo']!=photo:continue
            guards=guard_proposals(frame['candidates_PRIVATE'],(frame['x_private'],frame['y_private']),photo,zones)
            full_total+=len(guards);full_hits+=sum(x['fiducial_mark_zone_overlap'] for x in guards)
            full_arm+=sum(x['fiducial_arm_lookalike'] for x in guards)
    if len(rows)!=15:raise AssertionError('Expected 15 supplied fiducial keypoints on 2 source photos')
    graph_public={}
    if graph_manifest is not None:
        graphs=json.loads(graph_manifest.read_text())['graphs']
        selected_by_id={t['crop_id']:t for t in selected['tiles']}
        details=[]
        for g in graphs:
            tile=selected_by_id[g['crop_id']]
            if g['photo_id']!=tile['photo_id']:raise ValueError('Graph/photo source mismatch')
            annotation=guard_graph(g,tile['native_origin_xy_PRIVATE'],g['photo_id'],all_zones)
            details.append({'crop_id':g['crop_id'],'photo':g['photo_id'],
                            'machine_graph_risk_annotation_PRIVATE':annotation})
        sums={k:sum(gg['machine_graph_risk_annotation_PRIVATE'][k] for gg in details)
              for k in ('guarded_node_count','fiducial_risk_edges',
                        'fiducial_risk_among_previously_ranked_edges',
                        'unknown_schema_abstain_edges','preliminary_unambiguous_S39_edges')}
        graph_public={'total_graphs_checked':len(graphs),'graph_nodes':sum(len(g['nodes']) for g in graphs),
                      'candidate_relation_edges':sum(len(g['candidate_edges']) for g in graphs),
                      **sums,'automatic_physical_joins_created':0}
        (out_dir/'S40_GRAPH_FIDUCIAL_GUARDS_PRIVATE.json').write_text(
            json.dumps({'schema':'s40_graph_fiducial_guard_v1','graphs':details},ensure_ascii=False)+'\n')
    report={'schema':'s40_fiducial_machine_audit_public_v1','date':'2026-10-25',
        'physical_source_count':2,'known_supplied_fiducial_marks':len(rows),
        'supplied_source_labels_other_than_fiducials_not_treated_as_negative':True,
        'candidate_total_in_fiducial_diagnostic_crops':sum(z['candidate_count'] for z in rows),
        'candidate_flagged_near_fiducials_diagnostic_crops':sum(z['candidate_touches_known_fiducial_zone'] for z in rows),
        'machine_inferred_X_marker_pair_count':sum(z['estimated_fiducial_X_arms']==2 for z in rows),
        'candidate_fiducial_arm_lookalikes_diagnostic_crops':diag_arm,
        'candidate_any_fiducial_risk_diagnostic_crops':sum(z['candidate_any_fiducial_risk'] for z in rows),
        'selected_39_candidate_total':selected_total,'selected_39_fiducial_proximity_flagged':selected_hits,
        'selected_39_fiducial_arm_lookalikes':selected_arm,
        'unbiased_grid_128_candidate_total':full_total,'unbiased_grid_128_fiducial_proximity_flagged':full_hits,
        'unbiased_grid_128_fiducial_arm_lookalikes':full_arm,
        'fiducial_zone_ridge_activation_mean':round(float(np.mean([z['ridge_activation_in_marker_zone'] for z in rows])),5),
        'reference_annulus_ridge_activation_mean':round(float(np.mean([z['ridge_activation_in_outer_reference_annulus'] for z in rows])),5),
        'sample_rows_without_source_coordinates':rows,
        'auto_particle_links_created':0,'native_expert_particle_GT_labels':0,
        'graph_fiducial_guards':graph_public,
        'conclusion':'Only flags ambiguity near known calibration marks. No native particle GT, no mass/3D claims.'}
    (out_dir/'S40_FIDUCIAL_MACHINE_REPORT_PUBLIC.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out_dir/'S40_FIDUCIAL_CANDIDATES_PRIVATE.json').write_text(json.dumps({
        'fiducials_PRIVATE':all_zones,'source_regions_PRIVATE':region_records},ensure_ascii=False)+'\n')
    return report


def cli():
    a=argparse.ArgumentParser()
    a.add_argument('--labels-json',type=Path,required=True)
    a.add_argument('--bmp-dir',type=Path,required=True)
    a.add_argument('--s39-selected',type=Path,required=True)
    a.add_argument('--s39-fullgrid',type=Path,required=True)
    a.add_argument('--private-out-dir',type=Path,required=True)
    a.add_argument('--s39-graph',type=Path,default=None)
    z=a.parse_args()
    print(json.dumps(run(z.labels_json,z.bmp_dir,z.s39_selected,z.s39_fullgrid,z.private_out_dir,z.s39_graph),ensure_ascii=False,indent=2))
if __name__=='__main__':cli()
