#!/usr/bin/env python3
"""S51 reversible bridge/third-line ambiguity on native S39 + synthetic injection.

Private per-edge image coordinates are never written to public Git.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np

from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_simulated_gaps import extract_track_templates
from bubble_chamber.s35_fragment_link import optimal_endpoints, hermite_gap
from bubble_chamber.s51_topology_collision import (full_graph_diagnostic,collision_advisory,RADIUS_PX)
from scripts.research.bubble_chamber.s37_benchmark_review_risk import split_source_frames
from scripts.research.bubble_chamber.s50_benchmark_curvature_continuation import generate_cases

SCHEMA='s51_native_topology_uncertainty_and_synthetic_controls_v1'
SEEDS={'006': (20265101,20265102),'005':(20265111,20265112)}


def synthetic_scene_bridge_collisions(scene) -> dict:
    # Same length-matched candidate fragments as frozen S50 benchmark.
    from bubble_chamber.s50_curvature_continuation import matched_candidates
    left,right=matched_candidates(scene.left,scene.right)
    nodes=[{'points_xy':x.tolist()} for x in left+right]
    yes=no=wrong_decoy=0
    rows=[]
    for i,a in enumerate(left):
        for j,b in enumerate(right):
            connection=optimal_endpoints(a,b)
            path=hermite_gap(connection['endpoint_a'],connection['endpoint_b'],
                             connection['outward_a'],connection['outward_b'],n=21)
            edge={'i':i,'j':j+len(left),'hermite_bridge_pixels_PROPOSAL_ONLY':path.tolist(),
                  'physics_join_authorized':False}
            flag=collision_advisory(nodes,edge)['potential_third_line_overlap']
            truth=(i,j) in scene.correct_pairs
            rows.append({'true_SYNTHETIC_ONLY':truth,'flag':bool(flag),
                         'false_decoy':j==2})
    return {'rows':rows,'scenario':scene.scenario}


def synthetic_control(source:Path,tiles:Path,*,test_n=40)->dict:
    from bubble_chamber.s35_simulated_gaps import make_scene
    m=audited_large_manifest(source)
    train,cal,test,digest=split_source_frames(m)
    templates=extract_track_templates(m,chosen_frames=test)
    outcomes={}
    for photo in ('006','005'):
        paths=sorted(tiles.glob(f'H50Z_030_01_{photo}_*.png'))
        expected=18 if photo=='006' else 21
        if len(paths)!=expected:raise ValueError('Wrong native H50Z crop bank')
        for seed in SEEDS[photo]:
            rng=np.random.default_rng(seed)
            collected=[]
            for n in range(test_n):
                tile=paths[int(rng.integers(len(paths)))]
                trial=make_scene(tile,templates,seed=int(rng.integers(1,2**31-1)),
                    scenario=('parallel','crossing','scratched','empty_gap')[n%4])
                assert set(trial.source_frame_ids).issubset(test)
                collected.append(synthetic_scene_bridge_collisions(trial))
            ctr=defaultdict(lambda:{'true':0,'true_warn':0,'wrong':0,'wrong_warn':0,
                                     'hard_decoy':0,'hard_decoy_warn':0})
            for rec in collected:
                for e in rec['rows']:
                    for key in ('all',rec['scenario']):
                        d=ctr[key]
                        if e['true_SYNTHETIC_ONLY']:
                            d['true']+=1;d['true_warn']+=int(e['flag'])
                        else:
                            d['wrong']+=1;d['wrong_warn']+=int(e['flag'])
                            if e['false_decoy']:
                                d['hard_decoy']+=1;d['hard_decoy_warn']+=int(e['flag'])
            outcomes[f'{photo}_seed_{seed}']={'scenes':test_n,'counts':dict(ctr)}
            a=ctr['all'];print('S51 photo',photo,'seed',seed,'true warnings',a['true_warn'],'/',a['true'],
                              'false warnings',a['wrong_warn'],'/',a['wrong'],flush=True)
    return {'synthetic_scene_seeds':SEEDS,'synthetic_scenes_per_seed':test_n,
            'heldout_visualmetro_source_frames':len(test),
            'source_visualmetro_split_sha256':digest,
            'outcomes':outcomes,'physical_native_GT_count':0}


def native_graph_review(graph_source:Path,s49_private:Path,private_path:Path)->dict:
    g=json.loads(graph_source.read_text(encoding='utf-8'))
    old=json.loads(s49_private.read_text(encoding='utf-8'))
    graphs=g.get('graphs');before=old.get('tiles')
    if (old.get('native_GT_count')!=0 or old.get('physical_auto_links')!=0
            or not isinstance(graphs,list) or len(graphs)!=39 or len(before)!=39):
        raise ValueError('Missing S39 or S49 original untouched machine graph')
    rows=[]
    counts={'original_candidates':0,'original_edges':0,'old_S49_risky_edges':0,
            'S51_2d_third_line_warning':0,'new_S51_warning_outside_S49':0,
            'union_warning':0,'unflagged_UNVERIFIED':0,
            'auto_join_count':0,'expert_verified_track_count':0}
    for gr,tile in zip(graphs,before):
        if gr['crop_id']!=tile['crop_id'] or gr['photo_id']!=tile['photo']:
            raise ValueError('Graph tile changed since S49')
        if len(gr['candidate_edges'])!=len(tile['graph_edges']):
            raise ValueError('Edge count mismatched')
        audit=full_graph_diagnostic(gr)
        counts['original_candidates']+=len(gr['nodes'])
        tile_details=[]
        for old_edge,prior,now in zip(gr['candidate_edges'],tile['graph_edges'],audit['private_per_edge_advisories']):
            if ((old_edge['i'],old_edge['j'])!=(prior['i'],prior['j'])
                    or (now['i'],now['j'])!=(prior['i'],prior['j'])
                    or not isinstance(prior['risky'],bool)):
                raise ValueError('Native graph relation not source-aligned')
            before_risk=prior['risky']
            new_risk=now['potential_third_line_overlap']
            counts['original_edges']+=1
            counts['old_S49_risky_edges']+=int(before_risk)
            counts['S51_2d_third_line_warning']+=int(new_risk)
            counts['new_S51_warning_outside_S49']+=int(new_risk and not before_risk)
            counts['union_warning']+=int(new_risk or before_risk)
            counts['unflagged_UNVERIFIED']+=int(not (new_risk or before_risk))
            tile_details.append({'i':now['i'],'j':now['j'],
                'collision_warning':new_risk,'old_S49_warning':before_risk,
                'union_warning_UNVERIFIED':bool(new_risk or before_risk),
                'number_of_other_nodes':now['third_line_candidate_count'],
                'min_distance_px_PRIVATE':now['minimum_other_line_distance_px_PRIVATE'],
                'other_nodes_PRIVATE':now['other_line_node_indices_PRIVATE'],
                'native_track_relation_truth':'UNKNOWN','physics_auto_join_authorized':False})
        rows.append({'crop_id':gr['crop_id'],'photo':gr['photo_id'],
                     'edges_PRIVATE':tile_details})
    if counts['original_edges']!=182 or counts['old_S49_risky_edges']!=58:
        raise ValueError('S49 frozen graph/risk totals do not match')
    private_path.parent.mkdir(parents=True,exist_ok=True)
    private_path.write_text(json.dumps({'schema':SCHEMA,'details_PRIVATE':rows,
                     'native_no_physics_gt':True},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return counts


def run(args):
    counts=native_graph_review(args.s39_graph,args.s49_private,args.private_out)
    controlled=synthetic_control(args.visualmetro,args.photo_tiles,test_n=args.test_scenes)
    report={'schema':SCHEMA,'source_graph_schema':'s39_geometry_link_graph_proposals_v1',
            '3d_reconstruction_performed':False,'native_GT_available':False,
            '2d_crossing_can_be_real_physical_tracks':True,
            'bridges_with_warnings_NOT_false_join_truth':True,
            'S51_radius_px_FROZEN':RADIUS_PX,
            'native_machine_graph':counts,'new_synthetic_controls':controlled}
    args.public_out.parent.mkdir(parents=True,exist_ok=True)
    args.public_out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    for f in ('s39-graph','s49-private','visualmetro','photo-tiles','public-out','private-out'):
        p.add_argument('--'+f,type=Path,required=True)
    p.add_argument('--test-scenes',type=int,default=40)
    a=p.parse_args()
    result=run(a)
    print('S51 native graph summary',result['native_machine_graph'],flush=True)

if __name__=='__main__':main()
