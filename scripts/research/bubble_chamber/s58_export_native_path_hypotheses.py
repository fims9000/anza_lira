#!/usr/bin/env python3
"""S58 private machine path component export; no real-physics GT is created."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from bubble_chamber.s58_assemble_path_hypotheses import assemble_components


def run(s39:Path,s56:Path,tiles:Path,out_private:Path,out_public:Path,preview:Path):
    d=json.loads(s39.read_text());e=json.loads(s56.read_text())
    if (d.get('status')!='MACHINE_ONLY_NEVER_GROUND_TRUTH' or
        e.get('data_NOT_expert_GT') is not True or e.get('physics_auto_joins_authorized') is not False or
        len(d.get('graphs',[]))!=39 or len(e.get('graphs',[]))!=39):
        raise ValueError('Expected locked S39/S56 private source manifests')
    out=[];sizes=Counter(); joined=0
    biggest=None
    for original,new in zip(d['graphs'],e['graphs']):
        if original['crop_id']!=new['crop_id_PRIVATE'] or original['photo_id']!=new['photo_id']:
            raise ValueError('Unaligned graph tile IDs')
        comps=assemble_components(original['nodes'],new['graph_s56_provisional_edges_NOT_PHYSICS_GT'])
        joined+=sum(c['join_count'] for c in comps)
        sizes.update(c['node_count'] for c in comps)
        out.append({'crop_id_PRIVATE':original['crop_id'],'components_PRIVATE':comps,
                    'all_unselected_nodes_retained_in_original_graph':True,
                    'native_relation_truth':'UNKNOWN'})
        for c in comps:
            if biggest is None or c['node_count']>biggest[1]['node_count']:
                biggest=(original,c)
    if joined!=37 or sum(sizes.values())!=35 or sizes.get(4)!=1:
        raise ValueError('Unexpected S56 native graph components; no silent data change')
    private={'schema':'s58_reversible_native_machine_path_components_PRIVATE_v1',
             'native_expert_GT_count':0,'native_physical_auto_join_count':0,
             'user_manual_labeling_required':False,'tiles':out}
    public={'schema':'s58_aggregate_machine_only_open_paths_public_v1',
            'input_machine_lines':567,'input_graph_nodes_after_nms':511,
            'input_unverified_join_candidates':182,
            'S56_selected_machine_only_joins':joined,
            'connected_machine_path_hypotheses':sum(sizes.values()),
            'component_counts_by_fragment_count':{str(k):v for k,v in sorted(sizes.items())},
            'confirmed_native_physical_tracks':0,'confirmed_native_physical_joins':0,
            'predicted_bridge_pixels_not_observed':True,
            'physics_identity_of_all_candidates':'UNKNOWN'}
    out_private.parent.mkdir(parents=True,exist_ok=True)
    out_public.parent.mkdir(parents=True,exist_ok=True)
    out_private.write_text(json.dumps(private,ensure_ascii=False,indent=2)+'\n')
    out_public.write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n')
    if biggest:
        graph,c=biggest
        path=tiles/(graph['crop_id']+'.png')
        im=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if im is None or im.shape[:2]!=(512,512):raise ValueError('Missing original private H50Z tile')
        for s in c['segments_PRIVATE']:
            pts=np.rint(np.asarray(s['points_xy_PRIVATE'])).astype('int32')
            if s['kind'].startswith('OBSERVED'):
                cv2.polylines(im,[pts],False,(58,220,110),2,cv2.LINE_AA)
            else:
                for i in range(0,len(pts)-1,3):
                    cv2.line(im,tuple(pts[i]),tuple(pts[min(i+1,len(pts)-1)]),(208,90,215),2,cv2.LINE_AA)
        cv2.rectangle(im,(0,476),(512,511),(26,30,43),-1)
        cv2.putText(im,'MACHINE HYPOTHESIS / NOT PHYSICS GT',(7,497),
                    cv2.FONT_HERSHEY_SIMPLEX,.53,(255,240,255),1,cv2.LINE_AA)
        preview.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(preview),im):raise ValueError('Cannot save private preview')
    return public


def main():
    p=argparse.ArgumentParser()
    for k in ('s39-private','s56-private','photo-tiles','out-private','out-public','private-preview'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.s39_private,a.s56_private,a.photo_tiles,a.out_private,
                         a.out_public,a.private_preview),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
