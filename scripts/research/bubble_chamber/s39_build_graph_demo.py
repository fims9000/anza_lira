#!/usr/bin/env python3
"""Private Graph-LIRA hypothetical join renderings for all 39 real H50Z tiles."""
from pathlib import Path
from collections import Counter
import cv2,json,argparse,sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s39_graph_proposals import link_graph,render_graph


def run(source:Path,private_crops:Path,out:Path):
    z=json.loads(source.read_text())
    if z['schema']!='s39_machine_only_photo_hypotheses_v1' or z['native_expert_gt_created']!=0:
        raise ValueError('Input must be S39 machine-only')
    out.mkdir(parents=True,exist_ok=True)
    details=[];aggregate=Counter()
    for tile in z['tiles']:
        graph=link_graph(tile['proposals'],max_nodes=20)
        aggregate['tiles']+=1;aggregate['graph_nodes']+=len(graph['nodes'])
        aggregate['possible_endpoint_links']+=len(graph['candidate_edges'])
        aggregate['unambiguous_machine_suggestions']+=len(graph['unambiguous_ranked_edges_UNVERIFIED'])
        aggregate['competing_ambiguous_links']+=graph['ambiguous_edges_abstained']
        aggregate['conflicting_links_abstained']+=graph['conflicting_edges_abstained']
        details.append({'crop_id':tile['crop_id'],'photo_id':tile['photo_id'],**graph})
        gray=cv2.imread(str(private_crops/(tile['crop_id']+'.png')),cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape!=(512,512):raise ValueError('Private photo crop mismatch')
        (out/'private_graph_overlays').mkdir(exist_ok=True)
        cv2.imwrite(str(out/'private_graph_overlays'/f'{tile["crop_id"]}_machine_graph.png'),render_graph(gray,graph))
    (out/'S39_GRAPH_RELATIONS_PRIVATE.json').write_text(json.dumps({'graphs':details,'status':'MACHINE_ONLY_NEVER_GROUND_TRUTH'},ensure_ascii=False,indent=2)+'\n')
    summary={'schema':'s39_graph_lira_candidate_audit_public_v1',
             'algorithm':'local geometry link graph only; not CT28 trained Graph-LIRA head',
             'native_verified_joins':0,
             'automatic_physics_join_count':0,
             'original_S31_pairs_remaining_unverified':28,
             'all_links_status':'REVERSIBLE_HYPOTHESES_NOT_SAME_PARTICLE_LABELS',
             'totals':dict(aggregate)}
    (out/'S39_GRAPH_SUMMARY_PUBLIC.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--machine-manifest',type=Path,required=True)
    p.add_argument('--private-photo-crops',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.machine_manifest,a.private_photo_crops,a.out_dir),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
