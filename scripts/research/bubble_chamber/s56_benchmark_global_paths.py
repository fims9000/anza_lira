#!/usr/bin/env python3
"""S56 frozen multi-piece machine-only geometry experiment.

Strictly private measured source frames and photo pixels are never written
into public outputs. The test-only source frames cannot enter fit/calibration.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from collections import Counter

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_simulated_gaps import extract_track_templates
from bubble_chamber.s35_fragment_link import GEOMETRY_FIELDS, feature_matrix, calibrated_negative_threshold
from bubble_chamber.s56_three_piece_scenes import make_three_piece_scene,SCENARIOS,TRUE_EDGES
from bubble_chamber.s56_global_path_assembly import (
    pair_rows,scored_edges,select_node_mutual,select_port_greedy,
    select_global_forest,forest_components)
from scripts.research.bubble_chamber.s37_benchmark_review_risk import split_source_frames

SCHEMA='s56_frozen_global_three_piece_simulation_public_v1'
PROTOCOL={
    'train_scenes':64,'cal_scenes':64,'test_scenes_per_seed':64,
    'train_seeds':{'005':20265601,'006':20265611},
    'cal_seeds':{'005':20265602,'006':20265612},
    'test_seeds':{'005':(20265603,20265604),'006':(20265613,20265614)},
    'cal_negative_fpr':.05,'port_or_node_margin':.04,
    'global_objective_stability_margin':.04,
    'bootstrap_scene_iterations':1000,
}
ARMS=('node_mutual','port_greedy','global_forest')


def generate(tiles:Path,photo:str,templates:list[dict],*,n:int,seed:int,preview_dir:Path|None=None):
    photos=sorted(tiles.glob(f'H50Z_030_01_{photo}_*.png'))
    if len(photos)!=(21 if photo=='005' else 18):
        raise ValueError('S56 locked genuine H50Z crop bank not found')
    rng=np.random.default_rng(seed)
    result=[]
    for k in range(n):
        scene=make_three_piece_scene(photos[int(rng.integers(len(photos)))],templates,
                                    seed=int(rng.integers(1,2**31-1)),
                                    scenario=SCENARIOS[k%len(SCENARIOS)])
        rows=pair_rows(scene.nodes)
        for row in rows:
            row['label_SYNTHETIC_ONLY']=int((row['i'],row['j']) in TRUE_EDGES)
        if len(rows)!=21 or sum(x['label_SYNTHETIC_ONLY'] for x in rows)!=4:
            raise AssertionError('Frozen simulator topology corrupted')
        result.append({'rows':rows,'scenario':scene.scenario,
                       '_PRIVATE_source_frames':scene.private_source_frames,
                       '_PRIVATE_scene':scene if preview_dir is not None and k==0 else None})
    return result


def flatten(cases):
    r=[row for c in cases for row in c['rows']]
    return r,np.asarray([p['label_SYNTHETIC_ONLY'] for p in r],int)


def fit(rows,labels):
    features=feature_matrix(rows,GEOMETRY_FIELDS)
    scaler=StandardScaler().fit(features)
    model=LogisticRegression(C=.3,random_state=1337,class_weight='balanced',max_iter=600)
    model.fit(scaler.transform(features),labels)
    def score(x):
        return model.predict_proba(scaler.transform(feature_matrix(x,GEOMETRY_FIELDS)))[:,1]
    return score


def make_selections(edges,threshold):
    margin=PROTOCOL['port_or_node_margin']
    a=select_node_mutual(edges,threshold=threshold,margin=margin)
    b=select_port_greedy(edges,threshold=threshold,margin=margin)
    c,qa=select_global_forest(edges,threshold=threshold,
                             margin=PROTOCOL['global_objective_stability_margin'])
    if not qa['solver_success']:
        raise RuntimeError('Global path optimizer failed closed; report not complete')
    return dict(zip(ARMS,(a,b,c))),qa


def evaluate(cases,scores,cutoff):
    if len(scores)!=21*len(cases):raise ValueError('Need full scene score arrays')
    all_out={a:[] for a in ARMS}
    for k,c in enumerate(cases):
        edges=scored_edges(c['rows'],scores[k*21:(k+1)*21])
        selections,qa=make_selections(edges,cutoff)
        true=TRUE_EDGES
        for name,chosen in selections.items():
            got={(e.i,e.j) for e in chosen}
            false=got-true
            correct=got&true
            full_1={ (0,1),(1,2) }.issubset(got) and all(not ({e.i,e.j}&{0,1,2}) for e in chosen if (e.i,e.j) in false)
            full_2={ (3,4),(4,5) }.issubset(got) and all(not ({e.i,e.j}&{3,4,5}) for e in chosen if (e.i,e.j) in false)
            all_out[name].append({'true':len(correct),'wrong':len(false),'any_wrong':int(bool(false)),
                                  'suggestions':len(got),'full_trajectories':int(full_1)+int(full_2),
                                  'exact_complete_scene':int(len(correct)==4 and not false),
                                  'components':len(forest_components(7,chosen)),
                                  'solver_success':int(qa['solver_success'])})
    return all_out


def totals(records):
    n=len(records)
    v={k:int(sum(r[k] for r in records)) for k in records[0]}
    v.update(n_synthetic_scenes=n,
             synthetic_true_links_total=4*n,
             synthetic_full_paths_total=2*n,
             true_recovery=float(v['true']/(4*n)),
             wrong_joins_per_100_simulated_scenes=float(v['wrong']*100/n),
             synthetic_link_precision=float(v['true']/v['suggestions']) if v['suggestions'] else None,
             physically_verified_native_joins=0)
    return v


def paired_bootstrap(records_a,records_b,*,seed:int,n_boot:int=1000):
    if len(records_a)!=len(records_b) or len(records_a)==0:
        raise ValueError('Paired bootstrap requires corresponding scenes')
    rng=np.random.default_rng(seed)
    outputs={}
    for key in ('true','wrong','full_trajectories','any_wrong'):
        delta=np.asarray([b[key]-a[key] for a,b in zip(records_a,records_b)],float)
        ix=rng.integers(0,len(delta),size=(n_boot,len(delta)))
        samples=delta[ix].mean(axis=1)
        outputs[key]={'mean_delta_per_simulated_scene':float(delta.mean()),
                      'conditional_scene_bootstrap_95pct':[float(v) for v in np.quantile(samples,(.025,.975))]}
    return outputs


def render_train_preview(scene,all_rows,scores,cutoff,out_dir):
    out_dir.mkdir(parents=True,exist_ok=True)
    canvas=cv2.cvtColor(scene.gray,cv2.COLOR_GRAY2BGR)
    colors=((40,170,60),(40,170,60),(40,170,60),(90,150,230),(90,150,230),(90,150,230),(180,90,110))
    for i,p in enumerate(scene.nodes):
        cv2.polylines(canvas,[np.rint(p).astype('int32')],False,colors[i],1,cv2.LINE_AA)
        cv2.putText(canvas,str(i),tuple(np.clip(np.rint(p[0]).astype(int),0,180)),
                    cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1,cv2.LINE_AA)
    edges=scored_edges(all_rows,scores)
    chosen,qa=make_selections(edges,cutoff)
    for e in chosen['global_forest']:
        a,b=scene.nodes[e.i],scene.nodes[e.j]
        pa=a[e.end_i];pb=b[e.end_j]
        xx=np.linspace(pa,pb,13)
        for k in range(0,12,3):
            cv2.line(canvas,tuple(np.rint(xx[k]).astype(int)),tuple(np.rint(xx[k+1]).astype(int)),
                     (230,60,230),1,cv2.LINE_AA)
    cv2.imwrite(str(out_dir/'S56_PRIVATE_TRAIN_SYNTHETIC_GRAPH.png'),canvas)


def native_audit(private_graph:Path,source_models:dict,thresholds:dict,private_out:Path):
    data=json.loads(private_graph.read_text())
    if data.get('status')!='MACHINE_ONLY_NEVER_GROUND_TRUTH' or len(data.get('graphs',[]))!=39:
        raise ValueError('Need exact audited private S39 native machine graph')
    out=[];counts=Counter()
    for g in data['graphs']:
        photo=g['photo_id']
        if photo not in ('005','006'):raise ValueError('Unknown source photo')
        # Freeze transfer: fit and calibrate exclusively on the OPPOSITE photo.
        other='006' if photo=='005' else '005'
        score=source_models[other];threshold=thresholds[other]
        nodes=g['nodes'];rows=[]
        for e in g['candidate_edges']:
            i=int(e['i']);j=int(e['j'])
            # Only evaluate already proposed S39 pairs, not all O(N**2) pairs.
            from bubble_chamber.s35_fragment_link import pair_features,optimal_endpoints
            a=np.asarray(nodes[i]['points_xy'],float);b=np.asarray(nodes[j]['points_xy'],float)
            values=pair_features(a,b,evidence={})
            pick=optimal_endpoints(a,b)
            rows.append({'i':i,'j':j,'end_i':int(pick['endpoint_index_a']),
                         'end_j':int(pick['endpoint_index_b']),
                         **{field:float(values[field]) for field in GEOMETRY_FIELDS}})
        choices=[]
        if rows:
            scores=score(rows);edges=scored_edges(rows,scores)
            selections,qa=make_selections(edges,threshold)
            choices=[[int(e.i),int(e.j)] for e in selections['global_forest']]
            counts['advisory_selected_edges']+=len(choices)
            counts['possible_edges_scored']+=len(rows)
            counts['global_ambiguous_abstentions']+=qa['global_ambiguous_abstentions']
        counts['tiles']+=1;counts['source_nodes']+=len(nodes)
        out.append({'crop_id_PRIVATE':g['crop_id'], 'photo_id':photo,
                    'graph_s56_provisional_edges_NOT_PHYSICS_GT':choices,
                    'native_relation_truth':'UNKNOWN',
                    'original_s39_node_ids_unchanged':True})
    private_out.parent.mkdir(parents=True,exist_ok=True)
    private_out.write_text(json.dumps({'schema':'s56_private_synthetic_calibrated_native_advisory',
        'data_NOT_expert_GT':True,'physics_auto_joins_authorized':False,'graphs':out},indent=2))
    return {'counts':dict(counts),'all_native_labels_unknown':True,
            'native_physical_auto_joins':0,'trained_graph_lira_relation_head':False}


def run(visualmetro:Path,tiles:Path,out:Path,*,private_dir:Path|None=None,
        native_graph:Path|None=None, train_n=64,cal_n=64,test_n=64,bootstrap=1000,
        smoke=False):
    manifest=audited_large_manifest(visualmetro)
    train,cal,test,digest=split_source_frames(manifest)
    templates={k:extract_track_templates(manifest,chosen_frames=frames) for k,frames in
               (('train',train),('cal',cal),('test',test))}
    public={'schema':SCHEMA,'machine_only':True,'original_architectures_unchanged':True,
            'native_verified_trajectories':0,'native_verified_joins':0,
            'operator_source_frames_complete':len(manifest['frames']),
            'source_abstract_manifest_SHA256':manifest['provenance_sha256'],
            'photo_native_backgrounds_one_physical_run':True,
            'source_visualmetro_does_not_match_H50Z_images':True,
            'S37_frozen_frame_split_sha256':digest,
            'source_frame_split_counts':{'train':len(train),'cal':len(cal),'test':len(test)},
            'protocol':{**PROTOCOL, 'effective':{'train':train_n,'cal':cal_n,'test_per_seed':test_n},
                        'smoke_not_confirmatory':smoke},
            'synthetic_GT_not_native_physics':True,'directions':{}}
    fitted={};cutoffs={}
    for photo,target in (('005','006'),('006','005')):
        print(f'S56 {photo}->{target} source generator...',flush=True)
        ts=generate(tiles,photo,templates['train'],n=train_n,
                    seed=PROTOCOL['train_seeds'][photo] if not smoke else 20265691+(photo=='006'),
                    preview_dir=private_dir)
        cs=generate(tiles,photo,templates['cal'],n=cal_n,
                    seed=PROTOCOL['cal_seeds'][photo] if not smoke else 20265693+(photo=='006'))
        assert {f for c in ts for f in c['_PRIVATE_source_frames']}<=train
        assert {f for c in cs for f in c['_PRIVATE_source_frames']}<=cal
        tr,yt=flatten(ts);cr,yc=flatten(cs)
        scorer=fit(tr,yt)
        cutoff=calibrated_negative_threshold(scorer(cr),yc,max_fpr=PROTOCOL['cal_negative_fpr'])
        fitted[photo]=scorer;cutoffs[photo]=cutoff
        direction={'train_photo_id':photo,'heldout_test_photo_id':target,
                   'cal_negative_fpr_policy_only_not_physics':.05,
                   'calibration_cutoff':cutoff,'tests':{}}
        if private_dir is not None:
            if ts[0]['_PRIVATE_scene'] is None:raise RuntimeError('Private preview source missing')
            render_train_preview(ts[0]['_PRIVATE_scene'],ts[0]['rows'],
                                 scorer(ts[0]['rows']),cutoff,private_dir/photo)
        seeds=PROTOCOL['test_seeds'][photo] if not smoke else (20265695+(photo=='006'),)
        for seed in seeds:
            cases=generate(tiles,target,templates['test'],n=test_n,seed=seed)
            assert {f for c in cases for f in c['_PRIVATE_source_frames']}<=test
            rows,y=flatten(cases);scores=scorer(rows)
            selections=evaluate(cases,scores,cutoff)
            summary={'synthetic_pair_AUROC':float(roc_auc_score(y,scores)),
                     'synthetic_pair_AUPRC':float(average_precision_score(y,scores)),
                     'arms':{name:totals(selections[name]) for name in ARMS},
                     'paired_global_minus_node':paired_bootstrap(selections['node_mutual'],
                         selections['global_forest'],seed=seed+310,n_boot=bootstrap),
                     'paired_global_minus_port':paired_bootstrap(selections['port_greedy'],
                         selections['global_forest'],seed=seed+311,n_boot=bootstrap)}
            direction['tests'][str(seed)]=summary
            print(f"S56 {photo}->{target} test {seed}: " +
                  ' '.join(f"{arm} true={summary['arms'][arm]['true']}/"+
                           f"{4*test_n} wrong={summary['arms'][arm]['wrong']}" for arm in ARMS),
                  flush=True)
        public['directions'][f'{photo}_to_{target}']=direction
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(public,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if native_graph is not None:
        if private_dir is None:raise ValueError('Native per-photo coordinates require PRIVATE output')
        public['native_machine_advisory']=native_audit(native_graph,fitted,cutoffs,
            private_dir/'S56_NATIVE_PATH_GRAPH_PRIVATE.json')
        out.write_text(json.dumps(public,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return public


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--public-out',type=Path,required=True)
    p.add_argument('--private-dir',type=Path)
    p.add_argument('--native-graph',type=Path)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    n=6 if args.smoke else 64
    run(args.visualmetro,args.photo_tiles,args.public_out,private_dir=args.private_dir,
        native_graph=args.native_graph,train_n=n,cal_n=n,test_n=n,
        bootstrap=100 if args.smoke else 1000,smoke=args.smoke)
if __name__=='__main__':main()
