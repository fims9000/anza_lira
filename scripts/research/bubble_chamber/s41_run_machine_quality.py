#!/usr/bin/env python3
"""S41: reconcile S39 H50Z machine pseudo-labels using S40 known fiducials.

PRIVATE outputs retain photo pixels and machine coordinates. The public aggregate
contains only counts. No automatic native particle joins or GT assertions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s41_machine_quality import annotate_s41,SCHEMA

EXPECTED_S39_SOURCE_SHA="903cfae7e8742bab203ee12aecfa454b75a71454dbabaee630982557a84b13d7"


def _safe_cid(s: str) -> str:
    if (not s.startswith('H50Z_030_01_') or not s.replace('_','').isalnum()
            or '/' in s or '..' in s):
        raise ValueError('Unsafe H50Z crop id')
    return s


def combined_graph_guard(graph:dict, original_guard:dict, flags:dict[str,dict])->dict:
    """Abstain on known fiducial risk, never gate on photo stability.

    Negative scratch controls showed stability worsens decoy separation.
    Nodes/links remain native UNVERIFIED even when not fiducial-risk.
    """
    nodes=graph['nodes']; edges=graph['candidate_edges']
    prior=original_guard['candidate_edges']
    if len(edges)!=len(prior) or len(nodes)!=len(original_guard['nodes']):
        raise ValueError('S39/S40 graph provenance or ordering mismatch')
    node_checks=[]
    for node in nodes:
        source_ids=node['source_candidate_ids']
        if not source_ids or any(d not in flags for d in source_ids):
            raise ValueError('Missing S39 candidate in collapsed graph node')
        val=[flags[i] for i in source_ids]
        node_checks.append({"all_sources_stable":all(not v['unstable_photo_response'] for v in val),
                            "any_fiducial_risk":any(v['fiducial_risk'] for v in val)})
    selected={(e['i'],e['j']) for e in graph.get('unambiguous_ranked_edges_UNVERIFIED',[])}
    summary=Counter()
    annotations=[]
    for e, p in zip(edges,prior):
        i=e['i'];j=e['j']
        if p.get('i')!=i or p.get('j')!=j or not (0<=i<len(nodes) and 0<=j<len(nodes) and i!=j):
            raise ValueError('S40/S39 edge index or order mismatch')
        fid=bool(p['fiducial_overlap'])
        unstable=not (node_checks[i]['all_sources_stable'] and node_checks[j]['all_sources_stable'])
        status=('KNOWN_FIDUCIAL_RISK_ABSTAIN' if fid else
                'UNVERIFIED_CANDIDATE_NO_AUTOJOIN')
        summary['total_edges']+=1
        summary['fiducial_risk']+=int(fid)
        summary['unstable_node_edges']+=int(unstable)
        summary['stability_warning_advisory_only']+=int(unstable and not fid)
        summary['eligible_machine_only_edges_not_physics_gt']+=int(not fid)
        summary['preselected_machine_edge_abstained']+=int((i,j) in selected and fid)
        annotations.append({'i':i,'j':j,'status':status,'fiducial_risk':fid,
                            'unstable_endpoint_ADVISORY_ONLY':unstable,
                            'physical_same_particle_relation':'UNKNOWN',
                            'auto_join_performed':False})
    return {'edges':annotations,'summary':dict(summary),'native_physics_auto_links':0}


def run(*,s39_private:Path,s40_private:Path,s40_graph_private:Path,
        s39_graph_private:Path,photo_tiles:Path,s39_pseudo_masks:Path,
        output:Path)->dict:
    source=json.loads(s39_private.read_text())
    s40=json.loads(s40_private.read_text())
    guards=json.loads(s40_graph_private.read_text())
    graphs=json.loads(s39_graph_private.read_text())
    if (source.get('source_file_sha256')!=EXPECTED_S39_SOURCE_SHA
            or source.get('native_expert_gt_created')!=0 or len(source['tiles'])!=39):
        raise ValueError('Wrong S39 provenance / number of crops')
    zones=s40['fiducials_PRIVATE']
    if len(zones)!=15 or any(len(z.get('arms_angles_rad_PRIVATE',[]))!=2 for z in zones):
        raise ValueError('S40 fiducials must have 15 inferred two-arm crosses')
    graph_by_id={g['crop_id']:g for g in graphs['graphs']}
    guard_by_id={g['crop_id']:g['machine_graph_risk_annotation_PRIVATE'] for g in guards['graphs']}
    if len(graph_by_id)!=39 or len(guard_by_id)!=39:
        raise ValueError('Incomplete graph guard manifest')
    output.mkdir(parents=True,exist_ok=True)
    folder=(output/'S41_PRIVATE_MACHINE_DEMO');folder.mkdir(exist_ok=True)
    entries=[];private=[];aggregates=Counter();graph_sum=Counter();old_totals=Counter();new_totals=Counter()
    for ti,tile in enumerate(source['tiles']):
        cid=_safe_cid(tile['crop_id']);photo=tile['photo_id']
        image_path=photo_tiles/f'{cid}.png'
        with image_path.open('rb') as f:raw=f.read()
        original=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        if original is None or original.shape!=(512,512) or sha256(original.tobytes()).hexdigest()!=tile['source_gray_sha256']:
            raise ValueError('Private H50Z crop pixel hash mismatch')
        old_path=s39_pseudo_masks/f'{cid}_PSEUDO_NOT_GT.png'
        old=cv2.imread(str(old_path),cv2.IMREAD_GRAYSCALE)
        if old is None or old.shape!=original.shape:raise ValueError('Missing S39 pseudo-mask')
        result=annotate_s41(original,old,tile['proposals'],zones,
                            tile['native_origin_xy_PRIVATE'],photo)
        refined=result['refined_mask_PRIVATE']; protected=result['protected_mask_PRIVATE']
        if set(np.unique(refined))-{0,1,255}:
            raise AssertionError('Unexpected pseudo-mask class')
        cv2.imwrite(str(folder/f'{cid}_S41_PSEUDO_NOT_GT.png'),refined)
        color=cv2.cvtColor(original,cv2.COLOR_GRAY2BGR)
        for proposal, flag in zip(tile['proposals'],result['candidate_flags_PRIVATE']):
            coords=np.rint(np.asarray(proposal['points_xy'])).astype(np.int32)
            marker=(30,45,240) if flag['fiducial_risk'] else (
                (20,180,240) if flag['unstable_photo_response'] else (90,205,85))
            cv2.polylines(color,[coords],False,marker,2,cv2.LINE_AA)
        cv2.imwrite(str(folder/f'{cid}_S41_OVERLAY.png'),color)
        view=cv2.cvtColor(original,cv2.COLOR_GRAY2BGR)
        view[refined==1]=(80,205,65)
        view[refined==0]=(165,50,210)
        # Unknown is original grayscale: no fictitious negative information.
        cv2.imwrite(str(folder/f'{cid}_S41_MASK_VIEW.png'),view)
        support_map={p['draft_id']:f for p,f in zip(tile['proposals'],result['candidate_flags_PRIVATE'])}
        if len(support_map)!=len(tile['proposals']):raise ValueError('Duplicate S39 draft identifiers')
        graph_guard=combined_graph_guard(graph_by_id[cid],guard_by_id[cid],support_map)
        for k,v in graph_guard['summary'].items():graph_sum[k]+=v
        flags=result['candidate_flags_PRIVATE']
        aggregates['tiles']+=1
        aggregates['lines']+=len(flags)
        aggregates['stable_lines']+=result['machine_stable_lines_UNVERIFIED']
        aggregates['fiducial_risk_lines']+=result['fiducial_risk_lines']
        aggregates['uncertain_lines']+=result['uncertain_lines']
        aggregates['unstable_lines']+=sum(f['unstable_photo_response'] for f in flags)
        for k,v in result['changes']['before'].items():old_totals[k]+=v
        for k,v in result['changes']['after'].items():new_totals[k]+=v
        aggregates['protected_pixel_positions']+=result['changes']['protected_image_pixels']
        aggregates['positive_demoted_to_unknown']+=result['changes']['positive_demoted_to_unknown']
        aggregates['negative_demoted_to_unknown']+=result['changes']['negative_demoted_to_unknown']
        entries.append({'cid':cid,'photo':photo,'lines':len(flags),'stable':result['machine_stable_lines_UNVERIFIED'],
                        'risk':result['uncertain_lines'],'pos_before':result['changes']['before']['positive'],
                        'pos_after':result['changes']['after']['positive']})
        private.append({'crop_id':cid,'photo':photo,'source_image_sha256':tile['source_gray_sha256'],
                        'origin_xy_PRIVATE':tile['native_origin_xy_PRIVATE'],
                        'candidate_quality_NOT_GT':[
                            {'id':p['draft_id'],'stability':q,'flags':r}
                            for p,q,r in zip(tile['proposals'],result['candidate_support_PRIVATE'],flags)],
                        'graph_guard':graph_guard,'summary':result['changes']})
        print(f'S41 {ti+1:02d}/39 {cid} {result["machine_stable_lines_UNVERIFIED"]}/{len(flags)} stable candidates',flush=True)
    output_private={'schema':SCHEMA,'tiles':private,'model_has_native_physics_GT':False,
                    'native_verified_particle_links':0}
    (output/'S41_MACHINE_QUALITY_PRIVATE.json').write_text(json.dumps(output_private,ensure_ascii=False,indent=2)+'\n')
    public={'schema':SCHEMA,'source_photographs':2,'independent_physical_runs':1,
            'native_particle_GT_line_count':0,'validated_native_particle_links':0,
            'pseudo_labels_remain_machine_hypotheses':True,
            'original_machines_unchanged':True,'photos_annotated':39,
            'candidate_quality_totals':dict(aggregates),
            'old_pseudo_pixel_counts':dict(old_totals),'new_pseudo_pixel_counts':dict(new_totals),
            'graph_guard_totals':dict(graph_sum),
            'fixed_perturbations':['identity','gamma_0p78','gamma_1p30','blur_sigma_1p1'],
            'stability_gate':'baseline >=0.35 and min of all 4 supports >=0.25, 3.5 percent ridge activation, 2px tolerance',
            'no_physics_accuracy_claim':True,
            'limitation':'Stability rejects exposure sensitivity, NOT necessarily scratches or scanner artifacts.'}
    (output/'S41_PUBLIC_AGGREGATES.json').write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n')
    _write_html(folder,entries)
    return public


def _write_html(folder:Path,entries:list[dict]):
    cards=[]
    for d in entries:
        cid=html.escape(d['cid'])
        cards.append(f'''<article><h3>{cid}</h3>
        <p>Machine lines: {d['lines']} · stable: {d['stable']} · uncertain: {d['risk']} · pseudo-positive px: {d['pos_before']} → {d['pos_after']}</p>
        <div class="photos"><figure><img loading="lazy" src="{cid}_S41_OVERLAY.png" alt="Machine hypotheses"><figcaption>Machine lines (green stable / amber unstable / red fiducial overlap)</figcaption></figure>
        <figure><img loading="lazy" src="{cid}_S41_MASK_VIEW.png" alt="Machine mask"><figcaption>Green machine-positive / purple assumed-negative / gray unknown</figcaption></figure></div>
        </article>''')
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>ANZA–LIRA S41 / machine-only</title>
    <style>body{font:16px system-ui,sans-serif;background:#101825;color:#edf2ff;margin:0;padding:2rem}h1{margin-top:0}p{color:#bac9dc;line-height:1.4}small{color:#bed0d8}article{background:#1c293b;border:1px solid #405069;border-radius:13px;padding:1rem;margin:1.25rem 0}.photos{display:flex;flex-wrap:wrap;gap:1rem}figure{margin:0;max-width:480px}img{width:min(440px,100%);height:auto;border:1px solid #536780;border-radius:9px}figcaption{color:#bdcbdf}details{margin:1rem 0}summary{cursor:pointer}</style>
    <h1>ANZA–LIRA S41 — машинная проверка гипотез</h1>
    <p><b>ВНИМАНИЕ:</b> это НЕ физическая разметка, НЕ обнаруженные частицы и НЕ 3D-реконструкция. Две исходные фотографии принадлежат одному физическому запуску. Ни одна гипотеза соединения не подтверждена автоматически. Пользовательская разметка не нужна.</p>
    <p>Красные линии пересекают установленные реперные кресты, жёлтые нестабильны к контролируемым изменениям фотометрии, зелёные выдержали эвристическую проверку, но всё ещё могут быть царапинами. Фотометрический тест НЕ применяется как фильтр разметки: на искусственных царапинах он ухудшил AUC. В маске зелёный — предварительный 1, фиолетовый — предположительный 0, серый/исходный фон — неизвестно (255). Неизвестное никогда не преобразуется в отрицательный класс.</p>
    <details open><summary>39 реальных участков H50Z; машинная псевдоразметка S41</summary>
    '''+'\n'.join(cards)+'</details></html>'
    (folder/'index.html').write_text(page,encoding='utf8')


def cli():
    p=argparse.ArgumentParser()
    for s in ('s39-private','s40-private','s40-graph-private','s39-graph-private','photo-tiles','s39-pseudo-masks','output'):
        p.add_argument('--'+s,required=True,type=Path)
    a=p.parse_args()
    x=run(s39_private=a.s39_private,s40_private=a.s40_private,
          s40_graph_private=a.s40_graph_private,s39_graph_private=a.s39_graph_private,
          photo_tiles=a.photo_tiles,s39_pseudo_masks=a.s39_pseudo_masks,output=a.output)
    print(json.dumps(x,ensure_ascii=False,indent=2))


if __name__=='__main__':cli()
