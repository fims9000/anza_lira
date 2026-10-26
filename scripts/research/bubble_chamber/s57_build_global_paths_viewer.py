#!/usr/bin/env python3
"""Offline private S57 demonstration: S56 path suggestions + existing risk alerts.

Copies only already-private photographic imagery into private output. The PUBLIC
repo contains only this template builder, never per-photo pixel coordinates.
"""
from __future__ import annotations
import argparse
import json
from hashlib import sha256
from pathlib import Path
import shutil
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s52_build_machine_showcase import join_advisories

TEMPLATE=ROOT/'scripts/research/bubble_chamber/s49_offline_template.html'
SCHEMA='s57_machine_only_global_paths_PRIVATE_v1'


def _replace_exact(content,source,replacement):
    if content.count(source)!=1:
        raise ValueError(f'Unexpected template change: {source[:75]}')
    return content.replace(source,replacement)


def assemble(s49:dict,s51:dict,s56:dict)->tuple[dict,dict]:
    combined=join_advisories(s49,s51)
    graphs=s56.get('graphs')
    if (s56.get('data_NOT_expert_GT') is not True or
        s56.get('physics_auto_joins_authorized') is not False or
        not isinstance(graphs,list) or len(graphs)!=39):
        raise ValueError('Only audited private machine-only S56 input supported')
    counts={'machine_nodes':0,'original_edges':0,'global_synthetic_calibrated_suggestions':0,
            'global_with_previous_warning':0,'S51_warning_edges':0,'global_new_warning_overlap':0}
    for t,g in zip(combined['tiles'],graphs):
        if t['crop_id']!=g['crop_id_PRIVATE'] or t['photo']!=g['photo_id']:
            raise ValueError('Private graph/photo pairing changed')
        chosen={tuple(pair) for pair in g['graph_s56_provisional_edges_NOT_PHYSICS_GT']}
        if len(chosen)!=len(g['graph_s56_provisional_edges_NOT_PHYSICS_GT']):
            raise ValueError('Duplicated S56 suggestion')
        found=set()
        for edge in t['graph_edges']:
            ident=(edge['i'],edge['j'])
            edge['S56_PROVISIONAL_NOT_PHYSICS_GT']=ident in chosen
            edge['native_relation_truth']='UNKNOWN'
            edge['physical_auto_join_authorized']=False
            if ident in chosen:
                found.add(ident)
                counts['global_synthetic_calibrated_suggestions']+=1
                counts['global_with_previous_warning']+=int(edge['risky'])
                counts['global_new_warning_overlap']+=int(edge['s51_third_line_overlap_advisory'])
            counts['original_edges']+=1
            counts['S51_warning_edges']+=int(edge['s51_third_line_overlap_advisory'])
        if found!=chosen:
            raise ValueError('Selected link absent from original S39 graph')
    if (counts['original_edges']!=182 or counts['global_synthetic_calibrated_suggestions']!=37 or
        combined['summary']['risky_edges']!=76 or combined['summary']['machine_candidates']!=567):
        raise ValueError('Unexpected canonical native candidates or results')
    combined['schema']=SCHEMA
    combined['s56_global_path_proposals_NOT_NATIVE_PHYSICS_GT']=counts['global_synthetic_calibrated_suggestions']
    combined['native_GT_count']=0
    combined['physical_auto_links']=0
    combined['any_trained_relation_head']=False
    combined['all_labels_unknown_for_native_H50Z']=True
    combined['summary']['s56_advisory_links']=counts['global_synthetic_calibrated_suggestions']
    return combined,counts


def build_template():
    s=TEMPLATE.read_text(encoding='utf-8')
    replacements={
       'S35-S49 · исходные H50Z':'S35–S57 · исходные H50Z',
       "metric('Соединений с предупреждением',data.summary.risky_edges)":
         "metric('Соединений с предупреждением',data.summary.risky_edges),metric('Стыков S56 (не GT)',data.summary.s56_advisory_links)",
       '<label><input id="showGraphRiskOnly" type="checkbox"> Только рискованные связи</label>':
         '<label><input id="showGraphRiskOnly" type="checkbox"> Только рискованные связи</label>\n  <label><input id="showGlobalOnly" type="checkbox" checked> Глобальные гипотезы S56</label>\n  <label><input id="showAllEdges" type="checkbox"> Все предложения S39</label>',
       '<span style="--c:#85bdff">Предлагаемая связь, не физический трек</span>':
         '<span style="--c:#85bdff">Предлагаемая связь, не физический трек</span>\n<span style="--c:#d6a7ff">Выбрана S56 только машинным алгоритмом</span>',
       "if(el('showGraphRiskOnly').checked&&!e.risky)continue;drawPolyline(ctx,e.bridge,e.risky?'#ff6b90':'#85bdff',2.0,[5,4])":
         "if(el('showGraphRiskOnly').checked&&!e.risky)continue; if(el('showGlobalOnly').checked&&!el('showAllEdges').checked&&!e.S56_PROVISIONAL_NOT_PHYSICS_GT)continue; drawPolyline(ctx,e.bridge,e.S56_PROVISIONAL_NOT_PHYSICS_GT?(e.risky?'#ff6b90':'#d6a7ff'):(e.risky?'#ff6b90':'#85bdff'),e.S56_PROVISIONAL_NOT_PHYSICS_GT?3.0:1.6,[5,4])",
       'рискованных соединений: <b>${t.graph_edges.filter(x=>x.risky).length}</b>.':
         'рискованных соединений: <b>${t.graph_edges.filter(x=>x.risky).length}</b>; машинных гипотез S56: <b>${t.graph_edges.filter(x=>x.S56_PROVISIONAL_NOT_PHYSICS_GT).length}</b>.',
       "['showCandidates','onlyWarnings','showGraph','showGraphRiskOnly']":
         "['showCandidates','onlyWarnings','showGraph','showGraphRiskOnly','showGlobalOnly','showAllEdges']",
       "schema:'s49_machine_unverified_export'":"schema:'s57_machine_unverified_export'",
       "a.download='S49_MACHINE_HYPOTHESES_NOT_PHYSICS_GT.json'":
         "a.download='S57_MACHINE_HYPOTHESES_NOT_PHYSICS_GT.json'",
    }
    for a,b in replacements.items():
        s=_replace_exact(s,a,b)
    banner='''
<section class="notice"><h2>S56 — построение полных цепочек, только искусственный контроль</h2>
<p>Два трека по три отрезка и один ложный кандидат. На 256 синтетических тестовых сценах:
локальный S39: 88 правильных / 15 неправильных стыков, 0 полных цепочек;
жадный портовый: 441 / 36, 129 полных цепочек; глобальный S56: 246 / 13,
66 полных цепочек. Это НЕ доля правильно восстановленных реальных частиц.</p>
<p>Фиолетовый пунктир: 37 консервативно выбранных *машинных гипотез* из исходных 182
связей H50Z. Никаких реальных физических соединений не утверждается. Существующие
реперные, фотометрические и топологические предупреждения НЕ снимаются.</p>
<p class="muted">Числа и границы доверия в `S56_PUBLIC_GLOBAL_PATH_RESULTS.json`.</p></section>
'''
    s=_replace_exact(s,'<section id="details" aria-live="polite"></section>',
                      '<section id="details" aria-live="polite"></section>'+banner)
    s=_replace_exact(s, 'grid-template-columns:repeat(4,minmax(130px,1fr))',
                     'grid-template-columns:repeat(5,minmax(130px,1fr))')
    if s.count('__INLINE_PAYLOAD__')!=1:
        raise ValueError('Missing inline private data token')
    return s


def run(s49_path:Path,s51_path:Path,s56_path:Path,s49_images:Path,
        s56_public:Path,out_dir:Path,out_zip:Path)->dict:
    s49=json.loads(s49_path.read_text())
    s51=json.loads(s51_path.read_text())
    s56=json.loads(s56_path.read_text())
    data,counts=assemble(s49,s51,s56)
    out_dir.mkdir(parents=True,exist_ok=True)
    images=out_dir/'images';images.mkdir(exist_ok=True)
    copied=[]
    for t in data['tiles']:
        for k in ('source_image','other_image'):
            relative=Path(t[k])
            if (relative.parent!=Path('images') or not relative.name.startswith('S49_')
                or relative.suffix!='.jpg'):
                raise ValueError('Unexpected private asset locator')
            src=s49_images/relative.name
            if not src.is_file():raise ValueError('Missing genuine H50Z review asset')
            shutil.copyfile(src,images/relative.name)
            copied.append(relative.name)
    if len(set(copied))!=78:raise ValueError('Missing photo comparison panels')
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    html=build_template().replace('__INLINE_PAYLOAD__',payload)
    if any(x in html for x in ('__INLINE_PAYLOAD__','fetch(', 'https://','http://')):
        raise ValueError('Viewer must work fully offline')
    (out_dir/'index.html').write_text(html,encoding='utf-8')
    shutil.copyfile(s56_public,out_dir/'S56_PUBLIC_GLOBAL_PATH_RESULTS.json')
    (out_dir/'README_FIRST_RU.txt').write_text(
      'ANZA–LIRA S56/S57. ПРИВАТНЫЙ машинный просмотр реальных снимков H50Z.\n'
      'Распакуйте архив целиком и откройте index.html. Размечать ничего не нужно.\n'
      'Все 567 линий и 182 возможные связи сохранены. Глобальный алгоритм выделил 37\n'
      'не подтверждённых физически гипотез; они не являются восстановленными треками.\n'
      'Не публиковать исходные фотографии, координаты и JSON в публичных репозиториях.\n',
      encoding='utf-8')
    manifest={str(p.relative_to(out_dir)):sha256(p.read_bytes()).hexdigest()
              for p in sorted(out_dir.rglob('*')) if p.is_file()}
    (out_dir/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    out_zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out_dir.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out_dir.parent))
    with zipfile.ZipFile(out_zip) as z:
        if z.testzip() is not None:raise RuntimeError('Bad ZIP member checksum')
    return {'zip_file':str(out_zip),'zip_bytes':out_zip.stat().st_size,
            'private_H50Z_images':len(set(copied)),'tiles':len(data['tiles']),
            'native_lines':data['summary']['machine_candidates'],
            'native_candidate_links':data['summary']['graph_edges'],
            'S56_provisional_links':counts['global_synthetic_calibrated_suggestions'],
            'S56_overlap_prior_warning':counts['global_with_previous_warning'],
            'native_physics_join_count':0}


def main():
    p=argparse.ArgumentParser()
    for arg in ('s49-private','s51-private','s56-private','s49-images','s56-public','out-dir','out-zip'):
        p.add_argument('--'+arg,type=Path,required=True)
    x=p.parse_args()
    print(json.dumps(run(x.s49_private,x.s51_private,x.s56_private,x.s49_images,
                         x.s56_public,x.out_dir,x.out_zip),indent=2))
if __name__=='__main__':main()
