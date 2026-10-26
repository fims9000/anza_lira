#!/usr/bin/env python3
"""S52 offline S49+S51 reversible machine-only viewer.

Reads private source proposals and augments them without changing any original
line or edge coordinate. The output is private: it contains native H50Z pixels.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

TEMPLATE=(Path(__file__).resolve().parent/'s49_offline_template.html')
SCHEMA='s52_machine_only_risks_without_native_physics_GT_v1'


def join_advisories(s49:dict,s51:dict)->dict:
    if (s49.get('native_GT_count')!=0 or s49.get('physical_auto_links')!=0
        or s51.get('native_no_physics_gt') is not True
        or not isinstance(s49.get('tiles'),list)
        or not isinstance(s51.get('details_PRIVATE'),list)
        or len(s49['tiles'])!=len(s51['details_PRIVATE'])):
        raise ValueError('Unknown private machine-only source provenance')
    r=json.loads(json.dumps(s49))
    base_edges=0;old_risk=0;new_risk=0;union=0
    for tile,patch in zip(r['tiles'],s51['details_PRIVATE']):
        if tile.get('crop_id')!=patch.get('crop_id') or tile.get('photo')!=patch.get('photo'):
            raise ValueError('Wrong original tile identity')
        edges=tile.get('graph_edges');addon=patch.get('edges_PRIVATE')
        if not isinstance(edges,list) or not isinstance(addon,list) or len(edges)!=len(addon):
            raise ValueError('Bridge count changed between S49 and S51')
        for e,x in zip(edges,addon):
            if (e.get('i'),e.get('j'))!=(x.get('i'),x.get('j')) or not isinstance(e.get('risky'),bool):
                raise ValueError('Bridge IDs changed')
            if (x.get('physics_auto_join_authorized') is not False or
                    x.get('native_track_relation_truth')!='UNKNOWN'
                    or not isinstance(x.get('collision_warning'),bool)
                    or x.get('old_S49_warning') is not e['risky']):
                raise ValueError('Not a safe machine advisory')
            prev=e['risky']
            crossing=x['collision_warning']
            e['s49_prior_warning']=prev
            e['s51_third_line_overlap_advisory']=crossing
            e['risky']=bool(prev or crossing)
            e['native_relation_truth']='UNKNOWN'
            e['physical_auto_join_authorized']=False
            base_edges+=1;old_risk+=int(prev);new_risk+=int(crossing and not prev);union+=int(e['risky'])
    r['schema']=SCHEMA
    r['scientific_interpretation']='Machine-only image plane proposals; all native particle IDs and physical joins UNKNOWN'
    r['physical_auto_links']=0
    r['native_GT_count']=0
    r['allowed_to_use_as_heldout_photo_training_GT']=False
    r['summary']['S49_prior_warning_edges']=old_risk
    r['summary']['S51_additional_overlap_advisories']=new_risk
    r['summary']['risky_edges']=union
    r['summary']['graph_edges']=base_edges
    if base_edges>0 and (r['summary']['machine_candidates']<1 or r['summary']['crops']!=len(r['tiles'])):
        raise ValueError('Wrong candidate provenance')
    return r


def build_template():
    s=TEMPLATE.read_text(encoding='utf-8')
    replacements={
        'S35-S49 · исходные H50Z':'S35-S52 · исходные H50Z',
        "metric('Соединений с предупреждением',data.summary.risky_edges)":
           "metric('Соединений с предупреждением',data.summary.risky_edges),metric('Новые предупреждения S51',data.summary.S51_additional_overlap_advisories)",
        '<label><input id="showGraphRiskOnly" type="checkbox"> Только рискованные связи</label>':
           '<label><input id="showGraphRiskOnly" type="checkbox"> Только рискованные связи</label>\n  <label><input id="showCollisionOnly" type="checkbox"> Только пересечения S51</label>',
        '<span style="--c:#85bdff">Предлагаемая связь, не физический трек</span>':
           '<span style="--c:#85bdff">Предлагаемая связь, не физический трек</span>\n  <span style="--c:#ffe06e">Новый риск пересечения S51</span>',
        "if(el('showGraphRiskOnly').checked&&!e.risky)continue;drawPolyline(ctx,e.bridge,e.risky?'#ff6b90':'#85bdff',2.0,[5,4])":
           "if(el('showGraphRiskOnly').checked&&!e.risky)continue;if(el('showCollisionOnly').checked&&!e.s51_third_line_overlap_advisory)continue;drawPolyline(ctx,e.bridge,e.s51_third_line_overlap_advisory&&!e.s49_prior_warning?'#ffe06e':(e.risky?'#ff6b90':'#85bdff'),2.0,[5,4])",
        'рискованных соединений: <b>${t.graph_edges.filter(x=>x.risky).length}</b>.':
           'рискованных соединений: <b>${t.graph_edges.filter(x=>x.risky).length}</b>; пересечений с третьей линией S51: <b>${t.graph_edges.filter(x=>x.s51_third_line_overlap_advisory).length}</b>.',
        "['showCandidates','onlyWarnings','showGraph','showGraphRiskOnly']":
           "['showCandidates','onlyWarnings','showGraph','showGraphRiskOnly','showCollisionOnly']",
        "schema:'s49_machine_unverified_export'":
           "schema:'s52_machine_unverified_export'",
        "a.download='S49_MACHINE_HYPOTHESES_NOT_PHYSICS_GT.json'":
           "a.download='S52_MACHINE_HYPOTHESES_NOT_PHYSICS_GT.json'",
    }
    for before,after in replacements.items():
        if s.count(before)!=1:raise ValueError(f'Template changed or ambiguous marker: {before[:90]}')
        s=s.replace(before,after)
    # The visual UI explains the benchmark without turning synthetic-only metrics
    # into experimental physics results. Data are bundled locally as JSON files.
    addon='''
<section id="benchmark" class="notice">
<h2>S50: контрольная проверка кривизны (искусственные разрывы)</h2>
<p>Сопоставимые 31-точечные фрагменты, независимые кадры visualMetro и перенос между H50Z 005 и 006.
AUROC с кривизной выше в трёх из четырёх проверок, но при фиксированном пороге число ложных соединений
увеличилось. <strong>Кривизна не включена в автоматическое соединение.</strong></p>
<h2>S51: диагностический признак пересечений</h2>
<p>На контролируемых сценах сработал у 70/320 правильных и 140/640 неправильных искусственных связей.
Он не отличает правильный физический трек от пересекающей его линии. Жёлтый пунктир — только новый
тип предупреждения; все соединения остаются обратимыми машинными гипотезами.</p>
<p class="muted">Численные отчёты: S50_PUBLIC_LENGTH_MATCHED_CONTINUATION_RESULTS.json и
S51_PUBLIC_TOPOLOGY_COLLISION_RESULTS.json (рядом с index.html).</p>
</section>
'''
    target='<section id="details" aria-live="polite"></section>'
    if s.count(target)!=1:raise ValueError('Missing detail section in original viewer')
    s=s.replace(target,target+addon)
    s=s.replace('S49 ·', 'S52 ·')
    return s


def run(s49_file:Path,s51_private:Path,s49_images:Path,
        s50_public:Path,s51_public:Path,out_dir:Path,out_zip:Path)->dict:
    source=json.loads(s49_file.read_text(encoding='utf-8'))
    patch=json.loads(s51_private.read_text(encoding='utf-8'))
    combined=join_advisories(source,patch)
    if (len(combined['tiles'])!=39 or combined['summary']['graph_edges']!=182 or
        combined['summary']['machine_candidates']!=567 or
        combined['summary']['risky_edges']!=76):
        raise ValueError('Canonical S49/S51 native machine aggregates changed')
    out_dir.mkdir(parents=True,exist_ok=True)
    imgs=out_dir/'images'
    imgs.mkdir(exist_ok=True)
    asset_list=[]
    for i,t in enumerate(combined['tiles']):
        for key in ('source_image','other_image'):
            name=Path(t[key])
            if name.parent != Path('images') or not name.name.startswith('S49_') or name.suffix!='.jpg':
                raise ValueError('Unexpected image asset path')
            src=s49_images/name.name
            dst=imgs/name.name
            if not src.is_file():raise FileNotFoundError(src)
            shutil.copyfile(src,dst)
            asset_list.append(name.name)
    if len(set(asset_list))!=78:
        raise ValueError('Cannot map two different photo assets per tile')
    for origin in (s50_public,s51_public):
        shutil.copyfile(origin,out_dir/origin.name)
    payload=json.dumps(combined,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    page=build_template().replace('__INLINE_PAYLOAD__',payload)
    if '__INLINE_PAYLOAD__' in page or 'https://' in page or 'http://' in page or 'fetch(' in page:
        raise ValueError('Viewer must be a self-contained offline HTML')
    (out_dir/'index.html').write_text(page,encoding='utf-8')
    (out_dir/'README_FIRST_RU.txt').write_text(
        'ANZA-LIRA S52: машинный показ для физиков. НЕТ физически подтверждённых треков.\n'
        'Распакуйте ZIP целиком; откройте index.html в Chrome/Firefox.\n'
        'Все 39 участков/567 линий/182 связи сохранены; жёлтым отмечены новые проверки S51.\n'
        'Не используйте эти кадры, предупреждения или экспорт как экспертную GT.\n'
        'На исходных BMP не выполнялись 3D или mass/charge/PID измерения.\n',encoding='utf-8')
    manifest={}
    for path in sorted(out_dir.rglob('*')):
        if path.is_file():manifest[str(path.relative_to(out_dir))]=sha256(path.read_bytes()).hexdigest()
    (out_dir/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(out_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(out_dir.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(out_dir.parent))
    with zipfile.ZipFile(out_zip) as z:
        if z.testzip() is not None:raise ValueError('Private ZIP checksum failure')
    return {'zip':str(out_zip),'size_bytes':out_zip.stat().st_size,
            'image_files':len(set(asset_list)),'total_files':sum(1 for p in out_dir.rglob('*') if p.is_file()),
            'native_machine_only':True,'source_tiles':39,'lines':567,'edges':182,
            'previous_warning_edges':58,'new_warning_edges':18,'union_warning_edges':76,
            'native_GT_count':0,'auto_physical_joins':0}


def main():
    p=argparse.ArgumentParser()
    for arg in ('s49-private','s51-private','s49-images','s50-public','s51-public','out-dir','out-zip'):
        p.add_argument('--'+arg,required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(run(a.s49_private,a.s51_private,a.s49_images,
            a.s50_public,a.s51_public,a.out_dir,a.out_zip),indent=2))

if __name__=='__main__':main()
