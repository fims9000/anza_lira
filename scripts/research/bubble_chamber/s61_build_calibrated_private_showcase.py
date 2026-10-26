#!/usr/bin/env python3
"""Private S61 add-on: compare S56 baseline with CAL-only S60 exploratory joins.

No image/media/coordinates are committed: this builder is public-safe, but its
outputs are PRIVATE. New links never default to ON or imply physical GT.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import zipfile

SCHEMA='s61_private_machine_compare_conservative_exploratory_v1'
PAYLOAD=re.compile(r'(<script id="private-data" type="application/json">)(.*?)(</script>)',re.DOTALL)


def integrate_payload(s58_html:str,s60_private:dict)->tuple[str,dict]:
    found=PAYLOAD.search(s58_html)
    if not found:raise ValueError('Original S58 inline PRIVATE payload required')
    doc=json.loads(found.group(2))
    if (doc.get('schema')!='s57_machine_only_global_paths_PRIVATE_v1'
        or doc.get('native_GT_count')!=0 or doc.get('physical_auto_links')!=0
        or len(doc.get('tiles',[]))!=39):
        raise ValueError('S58 must contain unverified native machine-only evidence')
    if (s60_private.get('native_physics_GT_created')is not False or
        s60_private.get('native_physics_auto_join_authorized')is not False or
        len(s60_private.get('graphs',[]))!=39):
        raise ValueError('Only audited frozen-calibration native S60 accepted')
    counters={'tiles':0,'S56_baseline_hypotheses':0,'S60_cal_exploratory_hypotheses':0,
              'new_exploratory_only':0,'new_already_risky':0,'previous_machine_links_lost':0,
              'verified_physical_links':0}
    for tile,g in zip(doc['tiles'],s60_private['graphs']):
        if tile['crop_id']!=g['crop_id_PRIVATE'] or tile['photo']!=g['photo_id']:
            raise ValueError('Private photo identity changed')
        base={tuple(v) for v in g['S56_previous_suggestions_UNVERIFIED']}
        cal={tuple(v) for v in g['S60_suggestions_UNVERIFIED']}
        if not base<=cal:raise ValueError('Unexpected baseline removal must be analyzed separately')
        if not len(cal)==len(g['S60_suggestions_UNVERIFIED']):
            raise ValueError('Duplicate S60 machine pair')
        present=set()
        for edge in tile['graph_edges']:
            key=(edge['i'],edge['j']);present.add(key)
            if bool(edge['S56_PROVISIONAL_NOT_PHYSICS_GT'])!=(key in base):
                raise ValueError('S56 reference changed after freeze')
            edge['S60_CAL_SELECTED_UNVERIFIED']=key in cal
            edge['S60_EXPLORATORY_NEW_NOT_PHYSICS_GT']=key in (cal-base)
            edge['native_relation_truth']='UNKNOWN'
            edge['physical_auto_join_authorized']=False
            if key in cal-base:
                counters['new_exploratory_only']+=1
                counters['new_already_risky']+=int(bool(edge['risky']))
        if not cal<=present:raise ValueError('Experimental suggestion missing native S39 edge')
        counters['tiles']+=1
        counters['S56_baseline_hypotheses']+=len(base)
        counters['S60_cal_exploratory_hypotheses']+=len(cal)
        counters['previous_machine_links_lost']+=len(base-cal)
    if (counters['S56_baseline_hypotheses']!=37 or
        counters['S60_cal_exploratory_hypotheses']!=41 or
        counters['new_exploratory_only']!=4 or
        counters['previous_machine_links_lost']!=0):
        raise ValueError('S60 native aggregate no longer matches reproducible audit')
    doc['schema']=SCHEMA
    doc['physical_auto_links']=0
    doc['native_GT_count']=0
    doc['summary']['S60_exploratory_total']=41
    doc['summary']['S60_exploratory_new']=4
    doc['S60_note']='EXPERIMENTAL_ONLY: four new joins are NOT automatically authorized or physics verified'
    new_json=json.dumps(doc,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    return s58_html[:found.start(2)]+new_json+s58_html[found.end(2):],counters


def upgrade_html(html:str)->str:
    old='S35–S58 · исходные H50Z'
    if html.count(old)!=1:raise ValueError('Unexpected S58 title')
    html=html.replace(old,'S35–S61 · исходные H50Z')
    marker='<label><input id="showAllEdges" type="checkbox"> Все предложения S39</label>'
    if html.count(marker)!=1:raise ValueError('Expected S58 filter missing')
    html=html.replace(marker,'<label><input id="showS60Experimental" type="checkbox"> Дополнительные 4 S60 (эксперимент, не GT)</label>\n  '+marker)
    marker='<span style="--c:#d6a7ff">Выбрана S56 только машинным алгоритмом</span>'
    if html.count(marker)!=1:raise ValueError('Expected S56 legend missing')
    html=html.replace(marker,marker+'\n<span style="--c:#ffcf80">S60: дополнительно предложено, риск не исключён</span>')
    marker="metric('Стыков S56 (не GT)',data.summary.s56_advisory_links)"
    if html.count(marker)!=1:raise ValueError('Expected S56 metric missing')
    html=html.replace(marker,marker+",metric('Дополнительно S60 (не GT)',data.summary.S60_exploratory_new)")
    # A selected *experimental* pair may be included only if the user explicitly
    # enables its checkbox. Default mode is exactly the S56 conservative output.
    marker="if(el('showGlobalOnly').checked&&!el('showAllEdges').checked&&!e.S56_PROVISIONAL_NOT_PHYSICS_GT)continue;"
    if html.count(marker)!=1:raise ValueError('Expected S58 canvas filter missing')
    html=html.replace(marker,
        "if(el('showGlobalOnly').checked&&!el('showAllEdges').checked&&!e.S56_PROVISIONAL_NOT_PHYSICS_GT&&!(el('showS60Experimental').checked&&e.S60_EXPLORATORY_NEW_NOT_PHYSICS_GT))continue;"
        "if(e.S60_EXPLORATORY_NEW_NOT_PHYSICS_GT&&!el('showS60Experimental').checked)continue;")
    # Canvas marks new edges in amber; baseline purple/red remains unmodified.
    marker="e.S56_PROVISIONAL_NOT_PHYSICS_GT?(e.risky?'#ff6b90':'#d6a7ff'):(e.risky?'#ff6b90':'#85bdff')"
    if html.count(marker)!=1:raise ValueError('Expected baseline colors missing')
    html=html.replace(marker,"e.S60_EXPLORATORY_NEW_NOT_PHYSICS_GT?'#ffcf80':("+marker+")")
    marker="['showCandidates','onlyWarnings','showGraph','showGraphRiskOnly','showGlobalOnly','showAllEdges']"
    if html.count(marker)!=1:raise ValueError('Expected original event subscriptions missing')
    html=html.replace(marker,"['showCandidates','onlyWarnings','showGraph','showGraphRiskOnly','showGlobalOnly','showS60Experimental','showAllEdges']")
    marker="schema:'s57_machine_unverified_export'"
    if html.count(marker)!=1:raise ValueError('Expected S57 export missing')
    html=html.replace(marker,"schema:'s61_machine_unverified_export'")
    marker="a.download='S57_MACHINE_HYPOTHESES_NOT_PHYSICS_GT.json'"
    if html.count(marker)!=1:raise ValueError('Expected S57 export filename missing')
    html=html.replace(marker,"a.download='S61_MACHINE_HYPOTHESES_NOT_PHYSICS_GT.json'")
    marker='<!-- s61_insert -->'
    insert='''<section class="notice"><h2>S59–S61 · что реально проверено</h2>
<p><strong>S59:</strong> на 256 новых искусственных сценах расширенный поиск пар
не увеличил число выбранных глобальным алгоритмом стыков. На 39 H50Z
прежний набор 182 пар расширялся до 247, но исходные 37 предложений не изменились.</p>
<p><strong>S60:</strong> CAL-only ослабление порога для 005→006 повысило число
правильных ИСКУССТВЕННЫХ связей на новой тестовой выборке с 134 до 183,
НО ложных — с 11 до 15; полных искусственных цепочек с 38 до 58.
Утверждение о безопасности для настоящих частиц НЕ подтверждено.</p>
<p><strong>S61:</strong> 37 прежних осторожных гипотез показаны по умолчанию;
ещё 4 машинных стыка доступны только при явном включении жёлтого
экспериментального слоя. НИ ОДНОГО физически подтверждённого соединения.</p>
<p><a href="S59_PUBLIC_LOCKED_TEST_RESULTS.json">S59: агрегированные результаты</a> ·
<a href="S60_PUBLIC_LOCKED_TEST_RESULTS.json">S60: агрегированные результаты</a> ·
<a href="S60_NATIVE_MARGIN_ADVISORIES_PRIVATE.json">S60: приватные машинные координаты (UNKNOWN)</a></p></section>'''
    if html.count('<div class="notice">Повторяемость на двух фотографиях')!=1:
        raise ValueError('Expected S58 disclaimer missing')
    html=html.replace('<div class="notice">Повторяемость на двух фотографиях',insert+'\n<div class="notice">Повторяемость на двух фотографиях')
    if any(x in html for x in ('__INLINE_PAYLOAD__','fetch(', 'http://','https://')):
        raise ValueError('Viewer must remain fully offline')
    return html


def run(s58_dir:Path,s60_private:Path,s59_public:Path,s60_public:Path,
        out_dir:Path,out_zip:Path)->dict:
    source=s58_dir/'index.html'
    if any(not x.is_file() for x in (source,s60_private,s59_public,s60_public)):
        raise FileNotFoundError('Missing private S58/S60 or public-safe aggregate')
    s=source.read_text(encoding='utf-8')
    s,stats=integrate_payload(s,json.loads(s60_private.read_text()))
    s=upgrade_html(s)
    if out_dir.exists():shutil.rmtree(out_dir)
    shutil.copytree(s58_dir,out_dir)
    (out_dir/'index.html').write_text(s,encoding='utf-8')
    shutil.copyfile(s60_private,out_dir/'S60_NATIVE_MARGIN_ADVISORIES_PRIVATE.json')
    shutil.copyfile(s59_public,out_dir/'S59_PUBLIC_LOCKED_TEST_RESULTS.json')
    shutil.copyfile(s60_public,out_dir/'S60_PUBLIC_LOCKED_TEST_RESULTS.json')
    (out_dir/'README_FIRST_RU.txt').write_text(
      (out_dir/'README_FIRST_RU.txt').read_text(encoding='utf-8')+
      '\nS61: первая загрузка показывает ТОЛЬКО исходные 37 консервативных гипотез.\n'
      'Четыре дополнительные связи S60 доступны отдельно, имеют UNKNOWN-статус,\n'
      'и не являются подтверждёнными треками частиц. Не публиковать снимки и координаты.\n',
      encoding='utf-8')
    manifest={str(p.relative_to(out_dir)):sha256(p.read_bytes()).hexdigest()
              for p in sorted(out_dir.rglob('*')) if p.is_file() and p.name!='SHA256_MANIFEST.json'}
    (out_dir/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    out_zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out_dir.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out_dir.parent))
    with zipfile.ZipFile(out_zip) as z:
        if z.testzip() is not None:raise RuntimeError('Corrupted S61 private archive')
    return {**stats,'private_archive':str(out_zip),'archive_bytes':out_zip.stat().st_size,
            'real_physics_joins_authorized':0}


def main():
    p=argparse.ArgumentParser()
    for arg in ('s58-dir','s60-private','s59-public','s60-public','out-dir','out-zip'):
        p.add_argument('--'+arg,type=Path,required=True)
    x=p.parse_args()
    print(json.dumps(run(x.s58_dir,x.s60_private,x.s59_public,x.s60_public,
                         x.out_dir,x.out_zip),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
