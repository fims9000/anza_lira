#!/usr/bin/env python3
"""Build an OFFLINE PRIVATE source-locked H50Z S55 machine-only demonstrator.

Retains the complete S52 native graph and advisory coordinates, and adds only
synthetic renderer-origin domain-support warnings from S55. Never exports
native GT, native classes or authorized physical automatic joins.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

TEMPLATE=Path(__file__).with_name('s55_offline_machine_demo.html')
SOURCE_VIEWER='ANZA_LIRA_S52_MACHINE_ONLY_SHOWCASE'
OUT_VIEWER='ANZA_LIRA_S55_MACHINE_ONLY_SHOWCASE'
KIND_ORDER=('simulated_bead_track','continuous_hairline','periodic_dashes',
            'scattered_grains','same_bead_renderer_scratch','unseen_bead_clusters')


def combine(s52:dict,s55:dict)->dict:
    if (s52.get('native_GT_count')!=0 or s52.get('physical_auto_links')!=0
        or s55.get('native_expert_GT_count')!=0 or s55.get('physical_auto_joins')!=0
        or len(s52.get('tiles',[]))!=39 or len(s55.get('tiles',[]))!=39
        or s52.get('summary',{}).get('machine_candidates')!=567
        or s52.get('summary',{}).get('graph_edges')!=182
        or s52.get('summary',{}).get('risky_edges')!=76):
        raise ValueError('Unknown or unsafe authentic S52/S55 source')
    doc=json.loads(json.dumps(s52))
    counts={}
    total=0
    for tile,new in zip(doc['tiles'],s55['tiles']):
        if (tile['crop_id'],tile['photo'])!=(new['crop_id'],new['photo']):
            raise ValueError('Private source tile identity changed')
        rows=new['candidate_advisories']
        if len(rows)!=len(tile['candidates']):
            raise ValueError('Machine candidate lost')
        for original,additional in zip(tile['candidates'],rows):
            if (original['draft_id']!=additional['draft_id']
                or additional.get('native_track_identity')!='UNKNOWN'
                or additional.get('candidate_preserved') is not True
                or additional.get('physical_auto_join_authorized') is not False):
                raise ValueError('Machine candidate identity/physics status changed')
            status=additional['status']
            if not status.startswith('UNKNOWN_'):
                raise ValueError('Non-UNKNOWN machine prediction unauthorized')
            original['s55_status']=status
            original['s55_simulator_score']=additional.get('score_S55_SIMULATOR_ORIGIN_NOT_PHYSICS_PROB')
            counts[status]=counts.get(status,0)+1
            total+=1
        for edge in tile['graph_edges']:
            if edge.get('native_relation_truth')!='UNKNOWN' or edge.get('physical_auto_join_authorized') is not False:
                raise ValueError('A graph edge is not a machine-only hypothesis')
    if total!=567 or counts.get('UNKNOWN_UNSCORABLE_SOURCE_CONTEXT')!=141:
        raise AssertionError('Inconsistent S55 native support summary')
    doc['schema']='s55_offline_machine_hypotheses_WITH_SIMULATOR_ONLY_support_v1'
    doc['native_GT_count']=0
    doc['physical_auto_links']=0
    doc['manual_annotation_required']=False
    doc['s55_support_counts']=counts
    return doc


def make_controls(source:Path,out:Path):
    from PIL import Image
    labels=['Точки: генератор +','Сплошная линия −','Прерывистая −',
            'Точки со сдвигом −','ТОТ ЖЕ генератор −','Другие кластеры −']
    sources=[('005_to_006_20265503','Тест 006'),('006_to_005_20265513','Тест 005')]
    tile=205; top=43;row=240
    sheet=Image.new('RGB',(tile*6,row*2+28),(25,35,51))
    dr=ImageDraw.Draw(sheet)
    font_file='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    try:font=ImageFont.truetype(font_file,12)
    except OSError:font=ImageFont.load_default()
    for i,(sub,title) in enumerate(sources):
        dr.text((9,i*row+8),title+' — ТОЛЬКО ИСКУССТВЕННЫЕ ОБЪЕКТЫ',font=font,fill=(246,217,169))
        for j,kind in enumerate(KIND_ORDER):
            path=source/sub/f'scene_00_{kind}.png'
            if not path.is_file():raise FileNotFoundError(path)
            pic=Image.open(path).convert('RGB')
            pic=pic.resize((190,190))
            sheet.paste(pic,(j*tile+8,i*row+35))
            dr.text((j*tile+8,i*row+227),labels[j],font=font,fill=(233,240,248))
    sheet.save(out,quality=88,optimize=True)


def run(s52_zip:Path,s55_private:Path,s53_report:Path,
        s53_metrics:Path,s55_metrics:Path,s54_public:Path,s55_public:Path,
        s55_controls:Path,s53_controls:Path,source_private_dir:Path,
        out:Path,out_zip:Path)->dict:
    doc_s55=json.loads(s55_private.read_text())
    with zipfile.ZipFile(s52_zip) as source:
        html=source.read(f'{SOURCE_VIEWER}/index.html').decode('utf-8')
        match=re.search(r'<script id="private-data" type="application/json">(.*?)</script>',html,re.S)
        if match is None:raise ValueError('Private S52 viewer no longer has canonical JSON')
        s52=json.loads(match.group(1))
        merged=combine(s52,doc_s55)
        out.mkdir(parents=True,exist_ok=True)
        imgs=out/'images';imgs.mkdir(exist_ok=True)
        paths=set()
        for t in merged['tiles']:
            for key in ('source_image','other_image'):
                asset=Path(t[key])
                if asset.parent!=Path('images') or not asset.name.startswith('S49_') or asset.suffix!='.jpg':
                    raise ValueError('Unexpected source photo locator')
                paths.add(asset.name)
        if len(paths)!=78:raise ValueError('Wrong genuine H50Z image count')
        for name in sorted(paths):
            (imgs/name).write_bytes(source.read(f'{SOURCE_VIEWER}/images/{name}'))
    payload=json.dumps(merged,separators=(',',':'),ensure_ascii=False).replace('<','\\u003c')
    page=TEMPLATE.read_text(encoding='utf-8')
    if page.count('__INLINE_PAYLOAD__')!=1:
        raise ValueError('Unsafe/repeated inline JSON marker')
    page=page.replace('__INLINE_PAYLOAD__',payload)
    if any(x in page for x in ('__INLINE_PAYLOAD__','fetch(', 'https://', 'http://')):
        raise ValueError('Offline viewer must not fetch remote data')
    (out/'index.html').write_text(page,encoding='utf-8')
    (out/'S53_S55_RESEARCH_REPORT_RU.md').write_bytes(s53_report.read_bytes())
    for file,name in [(s53_metrics,'S53_SYNTHETIC_PUBLIC.json'),
         (s55_metrics,'S55_SYNTHETIC_PUBLIC.json'),
         (s54_public,'S54_CONTEXT_OOD_PUBLIC.json'),
         (s55_public,'S55_NATIVE_SUPPORT_PUBLIC.json')]:
        (out/name).write_bytes(file.read_bytes())
    controls=out/'simulator_checkpoints_PRIVATE'; controls.mkdir(exist_ok=True)
    for root, suffix in [(s53_controls,'S53'),(s55_controls,'S55')]:
        for path in sorted(root.rglob('*.json')):
            dst=controls/(suffix+'_'+path.parent.name+'_'+path.name)
            dst.write_bytes(path.read_bytes())
    private=out/'private_machine_records';private.mkdir(exist_ok=True)
    for path in sorted(source_private_dir.glob('S5*_PRIVATE.json')):
        if path.stat().st_size > 10_000_000:raise ValueError('Unexpected enormous private manifest')
        shutil.copyfile(path,private/path.name)
    make_controls(s55_controls,out/'S55_SYNTHETIC_CONTROLS.jpg')
    (out/'README_FIRST_RU.txt').write_text(
      'ANZA–Graph-LIRA S55. Только автоматический анализ.\n'
      'Распакуйте ZIP и откройте index.html. Интернет и ручная разметка не нужны.\n'
      '39 настоящих участков H50Z, все 567 машинных линий, все 182 гипотезы соединений.\n'
      'Симуляторные оценки не являются вероятностями физического трека.\n'
      'Положительные/отрицательные метки относятся только к нарисованным штрихам.\n'
      '141 линия не помещается в 192px окно; ещё 14 вне поддержки S55 симуляции.\n'
      'Ни одна линия не удалена; не было ни подтверждённых физических треков, ни авто-стыков.\n'
      'В каталоге private_machine_records содержатся приватные идентификаторы и внутренние оценки.\n'
      'Не публикуйте этот ZIP и изображения в открытом GitHub.\n',encoding='utf-8')
    checks={str(p.relative_to(out)):sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'SHA256_MANIFEST.json').write_text(json.dumps(checks,indent=2)+'\n')
    with zipfile.ZipFile(out_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
        for path in sorted(out.rglob('*')):
            if path.is_file():zipped.write(path,Path(OUT_VIEWER)/path.relative_to(out))
    with zipfile.ZipFile(out_zip) as zipped:
        if zipped.testzip() is not None:raise AssertionError('Private package CRC failed')
    return {'private_zip':str(out_zip),'bytes':out_zip.stat().st_size,
            'native_real_photo_images':len(paths),'native_lines':567,
            'candidate_joins':182,'previous_join_warnings':76,
            's55_context_unscorable':merged['s55_support_counts']['UNKNOWN_UNSCORABLE_SOURCE_CONTEXT'],
            's55_outside_simulator_support':merged['s55_support_counts']['UNKNOWN_OUTSIDE_SOURCE_SIMULATOR_SUPPORT'],
            'native_GT_count':0,'physical_auto_joins':0,'file_count':len(checks)+1}


def main():
    p=argparse.ArgumentParser()
    for arg in ('s52_zip','s55_private','report','s53_public','s55_public','s54_public','s55_ood_public','s55_controls','s53_controls','private_records','output_dir','output_zip'):
        p.add_argument('--'+arg.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    result=run(a.s52_zip,a.s55_private,a.report,a.s53_public,a.s55_public,
            a.s54_public,a.s55_ood_public,a.s55_controls,a.s53_controls,
            a.private_records,a.output_dir,a.output_zip)
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
