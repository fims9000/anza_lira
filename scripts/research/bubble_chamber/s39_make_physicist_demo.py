#!/usr/bin/env python3
"""Offline machine-only physicist demonstration; sources remain private."""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
from html import escape
import argparse,csv,gzip,json,shutil,zipfile

ROOT=Path(__file__).resolve().parents[3]

HTML='''<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ANZA–LIRA · Автоматический анализ H50Z</title>
<style>
:root {font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#18212b;background:#f6f8fa}
body{max-width:1390px;margin:auto;padding:18px} h1{margin:.5rem 0}h2{margin:.9rem 0}
.small{color:#475569;font-size:.93rem}.warn{border-left:4px solid #b45309;background:#fffbeb;padding:10px 16px;line-height:1.5;margin:16px 0}
.metrics{display:flex;gap:20px;flex-wrap:wrap;margin:16px 0}
.metric{background:white;border:1px solid #dbe2e9;padding:10px 16px;min-width:130px;border-radius:7px}
.metric b{font-size:1.45rem;display:block}.metric small{color:#475569}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:15px 0}
select,button{padding:9px;font-size:1rem;border-radius:6px;border:1px solid #bfcad6;background:white}
button{cursor:pointer}button:hover{background:#e8eef5} .panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
.panel{background:white;border:1px solid #dae2e7;border-radius:6px;padding:9px}.panel img{max-width:100%;width:100%;image-rendering:auto}
.panel h3{margin:2px 0 8px;font-size:1rem}.gallery{display:flex;gap:12px;flex-wrap:wrap}
.gallery img{height:260px;max-width:330px;border:1px solid #cbd5e1;object-fit:contain;background:white}
pre{font-size:12px;background:#f1f5f9;white-space:pre-wrap;overflow-wrap:anywhere;padding:12px}
@media (max-width:860px){.panels{grid-template-columns:1fr}.gallery img{height:190px}}
</style></head><body>
<h1>ANZA–LIRA · Автоматические гипотезы треков на H50Z</h1>
<p class="small">Приватная локальная демонстрация S39 · 25 октября 2026 · исходные фотографии и нейросетевые веса не публикуются</p>
<div class="warn"><strong>Научный статус: только машинные гипотезы.</strong> Зелёные линии — найденные алгоритмами фрагменты; пурпурные пунктирные соединения — геометрические предложения, <strong>не</strong> восстановленные физические траектории. Псевдомаски обучающие и могут содержать царапины, реперы и фон. Никакие настоящие particle ID, 3D координаты или массы не присвоены.</div>
<div class="metrics"><div class="metric"><b>2</b><small>BMP, один физический запуск</small></div><div class="metric"><b>39</b><small>детально просмотренных машиной участков</small></div><div class="metric"><b>567</b><small>машинных гипотез линий</small></div><div class="metric"><b>182</b><small>геометрически допустимых связей-кандидатов</small></div><div class="metric"><b>128</b><small>случайных участков полных BMP</small></div></div>
<h2>Настоящие фотографии · машинные линии · гипотезы связи</h2>
<div class="controls"><label>Фото <select id="photo"><option value="all">005 + 006</option><option>005</option><option>006</option></select></label>
<label>Тип участка <select id="stratum"><option value="all">Все</option></select></label>
<button id="prev">Назад</button><button id="next">Вперёд</button><select id="tile"></select></div>
<p id="desc" class="small"></p>
<div class="panels"><div class="panel"><h3>Оригинальный фрагмент H50Z (без обработки)</h3><img id="orig" loading="lazy"></div><div class="panel"><h3>Автоматические линии: S31 + S39</h3><img id="detect" loading="lazy"></div><div class="panel"><h3>Предполагаемые соединения, пунктир (без физического подтверждения)</h3><img id="links" loading="lazy"></div><div class="panel"><h3>Псевдомаска: зелёный — машинный позитив, розовый — условный фон, остальное — UNKNOWN</h3><img id="mask" loading="lazy"></div></div>
<h2>Слепая пространственная проверка на полных BMP</h2>
<p class="small">Ниже заранее выбранные области из равномерной выборки 64 участков каждой фотографии. Их не выбирали по красивому результату. Это иллюстрации машинного поиска, а не примеры доказанного распознавания частиц.</p>
<div class="gallery" id="randomGallery"></div>
<h2>Результат контролируемого эксперимента</h2><p>На 80 сценах с искусственно добавленными штрихами на настоящих фотографических фонах обычный локальный контраст и многомасштабная геометрия показали лучшую полноту, чем предварительно обученная AZUNet. Но те же простые фильтры выделяют искусственные царапины; без независимой стереоинформации физическая принадлежность фрагментов не установлена.</p>
<p class="small">Точные метрики: <code>S39_SYNTHETIC_HOLDOUT_PUBLIC.json</code>. Все машинные 2D-координаты: <code>S39_MACHINE_POINTS_PRIVATE.csv.gz</code>; локальные masks в <code>masks/</code>. Руководство visualMetro (Беляев, Бушмина, Клевцова, 2025) требует три проекции, реперы и калибровку для стереореконструкции. Адроида (Беляев, 2023) относится к особому случаю остановившегося трека, а не к универсальной 2D-разметке.</p>
<script id="manifest" type="application/json">__DATA__</script>
<script>
'use strict';
const rows=JSON.parse(document.querySelector('#manifest').textContent), sel=document.querySelector('#tile');
const photo=document.querySelector('#photo'),stratum=document.querySelector('#stratum');
for(const tag of [...new Set(rows.map(x=>x.stratum))].sort()){const o=document.createElement('option');o.textContent=tag;stratum.append(o);}
let visible=[];function updateOptions(){let current=sel.value;visible=rows.filter(x=>(photo.value==='all'||x.photo_id===photo.value)&&(stratum.value==='all'||x.stratum===stratum.value));sel.replaceChildren();for(const x of visible){const o=document.createElement('option');o.value=x.crop_id;o.textContent=x.crop_id;sel.append(o);}if(visible.some(x=>x.crop_id===current))sel.value=current;render();}
function render(){const x=visible.find(v=>v.crop_id===sel.value);if(!x){document.querySelector('#desc').textContent='Нет областей';return;}
for(const [id,prefix,suffix] of [['orig','crops/','.png'],['detect','overlays/','_AUTO.png'],['links','graph/','_machine_graph.png'],['mask','masks/','_PSEUDO_NOT_GT.png']]){document.getElementById(id).src=prefix+x.crop_id+suffix;document.getElementById(id).alt=x.crop_id+' / '+id;}
document.querySelector('#desc').textContent=`${x.crop_id}; страта ${x.stratum}; исходные S31: ${x.original}; новые S39: ${x.new}; предложенные машинные линии: ${x.proposals}; всё ещё без экспертной GT.`;
}
function jump(d){if(!visible.length)return;sel.selectedIndex=(sel.selectedIndex+d+visible.length)%visible.length;render();}
photo.addEventListener('change',updateOptions);stratum.addEventListener('change',updateOptions);sel.addEventListener('change',render);document.querySelector('#prev').onclick=()=>jump(-1);document.querySelector('#next').onclick=()=>jump(1);
const gallery=document.querySelector('#randomGallery');for(const p of __RANDOM__){const im=document.createElement('img');im.src='random/'+p;im.alt='Unbiased full-frame sample '+p;gallery.append(im);}
updateOptions();
</script></body></html>'''


def export_points(s39:dict,random:dict|None,out:Path):
    n=0
    with gzip.open(out,'wt',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        w.writerow(('source','photo_id','crop_id','candidate_id','point_number','x_native_photo_px','y_native_photo_px','algorithm','verified_physical_truth'))
        for tile in s39['tiles']:
            ox,oy=tile['native_origin_xy_PRIVATE']
            for cand in tile['proposals']:
                for idx,(x,y) in enumerate(cand['points_xy']):
                    w.writerow(('source_39_selected',tile['photo_id'],tile['crop_id'],cand['draft_id'],idx,round(x+ox,3),round(y+oy,3),cand['source_method'],'NO'))
                    n+=1
        if random is not None:
            for tile in random['frames']:
                for cand in tile['candidates_PRIVATE']:
                    for idx,(x,y) in enumerate(cand['points_xy']):
                        w.writerow(('unbiased_BMP_random_sample',tile['photo'],f"sample{tile['index']:03d}",cand['draft_id'],idx,round(x+tile['x_private'],3),round(y+tile['y_private'],3),cand['source_method'],'NO'))
                        n+=1
    return n


def run(source:Path,random_source:Path,private_crop_dir:Path,overlay_dir:Path,graph_dir:Path,mask_dir:Path,random_overlays:Path,results:Path,outzip:Path):
    man=json.loads(source.read_text())
    full=json.loads(random_source.read_text())
    if man['schema']!='s39_machine_only_photo_hypotheses_v1' or full.get('no_truth') is not True:
        raise ValueError('Wrong private source/provenance')
    results.mkdir(exist_ok=True,parents=True)
    for folder in ('crops','overlays','graph','masks','random'):(results/folder).mkdir(exist_ok=True)
    summary=[]
    for tile in man['tiles']:
        cid=tile['crop_id'];stat=tile['summary']
        summary.append({'crop_id':cid,'photo_id':tile['photo_id'],'stratum':tile['stratum'],
                        'original':stat['original_drafts'],'new':stat['new_ridge_proposals'],
                        'proposals':stat['original_drafts']+stat['new_ridge_proposals']})
        for origin,dst in ((private_crop_dir/f'{cid}.png',results/'crops'/f'{cid}.png'),
                           (overlay_dir/f'{cid}_AUTO.png',results/'overlays'/f'{cid}_AUTO.png'),
                           (graph_dir/f'{cid}_machine_graph.png',results/'graph'/f'{cid}_machine_graph.png'),
                           (mask_dir/f'{cid}_PSEUDO_NOT_GT.png',results/'masks'/f'{cid}_PSEUDO_NOT_GT.png')):
            if not origin.exists():raise FileNotFoundError(origin)
            shutil.copy2(origin,dst)
    sampled=sorted(random_overlays.glob('*_PRIVATE.png'))
    for p in sampled:shutil.copy2(p,results/'random'/p.name)
    embedded=json.dumps(summary,ensure_ascii=False).replace('<','\\u003c')
    random_names=json.dumps([p.name for p in sampled],ensure_ascii=False).replace('<','\\u003c')
    (results/'index.html').write_text(HTML.replace('__DATA__',embedded).replace('__RANDOM__',random_names),encoding='utf-8')
    n=export_points(man,full,results/'S39_MACHINE_POINTS_PRIVATE.csv.gz')
    public_files=['S39_MACHINE_SUMMARY_PUBLIC.json','S39_SYNTHETIC_HOLDOUT_PUBLIC.json']
    for f in public_files:
        fp=source.parent/f
        if not fp.exists():raise FileNotFoundError(fp)
        shutil.copy2(fp,results/f)
    for f in ['S39_GRAPH_SUMMARY_PUBLIC.json']:
        fp=graph_dir.parent/f
        shutil.copy2(fp,results/f)
    fp=random_source.parent/'S39_FULL_BMP_AUDIT_PUBLIC.json'
    shutil.copy2(fp,results/fp.name)
    (results/'README_FIRST_RU.txt').write_text(
        'ANZA-LIRA S39: МАШИННЫЙ ДЕМОНСТРАТОР, НЕ ФИЗИЧЕСКАЯ GT.\n'
        'Распакуйте ZIP полностью. Откройте index.html в браузере.\n'
        'Все исходные PNG, автоматически найденные центральные линии, обратимые кандидатные связи и трёхсостояние псевдомасок находятся рядом с HTML.\n'
        'Все координаты пикселей в CSV относятся к оригинальным BMP.\n'
        'ВНИМАНИЕ: стрелки/массы/PID/трёхмерные траектории отсутствуют, связи не разрешены автоматически.\n',encoding='utf-8')
    with zipfile.ZipFile(outzip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(results.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(results.parent))
    return {'offline_demo':str(outzip),'zip_size':outzip.stat().st_size,'private_machine_2D_point_rows':n,
            'source_crop_count':len(man['tiles']),'unbiased_random_spatial_tiles':len(full['frames']),
            'gallery_random_pngs':len(sampled),'original_BMPs_included':False}


def main():
    p=argparse.ArgumentParser()
    for key in ['machine-manifest','full-frame-manifest','private-crop-dir','overlay-dir','graph-dir','mask-dir','random-overlays','out-dir','out-zip']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.machine_manifest,a.full_frame_manifest,a.private_crop_dir,a.overlay_dir,a.graph_dir,a.mask_dir,a.random_overlays,a.out_dir,a.out_zip),indent=2))
if __name__=='__main__':main()
