#!/usr/bin/env python3
"""Offline S62 private S61 viewer + synthetic-on-H50Z detector diagnostic.

No private photo or per-shot geometry is imported by this *source* script.
Output bundles must NEVER be uploaded to public GitHub.
"""
from __future__ import annotations
import argparse
import json
from hashlib import sha256
from pathlib import Path
import shutil
import zipfile

SCHEMA='s62_image_only_candidate_upper_bound_SYNTHETIC_ONLY_v1'
SEEDS=(20266203,20266204,20266213,20266214)


def assert_public_result(data:dict)->None:
    if (data.get('schema')!=SCHEMA or data.get('smoke_only_NOT_confirmatory')is not False
        or data.get('native_H50Z_physical_track_GT_count')!=0 or
        data.get('source_polylines_never_provided_to_pixel_detectors')is not True):
        raise ValueError('Only frozen image-only synthetic S62 results accepted')
    found={int(s) for d in data['directions'].values() for s in d['seeds']}
    if found!=set(SEEDS):raise ValueError('Expected full frozen S62 test-seed set')


def run(s61_dir:Path,s62_public:Path,private_previews:Path,out_dir:Path,out_zip:Path)->dict:
    original=s61_dir/'index.html'
    if not original.is_file():raise FileNotFoundError('Missing private S61 gallery')
    result=json.loads(s62_public.read_text())
    assert_public_result(result)
    previews=sorted(private_previews.glob('S62_PRIVATE_SYNTHETIC_*_seed*.png'))
    if len(previews)!=4 or any(not any(f'seed{seed}' in p.name for p in previews) for seed in SEEDS):
        raise ValueError('Need exactly four private synthetic-on-H50Z previews')
    if out_dir.exists():shutil.rmtree(out_dir)
    shutil.copytree(s61_dir,out_dir,ignore=shutil.ignore_patterns('S61_PRIVATE_BROWSER_QA.png'))
    (out_dir/'images'/'synthetic_s62').mkdir(exist_ok=True)
    for p in previews:shutil.copyfile(p,out_dir/'images'/'synthetic_s62'/p.name)
    shutil.copyfile(s62_public,out_dir/'S62_PUBLIC_LOCKED_IMAGE_ONLY_RESULTS.json')
    old=(out_dir/'index.html').read_text(encoding='utf-8')
    anchor='<div class="notice">Повторяемость на двух фотографиях'
    if old.count(anchor)!=1:raise ValueError('Missing S61 scientific disclaimer')
    cards='\n'.join('<figure style="margin:5px;max-width:420px"><img loading="lazy" '
        f'src="images/synthetic_s62/{p.name}" style="max-width:100%;border-radius:6px" '
        'alt="Синтетически добавленные треки и машинные полилинии на настоящем фоне H50Z">'
        f'<figcaption>{p.stem} — только искусственные линии</figcaption></figure>' for p in previews)
    block='''<section class="notice"><h2>S62 — проверка детектора прямо на пикселях</h2>
<p><strong>ВНИМАНИЕ: линии ниже искусственно добавлены на настоящий H50Z-фон,
не являются физически подтверждёнными треками!</strong> Отдельный детектор
получает только 192×192 изображение; координаты генератора использованы
исключительно после извлечения линий для расчёта контрольных показателей.</p>
<table><thead><tr><th>Метод</th><th>Доступных искусственных стыков</th>
<th>Условно полных 3-фрагментных путей</th></tr></thead><tbody>
<tr><td>Многомасштабная маска</td><td>149 / 384</td><td>56 / 192</td></tr>
<tr><td>Локальный контраст</td><td>170 / 384</td><td>64 / 192</td></tr>
<tr><td><strong>Реально извлечённые полилинии из пикселей</strong></td>
<td><strong>73 / 384</strong></td><td><strong>19 / 192</strong></td></tr>
</tbody></table>
<p>Результат указывает на недостаточную полноту извлечения полилиний:
хорошие значения графовой оптимизации с заранее заданными координатами
НЕ являются end-to-end качеством. <a href="S62_PUBLIC_LOCKED_IMAGE_ONLY_RESULTS.json">Агрегированные результаты и условные интервалы</a>.</p>
<div style="display:flex;flex-wrap:wrap;gap:12px">'''+cards+'''</div></section>'''
    updated=old.replace(anchor,block+'\n'+anchor)
    if old.count('S35–S61 ·')!=1:raise ValueError('Unexpected old S61 title')
    updated=updated.replace('S35–S61 ·','S35–S62 ·')
    if any(tag in updated for tag in ('fetch(', 'http://','https://', '__INLINE_PAYLOAD__')):
        raise ValueError('Private S62 viewer must require no internet')
    (out_dir/'index.html').write_text(updated,encoding='utf-8')
    (out_dir/'README_FIRST_RU.txt').write_text(
        (out_dir/'README_FIRST_RU.txt').read_text(encoding='utf-8')+
        '\nS62: раздел «Проверка детектора прямо на пикселях» показывает 4\n'
        'синтетических наложения на настоящие H50Z-фоны и таблицу потерь\n'
        'кандидатов. Это НЕ физически размеченные траектории.\n',encoding='utf-8')
    manifest={str(p.relative_to(out_dir)):sha256(p.read_bytes()).hexdigest()
              for p in sorted(out_dir.rglob('*')) if p.is_file() and p.name!='SHA256_MANIFEST.json'}
    (out_dir/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    out_zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in sorted(out_dir.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out_dir.parent))
    with zipfile.ZipFile(out_zip) as z:
        if z.testzip() is not None:raise RuntimeError('Corrupted private research pack')
    return {'private_archive':str(out_zip),'bytes':out_zip.stat().st_size,
            'private_synthetic_overlays':4,'native_expert_GT':0,'physics_auto_joins_authorized':0,
            'SHA256':sha256(out_zip.read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser()
    for k in ('s61-dir','s62-public','private-previews','out-dir','out-zip'):
        p.add_argument('--'+k,type=Path,required=True)
    x=p.parse_args()
    print(json.dumps(run(x.s61_dir,x.s62_public,x.private_previews,x.out_dir,x.out_zip),indent=2))
if __name__=='__main__':main()
