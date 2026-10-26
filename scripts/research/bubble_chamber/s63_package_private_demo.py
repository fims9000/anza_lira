#!/usr/bin/env python3
"""S63 offline H50Z private-viewer addendum; never to public repository."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

SCHEMA='s63_machine_only_curves_CAL_then_fresh_TEST_SYNTHETIC_ONLY_v1'


def run(src:Path,calfile:Path,testfile:Path,out_dir:Path,out_zip:Path)->dict:
    cal=json.loads(calfile.read_text())
    test=json.loads(testfile.read_text())
    if cal['schema']!=SCHEMA or test['schema']!=SCHEMA or cal['phase']!='cal' or test['phase']!='test':
        raise ValueError('Wrong S63 frozen phases')
    if test['calibration_sha256']!=hashlib.sha256(calfile.read_bytes()).hexdigest():
        raise ValueError('S63 tested with a different CAL artifact')
    chosen=cal['cal_only_decision']['chosen_on_CAL_only']
    if chosen!=test['selected_method'] or chosen!='junction_less':
        raise ValueError('Unexpected research run / CAL selection')
    base,selected=test['total']['s39_original'],test['total'][chosen]
    if not (base['scenes']==selected['scenes']==96 and
            base['true_links']==76 and selected['true_links']==85 and
            base['decoy_supported']==28 and selected['decoy_supported']==34):
        raise ValueError('Wrong S63 locked test metrics')
    if not (src/'index.html').is_file():raise FileNotFoundError('Prior private S62 viewer missing')
    if out_dir.exists():shutil.rmtree(out_dir)
    shutil.copytree(src,out_dir)
    shutil.copyfile(calfile,out_dir/'S63_FROZEN_CAL_PUBLIC.json')
    shutil.copyfile(testfile,out_dir/'S63_FROZEN_NEW_TEST_PUBLIC.json')
    original=(out_dir/'index.html').read_text(encoding='utf8')
    anchor='<section class="notice"><h2>S62 — проверка детектора прямо на пикселях</h2>'
    if original.count(anchor)!=1 or original.count('S35–S62 ·')!=1:raise ValueError('S62 HTML changed')
    block='''<section class="notice"><h2>S63 — больше машинных линий, но и больше ложных (строгий отложенный контроль)</h2>
<p><strong>НЕТ автоматического перевода в основной режим:</strong> решение о варианте «меньше разрезать пересечения»
принято на 48 искусственных калибровочных сценах. На 96 новых тестовых сценах восстановилось больше
геометрических фрагментов и связей, но появились дополнительные искусственные ложные фрагменты.
Настоящих подтверждённых физических треков среди этих меток нет.</p>
<table><thead><tr><th>S63 synthetic-on-H50Z test, 96 scenes</th><th>Осторожный S39</th><th>S63 исследовательский</th></tr></thead><tbody>
<tr><td>Поддержано из 576 искусственных фрагментов</td><td>198</td><td>208</td></tr>
<tr><td>Условно доступны из 384 правильных стыков</td><td>76</td><td>85</td></tr>
<tr><td>Условно доступны из 192 полных цепочек</td><td>22</td><td>28</td></tr>
<tr><td>Поддержано из 96 искусственных ложных фрагментов</td><td>28</td><td><strong>34 — ухудшение</strong></td></tr>
</tbody></table><p>Эти показатели касаются только изображений с программно внесёнными линиями. Они НЕ доказывают
точность физического трекинга. Исходные 39 фотографических участков и 567 машинных линий ниже сохранены
без изменений, 37 осторожных стыков остаются режимом по умолчанию.</p>
<p><a href="S63_FROZEN_CAL_PUBLIC.json">Протокол CAL с замороженным выбором</a> ·
<a href="S63_FROZEN_NEW_TEST_PUBLIC.json">Все агрегированные результаты нового TEST</a></p>
</section>'''
    updated=original.replace(anchor,block+'\n'+anchor).replace('S35–S62 ·','S35–S63 ·')
    if any(token in updated for token in ('fetch(', 'http://', 'https://')):
        raise ValueError('Offline viewer cannot fetch from Internet')
    (out_dir/'index.html').write_text(updated,encoding='utf8')
    with (out_dir/'README_FIRST_RU.txt').open('a',encoding='utf8') as f:
        f.write('\nS63: CAL-only выбор разрезания пересечений увеличил искусственную\n'
                'полноту на TEST, но увеличил и ложные фрагменты. В основном\n'
                'режиме исходный S39; S63 в интерфейсе только диагностический.\n')
    manifest={str(p.relative_to(out_dir)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in out_dir.rglob('*') if p.is_file() and p.name!='SHA256_MANIFEST.json'}
    (out_dir/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(out_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in sorted(out_dir.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out_dir.parent))
    with zipfile.ZipFile(out_zip) as z:
        assert z.testzip() is None
    return {'zip':str(out_zip),'file_count':len(manifest)+1,'size_bytes':out_zip.stat().st_size,
            'sha256':hashlib.sha256(out_zip.read_bytes()).hexdigest(),
            'default_physics_auto_joins':0,'native_GT':0}


def main():
    p=argparse.ArgumentParser()
    for key in ('source-s62','calibration','test','out-dir','out-zip'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(run(args.source_s62,args.calibration,args.test,args.out_dir,args.out_zip),indent=2))
if __name__=='__main__':main()
