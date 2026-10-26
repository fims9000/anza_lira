#!/usr/bin/env python3
"""Package S39 machine-only visualization + source-fiducial risk QA privately.

Never add this generated ZIP or images to public Git. All sample pixels and
original-photo coordinates remain in the private package, not source tree.
"""
from __future__ import annotations
import argparse,html,json,shutil,zipfile,hashlib
from pathlib import Path


def package(s39_demo_root:Path,s40_private_dir:Path,out_zip:Path)->dict:
    if not (s39_demo_root/'index.html').is_file():raise ValueError('Missing S39 offline demo')
    report=json.loads((s40_private_dir/'S40_FIDUCIAL_MACHINE_REPORT_PUBLIC.json').read_text())
    imgs=sorted(s40_private_dir.glob('S40_PRIVATE_FID_*.png'))
    if len(imgs)!=15 or report['known_supplied_fiducial_marks']!=15:
        raise ValueError('Incomplete 15-fiducial source set')
    root=out_zip.with_suffix('')
    if root.exists():shutil.rmtree(root)
    root.mkdir(parents=True)
    inner=root/'S39_MACHINE_ONLY_DEMO'
    shutil.copytree(s39_demo_root,inner)
    qc=root/'S40_FIDUCIAL_SAFETY'
    qc.mkdir()
    for path in imgs:shutil.copy2(path,qc/path.name)
    shutil.copy2(s40_private_dir/'S40_FIDUCIAL_MACHINE_REPORT_PUBLIC.json',qc/'S40_FIDUCIAL_MACHINE_REPORT_PUBLIC.json')
    shutil.copy2(s40_private_dir/'S40_GRAPH_FIDUCIAL_GUARDS_PRIVATE.json',qc/'S40_GRAPH_FIDUCIAL_GUARDS_PRIVATE.json')
    # Browser contains SOURCE PIXELS and must remain strictly private.
    cards=[]
    by_photo={r['photo']+'_'+r['fiducial']:r for r in report['sample_rows_without_source_coordinates']}
    for f in imgs:
        v=f.stem.replace('S40_PRIVATE_FID_','')
        row=by_photo.get(v,{})
        cards.append(f'''<article><h3>Камера {html.escape(v)}</h3><img src="{html.escape(f.name)}" width="480" height="480" loading="lazy" alt="Крест {html.escape(v)}"/>
        <p>Машинные линии: {row.get('candidate_count','?')}. Зоны/лучи репера: {row.get('candidate_any_fiducial_risk','?')}.</p></article>''')
    style='''<style>body{background:#0c1521;color:#ecf4fa;font:16px system-ui,sans-serif;margin:0 auto;padding:32px;max-width:1600px}a{color:#93caff}h1{font-size:30px}p{line-height:1.6}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}article{background:#17283c;border:1px solid #2c4964;padding:14px;border-radius:10px}article img{width:100%;height:auto}small{color:#bed0dd}strong{color:#ffe6a3}</style>'''
    page='''<!doctype html><html lang="ru"><meta charset="UTF-8"><title>ANZA S40 — проверка крестов</title>'''+style+f'''<body><a href="../index.html">← К выбору демо</a><h1>S40 — автоматическая проверка реперных крестов</h1><p>
      15 служебных крестов из исходных H50Z-координат. Жёлтый круг — зона неопределённости около известного репера;
      сиреневые линии — найденные направления креста; <strong>красный</strong> — машинная линия, похожая на репер;
      бирюзовый — остальные предложенные линии. Эти цвета НЕ означают, что физические треки подтверждены или отвергнуты.</p>
      <p><strong>{report['graph_fiducial_guards']['fiducial_risk_edges']}</strong> из {report['graph_fiducial_guards']['candidate_relation_edges']}
      графовых гипотез требуют повышенной осторожности из-за реперов. {report['graph_fiducial_guards']['fiducial_risk_among_previously_ranked_edges']}
      из {report['graph_fiducial_guards']['preliminary_unambiguous_S39_edges']} «однозначных по геометрии» связей попали в эту группу.</p>
      <small>Только машинный анализ. Нет экспертной GT, нет автоматических физических соединений, PID/массы или 3D-реконструкции. Фото и ID остаются в этом приватном ZIP.</small>
      <section>{''.join(cards)}</section></body></html>'''
    (qc/'index.html').write_text(page,encoding='utf-8')
    (root/'index.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8">'+style+'<body><h1>ANZA–LIRA · автоматическое демо S39–S40</h1><p>Это машинные гипотезы, не подтверждённые треки частиц. Без ручной разметки.</p><p><a href="S39_MACHINE_ONLY_DEMO/index.html">S39: 39 реальных областей, предлагаемые линии и связность, 128 дополнительных областей</a></p><p><a href="S40_FIDUCIAL_SAFETY/index.html">S40: как алгоритм выявляет служебные реперные кресты и блокирует опасные соединения</a></p></body></html>',encoding='utf-8')
    (root/'README_FIRST_RU.txt').write_text('АНЗА-LIRA S39–S40: только машинный анализ. Распакуйте ZIP ЦЕЛИКОМ и откройте index.html в браузере.\nВсю исходную фотометрию, coords и граф оставьте приватными.\nСвязи, найденные машиной, НЕ являются физически подтвержденными.\n',encoding='utf-8')
    out_zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(root.parent))
    with zipfile.ZipFile(out_zip) as z:
        corrupt=z.testzip()
        count=len(z.namelist())
    if corrupt:raise AssertionError('Corrupt demo asset '+corrupt)
    return {'zip':str(out_zip),'size_bytes':out_zip.stat().st_size,
            'sha256':hashlib.sha256(out_zip.read_bytes()).hexdigest(),
            'members':count,'known_fiducials':len(imgs),
            'graph_relation_risk_count':report['graph_fiducial_guards']['fiducial_risk_edges'],
            'qa':'ZIP CRC verified; no physical particle GT'}


def cli():
    p=argparse.ArgumentParser()
    p.add_argument('--s39-demo-root',type=Path,required=True)
    p.add_argument('--s40-private-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(package(a.s39_demo_root,a.s40_private_dir,a.output),ensure_ascii=False,indent=2))
if __name__=='__main__':cli()
