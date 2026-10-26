#!/usr/bin/env python3
"""Build a fully offline PRIVATE S58 upgrade to the already validated S57 UI."""
from __future__ import annotations
import argparse,json,shutil,zipfile
from hashlib import sha256
from pathlib import Path


def run(s57_dir:Path,s58_private:Path,s58_public:Path,preview:Path,
        out_dir:Path,out_zip:Path)->dict:
    if not (s58_private.is_file() and s58_public.is_file() and preview.is_file()):
        raise ValueError('Missing private draft paths or preview')
    private=json.loads(s58_private.read_text())
    public=json.loads(s58_public.read_text())
    if (private.get('native_expert_GT_count')!=0 or
        private.get('native_physical_auto_join_count')!=0 or
        public.get('confirmed_native_physical_tracks')!=0 or
        public.get('confirmed_native_physical_joins')!=0 or
        public.get('component_counts_by_fragment_count')!={'2':34,'4':1}):
        raise ValueError('Only frozen machine-only S58 export is accepted')
    if not (s57_dir/'index.html').is_file():raise ValueError('Missing S57 provenance')
    if out_dir.exists():shutil.rmtree(out_dir)
    shutil.copytree(s57_dir,out_dir)
    shutil.copyfile(s58_private,out_dir/'S58_MACHINE_PATH_COMPONENTS_PRIVATE.json')
    shutil.copyfile(s58_public,out_dir/'S58_PUBLIC_MACHINE_PATH_COMPONENTS.json')
    shutil.copyfile(preview,out_dir/'S58_LONGEST_MACHINE_PATH_PRIVATE.jpg')
    s=(out_dir/'index.html').read_text(encoding='utf-8')
    insert='''<section class="notice">
<h2>S58 — упорядоченные машинные цепочки (НЕ физическая GT)</h2>
<p>Из 37 стыков S56 построено 35 связанных 2D-гипотез: 34 пары фрагментов и
одна цепочка из четырёх фрагментов. В исходных фотографиях нет экспертных
меток принадлежности линий одной частице. Пунктирные пробелы вычислены,
<strong>а не наблюдались</strong> на плёнке.</p>
<p><a href="S58_MACHINE_PATH_COMPONENTS_PRIVATE.json" download style="color:#bdcfff">Сохранить все цепочки как приватный JSON с происхождением сегментов</a></p>
<img src="S58_LONGEST_MACHINE_PATH_PRIVATE.jpg" alt="Самая длинная машинная гипотеза из четырёх фрагментов, без физического подтверждения" style="max-width:512px;width:100%;border-radius:6px">
</section>'''
    token='<div class="notice">Повторяемость на двух фотографиях'
    if s.count(token)!=1:raise ValueError('Unexpected original S57 HTML')
    s=s.replace(token,insert+'\n'+token)
    if s.count('S35–S57 ·')!=1:raise ValueError('Unexpected prior showcase heading')
    s=s.replace('S35–S57 ·','S35–S58 ·')
    (out_dir/'index.html').write_text(s,encoding='utf-8')
    (out_dir/'README_FIRST_RU.txt').write_text(
       (out_dir/'README_FIRST_RU.txt').read_text(encoding='utf-8')+
       '\nS58: 35 групп машинных фрагментов. JSON отдельно отличает исходные линии от\n'
       'искусственно предполагаемых промежутков. Нет физически подтверждённых треков.\n',encoding='utf-8')
    manifest={str(p.relative_to(out_dir)):sha256(p.read_bytes()).hexdigest()
              for p in sorted(out_dir.rglob('*')) if p.is_file() and p.name!='SHA256_MANIFEST.json'}
    (out_dir/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    out_zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out_dir.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out_dir.parent))
    with zipfile.ZipFile(out_zip) as z:
        if z.testzip() is not None:raise RuntimeError('Corrupted private archive')
    return {'archive':str(out_zip),'bytes':out_zip.stat().st_size,
            'S58_private_fragments':sum(1 for t in private['tiles'] for c in t['components_PRIVATE']),
            'real_physics_joins_authorized':0}


def main():
    p=argparse.ArgumentParser()
    for k in ('s57-dir','s58-private','s58-public','preview','out-dir','out-zip'):
        p.add_argument('--'+k,type=Path,required=True)
    x=p.parse_args()
    print(json.dumps(run(x.s57_dir,x.s58_private,x.s58_public,x.preview,x.out_dir,x.out_zip),indent=2))
if __name__=='__main__':main()
