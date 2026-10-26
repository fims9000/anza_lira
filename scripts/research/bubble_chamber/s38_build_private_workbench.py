#!/usr/bin/env python3
"""Build a PRIVATE independently blinded H50Z offline reviewer; code is safe for public Git.

The output includes original H50Z image pixels, unverified line coordinates and
image-attached metadata. NEVER commit the generated zip or PNGs to public GitHub.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from pathlib import Path
import json
import zipfile
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

import cv2
import numpy as np
from bubble_chamber.s38_review_protocol import review_queues, source_statistics, canonical_json_sha

ROOT=Path(__file__).resolve().parent


def build(source:Path, target:Path, public_metrics:Path|None=None, *, expected_source_sha256:str|None=None):
    actual_source_sha=sha256(source.read_bytes()).hexdigest()
    if expected_source_sha256 is not None and actual_source_sha.lower()!=expected_source_sha256.lower():
        raise ValueError('Source ZIP SHA-256 does not match fixed expected experiment archive')
    with zipfile.ZipFile(source) as z:
        drafts=json.loads(z.read('annotations/S31_MACHINE_DRAFTS.json'))
        evidence=json.loads(z.read('review/S35_28_REAL_H50Z_PAIR_QUESTIONS_PRIVATE.json'))
        template=json.loads(z.read('annotations/S31_EMPTY_CORRECTIONS_TEMPLATE.json'))
        locked_crops={c['crop_id']:c for c in template['crops']}
        if len(locked_crops)!=39 or len(drafts['photos'])!=39:
            raise ValueError('Expected 39 source crop provenance entries')
        queues=review_queues(drafts,evidence)
        payload={'schema':'s38_private_reviewer_source_v1','photos':drafts['photos'],
                 'relations':evidence['rows'],'template':template,'queues':queues}
        inventory=source_statistics(drafts,evidence,queues)
        payload_sha=canonical_json_sha(payload)
        html=ROOT.joinpath('s38_review_workbench.html').read_text(encoding='utf8')
        if html.count('__PRIVATE_JSON__')!=1:raise ValueError('Missing unique payload slot')
        # Escaping the opening bracket prevents a JSON string from ending script element.
        page=html.replace('__PRIVATE_JSON__', json.dumps(payload,ensure_ascii=True,separators=(',',':')).replace('<','\\u003c'))
        target.parent.mkdir(parents=True,exist_ok=True)
        integrity={'source_archive_sha256':actual_source_sha,
                   'payload_json_canonical_sha256':payload_sha,
                   'source_photo_decoded_pixel_sha256':{},
                   'source_label_status':'MACHINE_DRAFT_UNREVIEWED',
                   'physical_joins_authorized':False}
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as out:
            out.writestr('S38_REVIEWER/index.html', page.encode('utf8'))
            for photo in drafts['photos']:
                name=photo['crop_id']+'.png'
                raw=z.read('source_39_H50Z_crops/'+name)
                frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
                if frame is None or frame.shape!=(512,512):raise ValueError('Missing original pixels '+name)
                digest=sha256(frame.tobytes()).hexdigest()
                if photo['crop_id'] not in locked_crops:
                    raise ValueError('Crop absent from original source template')
                if locked_crops[photo['crop_id']].get('source_gray_pixels_sha256',digest)!=digest:
                    raise ValueError('Template/image pixel SHA-256 mismatch')
                if digest!=photo['source_gray_pixels_sha256']:
                    raise ValueError('Source image checksum mismatch '+name)
                integrity['source_photo_decoded_pixel_sha256'][name]=digest
                out.writestr('S38_REVIEWER/crops/'+name,raw)
            out.writestr('S38_REVIEWER/integrity.json',json.dumps(integrity,ensure_ascii=False,indent=2).encode('utf8'))
            out.writestr('S38_REVIEWER/README_RU.txt',(
                'S38: ПРИВАТНЫЙ разметчик АНЗА / Graph-LIRA\n'
                'Распакуйте весь архив целиком; откройте index.html в браузере.\n'
                'Каждый из двух проверяющих работает независимо (отдельный код; раунды A и B).\n'
                'Сначала просмотрите ВСЕ 39 участков, особенно 17 участков без машинных линий.\n'
                'Если найден пропущенный след — нарисуйте его вручную, не создавая фиктивную метку 0.\n'
                'Затем оцените 133 линии и 28 связей. Связь same/different требует проверки обеих линий.\n'
                'Оценки модели и название метода скрыты; их раскрытие фиксируется в JSON.\n'
                'Экспортируйте PRIVATE JSON перед закрытием. Не отправляйте PNG, ZIP и JSON в публичный GitHub.\n'
                'Всё остаётся LOCAL PROVISIONAL; для физической истины нужна отдельная сертификация.\n'
            ).encode('utf8'))
            out.writestr('S38_REVIEWER/README.txt',
                ('PRIVATE OFFLINE REVIEWER. Unzip completely and open index.html. '\
                 'Round A and round B must be done independently by different reviewers. '\
                 'Every proposal is MACHINE_DRAFT_UNREVIEWED; no physical certification. '\
                 'Export private JSON before exit. NEVER PUSH THIS PACKAGE TO PUBLIC GIT.\n').encode('utf8'))
    if public_metrics:
        public_metrics.parent.mkdir(parents=True,exist_ok=True)
        public_metrics.write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return inventory


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--private-source-zip',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--public-inventory',type=Path)
    ap.add_argument('--expected-source-sha256',type=str)
    a=ap.parse_args()
    print(json.dumps(build(a.private_source_zip,a.out,a.public_inventory,expected_source_sha256=a.expected_source_sha256),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
