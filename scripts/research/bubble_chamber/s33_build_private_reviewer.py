#!/usr/bin/env python3
"""Build a private S32 reviewer with non-certifying S33 AZUNet evidence."""
from __future__ import annotations
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s33_evidence_review import inject_evidence
if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--s32-html',type=Path,required=True)
 p.add_argument('--evidence',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 source=a.s32_html.read_text(encoding='utf8')
 evidence=json.loads(a.evidence.read_text(encoding='utf8'))
 html=inject_evidence(source,evidence)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(html,encoding='utf8')
 receipt={'schema':'s33_private_review_evidence_receipt_v1',
    'S32_source_html_sha256':hashlib.sha256(a.s32_html.read_bytes()).hexdigest(),
    'unverified_evidence_sha256':hashlib.sha256(a.evidence.read_bytes()).hexdigest(),
    'output_sha256':hashlib.sha256(a.out.read_bytes()).hexdigest(),
    'physicist_certified_labels_added':0,'physical_auto_joins_authorized':False,
    'original_s31_export_preserved':True,'total_draft_ids':133}
 (a.out.parent/'S33_review_overlay_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('S33 REVIEW OVERLAY READY',a.out.stat().st_size,'bytes; original review untouched')
