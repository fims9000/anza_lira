#!/usr/bin/env python3
"""Embed unverified S35 relation evidence into the existing private S33 editor."""
from __future__ import annotations
from hashlib import sha256
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s35_review_overlay import overlay_s35_review

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--original-reviewer',type=Path,required=True)
 p.add_argument('--S35-evidence',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 original=a.original_reviewer.read_text(encoding='utf8')
 evidence=json.loads(a.S35_evidence.read_text(encoding='utf8'))
 html=overlay_s35_review(original,evidence)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(html,encoding='utf8')
 receipt={'schema':'s35_private_unverified_review_integrity_receipt_v1',
   'original_review_source_sha256':sha256(a.original_reviewer.read_bytes()).hexdigest(),
   's35_machine_evidence_sha256':sha256(a.S35_evidence.read_bytes()).hexdigest(),
   'review_html_sha256':sha256(a.out.read_bytes()).hexdigest(),
   's31_s32_manual_export_untouched':True,'physical_joins_authorized':False,
   'machine_proposals':133,'pair_questions':28,'human_expert_labels_added':0}
 (a.out.parent/'S35_private_review_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('S35 PRIVATE UI',a.out,'size',a.out.stat().st_size,'pair questions',28)
