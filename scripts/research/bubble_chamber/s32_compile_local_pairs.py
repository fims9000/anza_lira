#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s32_relation_bridge import compile_local_relation_reviews
if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--drafts',type=Path,required=True);p.add_argument('--corrections',type=Path)
 p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 d=json.loads(a.drafts.read_text())
 c=json.loads(a.corrections.read_text()) if a.corrections else None
 report=compile_local_relation_reviews(d,c)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print({k:report[k] for k in ('total_candidate_questions','locally_reviewed_same','locally_reviewed_different',
      'expert_certified_relation_count','eligible_for_provisional_graph_training')})
