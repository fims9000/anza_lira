#!/usr/bin/env python3
"""Build prioritized S19 review HTML from the original provenance-locked package."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s19_real_review import load_package,make_offline_review_html
from bubble_chamber.s20_review_priority import rank_review_questions,integrate_priority_ui


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--proposals',required=True,type=Path)
    p.add_argument('--tiles',required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    args=p.parse_args()
    package=load_package(args.proposals,args.tiles)
    priority=rank_review_questions(package)
    html=integrate_priority_ui(make_offline_review_html(package,args.tiles),priority)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'S20_prioritized_expert_workbench.html').write_text(html,encoding='utf-8')
    (args.out_dir/'S20_review_questions.json').write_text(json.dumps(priority,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'candidate_questions':len(priority['candidate_questions']),
                      'pair_questions':len(priority['pair_questions']),
                      'expert_confirmed':0,'html':str(args.out_dir/'S20_prioritized_expert_workbench.html')},indent=2))
if __name__=='__main__':main()
