"""Uniform-green research review layer for S118, keeping every original path."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from bubble_chamber.s120_reviewer import main


def run(bmp,run_dir,features,model,out,manual=None,training_report=None):
    z=np.load(features)
    cfg=json.loads((run_dir/'RUN_CONFIG.json').read_text())
    if str(z['source_sha256'])!=cfg['source_image_sha256']:
        raise ValueError('S120 features come from a different source BMP')
    result=main(bmp,run_dir,features,model,out,manual)
    if training_report:
        trained=json.loads(training_report.read_text())
        result['model_training_source_sha256']=trained['label_source_sha256']
        result['evaluation_on_training_source']=trained['label_source_sha256']==cfg['source_image_sha256']
        result['manual_precision_recall_not_independent']=True
        (out/'S120_REVIEW_REPORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bmp',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manual-for-qa',type=Path)
    p.add_argument('--training-report',type=Path)
    a=p.parse_args();run(a.bmp,a.run_dir,a.features,a.model,a.out,a.manual_for_qa,a.training_report)
