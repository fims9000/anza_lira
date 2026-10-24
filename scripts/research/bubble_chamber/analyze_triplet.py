#!/usr/bin/env python3
"""Run *independent 2D* extraction on three checksum-verified film views.

3-view output is NOT a matched 3D event, even if calibration files exist.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import cv2
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bubble_chamber.stereo_dataset import verified_triplet, get_event
from bubble_chamber.pilot import PilotConfig, analyze_frame


def analyze(manifest: dict, event_id: str, out_dir: Path, cfg: PilotConfig) -> dict:
    frames = verified_triplet(manifest, event_id)
    event = get_event(manifest, event_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for i, frame in enumerate(frames, start=1):
        result, mask, overlay = analyze_frame(frame, cfg)
        target = out_dir / f'view{i}'
        target.mkdir(parents=True, exist_ok=True)
        (target / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        cv2.imwrite(str(target / 'mask.png'), mask)
        cv2.imwrite(str(target / 'overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        outputs.append({'view_id':i,'photo_sha256':event['views'][i-1]['sha256'],
                        'fragments':result['fragment_count'], 'review_only_joins':result['proposed_joins_count'],
                        'relative_output':f'view{i}/result.json'})
    summary={'schema':'bubble_independent_three_views_v1','event_id':event_id, 'group_id':event['group_id'],
             'source_split':event['split'], 'calibration_status':event['calibration_status'],
             'physical_result':'NO_STEREO_MATCHING_NO_3D_NO_MOMENTUM',
             'auto_joins':0,'views':outputs}
    (out_dir/'triplet_result.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--event-id',required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--segmentation',choices=('contrast','frangi'),default='frangi')
    parser.add_argument('--ridge-fraction',type=float,default=.16)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    cfg=PilotConfig(segmentation_method=args.segmentation,ridge_response_fraction=args.ridge_fraction)
    summary=analyze(manifest,args.event_id,args.out_dir,cfg)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
