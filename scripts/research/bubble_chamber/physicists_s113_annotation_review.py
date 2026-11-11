#!/usr/bin/env python3
"""Build private, offline S113 expert overlays for original H50Z sampled ROIs."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s113_review_output import build_annotation,render_png,render_html,validate_review_decisions

CASES=('005_start2_1200','005_start3_1200','005_start5_1200',
       '006_start1_1200','006_start2_1200','006_start3_1200')


def run(root:Path,out:Path,selected=CASES,labels_json:Path|None=None):
    out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for roi in selected:
        report=build_annotation(root,roi,labels_json)
        gray=cv2.imread(str(root/'s108_results_all'/roi/'06_SOURCE_ORIGINAL_GRAY.png'),0)
        target=out/roi;target.mkdir(parents=True,exist_ok=True)
        (target/'tracks_crosses_annotation_NOT_GT.json').write_text(
            json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (target/'interactive_annotation_review.html').write_text(render_html(report,gray),encoding='utf-8')
        cv2.imwrite(str(target/'ANNOTATION_A_B_CROSSES_TRACKS.png'),render_png(report,gray))
        cv2.imwrite(str(target/'ANNOTATION_ALL_ENDS_DETAILED.png'),render_png(report,gray,end_marker_limit=None))
        summary.append({k:report[k] for k in ('roi','source_png_sha256','S111_global_group_count',
                      'S111_curve_count','protected_zone_candidates','small_protected_fiducial_candidates','large_scan_artifact_zones',
                      'tracks_with_protected_region_collision','geometry_endpoints_explicitly_available')})
    (out/'overview.json').write_text(json.dumps({'schema':'S113_ROI_AUDIT_V1',
                'all_frame_pixels_evaluated':False,'only_six_ROI_audited':True,
                'summary':summary},ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out',type=Path,default=ROOT/'s113_annotation_results')
    p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    p.add_argument('--operator-labels',type=Path,help='PRIVATE original H50Z labels.json')
    p.add_argument('--import-review',type=Path,help='Validate existing expert JSON (never train from raw machine labels)')
    args=p.parse_args()
    if args.import_review:
        data=json.loads(args.import_review.read_text(encoding='utf-8'))
        print(json.dumps(validate_review_decisions(build_annotation(args.root,data['roi'],args.operator_labels),data),ensure_ascii=False,indent=2))
    else:
        print(json.dumps(run(args.root,args.out,args.cases,args.operator_labels),ensure_ascii=False,indent=2))

if __name__=='__main__':main()