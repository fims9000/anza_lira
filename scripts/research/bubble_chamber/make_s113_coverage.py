#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s113_frame_coverage import frame_roi_coverage,render_coverage,lowres_annotated_bmp_from_7z
from bubble_chamber.s113_fiducials import detect_cross_candidates,draw_cross_candidates

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--annotations',type=Path,default=ROOT/'s113_annotation_results')
 p.add_argument('--out',type=Path,default=ROOT/'s113_annotation_results')
 p.add_argument('--historical-annotated-005-7z',type=Path)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rows=[json.loads(z.read_text(encoding='utf-8')) for z in sorted(a.annotations.glob('*/tracks_crosses_annotation_NOT_GT.json'))]
 if not rows:raise ValueError('First run physicists_s113_annotation_review.py on source ROIs')
 coverage=frame_roi_coverage(rows)
 (a.out/'FULL_FRAME_COVERAGE_NOT_FULL_MODEL_RUN.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2),encoding='utf-8')
 render_coverage(coverage,a.out/'FULL_FRAME_ROI_COVERAGE_005_006.png')
 if a.historical_annotated_005_7z:
  mosaic,diag=lowres_annotated_bmp_from_7z(a.historical_annotated_005_7z,stride=12)
  cv2.imwrite(str(a.out/'FULL_005_HISTORICALLY_OVERPAINTED_LOWRES_ONLY.png'),mosaic)
  (a.out/'FULL_005_MOSAIC_PROVENANCE.json').write_text(json.dumps(diag,indent=2),encoding='utf-8')
  markers=detect_cross_candidates(cv2.cvtColor(mosaic,cv2.COLOR_BGR2GRAY),min_score=.66)
  cv2.imwrite(str(a.out/'FULL_005_X_CANDIDATES_NOT_FIDUCIAL_GT.png'),draw_cross_candidates(mosaic,markers))
  (a.out/'FULL_005_X_CANDIDATES_NOT_GT.json').write_text(json.dumps({
      'schema':'S113_OLD_PAINTED_THUMBNAIL_X_CANDIDATES_NOT_GT',
      'source':'H50Z_030_01_005_classification_segmentation.7z',
      'downsampling_stride':12,'candidate_count':len(markers),
      'operator_verified_fiducials':0,
      'not_usable_for_automatic_training':True,
      'candidates':markers},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:{p:v for p,v in f.items() if p!='rectangles_xyxy'} for k,f in coverage.items()},indent=2))
if __name__=='__main__':main()