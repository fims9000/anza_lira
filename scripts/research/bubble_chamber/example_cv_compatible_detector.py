#!/usr/bin/env python3
"""OpenCV LSD/Hough v2 (both return layouts) generated-only comparison. NO CAL."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from bubble_chamber.interop_api import process_image,ScoreBinding
from bubble_chamber.interop_cv_compat import OpenCVLSDCompatibleDetector,OpenCVHoughCompatibleDetector


def main():
    image=np.full((192,192),207,np.uint8)
    for y in (52,114):
        cv2.line(image,(19,y),(168,y+3),28,2)
    outs=[]
    for cls in (OpenCVLSDCompatibleDetector,OpenCVHoughCompatibleDetector):
        detector=cls(max_lines=12)
        binding=ScoreBinding('TOY_ONLY_NO_TRAINED_GEOMETRY',detector.detector_id,
                 lambda rows:np.zeros(len(rows),float),cutoff=1.,
                 source_cal_id='NOT_CALIBRATED',strategy='none')
        out=process_image(image,detector,binding)
        outs.append({**out.as_public_dict(),'generated_input_only':True,'trained':False})
    print(json.dumps(outs,ensure_ascii=False,indent=2))
    assert all(not z['physical_auto_joins_authorized'] and z['advisory_joins_UNKNOWN']==0 for z in outs)
if __name__=='__main__':main()
