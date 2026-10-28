#!/usr/bin/env python3
"""Runnable integration example: generated image and uncalibrated toy heuristic.

For a real detector replace `OpenCVLSDDetector` and provide a detector-specific
verified CalibrationReceipt + private score model. No H50Z accuracy claims.
"""
from __future__ import annotations
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import json
import cv2
import numpy as np
from bubble_chamber.interop_api import ScoreBinding,process_image
from bubble_chamber.interop_detectors import OpenCVLSDDetector
from scripts.research.bubble_chamber.interop_comparison import score_geometry_for_QA


def main():
    gray=np.full((192,192),208,np.uint8)
    cv2.line(gray,(22,55),(87,57),25,2)
    cv2.line(gray,(100,58),(164,60),28,2)
    detector=OpenCVLSDDetector(max_lines=12)
    # Only to illustrate the API. This is NOT a trained S56 score nor CAL.
    binding=ScoreBinding('TOY-UNTRAINED-GEOMETRY',detector.detector_id,
                         score_geometry_for_QA,cutoff=.70,margin=.04,
                         source_cal_id='TOY-NOT-CALIBRATED',strategy='port_greedy')
    result=process_image(gray,detector,binding)
    print(json.dumps({**result.as_public_dict(),
         'input_is_generated':True,'model_calibrated':False,
         'ready_for_real_physics':False},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
