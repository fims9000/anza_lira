#!/usr/bin/env python3
"""RUNNABLE toy-only receipt/ROI/swap demonstration. NO real H50Z GT.

The mock files below are generated locally and deleted. This example does
NOT claim an actual trained score; pass a real trusted scorer, frozen TRAIN
weights and CAL artifact in your application instead.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2
import numpy as np

from bubble_chamber.interop_calibration import CalibrationReceipt
from bubble_chamber.interop_coordinates import CropTransform,private_review_payload
from bubble_chamber.interop_detectors import OpenCVLSDDetector
from bubble_chamber.interop_verified_runtime import process_verified


def main():
    # No actual H50Z: generated 512x512 input, explicit 192px center crop.
    source=np.full((512,512),208,dtype=np.uint8)
    cv2.line(source,(186,213),(278,211),36,2)
    cv2.line(source,(298,209),(413,211),24,2)
    crop=CropTransform(x0=160,y0=160,full_width=512,full_height=512)
    detector=OpenCVLSDDetector(max_lines=12)
    with tempfile.TemporaryDirectory(prefix='S85_TOY_ONLY_RECEIPT_') as temp:
        model=Path(temp)/'TOY_MODEL_BYTES_NOT_TRAINED'
        cal=Path(temp)/'TOY_SOURCE_CAL_NOT_H50Z'
        model.write_bytes(b'NEVER_DEPLOY_UNTRAINED_TOY_SCORER')
        cal.write_bytes(b'NO_CAL_SOURCE_DATA_TOY_EXAMPLE')
        receipt=CalibrationReceipt(
            detector.detector_id,'TOY-UNTRAINED-PAIR-RECEIPT',sha256(model.read_bytes()).hexdigest(),
            sha256(cal.read_bytes()).hexdigest(),'SYNTHETIC_UI_DEMO_ONLY',
            'GENERATED_512_GRAY_DOMAIN',cutoff=.85,margin=.04,strategy='none',
            validation_level='toy_only')
        result=process_verified(crop.extract(source),detector,receipt=receipt,
             model_file=model,source_cal_file=cal,
             score_pairs=lambda rows:np.zeros((len(rows),),dtype=float),
             expected_training_domain='GENERATED_512_GRAY_DOMAIN',
             minimum_evidence='toy_only')
        # PRIVATE coords exist only inside this process; not printed or committed.
        private=private_review_payload(result,crop,source_image_id='PRIVATE_GENERATED_TOY')
        assert private['publication_safe'] is False
        print(json.dumps({**result.as_public_dict(),
             'source_input_generated_not_H50Z':True,
             'source_roi_xy':[crop.x0,crop.y0],
             'verified_receipt_sha256':receipt.model_sha256,
             'private_review_positions_exported':False,
             'known_real_physics_tracks':0,
             'this_mock_scorer_is_not_trained':True},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
