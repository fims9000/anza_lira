#!/usr/bin/env python3
"""Original ANZA AZUNet -> 2D line adapter -> advisory Graph-LIRA example.

Without --checkpoint, the ORIGINAL ANZA architecture is randomly initialized,
only to exercise code and shape contracts. No pixel GT, trained model, CAL or
native particle claim exists. Even with a private checkpoint this example uses
strategy=none; use verified_anza_to_graph() with a detector-specific frozen
S56 TRAIN model and CAL receipt before research-only join proposals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch

from bubble_chamber.interop_anza_frontend import OriginalAZUNetDetector
from bubble_chamber.interop_api import ScoreBinding, process_image
from bubble_chamber.s14_anza_frontend import AZInferenceConfig


def generated_gray():
    gray=np.full((192,192),207,np.uint8)
    for start,stop in ((24,78),(95,159)):
        cv2.line(gray,(start,64),(stop,61),29,1)
        cv2.line(gray,(start,127),(stop,130),39,1)
    return gray


def main(argv=None):
    p=argparse.ArgumentParser(description='S85 ANZA AZUNet -> Graph-LIRA, research-only demo')
    p.add_argument('--checkpoint',type=Path,
                   help='Trusted original AZUNet checkpoint; never included with public source')
    a=p.parse_args(argv)
    torch.set_num_threads(min(torch.get_num_threads(),2))
    if a.checkpoint is not None:
        detector,metadata=OriginalAZUNetDetector.from_private_checkpoint(a.checkpoint)
        has_private_checkpoint=True
    else:
        from models.segmentation import AZUNet
        from models.azconv import AZConvConfig
        torch.manual_seed(14085)
        # This creates the REAL original source architecture, NOT fitted weights.
        model=AZUNet(in_channels=1,out_channels=1,num_rules=2,widths=(4,4,4,4),
                     cfg=AZConvConfig(geometry_mode='local_hyperbolic',use_fuzzy=True,
                                      use_anisotropy=True))
        detector=OriginalAZUNetDetector(model,config=AZInferenceConfig(),max_lines=12)
        metadata={}
        has_private_checkpoint=False
    # A detector-specific TRAIN/CAL score model is not supplied to this demo.
    # Explicit NONE avoids pretending that .85 is a calibrated Graph-LIRA cutoff.
    no_scorer=ScoreBinding('TOY_ONLY_NO_LEARNED_GEOMETRY',detector.detector_id,
                           score_pairs=lambda rows: np.zeros(len(rows),dtype=float),
                           cutoff=1.0,margin=.04,source_cal_id='NOT_CALIBRATED',
                           strategy='none')
    result=process_image(generated_gray(),detector,no_scorer)
    print(json.dumps({**result.as_public_dict(),
        'source_model_architecture':'ORIGINAL_models.segmentation.AZUNet',
        'ANZA_checkpoint_loaded':has_private_checkpoint,
        'ANZA_detector_id':detector.detector_id,
        'ANZA_checkpoint_sha256':metadata.get('checkpoint_sha256'),
        'input_generated_not_H50Z':True,
        'graph_scorer_trained':False,
        'graph_source_calibrated':False,
        'native_track_accuracy_evaluated':False,
        'research_only':True},ensure_ascii=False,indent=2))
    assert len(result.joins)==0 and not result.physical_auto_joins_authorized
    return result

if __name__=='__main__':main()
