"""Validated developer-facing adapter: use the receipt, not an unbound scorer.

Only permits read-only **research** results. Caller must implement and load a
trusted scorer for the explicitly hashed model; this module verifies file
identities and provenance metadata, not the internal callback's semantics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import numpy as np

from .interop_api import Detector,process_image,ResearchResult
from .interop_calibration import CalibrationReceipt


def process_verified(gray:np.ndarray, detector:Detector, *, receipt:CalibrationReceipt,
                     model_file:Path, source_cal_file:Path,
                     score_pairs:Callable, expected_training_domain:str,
                     minimum_evidence:str='synthetic_on_real_photo',
                     solver_timeout:float=3.)->ResearchResult:
    """Prevent accidental old model/CAL policy reuse after detector swap.

    Physics auto-joins are **never** permitted, including expert-labeled
    receipt level. An evidence flag is a self-declared receipt, not proof.
    """
    if not isinstance(receipt,CalibrationReceipt):
        raise ValueError('A fully parsed detector-specific CalibrationReceipt is mandatory')
    if not isinstance(expected_training_domain,str) or not expected_training_domain or \
       receipt.training_domain_id != expected_training_domain:
        raise ValueError('Wrong source TRAIN/CAL domain, rerun calibration for target detector')
    levels={'toy_only':0,'synthetic_on_real_photo':1,'expert_labeled_native':2}
    if minimum_evidence not in levels:
        raise ValueError('Unknown minimum evidence policy')
    if levels[receipt.validation_level]<levels[minimum_evidence]:
        raise ValueError('A toy-only callback is not verified for target-photo research')
    binding=receipt.binding(detector_id=getattr(detector,'detector_id',None),
                            model_file=Path(model_file),
                            calibration_file=Path(source_cal_file),scorer=score_pairs)
    result=process_image(gray,detector,binding,solver_timeout=solver_timeout)
    assert result.native_physics_gt_count==0 and not result.physical_auto_joins_authorized
    return result
