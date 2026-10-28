"""Serializable detector/geometry-scorer binding receipt for external teams.

The receipt binds an external model and its numeric CAL threshold to one
specific pixel detector and training domain. It does not itself contain
private weights, claim native track truth, or authorize automatic joins.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import re
from pathlib import Path
from typing import Callable

from .interop_api import IMAGE_SHAPE, ScoreBinding, STRATEGIES

SCHEMA='ANZA-LIRA-DET-SCORE-CAL-RECEIPT-V1'
VALIDATION=('toy_only','synthetic_on_real_photo','expert_labeled_native')
HEX64=re.compile(r'[0-9a-f]{64}\Z')


@dataclass(frozen=True)
class CalibrationReceipt:
    detector_id: str
    scorer_id: str
    model_sha256: str
    source_cal_sha256: str
    source_cal_id: str
    training_domain_id: str
    cutoff: float
    margin: float
    strategy: str
    validation_level: str
    image_height: int = IMAGE_SHAPE[0]
    image_width: int = IMAGE_SHAPE[1]
    schema: str = SCHEMA

    def __post_init__(self):
        if self.schema!=SCHEMA or (self.image_height,self.image_width)!=IMAGE_SHAPE:
            raise ValueError('Unrecognized source policy/image contract version')
        if any(not isinstance(x,str) or not x or len(x)>256 for x in (
             self.detector_id,self.scorer_id,self.source_cal_id,self.training_domain_id)):
            raise ValueError('Require nonempty detector/scorer/domain/calibration IDs')
        if not HEX64.fullmatch(self.model_sha256) or not HEX64.fullmatch(self.source_cal_sha256):
            raise ValueError('Model and CAL SHA256 must be lowercase 64-char checksums')
        if self.validation_level not in VALIDATION:
            raise ValueError('Unknown evidence level; no claimed physical validation by default')
        if self.strategy not in STRATEGIES or not 0<=self.cutoff<=1 or self.margin<0:
            raise ValueError('Unsupported fixed policy or threshold')
        import math
        if not math.isfinite(self.cutoff) or not math.isfinite(self.margin):
            raise ValueError('Policy must be finite')

    @classmethod
    def from_file(cls,path:Path):
        data=json.loads(Path(path).read_text(encoding='utf8'))
        if not isinstance(data,dict) or set(data)!={f.name for f in fields(cls)}:
            raise ValueError('Manifest must contain EXACTLY the documented schema keys')
        return cls(**data)

    def binding(self, *, detector_id:str, model_file:Path, calibration_file:Path, scorer:Callable):
        if detector_id!=self.detector_id:
            raise ValueError('Detector identity changed; this learned CAL policy is not portable')
        if sha256(Path(model_file).read_bytes()).hexdigest()!=self.model_sha256:
            raise ValueError('Model checksum mismatch (do not silently load other coefficients)')
        if sha256(Path(calibration_file).read_bytes()).hexdigest()!=self.source_cal_sha256:
            raise ValueError('Source-CAL checksum mismatch (no calibration transfer)')
        return ScoreBinding(self.scorer_id,self.detector_id,scorer,self.cutoff,self.margin,
                            self.source_cal_id,self.strategy)

    def public_dict(self):
        out=asdict(self)
        out.update(native_physics_gt_count=0,physical_auto_joins_authorized=False)
        return out
