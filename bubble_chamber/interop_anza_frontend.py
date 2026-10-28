"""Original ANZA (AZUNet) -> skeleton 2D polylines -> Graph-LIRA interop.

The segmentation model is the *actual* models.segmentation.AZUNet, not an
invented ANZA substitute. A verified private checkpoint is mandatory for an
identity that can be detector-specifically calibrated. In-memory toy models
are explicitly marked unverified and are suitable ONLY for interface tests.

All proposed 2D fragment identities and joins remain UNKNOWN on real images.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from hashlib import sha256
from pathlib import Path
import json
from typing import Any

import numpy as np

from .interop_api import MAX_LINES, validate_lines, validate_pixels
from .s14_anza_frontend import AZInferenceConfig, az_mask_to_segments, load_original_az_checkpoint


def _state_sha256(model) -> str:
    """Fingerprint current in-memory weights; guards against silent model swaps."""
    digest=sha256()
    for key,tensor in sorted(model.state_dict().items()):
        a=tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode('utf8')+b'\0')
        digest.update(str(a.dtype).encode('ascii')+b'\0')
        digest.update(str(a.shape).encode('ascii')+b'\0')
        digest.update(a.tobytes())
    return digest.hexdigest()


def _fingerprint(config: AZInferenceConfig, max_lines: int) -> str:
    """Stable fingerprint for every pixel-to-polyline parameter."""
    return sha256(json.dumps({"config": asdict(config), "max_lines": max_lines},
                             sort_keys=True, separators=(',', ':')).encode('utf8')).hexdigest()


@dataclass
class OriginalAZUNetDetector:
    """Adapter for the original private AZUNet model with conservative output.

    The provided `model` MUST correspond to detector_id. For a verified,
    immutable identity, construct via `.from_private_checkpoint()`, which
    includes the raw checkpoint SHA-256 and all extraction parameters in ID.
    In-memory models are only allowed when `toy_only=True`; the ID is then
    prefixed with `TOY_ONLY` and is not eligible for trained-evidence claims.
    """
    model: Any = field(repr=False)
    detector_id: str = 'TOY_ONLY-unverified-AZUNet'
    config: AZInferenceConfig = AZInferenceConfig()
    max_lines: int = 12
    checkpoint_sha256: str | None = None
    model_state_sha256: str | None = None
    toy_only: bool = True

    def __post_init__(self):
        if not isinstance(self.detector_id, str) or not self.detector_id or len(self.detector_id) > 128:
            raise ValueError('Stable nonempty detector_id required')
        if not isinstance(self.max_lines, int) or not (1 <= self.max_lines <= MAX_LINES):
            raise ValueError('AZUNet detector line budget must be within 1..32')
        if not isinstance(self.config, AZInferenceConfig):
            raise ValueError('AZUNet requires the original AZInferenceConfig')
        self.config.validate()
        if not callable(getattr(self.model, 'eval', None)) or not callable(self.model):
            raise ValueError('An actual loaded torch segmentation model is required')
        if self.toy_only:
            if self.checkpoint_sha256 is not None or self.model_state_sha256 is not None or not self.detector_id.startswith('TOY_ONLY-'):
                raise ValueError('Toy in-memory model MUST be explicitly marked TOY_ONLY')
        else:
            if not self.checkpoint_sha256 or len(self.checkpoint_sha256) != 64 or \
               not self.model_state_sha256 or len(self.model_state_sha256) != 64:
                raise ValueError('Verified private checkpoint digest and model-state digest required for non-toy detector')
            fpr = _fingerprint(self.config, self.max_lines)
            expected = 'original-azunet-' + sha256((self.checkpoint_sha256 + fpr).encode()).hexdigest()[:32]
            if self.detector_id != expected or _state_sha256(self.model)!=self.model_state_sha256:
                raise ValueError('Changed AZUNet checkpoint/config/weights without detector ID update')

    @classmethod
    def from_private_checkpoint(cls, path: Path, *, config: AZInferenceConfig=AZInferenceConfig(),
                                max_lines: int=12):
        """Load only a locally trusted checkpoint with frozen original source hashes.

        Important: torch deserialization is for files trusted by the caller;
        the source loader uses weights_only=True and checks original code.
        Source weights and the file itself MUST stay out of public Git.
        """
        path=Path(path)
        if not path.is_file():
            raise FileNotFoundError('ANZA model checkpoint not supplied in source-only archive')
        if not 1 <= max_lines <= MAX_LINES or not isinstance(max_lines,int):
            raise ValueError('Bad detector line budget')
        config.validate()
        digest=sha256(path.read_bytes()).hexdigest()
        model,metadata=load_original_az_checkpoint(path)
        fpr=_fingerprint(config, max_lines)
        detector_id='original-azunet-' + sha256((digest+fpr).encode('utf8')).hexdigest()[:32]
        weights_digest=_state_sha256(model)
        obj=cls(model,detector_id=detector_id,config=config,max_lines=max_lines,
                checkpoint_sha256=digest,model_state_sha256=weights_digest,toy_only=False)
        # This metadata is a reference to weights already held privately.
        return obj, {**metadata, 'detector_id':detector_id,
                     'checkpoint_sha256':digest, 'model_state_sha256':weights_digest,
                     'extractor_config_sha256':fpr,
                     'native_physics_gt_count':0, 'automatic_real_joins_authorized':False}

    def detect(self, gray: np.ndarray) -> list[np.ndarray]:
        """Return finite Nx2 (x,y) open curves in 192px crop coordinates.

        This is purely image-driven. The raw AZUNet mask is skeletonized and
        splits at junctions; no graph edge is selected by the detector.
        """
        validate_pixels(gray)
        if not self.toy_only and _state_sha256(self.model)!=self.model_state_sha256:
            raise ValueError('Loaded ANZA model weights changed; refuse stale pair-scorer/CAL policy')
        # isolate input: the host app and its images are never modified.
        safe_image=gray.copy()
        res,mask,_overlay=az_mask_to_segments(safe_image,self.model,self.config)
        if mask.shape!=(192,192) or res.get('automatic_join_count')!=0:
            raise ValueError('ANZA segmentation unexpectedly changed source/auto-join policy')
        raw=[]
        for record in res['segments']:
            data=record.get('measurement_xy_px')
            pts=np.asarray(data,dtype=np.float64)
            if pts.ndim!=2 or pts.shape[1]!=2 or not np.isfinite(pts).all():
                raise ValueError('Malformed AZUNet pixel-derived polyline')
            if len(pts)<8 or np.any(pts<0) or np.any(pts>191):
                continue
            if np.linalg.norm(pts[-1]-pts[0])<4:
                # Skeleton loops with no open endpoint are not Graph-LIRA joins.
                continue
            length=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum())
            if length<6:
                continue
            raw.append((length,pts))
        # Original AZInferenceConfig can output arbitrarily many short curves;
        # choose long candidates deterministically, never use GT to select.
        raw.sort(key=lambda p:(-p[0],round(float(p[1][0,0]),5),round(float(p[1][0,1]),5)))
        candidates=[p for _,p in raw[:self.max_lines]]
        verified=validate_lines(candidates,max_lines=self.max_lines)
        return list(verified)

    @property
    def research_only(self)->bool:
        return True


def verified_anza_to_graph(gray, detector:OriginalAZUNetDetector, *, receipt,
                           model_file:Path, source_cal_file:Path, score_pairs,
                           expected_training_domain:str, solver_timeout:float=3.):
    """Only trusted AZUNet checkpoint + separate detector-specific Graph CAL.

    A receipt does NOT license automatic physics joins, even when the model
    file was trained with certified positive/negative masks. The score model
    and CAL are separately hashed by `process_verified`.
    """
    if not isinstance(detector,OriginalAZUNetDetector) or detector.toy_only:
        raise ValueError('Unverified/toy AZUNet cannot be used with verified research API')
    from .interop_verified_runtime import process_verified
    return process_verified(gray,detector,receipt=receipt,model_file=model_file,
          source_cal_file=source_cal_file,score_pairs=score_pairs,
          expected_training_domain=expected_training_domain,
          minimum_evidence='synthetic_on_real_photo',solver_timeout=solver_timeout)
