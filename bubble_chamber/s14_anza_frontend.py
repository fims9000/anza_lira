"""Original AZUNet mask -> auditable P0-compatible fragment/endpoint proposals.

No physics inference: even a trained segmenter has not certified particle
identities, known magnetic field or true source/start/stop orientation.
All 2D joins are proposals only. Works on small HxW crops; do not load full
hundreds-of-megabytes films into naive CPU inference.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import numpy as np
import cv2
import torch
from skimage.morphology import skeletonize
from .pilot import extract_skeleton_segments, arc_sample, fit_quadratic, propose_joins


@dataclass(frozen=True)
class AZInferenceConfig:
    segmentation_threshold:float=.5
    min_segment_length_px:float=12
    points_per_fragment:int=14
    max_input_pixels:int=1048576
    pad_divisor:int=8

    def validate(self):
        if not (.05<=self.segmentation_threshold<=.95 and self.min_segment_length_px>=4
                and self.points_per_fragment>=5 and self.max_input_pixels>=1024 and self.pad_divisor>=8):
            raise ValueError('AZ inference configuration invalid')


def load_original_az_checkpoint(path:Path):
    """Only load trusted, locally produced torch checkpoints!"""
    from models.segmentation import AZUNet
    from models.azconv import AZConvConfig
    ck=torch.load(Path(path),map_location='cpu',weights_only=True)
    if ck.get('architecture')!='azunet' or ck.get('source_git_blobs') is None:
        raise ValueError('source-locked S14 AZUNet checkpoint required')
    from scripts.research.bubble_chamber.s14_train_sparse_anza import MODEL_SOURCE_BLOBS,verify_model_source
    verify_model_source()
    if ck['source_git_blobs'] != MODEL_SOURCE_BLOBS:
        raise ValueError('checkpoint from different source geometry')
    model=AZUNet(in_channels=1,out_channels=1,num_rules=int(ck['rules']),
                 widths=tuple(ck['architecture_widths']),
                 cfg=AZConvConfig(geometry_mode='local_hyperbolic',use_fuzzy=True,use_anisotropy=True))
    model.load_state_dict(ck['model_weights']);model.eval()
    return model,{'training_manifest_sha256':ck['dataset_fingerprint'],
                  'checkpoint_epoch':ck['epoch'],
                  'source_git_blobs':ck['source_git_blobs']}


def az_mask_to_segments(image:np.ndarray,model,config:AZInferenceConfig=AZInferenceConfig()):
    config.validate()
    if image.ndim!=2 or image.dtype!=np.uint8:raise ValueError('AZ front-end needs grayscale uint8')
    h,w=image.shape
    if h*w>config.max_input_pixels:
        raise ValueError('image too large for this CPU path; use an ROI/crop of <= max_input_pixels')
    dh=(-h)%config.pad_divisor;dw=(-w)%config.pad_divisor
    padded=np.pad(image,((0,dh),(0,dw)),mode='edge')
    inp=torch.from_numpy(((255-padded).astype('float32')/255)[None,None])
    model.eval()
    with torch.no_grad():
        prob=torch.sigmoid(model(inp))[0,0,:h,:w].cpu().numpy()
    # Classifier threshold uncalibrated on real dense masks; the binary mask is
    # only an automated candidate extraction, not a trusted annotation.
    mask=prob>=config.segmentation_threshold
    skeleton=skeletonize(mask)
    chains=extract_skeleton_segments(skeleton, min_length_px=config.min_segment_length_px)
    records=[];raw=[]
    for chain in chains:
        if len(chain)<4:continue
        sample=arc_sample(chain,config.points_per_fragment)
        fit=fit_quadratic(sample)
        sid=len(records)
        records.append({'segment_id':sid,'measurement_xy_px':sample[:,::-1].tolist(),
                        'fit':fit,'point_source':'AZUNet pixel-mask skeleton / NOT expert measurements',
                        'real_particle_identity_verified':False})
        raw.append(chain)
    proposals=propose_joins(raw)
    # Returned proposal IDs correspond to `raw` (same order as records).
    result={'pipeline':'s14_original_azunet_sparse_review_v1',
            'source_sha256_grayscale_pixels':sha256(image.tobytes()).hexdigest(),
            'frame_shape_yx':[h,w],
            'fragment_count':len(records),'proposed_joins_count':len(proposals),
            'automatic_join_count':0,
            'segments':records,'join_proposals':proposals,
            'segmentation_threshold':config.segmentation_threshold,
            'probabilities_calibrated_on_dense_real_masks':False,
            'supports_real_particle_identification':False}
    overlay=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    overlay[skeleton]=[64,224,110]
    return result,mask.astype(np.uint8),overlay
