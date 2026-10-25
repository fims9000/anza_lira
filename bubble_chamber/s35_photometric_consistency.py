"""S35 controlled photo-view consistency for source-locked original AZUNet/U-Net.

The labels are S31 MACHINE hypotheses, not human-certified H50Z tracks.
Unknown photo pixels carry no supervised BCE; an optional transform-consistency
constraint uses them ONLY as unlabeled repeated observations of the same crop.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from bubble_chamber.s32_confidence import confidence_bce
from bubble_chamber.s33_photometric_domain import image_input


@dataclass(frozen=True)
class S35Protocol:
    train_seeds: tuple[int, ...] = (20261025, 20261026)
    test_forward_seeds: tuple[int, ...] = (20261041, 20261042, 20261043, 20261044)
    test_reverse_seeds: tuple[int, ...] = (20261045, 20261046, 20261047, 20261048)
    epochs: int = 2
    positives_per_crop: int = 5
    negatives_per_crop: int = 5
    batch_size: int = 4
    lambda_consistency: float = .10
    soft_confidence_margin: float = .08
    model_widths: tuple[int, ...] = (4, 6, 8, 12)
    soft_supervision_source: str = 'machine_provisional_S31_ONLY'
    test_real_native_track_truth: bool = False
    auto_physics_join_authorized: bool = False

    def check(self):
        if self.lambda_consistency != .10 or self.soft_confidence_margin != .08:
            raise ValueError('S35 lambda and weak confidence margin are preregistered and fixed')
        if (self.epochs != 2 or self.batch_size != 4 or self.train_seeds != (20261025,20261026)
            or self.positives_per_crop != 5 or self.negatives_per_crop != 5
            or self.model_widths != (4,6,8,12)):
            raise ValueError('S35 training budget is preregistered and fixed')
        if (self.test_forward_seeds != (20261041,20261042,20261043,20261044)
            or self.test_reverse_seeds != (20261045,20261046,20261047,20261048)):
            raise ValueError('S35 heldout test seeds are preregistered and fixed')
        if self.test_real_native_track_truth is not False or self.auto_physics_join_authorized is not False:
            raise ValueError('Attempt to certify machine predictions as real physical truth')
        return True


def paired_exposure_views(gray: np.ndarray, mode: str, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Same spatial pixels, different randomly exposed cameras; no target shifting.

    First view is the exact nonaugmented S33 input. Second is randomized
    affine exposure, clipped to valid uint8. The original spatial geometry
    and SAME 0/1/255 target are kept for both views. A true self-supervision
    target would NOT be given by the synthetic line labels, which are separate.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 32:
        raise ValueError('Require real H50Z gray patch')
    if mode not in ('quantile','local'):
        raise ValueError('Unregistered S35 photo mode')
    rng=np.random.default_rng(seed)
    a=float(rng.uniform(.74,1.26));b=float(rng.uniform(-28,28))
    observed=np.clip(a*gray.astype(np.float32)+b,0,255).astype(np.uint8)
    return image_input(gray,mode),image_input(observed,mode)


def consistency_loss(logits_a:torch.Tensor,logits_b:torch.Tensor,
                     target:torch.Tensor,confidence:torch.Tensor,*,
                     alpha:float=.10,confidence_margin:float=.08,
                     include_consistency:bool=True):
    """Matched-compute supervised objective + optional detached unlabeled view.

    L = (L_weighted_BCE(f(x),y) + L_weighted_BCE(f(Tx),y))/2
        + alpha E[mask_unknown * (p(Tx) - stopgrad(p(x)))**2]

    Only UNKNOWN photo pixels appear in consistency; no synthetic labels are
    imputed. `include_consistency=False` is a perfectly matched two-view
    photometric augmentation baseline, not an easier single-view control.
    """
    if not (logits_a.shape==logits_b.shape==target.shape==confidence.shape):
        raise ValueError('S35 views/target/weights must have same shapes')
    if not 0<=alpha<=.5 or not 0<=confidence_margin<=.45:
        raise ValueError('Invalid regularization coefficient')
    la=confidence_bce(logits_a,target,confidence)
    lb=confidence_bce(logits_b,target,confidence)
    if la is None or lb is None:raise ValueError('At least one machine labeled pixel required')
    supervised=(la+lb)*.5
    # Need no labels at the UNKNOWN sites: only photometrically paired views.
    unknown=(target==255)&(confidence==0)
    p=torch.sigmoid(logits_a).detach()
    eligible=unknown&(torch.abs(p-.5)>=confidence_margin)
    if bool(eligible.any()) and include_consistency:
        regularizer=F.mse_loss(torch.sigmoid(logits_b)[eligible],p[eligible])
    else:
        regularizer=logits_b.sum()*0.
    total=supervised+(alpha*regularizer if include_consistency else 0.)
    return total,{'supervised':float(supervised.detach()),
                  'consistency':float(regularizer.detach()),
                  'consistency_unknown_pixels':int(eligible.sum().item()) if include_consistency else 0,
                  'supervised_unknown_count':0}


def view_disagreement(p:np.ndarray,q:np.ndarray)->dict:
    """Pixel consistency on same source image; not a track-classification score."""
    p=np.asarray(p,dtype=np.float32);q=np.asarray(q,dtype=np.float32)
    if p.shape!=q.shape or p.ndim!=2 or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Noncomparable or nonfinite photo probabilities')
    if p.size==0 or p.min()<0 or q.min()<0 or p.max()>1 or q.max()>1:
        raise ValueError('Expected probability maps')
    return {'mean_absolute_probability_delta':float(np.mean(np.abs(p-q))),
            'fraction_binary_decision_flipped_at_0_5':float(np.mean((p>=.5)!=(q>=.5))),
            'n_photo_pixels':int(p.size),
            'real_track_accuracy':'NOT_MEASURED'}
