"""S32: risk-aware sampling and *actual* confidence-weighted machine labels.

All targets are S31 provisional machine inferences, never physics ground truth.
Unknown=255 has exactly zero loss and no path to a pseudo-negative assignment.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import numpy as np
import torch
from torch.nn import functional as F


def confidence_bce(logits: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor,
                   *, negative_floor: float = .12, negative_cap: float = .35) -> torch.Tensor | None:
    """Class-balanced, confidence-aware BCE with bounded uncertain-negative influence.

    Each class is normalized by its population of observed marks (not by its
    total confidence), then multiplied by its mean reliability. The uncertain
    negative coefficient is clipped to [floor, cap] RELATIVE TO the foreground
    coefficient to avoid a near all-positive collapse when machines give every
    pseudo-negative a very small confidence. This floor is an experimental
    regularizer, NOT evidence negatives are truly background.
    """
    if logits.shape != target.shape or logits.shape != confidence.shape:
        raise ValueError('S32 logits / targets / confidence shape mismatch')
    if torch.any((target != 0) & (target != 1) & (target != 255)):
        raise ValueError('Only machine marks 0, 1, 255 permitted')
    if not bool(torch.isfinite(confidence).all()) or bool((confidence < 0).any()) or bool((confidence > 1).any()):
        raise ValueError('Machine confidence must be finite in [0,1]')
    unknown = target == 255
    if bool((confidence[unknown] != 0).any()):
        raise ValueError('Unknown source photo pixels must have zero weight')
    observed = (target != 255) & (confidence > 0)
    if not bool(observed.any()):
        return None
    y = torch.where(target == 1, .92, .08).to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    positive = (target == 1) & observed
    negative = (target == 0) & observed
    if bool(positive.any()) and bool(negative.any()):
        wp = confidence[positive]
        wn = confidence[negative]
        # Weights genuinely affect within-class training, unlike S31's ignored
        # confidence argument. Explicit normalizer avoids overwhelming rare marks.
        lp = (bce[positive] * wp).mean()
        ln = (bce[negative] * wn).mean()
        relative = (wn.mean() / wp.mean()).clamp(min=negative_floor, max=negative_cap)
        return (lp / wp.mean() + relative * (ln / wn.mean())) / (1 + relative)
    if bool(positive.any()):
        wp = confidence[positive]
        return (bce[positive] * wp).mean() / wp.mean()
    wn = confidence[negative]
    return (bce[negative] * wn).mean() / wn.mean()


def balanced_weak_windows(rows, *, patch: int = 64, positives_per_crop: int = 7,
                          negatives_per_crop: int = 7, seed: int = 20261025):
    """Source-safe patch selection around both machine positive and unsure negative pixels.

    Does NOT create marks: entire patch is the original 0/1/255 mask. Does not
    mix source photos. Deterministic under the seed and crop-ID SHA.
    """
    if patch != 64 or positives_per_crop < 0 or negatives_per_crop < 0:
        raise ValueError('Fixed 64px windows and nonnegative selection budgets')
    result=[]
    for image, mask, confidence, rec in rows:
        if image.shape != (512,512) or mask.shape != image.shape or confidence.shape != image.shape:
            raise ValueError('Unexpected real H50Z crop shape')
        if np.any(confidence[mask==255]!=0) or not np.isfinite(confidence).all():
            raise ValueError('UNKNOWN pixels cannot have confidence')
        cid=rec['crop_id'];seed_val=int(sha256(f'S32|{seed}|{cid}'.encode()).hexdigest()[:16],16)
        rng=np.random.default_rng(seed_val)
        for category,count in ((1,positives_per_crop),(0,negatives_per_crop)):
            coords=np.argwhere((mask==category)&(confidence>0))
            if not len(coords):continue
            picks=coords[rng.integers(len(coords),size=count)]
            for cy,cx in picks:
                y=int(np.clip(int(cy)-patch//2,0,512-patch))
                x=int(np.clip(int(cx)-patch//2,0,512-patch))
                region=mask[y:y+patch,x:x+patch].copy()
                assert (region==category).any()
                result.append((image[y:y+patch,x:x+patch].copy(),region,
                               confidence[y:y+patch,x:x+patch].copy(),cid,category))
    return result


def patch_coverage(patches):
    classes=Counter(int(p[4]) for p in patches)
    counts=Counter()
    for _,mark,weight,_,_ in patches:
        counts['pseudo_positive']+=int((mark==1).sum())
        counts['assumed_negative']+=int((mark==0).sum())
        counts['unknown']+=int((mark==255).sum())
        assert not np.any(weight[mark==255]>0)
    return {'windows':len(patches),'positive_centered':classes[1],
            'negative_centered':classes[0],**dict(counts),
            'human_confirmed_labels':0,'physical_validity':'UNKNOWN'}
