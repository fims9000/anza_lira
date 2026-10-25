"""S36: fit-free saturation-aware normalizer for original AZConv2d.

When a camera/exposure perturbation clips gray values to 0/255, quantile
normalization is no longer affine-invariant. This mixes two existing S33
single-channel inputs using only the image itself, never test labels.
"""
from __future__ import annotations
import numpy as np
from bubble_chamber.s33_photometric_domain import image_input


S36_TEST_FORWARD_SEEDS=(20261049,20261050,20261051,20261052)
S36_TEST_REVERSE_SEEDS=(20261053,20261054,20261055,20261056)
S36_ALPHA_START=.02
S36_ALPHA_FULL=.20


def saturation_blend_input(gray:np.ndarray, *,return_diagnostics=False):
    """z=(1-alpha) z_quantile + alpha z_local.

    alpha = clip((mean[gray<=1 or gray>=254] - .02)/(.20-.02),0,1)

    Clipped sensor values irrecoverably lose signal. This is *only* a reversible
    preprocessing choice, not a calibrated confidence score or physical trace
    detector. Constant images are stable. Both source modes are fit-free.
    """
    img=np.asarray(gray)
    if img.ndim!=2 or img.dtype!=np.uint8 or min(img.shape)<32:raise ValueError('Expected source H50Z gray crop')
    s=float(np.mean((img<=1)|(img>=254)))
    alpha=float(np.clip((s-S36_ALPHA_START)/(S36_ALPHA_FULL-S36_ALPHA_START),0.,1.))
    quantile=image_input(img,'quantile')
    local=image_input(img,'local')
    out=((1.-alpha)*quantile+alpha*local).astype(np.float32)
    if not np.isfinite(out).all() or out.min()<0 or out.max()>1:raise ValueError('Invalid saturation-blended input')
    if return_diagnostics:
        return out,{'fraction_zero_or_255':s,'local_branch_weight_uncalibrated':alpha,
                    'physical_track_probability':False}
    return out


def matched_s36_photo_views(gray:np.ndarray, seed:int):
    if gray.dtype!=np.uint8 or gray.ndim!=2:raise ValueError('Bad photo patch')
    rng=np.random.default_rng(seed)
    a=float(rng.uniform(.74,1.26));b=float(rng.uniform(-28.,28.))
    observed=np.clip(a*gray.astype(np.float32)+b,0,255).astype(np.uint8)
    return saturation_blend_input(gray),saturation_blend_input(observed)


def holdout_exposure(gray:np.ndarray, mode:str):
    """Fixed, nonlearned exposures; highlight_clip is extra S36 stress test."""
    if gray.dtype!=np.uint8 or gray.ndim!=2:raise ValueError('Expected actual source pixel image')
    if mode=='identity':return gray
    if mode=='bright':return np.clip(gray.astype(np.float32)*.80+30,0,255).astype(np.uint8)
    if mode=='dark':return np.clip(gray.astype(np.float32)*1.15-28,0,255).astype(np.uint8)
    if mode=='highlight_clip':return np.clip(gray.astype(np.float32)*1.18+15,0,255).astype(np.uint8)
    raise ValueError('Unregistered S36 exposure stress scenario')
