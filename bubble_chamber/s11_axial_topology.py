"""S11 differentiable topology + existing AZConv axis supervision.

S11 trains ORIGINAL AZUNet modules. This extension lives outside models/,
without editing their kernels, and is valid only for synthetic pixel masks
until real expert-confirmed masks are available.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class StructuralWeights:
    cldice: float = 0.20
    axis: float = 0.05
    skeleton_iterations: int = 4
    axis_iterations: int = 4

    def validate(self) -> None:
        if not (0 <= self.cldice <= 2.0 and 0 <= self.axis <= 2.0):
            raise ValueError('Structural loss weights must lie in [0, 2]')
        if not 1 <= self.skeleton_iterations <= 20 or not 1 <= self.axis_iterations <= 20:
            raise ValueError('Invalid soft-skeleton iteration count')


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    r = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    c = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(r, c)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, (3, 3), stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def soft_skeleton(x: torch.Tensor, iterations: int = 4) -> torch.Tensor:
    """Differentiable morphological thinning, no hard threshold at training."""
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError('Expected [B,1,H,W] probability maps')
    if iterations < 1:
        raise ValueError('Need positive skeleton iterations')
    skel = torch.relu(x - _soft_open(x))
    current = x
    for _ in range(iterations):
        current = _soft_erode(current)
        delta = torch.relu(current - _soft_open(current))
        skel = skel + torch.relu(delta - skel * delta)
    return skel.clamp(0, 1)


def cldice_loss(logits: torch.Tensor, target: torch.Tensor, *, iterations: int = 4,
                smooth: float = 1.0) -> torch.Tensor:
    """Soft connectivity Dice; exact target foreground is simulated, not real."""
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError('Invalid segmentation target shape')
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError('Nonfinite segmentation values')
    if bool(((target < 0) | (target > 1)).any()):
        raise ValueError('Target must be foreground in [0,1]')
    prob = torch.sigmoid(logits)
    pred_s = soft_skeleton(prob, iterations)
    truth_s = soft_skeleton(target, iterations)
    reduce = (1, 2, 3)
    precision = ((pred_s * target).sum(dim=reduce) + smooth) / (pred_s.sum(dim=reduce) + smooth)
    sensitivity = ((truth_s * prob).sum(dim=reduce) + smooth) / (truth_s.sum(dim=reduce) + smooth)
    cld = (2 * precision * sensitivity + 1e-8) / (precision + sensitivity + 1e-8)
    return (1 - cld).mean()


def structural_training_loss(
    logits: torch.Tensor, target: torch.Tensor, model: torch.nn.Module,
    base_loss: torch.Tensor, weights: StructuralWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """BCE+Dice plus clDice and original AZ kernel direction alignment.

    target must be a dense CERTIFIED simulated pixel mask. On sparse reviewed
    real labels, this loss is invalid because unknown pixels cannot become bg.
    """
    weights.validate()
    if logits.shape != target.shape:
        raise ValueError('Prediction/target mismatch')
    topo = base_loss.new_zeros(())
    axis = base_loss.new_zeros(())
    if weights.cldice > 0:
        topo = cldice_loss(logits, target, iterations=weights.skeleton_iterations)
    if weights.axis > 0:
        from models.azconv import AZConv2d
        if not any(isinstance(m, AZConv2d) for m in model.modules()):
            raise ValueError('Axis alignment is only defined for authentic AZConv modules')
        func = getattr(model, 'axis_alignment_loss', None)
        if not callable(func):
            raise ValueError('The selected model does not expose original AZ axis alignment')
        axis = func(target, torch.ones_like(target), num_iters=weights.axis_iterations)
        if axis.ndim != 0 or not torch.isfinite(axis):
            raise ValueError('AZ axis alignment returned invalid loss')
    return base_loss + weights.cldice * topo + weights.axis * axis, {'base':base_loss.detach(),
        'cldice':topo.detach(),'axis':axis.detach()}


def binary_cldice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Per-image hard clDice, penalizes missing centerlines and false bridges."""
    p = np.asarray(pred, dtype=bool)
    y = np.asarray(truth, dtype=bool)
    if p.shape != y.shape or p.ndim != 2:
        raise ValueError('Expected aligned HxW binary masks')
    if not (p.any() or y.any()):
        return 1.
    if not p.any() or not y.any():
        return 0.
    sp = skeletonize(p)
    st = skeletonize(y)
    precision = (sp & y).sum() / max(sp.sum(), 1)
    sensitivity = (st & p).sum() / max(st.sum(), 1)
    return float(2 * precision * sensitivity / max(precision + sensitivity, 1e-12))


def per_frame_confusion(pred: np.ndarray, truth: np.ndarray, groups: list[str]) -> dict[str, np.ndarray]:
    """Frame-blocked stats for valid paired clustered inference."""
    if pred.shape != truth.shape or pred.ndim != 3 or len(pred) != len(groups):
        raise ValueError('Mismatched test arrays/groups')
    out: dict[str, np.ndarray] = {}
    for p, y, group in zip(pred.astype(bool), truth.astype(bool), groups):
        vec = np.array([int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())], dtype=np.int64)
        if group not in out:
            out[group] = np.zeros(3, np.int64)
        out[group] += vec
    return out


def count_metrics(rows: dict[str, np.ndarray]) -> dict[str, float | int]:
    c = np.stack(list(rows.values())).sum(axis=0)
    tp, fp, fn = map(int, c)
    return {'tp':tp,'fp':fp,'fn':fn,
            'dice':2*tp / max(2*tp+fp+fn,1),
            'precision':tp/max(tp+fp,1),
            'recall':tp/max(tp+fn,1)}


def paired_frame_bootstrap(base: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
                           *, seed: int = 20261024, draws: int = 1500) -> dict[str, object]:
    groups = sorted(base)
    if len(groups) < 5 or groups != sorted(candidate) or draws < 30:
        raise ValueError('Need >=5 identical frames and >=30 bootstrap draws')
    a = np.stack([base[g] for g in groups])
    b = np.stack([candidate[g] for g in groups])
    rng=np.random.default_rng(seed)
    chosen=rng.integers(0,len(groups),size=(draws,len(groups)))
    aa = a[chosen].sum(axis=1)
    bb = b[chosen].sum(axis=1)
    da=2*aa[:,0]/np.maximum(2*aa[:,0]+aa[:,1]+aa[:,2],1)
    db=2*bb[:,0]/np.maximum(2*bb[:,0]+bb[:,1]+bb[:,2],1)
    ci = np.quantile(db-da,[.025,.975])
    return {'candidate_minus_baseline_dice':float(count_metrics(candidate)['dice']-count_metrics(base)['dice']),
            'ci95':list(map(float,ci)), 'bootstrap_unit':'source_frame',
            'frames':len(groups),'draws':draws}
