"""S27: seam-safe overlapping-tile inference for source-locked ANZA / U-Net.

This changes ONLY inference aggregation. It does not train the architecture,
create expert pixel labels, join tracks, or claim physical accuracy.
"""
from __future__ import annotations

from collections.abc import Callable
import numpy as np
import torch


def tile_starts(length: int, tile: int=64, stride: int=32) -> list[int]:
    if tile<16 or stride<1 or stride>tile or length<tile:
        raise ValueError('invalid tile, stride, or image dimensions')
    pos=list(range(0, length-tile+1, stride))
    if pos[-1]!=length-tile:pos.append(length-tile)
    return pos


def blend_weights(tile: int=64, floor: float=.05) -> np.ndarray:
    if tile<16 or not 0<floor<=1:raise ValueError('invalid blend parameters')
    w=np.hanning(tile).astype('float32')
    return np.maximum(w[:,None]*w[None,:],floor).astype('float32')


def overlap_predict(gray: np.ndarray, model: torch.nn.Module,
                    preprocess: Callable[[np.ndarray],np.ndarray], *,
                    tile: int=64, stride: int=32, batch_size: int=32,
                    floor: float=.05) -> np.ndarray:
    """Overlap-averaged [H,W] sigmoid probabilities with strictly positive weights.

    No padding/rescaling of real photos; image and output share the exact pixel
    coordinates and shape. The output is a probability *candidate map*, not
    particle identity or calibrated segmentation.
    """
    if gray.ndim!=2 or gray.dtype!=np.uint8:raise ValueError('grayscale uint8 required')
    h,w=gray.shape
    if batch_size<1:raise ValueError('invalid batch size')
    ys=tile_starts(h,tile,stride);xs=tile_starts(w,tile,stride)
    weight=blend_weights(tile,floor)
    numerator=np.zeros((h,w),dtype='float32')
    denominator=np.zeros((h,w),dtype='float32')
    windows=[(y,x) for y in ys for x in xs]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(windows),batch_size):
            block=windows[start:start+batch_size]
            features=[]
            for y,x in block:
                patch=gray[y:y+tile,x:x+tile]
                arr=np.asarray(preprocess(patch),dtype='float32')
                if arr.ndim==2:arr=arr[None]
                if arr.ndim!=3 or arr.shape[1:]!=(tile,tile) or not np.isfinite(arr).all():
                    raise ValueError('preprocessor produced invalid shape or nonfinite features')
                features.append(arr)
            sample=torch.from_numpy(np.stack(features))
            outputs=model(sample).sigmoid()
            if outputs.shape!=(len(block),1,tile,tile):
                raise ValueError('invalid network output shape')
            scores=outputs[:,0].cpu().numpy().astype('float32')
            for (y,x),prediction in zip(block,scores):
                numerator[y:y+tile,x:x+tile] += prediction*weight
                denominator[y:y+tile,x:x+tile] += weight
    if not np.all(denominator>0):raise AssertionError('incomplete image coverage')
    result=numerator/denominator
    if not np.isfinite(result).all() or np.min(result)<0 or np.max(result)>1:
        raise AssertionError('invalid blended probabilities')
    return result


def tile_boundary_jump(score: np.ndarray, tile: int=64) -> dict:
    """Diagnostic only: discontinuities at previous 64px inference boundaries.

    The comparison is absolute mean jumps, not track accuracy. Bright native
    film marks may legitimately cross some boundaries.
    """
    a=np.asarray(score,dtype='float32')
    if a.ndim!=2 or min(a.shape)<2*tile or not np.isfinite(a).all():
        raise ValueError('finite 2D probability map required')
    vert=[];horiz=[];controls=[]
    for x in range(tile,a.shape[1],tile):
        vert.append(np.abs(a[:,x]-a[:,x-1]).mean())
        for shift in (-tile//2,tile//2):
            k=x+shift
            if 1<=k<a.shape[1]:controls.append(np.abs(a[:,k]-a[:,k-1]).mean())
    for y in range(tile,a.shape[0],tile):
        horiz.append(np.abs(a[y,:]-a[y-1,:]).mean())
        for shift in (-tile//2,tile//2):
            k=y+shift
            if 1<=k<a.shape[0]:controls.append(np.abs(a[k,:]-a[k-1,:]).mean())
    actual=float(np.mean(vert+horiz));typical=float(np.mean(controls))
    return {'mean_boundary_jump':actual,'mean_control_jump':typical,
            'boundary_to_control_ratio':actual/max(typical,1e-6),
            'mean_x_boundary_jump':float(np.mean(vert)),
            'mean_y_boundary_jump':float(np.mean(horiz)),
            'interpretation':'IMAGE INFERENCE SEAM PROXY, NOT HUMAN TRACK ACCURACY'}


def disagreement_stats(old: np.ndarray,new: np.ndarray,threshold: float)->dict:
    if old.shape!=new.shape or old.ndim!=2 or not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('invalid probability pair')
    if not 0<threshold<=1:raise ValueError('invalid existing validation threshold')
    a=old>=threshold;b=new>=threshold
    inter=int((a&b).sum());union=int((a|b).sum())
    return {'mean_absolute_score_change':float(np.abs(old-new).mean()),
            'original_selected_fraction_UNVERIFIED':float(a.mean()),
            'overlap_selected_fraction_UNVERIFIED':float(b.mean()),
            'iou_between_own_candidate_masks_NOT_GT':float(inter/union) if union else None,
            'candidate_pixel_fraction_changed':float(np.mean(a^b)),
            'no_real_expert_track_labels':True}


def overlap_predict_global_features(gray: np.ndarray, model: torch.nn.Module,
                                   preprocess: Callable[[np.ndarray],np.ndarray], *,
                                   tile: int=64,stride:int=32,batch_size:int=32,
                                   floor:float=.05)->np.ndarray:
    """One full-image contrast/ridge pass, then tiled neural receptive fields.

    Unlike per-window contrast preprocessing, this avoids artificial changes
    to input features at 64px tile edges. Model weights are untouched.
    A 512px crop is not physically rescaled.
    """
    if gray.ndim!=2 or gray.dtype!=np.uint8:raise ValueError('gray uint8 image required')
    h,w=gray.shape
    features=np.asarray(preprocess(gray),dtype='float32')
    if features.ndim==2:features=features[None]
    if features.ndim!=3 or features.shape[1:]!=(h,w) or not np.isfinite(features).all():
        raise ValueError('global feature extractor shape/values invalid')
    if batch_size<1:raise ValueError('invalid batch size')
    ys=tile_starts(h,tile,stride);xs=tile_starts(w,tile,stride)
    positions=[(y,x) for y in ys for x in xs]
    numerator=np.zeros((h,w),'float32');denom=np.zeros((h,w),'float32')
    weight=blend_weights(tile,floor)
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(positions),batch_size):
            loc=positions[start:start+batch_size]
            xx=torch.from_numpy(np.stack([features[:,y:y+tile,x:x+tile] for y,x in loc]))
            prob=model(xx).sigmoid()
            if prob.shape!=(len(loc),1,tile,tile):
                raise ValueError('unexpected model output')
            for (y,x),z in zip(loc,prob[:,0].cpu().numpy()):
                numerator[y:y+tile,x:x+tile]+=z*weight
                denom[y:y+tile,x:x+tile]+=weight
    if not (denom>0).all():raise AssertionError('hole in blending coverage')
    answer=numerator/denom
    if not np.isfinite(answer).all() or answer.min()<0 or answer.max()>1:
        raise AssertionError('invalid output range')
    return answer
