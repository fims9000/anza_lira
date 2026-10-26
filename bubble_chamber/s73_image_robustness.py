"""S73 frozen sensor/photo stressors, strictly pixel-only and deterministic.

No synthetic truth, native film labels, particle IDs or reference coordinates
enter any perturbation. This is a diagnostic transformation, NOT a simulator
of verified physical bubble-chamber acquisition artifacts.
"""
from __future__ import annotations
import cv2
import numpy as np

CONDITIONS=('clean','contrast_072','blur_sigma_11','grain_sigma_9','two_film_streaks')


def perturb(gray:np.ndarray,condition:str,seed:int)->np.ndarray:
    if not isinstance(gray,np.ndarray) or gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('S73 uses uint8 192x192 raster pixels only')
    if condition not in CONDITIONS:raise ValueError('Unfrozen domain shift')
    if not isinstance(seed,(int,np.integer)) or not 0<=seed<2**32:
        raise ValueError('Need finite explicit deterministic seed')
    if condition=='clean':return gray.copy()
    if condition=='contrast_072':
        return np.clip(np.rint((gray.astype('float32')-128.)*.72+128.),0,255).astype('uint8')
    if condition=='blur_sigma_11':
        return cv2.GaussianBlur(gray,(0,0),sigmaX=1.1,sigmaY=1.1).astype('uint8')
    rng=np.random.default_rng(int(seed))
    if condition=='grain_sigma_9':
        out=gray.astype('float32')+rng.normal(0.,9.,size=gray.shape)
        return np.clip(np.rint(out),0,255).astype('uint8')
    out=gray.copy()
    # Random synthetic film-like streaks. Position selection knows no curve GT.
    for i in range(2):
        a=(int(rng.integers(8,184)),int(rng.integers(8,184)))
        b=(int(rng.integers(8,184)),int(rng.integers(8,184)))
        value=20 if i==0 else 235
        cv2.line(out,a,b,value,thickness=1,lineType=cv2.LINE_8)
    return out
