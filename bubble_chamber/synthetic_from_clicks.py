"""Synthetic film patches from REAL measured track geometries (NOT real images).

Source point geometries are from visualMetro but all rendered intensities,
widths, defects and masks are artificial. A test F1 here can only show that a
model learned this generator; it cannot validate real bubble-film segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticFilmConfig:
    size: int = 128
    min_clicks: int = 7
    min_contrast: float = 35.0
    max_contrast: float = 100.0
    sample_seed: int = 20261024


def _seed(*parts) -> int:
    return int.from_bytes(sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


def render_measured_track(track: dict, view_id: int, sample_index: int, cfg: SyntheticFilmConfig = SyntheticFilmConfig()):
    """Return synthetic image and GENERATED pixel mask (uint8, HxW).

    Source real click coordinates are the only grounded component. Connecting
    them as a polyline is an **artificial** image construction, not an observed
    centerline or a surrogate real human pixel mask.
    """
    if cfg.size < 48 or cfg.size % 16 or not (0 < cfg.min_contrast <= cfg.max_contrast):
        raise ValueError('invalid simulated patch configuration')
    if view_id not in (1, 2, 3):
        raise ValueError('unknown view')
    points=np.asarray(track['measured_pixel_xy'][str(view_id)],dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<cfg.min_clicks or not np.isfinite(points).all():
        raise ValueError('track has too few valid clicks')
    rng=np.random.default_rng(_seed(cfg.sample_seed,track['frame_id'],track['track_id'],view_id,sample_index))
    pts=points-points.mean(0)
    phi=rng.uniform(-0.65,0.65)
    rot=np.asarray([[math.cos(phi),-math.sin(phi)],[math.sin(phi),math.cos(phi)]])
    pts=pts@rot.T
    span=np.ptp(pts,axis=0).max()
    if span<1e-6:
        raise ValueError('degenerate clicked geometry')
    pts=pts*(cfg.size*0.65/span)
    pts+=cfg.size/2+rng.uniform(-cfg.size*0.045,cfg.size*0.045,2)
    coords=np.rint(pts).astype(np.int32)
    mask=np.zeros((cfg.size,cfg.size),dtype=np.uint8)
    # Physical measured points are joined only in this synthetic renderer.
    thickness=int(rng.integers(1,4))
    cv2.polylines(mask,[coords.reshape(-1,1,2)],False,1,thickness,cv2.LINE_8)
    # Varying tone + granular artifacts; none of these were measured on film.
    yy,xx=np.mgrid[:cfg.size,:cfg.size]
    base=rng.uniform(190,235)+rng.uniform(-14,14)*xx/cfg.size+rng.uniform(-14,14)*yy/cfg.size
    base=base+rng.normal(0,rng.uniform(2,10),(cfg.size,cfg.size))
    contrast=rng.uniform(cfg.min_contrast,cfg.max_contrast)
    base-=contrast*mask
    speckle=(rng.random((cfg.size,cfg.size))<rng.uniform(0.001,0.007))
    base[speckle]-=rng.uniform(35,100)
    # Short scratches are distractors NOT positives in synthetic truth.
    for _ in range(int(rng.integers(0,3))):
        scratch=np.zeros_like(mask)
        sx=int(rng.integers(3,cfg.size-3));sy=int(rng.integers(3,cfg.size-3))
        ex=int(np.clip(sx+rng.integers(-25,26),0,cfg.size-1));ey=int(np.clip(sy+rng.integers(-25,26),0,cfg.size-1))
        cv2.line(scratch,(sx,sy),(ex,ey),1,1)
        base-=scratch*rng.uniform(15,35)
    gray=cv2.GaussianBlur(np.clip(base,0,255).astype(np.uint8),(0,0),rng.uniform(.35,1.15))
    return gray,mask,{'source_frame':track['frame_id'], 'source_track_id':track['track_id'],
                      'view_id':view_id,'source':'synthetic pixels; measured expert click geometry only',
                      'thin_stroke_width_simulated_px':thickness, 'sample_index':sample_index}


def frame_render_dataset(manifest: dict, permitted_frames: list[str], cfg:SyntheticFilmConfig = SyntheticFilmConfig(), repeats:int=2):
    """Ordered deterministic lazy iterator; never mix frame splits."""
    if manifest.get('audit_schema')!='visualmetro_large_audit_v1':
        raise ValueError('audited source required')
    allowed=set(permitted_frames)
    for frame in manifest['frames']:
        if frame['frame_id'] not in allowed:continue
        for track in frame['tracks']:
            if track['n_points']<cfg.min_clicks:continue
            for repeat in range(repeats):
                view=(1,2,3)[repeat%3]
                yield render_measured_track(track,view,repeat,cfg)
