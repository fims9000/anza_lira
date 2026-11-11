"""Physical 3D CCTA candidate patch extraction for controlled ANZA/CNN tests.

Input is the verified HU-valued CT volume and candidate endpoint LPS positions;
ground-truth labels are never used in patch generation. A frozen candidate may
have 2 (PAIR) or 3 (JUNCTION) endpoints. Both baselines use the *same* patch.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class PatchConfig:
    side: int = 24
    voxel_mm: float = 1.0
    endpoint_sigma_mm: float = 1.5
    hu_min: float = -300.0
    hu_max: float = 1200.0
    frame_mode: str = "canonical"  # frozen historical behavior; optional fixed LPS ablation

    def __post_init__(self):
        if self.side < 8 or self.side > 64 or self.side % 2:
            raise ValueError("side must be even in [8,64]")
        if not all(math.isfinite(x) for x in (
            self.voxel_mm,self.endpoint_sigma_mm,self.hu_min,self.hu_max
        )):
            raise ValueError("Nonfinite patch config")
        if self.voxel_mm <= 0 or self.endpoint_sigma_mm <= 0 or self.hu_max <= self.hu_min:
            raise ValueError("Invalid patch physical scales")
        if self.frame_mode not in {"canonical", "fixed_lps"}:
            raise ValueError("frame_mode must be canonical or fixed_lps")


def local_frame(endpoints_lps: np.ndarray, frame_mode: str = "canonical") -> tuple[np.ndarray,np.ndarray]:
    """Centroid and right-handed local orthonormal basis in physical LPS mm.

    fixed_lps is an experimental orientation-stable control: no canonical
    endpoint sorting can rotate the frame under a small coordinate perturbation.
    It does not guarantee candidate coverage or improve the neural model.
    """
    pts=np.asarray(endpoints_lps,dtype=np.float64)
    if pts.shape not in ((2,3),(3,3)) or not np.isfinite(pts).all():
        raise ValueError("Expected finite 2- or 3-endpoint LPS coordinate array")
    # Physical-coordinate canonicalization makes the representation
    # invariant to candidate endpoint-ID renaming and row permutations.
    # This is critical because the original CT28 scenes use deterministic
    # endpoint positions in their ground-truth examples.
    if frame_mode not in {"canonical", "fixed_lps"}:
        raise ValueError("Unknown local frame mode")
    pts=pts[np.lexsort((pts[:,2],pts[:,1],pts[:,0]))]
    direction=pts[1]-pts[0]
    length=float(np.linalg.norm(direction))
    if length <= 1e-6:
        raise ValueError("Coincident first two endpoints")
    if frame_mode == "fixed_lps":
        return pts.mean(axis=0),np.eye(3,dtype=np.float64)
    forward=direction/length
    canonical_axes=np.eye(3)
    reference=canonical_axes[int(np.argmin(abs(canonical_axes@forward)))]
    second=np.cross(forward,reference)
    second/=np.linalg.norm(second)
    third=np.cross(forward,second)
    frame=np.column_stack((forward,second,third))
    if abs(np.linalg.det(frame)-1.0)>1e-8 or not np.allclose(
        frame.T@frame,np.eye(3),atol=1e-8
    ):
        raise RuntimeError("3D physical candidate frame is not orthonormal")
    return pts.mean(axis=0),frame


def extract_candidate_patch(
    ct:np.ndarray,
    voxel_to_ras:np.ndarray,
    endpoints_lps:np.ndarray,
    config:PatchConfig=PatchConfig(),
) -> tuple[np.ndarray,dict]:
    """Return [2,side,side,side] normalized HU+endpoint heatmap, float32."""
    if ct.ndim!=3:
        raise ValueError("Expected 3D CT")
    if voxel_to_ras.shape!=(4,4) or not np.isfinite(voxel_to_ras).all():
        raise ValueError("Expected finite 4x4 affine")
    if not np.allclose(voxel_to_ras[3],[0,0,0,1]):
        raise ValueError("Non-homogeneous CT affine")
    if abs(np.linalg.det(voxel_to_ras[:3,:3]))<1e-8:
        raise ValueError("Singular CCTA voxel-to-RAS transform")
    center,frame=local_frame(endpoints_lps,config.frame_mode)
    axes=(np.arange(config.side,dtype=np.float32)-(config.side-1)/2)*config.voxel_mm
    grid=np.stack(np.meshgrid(axes,axes,axes,indexing="ij"),axis=0)
    flattened=grid.reshape(3,-1)
    pts_lps=center[:,None]+frame@flattened
    pts_ras=pts_lps*np.array([[-1.],[-1.],[1.]])
    inverse=np.linalg.inv(voxel_to_ras)
    coords=inverse[:3,:3]@pts_ras+inverse[:3,3,None]
    valid=np.all((coords>=0)&(coords<=np.asarray(ct.shape)[:,None]-1),axis=0)
    hu=map_coordinates(
        ct,coords,order=1,mode="constant",cval=-1000.0,
    ).reshape((config.side,)*3)
    clip=np.clip(hu,config.hu_min,config.hu_max)
    normalized=(2*(clip-config.hu_min)/(config.hu_max-config.hu_min)-1)
    # Gaussian endpoint locations in local physical coordinates; no endpoint
    # names, labels or scene type are provided as network input.
    pts=np.asarray(endpoints_lps,dtype=np.float64)
    pts=pts[np.lexsort((pts[:,2],pts[:,1],pts[:,0]))]
    local=frame.T@(pts.T-center[:,None])
    physical_half_extent=(config.side-1)*config.voxel_mm/2.
    endpoint_patch_margin_mm=float(
        physical_half_extent-np.abs(local).max()
    )
    squared=np.zeros((config.side,)*3,dtype=np.float32)
    heat=np.zeros_like(squared)
    for pos in local.T:
        squared[:]=sum((grid[axis]-pos[axis])**2 for axis in range(3))
        heat+=np.exp(-squared/(2*config.endpoint_sigma_mm**2))
    heat=np.clip(heat,0,1)
    out=np.stack((normalized,heat),axis=0).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite CCTA candidate patch")
    return out,{
        "candidate_degree":len(endpoints_lps),
        "frame_mode":config.frame_mode,
        "sampling_inside_volume_fraction":float(valid.mean()),
        "patch_center_lps_mm":center.tolist(),
        "voxel_mm":config.voxel_mm,
        "side":config.side,
        "orientation_det":float(np.linalg.det(frame)),
        "endpoint_patch_margin_mm":endpoint_patch_margin_mm,
        "all_endpoints_inside_patch":bool(endpoint_patch_margin_mm>=0.),
    }
