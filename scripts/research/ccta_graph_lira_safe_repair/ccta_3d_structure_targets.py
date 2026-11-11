"""Train-only segmentation supervision aligned with frozen CT28 physical patches.

The output is an auxiliary target for topology-aware training, NEVER a
model input. Validation/test reference masks must not be used to fit weights.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from ccta_3d_an_patches import PatchConfig, local_frame


def sample_vessel_reference(
    reference:np.ndarray,
    voxel_to_ras:np.ndarray,
    endpoints_lps:np.ndarray,
    config:PatchConfig,
)->np.ndarray:
    """Return binary [D,H,W] reference patch at the exact CT patch coordinates."""
    if reference.ndim!=3 or not np.isfinite(reference).all():
        raise ValueError("Expected finite 3D coronary reference")
    if voxel_to_ras.shape!=(4,4) or not np.isfinite(voxel_to_ras).all():
        raise ValueError("Expected finite mask affine")
    if not np.allclose(voxel_to_ras[3],[0,0,0,1],atol=1e-8):
        raise ValueError("Not a homogeneous image affine")
    if abs(np.linalg.det(voxel_to_ras[:3,:3]))<1e-8:
        raise ValueError("Singular reference image affine")
    center,rotation=local_frame(endpoints_lps,config.frame_mode)
    position=(np.arange(config.side,dtype=np.float64)
              -(config.side-1)/2)*config.voxel_mm
    grid=np.stack(np.meshgrid(position,position,position,indexing="ij"),axis=0)
    voxel_lps=center[:,None]+rotation@grid.reshape(3,-1)
    voxel_ras=voxel_lps*np.array([[-1.],[-1.],[1.]])
    inv=np.linalg.inv(voxel_to_ras)
    ijk=inv[:3,:3]@voxel_ras+inv[:3,3,None]
    target=map_coordinates(
        (reference>0).astype(np.uint8),ijk,order=0,
        mode="constant",cval=0,
    ).reshape((config.side,)*3)
    return target.astype(np.uint8)


def load_aligned_train_reference(
    patient:int,ct_shape:tuple[int,int,int],ct_affine:np.ndarray,
    segmentation_dir,
    expected_train_ids:set[int],
)->tuple[np.ndarray,str]:
    """Load only TRAIN original ImageCAS-X masks after strict physical QA."""
    from pathlib import Path
    import hashlib
    import nibabel as nib

    if patient not in expected_train_ids:
        raise ValueError("Dense vessel supervision strictly forbidden outside CT28 train")
    path=Path(segmentation_dir)/f"{patient}.coronary.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    image=nib.load(str(path))
    if tuple(image.shape)!=tuple(ct_shape) or not np.allclose(
        np.asarray(image.affine),ct_affine,atol=1e-6,rtol=0
    ):
        raise ValueError("Original CT and coronary reference geometry do not align")
    mask=np.asanyarray(image.dataobj)
    if mask.ndim!=3 or not np.isfinite(mask).all():
        raise ValueError("Invalid reference mask")
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda:stream.read(1024*1024),b""):
            h.update(part)
    return mask>0,h.hexdigest()
