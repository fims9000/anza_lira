"""S20: strictly label-free H50Z photo-domain masked reconstruction.

Real grayscale image pixels can be self-supervised by hiding them. They are
NOT expert track labels; this pretext score is NOT segmentation quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_TILE_RE = re.compile(r'^(H50Z_\d+_\d+_\d+)_start\d+_tile\.png$')


@dataclass(frozen=True)
class PhotoTile:
    path: Path
    photo_id: str
    sha256: str
    shape: tuple[int, int]


def discover_real_tiles(folder: Path) -> tuple[list[PhotoTile], dict]:
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f'Real H50Z tile folder missing: {folder}')
    files = sorted(folder.glob('*.png'))
    if not files:
        raise ValueError('No original grayscale tiles available')
    photos: list[PhotoTile] = []
    for path in files:
        found = _TILE_RE.fullmatch(path.name)
        if not found:
            raise ValueError(f'Unrecognized source photo or cropped tile: {path.name}')
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint8 or image.ndim != 2 or min(image.shape) < 64:
            raise ValueError(f'Real tile must be 8-bit single-channel and >=64px: {path.name}')
        photos.append(PhotoTile(path, found.group(1), sha256(path.read_bytes()).hexdigest(), tuple(image.shape)))
    ids = sorted({t.photo_id for t in photos})
    if len(ids) != 2:
        raise ValueError('Expected exactly two source photo groups; never split tiles from one original photo')
    train = [t for t in photos if t.photo_id == ids[0]]
    val = [t for t in photos if t.photo_id == ids[1]]
    if len(train) < 1 or len(val) < 1:
        raise ValueError('Source-photo groups must be nonempty')
    report = {
        'schema': 's20_source_photo_split_v1',
        'real_unlabeled_photo_ids': ids,
        'train_photo_id': ids[0], 'validation_photo_id': ids[1],
        'train_crops': len(train), 'validation_crops': len(val),
        'different_physical_run_for_test': False,
        'real_human_annotation_count': 0,
        'real_pixel_track_ground_truth': False,
        'photo_tiles': [{'name': t.path.name, 'sha256': t.sha256, 'photo_id': t.photo_id,
                         'shape': list(t.shape)} for t in photos],
    }
    return photos, report


def mask_blocks(size: int, rng: np.random.Generator, fraction: float = .22, block: int = 6) -> np.ndarray:
    """Masks are synthetic missing-pixel positions, NEVER track/background labels."""
    if size < 32 or not 0.01 <= fraction <= 0.6 or not 1 <= block <= size // 4:
        raise ValueError('Invalid patch/missing-fraction/block size')
    holes = np.zeros((size, size), dtype=np.bool_)
    target = int(size * size * fraction)
    for _ in range(size * size):
        y = int(rng.integers(0, size - block + 1)); x = int(rng.integers(0, size - block + 1))
        holes[y:y+block, x:x+block] = True
        if int(holes.sum()) >= target:
            break
    if not (0.01 <= holes.mean() <= .7):
        raise RuntimeError('Bad artificial reconstruction holes')
    return holes


def render_masked_sample(image: np.ndarray, holes: np.ndarray):
    if image.ndim != 2 or image.dtype != np.uint8 or image.shape != holes.shape:
        raise ValueError('Expected uint8 image + matching synthetic holes')
    target = (255. - image.astype('float32')) / 255.
    damaged = target.copy()
    damaged[holes] = 0.5
    return damaged, target, holes


class UnlabeledPhotoPatches(Dataset):
    """Deterministic patches; each source photo belongs to one split only."""

    def __init__(self, tiles: list[PhotoTile], *, patch: int = 64, per_tile: int = 24,
                 seed: int = 20261025, fraction: float = .22, block: int = 6,
                 brightness_jitter_px: float = 0., contrast_jitter: float = 0.):
        if patch < 32 or patch % 16 or per_tile < 1:
            raise ValueError('Patch must be multiple of 16, and per_tile positive')
        if not tiles:
            raise ValueError('Empty source group')
        if brightness_jitter_px<0 or brightness_jitter_px>90 or not 0<=contrast_jitter<=.6:
            raise ValueError('Invalid photometric augmentation range')
        self.provenance = []
        self.x = []; self.y = []; self.hole = []
        for item in tiles:
            if sha256(item.path.read_bytes()).hexdigest() != item.sha256:
                raise ValueError('Real image source changed after audit')
            im = cv2.imread(str(item.path), cv2.IMREAD_GRAYSCALE)
            if im is None or im.shape != item.shape:
                raise ValueError('Source grayscale image corrupted')
            if min(im.shape) < patch:
                raise ValueError('Real crop smaller than patch')
            for j in range(per_tile):
                # Stable independent hash avoids cross-machine hash randomization.
                key = f'{seed}:{item.sha256}:{j}'
                rng = np.random.default_rng(int.from_bytes(bytes.fromhex(sha256(key.encode()).hexdigest())[:8],'little'))
                sy = int(rng.integers(0, im.shape[0]-patch+1))
                sx = int(rng.integers(0, im.shape[1]-patch+1))
                original = im[sy:sy+patch, sx:sx+patch]
                k = int(rng.integers(4))
                original = np.rot90(original, k).copy()
                if rng.integers(2): original = np.flip(original, axis=1).copy()
                if brightness_jitter_px>0 or contrast_jitter>0:
                    # This is input/target appearance augmentation, not a real track label.
                    gain=float(rng.uniform(1.-contrast_jitter,1.+contrast_jitter))
                    offset=float(rng.uniform(-brightness_jitter_px,brightness_jitter_px))
                    original=np.uint8(np.clip(original.astype('float32')*gain+offset,0,255))
                holes = mask_blocks(patch, rng, fraction, block)
                x,y,m = render_masked_sample(original, holes)
                self.x.append(x[None]);self.y.append(y[None]);self.hole.append(m[None])
                self.provenance.append({'photo_id':item.photo_id,'tile_sha256':item.sha256,
                                        'tile':item.path.name,'xy':(sx,sy),'rot90':k,
                                        'synthetic_missing_pixels':int(m.sum())})
        self.x = torch.from_numpy(np.stack(self.x))
        self.y = torch.from_numpy(np.stack(self.y))
        self.hole = torch.from_numpy(np.stack(self.hole))
        assert self.x.shape == self.y.shape == self.hole.shape

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.hole[idx]


def reconstruction_loss(logits: torch.Tensor, target: torch.Tensor, holes: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape or logits.shape != holes.shape:
        raise ValueError('Reconstruction shapes mismatch')
    if not bool(holes.any()):
        raise ValueError('No hidden pixels in self-supervised batch')
    return torch.nn.functional.smooth_l1_loss(torch.sigmoid(logits)[holes], target[holes], beta=.1)


def edge_weights(target: torch.Tensor):
    """Gradient weights are derived from withheld targets FOR LOSS ONLY."""
    if target.ndim!=4 or target.shape[1]!=1:
        raise ValueError('Expected [B,1,H,W] images')
    padded=torch.nn.functional.pad(target,(1,1,1,1),mode='replicate')
    gx=(padded[:,:,1:-1,2:]-padded[:,:,1:-1,:-2]).abs()*.5
    gy=(padded[:,:,2:,1:-1]-padded[:,:,:-2,1:-1]).abs()*.5
    return ((gx+gy)/.3).clamp(0,1)


def masked_metrics(reconstruction: np.ndarray, target: np.ndarray, holes: np.ndarray) -> dict:
    if not (reconstruction.shape == target.shape == holes.shape):
        raise ValueError('All arrays must have identical shapes')
    diff = reconstruction[np.asarray(holes,bool)] - target[np.asarray(holes,bool)]
    if not len(diff): raise ValueError('Empty synthetic missing-pixel area')
    mae = float(np.mean(np.abs(diff))); mse = float(np.mean(diff**2))
    return {'masked_mae':mae,'masked_mse':mse,'masked_psnr_db':float(-10*np.log10(max(mse,1e-12))),
            'pixels_scored':int(diff.size)}


def inpaint_baseline(x: np.ndarray, holes: np.ndarray):
    """Classic TELEA inpainting, an important non-neural reference."""
    src = np.asarray(np.rint(np.clip(x,0,1)*255),np.uint8)
    mk = np.asarray(holes,np.uint8)*255
    return cv2.inpaint(src,mk,3,cv2.INPAINT_TELEA).astype('float32') / 255
