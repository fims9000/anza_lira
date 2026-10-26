"""S44 exact BMP pixel access and same-coordinate structure check, NOT physics GT."""
from __future__ import annotations
from pathlib import Path
import struct
import cv2
import numpy as np
from .s39_machine_auto import ridge_evidence, sample_bilinear


class ExactBMP8:
    """Read only ROI via memory mapping: native files are ~230 MB each."""
    def __init__(self, path: Path):
        path = Path(path)
        header = path.open('rb').read(54)
        if len(header) != 54 or header[:2] != b'BM':
            raise ValueError('Not a BMP')
        offset = struct.unpack_from('<I', header, 10)[0]
        width, height = struct.unpack_from('<ii', header, 18)
        bpp = struct.unpack_from('<H', header, 28)[0]
        compression = struct.unpack_from('<I', header, 30)[0]
        if not (width > 1000 and height > 1000 and bpp == 8
                and compression == 0 and 0 < offset < 8192):
            raise ValueError('Unrecognized/unsafe native uncompressed BMP layout')
        stride = ((width + 3) // 4) * 4
        if path.stat().st_size < offset + stride * height:
            raise ValueError('Truncated BMP')
        self.width, self.height = width, height
        self._raw = np.memmap(path, dtype=np.uint8, mode='r', offset=offset,
                              shape=(height, stride))

    def crop(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        x, y = int(x), int(y)
        if not (0 <= x < self.width and 0 <= y < self.height
                and width > 0 and height > 0 and x + width <= self.width
                and y + height <= self.height):
            raise ValueError('Out-of-frame original BMP crop')
        return np.asarray(self._raw[self.height-y-height:self.height-y,
                                    x:x+width][::-1]).copy()


def fit_common_fiducials(labels: dict) -> dict:
    """Never blindly pair f8(005) with f8(006); index coverage differs."""
    def fids(k):
        name = f'H50Z_030_01_{k}.bmp'
        if name not in labels:
            raise ValueError('Missing source marker annotations')
        return {r['label']: np.asarray((r['x_px'], r['y_px']), dtype=float)
                for r in labels[name]['keypoints'] if str(r['label']).startswith('f')}
    src, dst = fids('005'), fids('006')
    keys = ('f1', 'f2', 'f3', 'f4', 'f5', 'f6')
    if any(k not in src or k not in dst for k in keys):
        raise ValueError('Six common known fiducial identities required')
    a = np.asarray([src[k] for k in keys], dtype=np.float32)
    b = np.asarray([dst[k] for k in keys], dtype=np.float32)
    transform, inliers = cv2.estimateAffinePartial2D(
        a, b, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if transform is None or inliers is None or not np.all(inliers):
        raise ValueError('Unstable calibration fiducial fit')
    residual = np.linalg.norm((a @ transform[:, :2].T + transform[:, 2]) - b, axis=1)
    if not np.isfinite(transform).all() or np.max(residual) > 4.0:
        raise ValueError('Fiducial registration residual too large')
    invert = cv2.invertAffineTransform(transform)
    # An additional cross-ID geometric check, NOT an assumption of true ID.
    extra = None
    if 'f8' in src and 'f7' in dst:
        pred = transform[:, :2] @ src['f8'] + transform[:, 2]
        extra = float(np.linalg.norm(pred - dst['f7']))
    return {'forward_005_to_006_PRIVATE': transform,
            'inverse_006_to_005_PRIVATE': invert,
            'max_common_residual_px': float(residual.max()),
            'mean_common_residual_px': float(residual.mean()),
            'additional_f8_005_to_f7_006_residual_px': extra,
            'common_fiducial_count': len(keys)}


def warp_corresponding_window(destination: ExactBMP8, origin_xy: list | tuple,
                              transform: np.ndarray, size: int = 512) -> np.ndarray:
    """Return destination scan reprojected into the source crop coordinate grid."""
    if transform.shape != (2, 3) or not np.isfinite(transform).all():
        raise ValueError('Bad registered transform')
    ox, oy = map(float, origin_xy)
    corners = np.array([[ox, oy], [ox+size-1, oy],
                        [ox+size-1, oy+size-1], [ox, oy+size-1]], dtype=float)
    target = corners @ transform[:, :2].T + transform[:, 2]
    x0, y0 = np.floor(target.min(axis=0) - 3).astype(int)
    x1, y1 = np.ceil(target.max(axis=0) + 4).astype(int)
    if x0 < 0 or y0 < 0 or x1 > destination.width or y1 > destination.height:
        raise ValueError('Aligned H50Z paired view outside counterpart scan')
    target_patch = destination.crop(x0, y0, x1-x0, y1-y0)
    xs = np.arange(size, dtype=np.float32) + ox
    ys = np.arange(size, dtype=np.float32) + oy
    gx, gy = np.meshgrid(xs, ys)
    mx = transform[0, 0] * gx + transform[0, 1] * gy + transform[0, 2] - x0
    my = transform[1, 0] * gx + transform[1, 1] * gy + transform[1, 2] - y0
    return cv2.remap(target_patch, mx.astype('float32'), my.astype('float32'),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def ridge_near_path(score_map: np.ndarray, pts: np.ndarray,
                    *, percentile: float = 95., radius: int = 2) -> float:
    if radius not in (0, 1, 2) or score_map.ndim != 2:
        raise ValueError('Unsupported path support')
    p = np.asarray(pts, dtype='float64')
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 4 or not np.isfinite(p).all():
        raise ValueError('Invalid path')
    value = float(np.percentile(score_map, percentile))
    if value <= 0:
        positive = score_map[score_map > 0]
        if positive.size == 0:
            return 0.
        # Sparse physically meaningful ridge maps can have a zero global p95.
        # Never mark all pixels as ridges when threshold==0.
        value = float(np.quantile(positive, .30))
    signal = ((score_map >= value) & (score_map > 0)).astype(np.uint8)
    if radius:
        signal = cv2.dilate(signal, np.ones((2*radius+1, 2*radius+1), dtype='uint8'))
    return float(np.mean(sample_bilinear(signal, p) >= .5))


def compare_registered_ridges(gray_src: np.ndarray, gray_dst_aligned: np.ndarray,
                              candidates: list[dict]) -> list[dict]:
    if gray_src.shape != gray_dst_aligned.shape or gray_src.dtype != np.uint8 or gray_dst_aligned.dtype != np.uint8:
        raise ValueError('Image provenance missing')
    es = ridge_evidence(gray_src)['score']
    ed = ridge_evidence(gray_dst_aligned)['score']
    ranked = []
    for candidate in candidates:
        p = np.asarray(candidate['points_xy'], dtype=float)
        s = ridge_near_path(es, p)
        d = ridge_near_path(ed, p)
        # Within-paired-window negative control: displaced but otherwise
        # identical path geometry. A fixed ordering prevents cherry-picking
        # the shift with the lowest response.
        controls = ((0., 70.), (0., -70.), (70., 0.), (-70., 0.),
                    (50., 50.), (-50., -50.))
        ctrl = None
        for dx, dy in controls:
            q = p + np.array([dx, dy])
            if (np.all(q >= 7) and np.all(q[:, 0] < gray_dst_aligned.shape[1] - 7)
                    and np.all(q[:, 1] < gray_dst_aligned.shape[0] - 7)):
                ctrl = ridge_near_path(ed, q)
                break
        ranked.append({
            'draft_id': candidate['draft_id'],
            'source_multiscale_ridge_coverage': round(s, 6),
            'counterpart_same_registered_path_coverage': round(d, 6),
            'displaced_same_shape_counterpart_control_coverage': round(ctrl, 6) if ctrl is not None else None,
            'both_path_coverage_advisory': round(min(s, d), 6),
            'native_particle_truth': 'UNKNOWN',
            'candidate_kept': True,
            'physical_auto_join_allowed': False,
        })
    return ranked
