"""Private, read-only full-BMP candidate segmentation without physical track claims.

Compute each core pixel exactly once using halo-context inference. All exported
line identities and seam adjacencies are MACHINE UNKNOWN, never particle GT.
Keep private photo pixels and source coordinates outside the Git repository.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import label as component_label
from skimage.morphology import skeletonize

from .bitmap_tiles import GrayBMP8, sha256_file
from .pilot import arc_sample, extract_skeleton_segments, fit_quadratic
from .ridges import frangi_dark_ridges

SCHEMA = 'ANZA_LIRA_PRIVATE_TILED_SEGMENTATION_V1'
METHODS = ('contrast', 'frangi', 'azunet')


@dataclass(frozen=True)
class FullBMPConfig:
    method: str = 'frangi'
    core_size: int = 384
    halo: int = 48
    contrast_threshold: float = 18.0
    ridge_response_fraction: float = 0.12
    az_threshold: float = 0.5
    min_component_area: int = 15
    min_segment_length_px: float = 12.0
    max_skeleton_pixels: int = 30000
    max_fragments_per_tile: int = 512
    sample_points: int = 14
    preview_limit: int = 16
    full_mask_npy: bool = False

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError('unknown segmentation method')
        if self.core_size < 64 or self.core_size > 896 or self.core_size % 16:
            raise ValueError('core_size must be a multiple of 16 in 64..896')
        if self.halo < 16 or self.halo > self.core_size // 2:
            raise ValueError('halo must be 16..core_size/2')
        if (self.core_size + 2 * self.halo + 7) ** 2 > 1048576:
            raise ValueError('window too large for bounded AZUNet inference')
        if not np.isfinite(self.contrast_threshold) or self.contrast_threshold <= 0:
            raise ValueError('contrast threshold must be positive')
        if not 0 < self.ridge_response_fraction < 1 or not 0.05 <= self.az_threshold <= 0.95:
            raise ValueError('invalid segmentation threshold')
        if self.min_component_area < 1 or self.min_segment_length_px < 4:
            raise ValueError('invalid component or fragment minimum')
        if self.max_skeleton_pixels < 100 or self.max_fragments_per_tile < 1 or self.sample_points < 5:
            raise ValueError('invalid extraction budget')
        if not 0 <= self.preview_limit <= 256:
            raise ValueError('invalid preview limit')


def core_rectangles(width: int, height: int, size: int):
    """Non-overlapping, lossless cover. Edge cores may be smaller."""
    for y0 in range(0, height, size):
        for x0 in range(0, width, size):
            yield x0, y0, min(width, x0 + size), min(height, y0 + size)


def _window(bmp: GrayBMP8, core: tuple[int, int, int, int], halo: int):
    x0, y0, x1, y1 = core
    wx0, wy0 = max(0, x0 - halo), max(0, y0 - halo)
    wx1, wy1 = min(bmp.width, x1 + halo), min(bmp.height, y1 + halo)
    # GrayBMP8 verifies uncompressed identity-grayscale BMP and owns a read-only
    # memory map. Rectangular views avoid allocating an enormous source frame.
    pixels = np.asarray(bmp._data[wy0:wy1, wx0:wx1], dtype=np.uint8).copy()
    return pixels, (wx0, wy0, wx1, wy1)


def _filter_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    components, _ = component_label(mask, structure=np.ones((3, 3), np.uint8))
    sizes = np.bincount(components.ravel())
    return mask & (components > 0) & (sizes[components] >= min_area)


def predict_mask(gray: np.ndarray, cfg: FullBMPConfig, *, az_model=None) -> np.ndarray:
    """Binary candidate response; 0 is NOT a certified physical negative."""
    cfg.validate()
    if gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
        raise ValueError('expected nonempty uint8 grayscale image')
    if cfg.method == 'contrast':
        smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.7)
        background = cv2.GaussianBlur(smooth, (0, 0), sigmaX=14.0)
        contrast = cv2.subtract(background, smooth)
        return _filter_components(contrast >= cfg.contrast_threshold, cfg.min_component_area)
    if cfg.method == 'frangi':
        mask, _ = frangi_dark_ridges(gray, response_fraction=cfg.ridge_response_fraction,
                                     min_area=cfg.min_component_area)
        return mask
    if az_model is None:
        raise ValueError('azunet requires a source-locked private checkpoint')
    # Same original AZUNet input convention as s14_anza_frontend; do not create
    # pair proposals or invoke Graph-LIRA in a pixel segmentation pass.
    import torch
    height, width = gray.shape
    dh, dw = (-height) % 8, (-width) % 8
    image = np.pad(gray, ((0, dh), (0, dw)), mode='edge')
    tensor = torch.from_numpy(((255 - image).astype('float32') / 255)[None, None])
    az_model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(az_model(tensor))[0, 0, :height, :width].cpu().numpy()
    if prob.shape != gray.shape or not np.isfinite(prob).all():
        raise ValueError('invalid AZUNet prediction')
    return prob >= cfg.az_threshold


def _tile_id(core: tuple[int, int, int, int]) -> str:
    return f'y{core[1]:06d}_x{core[0]:06d}'


def _edge_point(xy: list[int], core: tuple[int, int, int, int]) -> bool:
    x, y = xy
    return x in (core[0], core[2] - 1) or y in (core[1], core[3] - 1)


def _fragments(core_skeleton: np.ndarray, core: tuple[int, int, int, int],
               cfg: FullBMPConfig):
    skeleton_count = int(np.count_nonzero(core_skeleton))
    if skeleton_count > cfg.max_skeleton_pixels:
        return [], 'skeleton_pixel_budget', skeleton_count
    paths = extract_skeleton_segments(core_skeleton, cfg.min_segment_length_px)
    if len(paths) > cfg.max_fragments_per_tile:
        return [], 'fragment_count_budget', skeleton_count
    fragments = []
    for points in paths:
        if len(points) < 4:
            continue
        xy = (points[:, ::-1] + np.array([core[0], core[1]])).astype(np.int32)
        sampled = arc_sample(points + np.array([core[1], core[0]]), cfg.sample_points)
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        fragments.append({
            'fragment_id': f'{_tile_id(core)}_f{len(fragments):04d}',
            'centerline_xy_source_px': xy.tolist(),
            'sampled_xy_source_px': np.round(sampled[:, ::-1], 3).tolist(),
            'observed_length_px': round(length, 3),
            'geometric_fit': fit_quadratic(points + np.array([core[1], core[0]])),
            'touches_tile_boundary': bool(_edge_point(xy[0].tolist(), core)
                                          or _edge_point(xy[-1].tolist(), core)),
            'status': 'UNKNOWN',
        })
    return fragments, None, skeleton_count


def _save_json_atomic(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n',
                   encoding='utf-8')
    tmp.chmod(0o600)
    os.replace(tmp, path)


def _save_png_atomic(path: Path, image: np.ndarray) -> None:
    tmp = path.with_name(path.stem + '.tmp.png')
    if not cv2.imwrite(str(tmp), image):
        raise OSError('PNG write failed: ' + str(path))
    tmp.chmod(0o600)
    os.replace(tmp, path)


def _process_tile(bmp: GrayBMP8, core, cfg: FullBMPConfig, out: Path,
                  preview: bool, az_model=None) -> dict[str, Any]:
    pixels, win = _window(bmp, core, cfg.halo)
    foreground = predict_mask(pixels, cfg, az_model=az_model)
    a, b = core[1] - win[1], core[0] - win[0]
    h, w = core[3] - core[1], core[2] - core[0]
    core_mask = foreground[a:a + h, b:b + w]
    skeleton = skeletonize(foreground)[a:a + h, b:b + w]
    fragments, skip, count = _fragments(skeleton, core, cfg)
    tid = _tile_id(core)
    mask_rel = f'masks/{tid}.png'
    _save_png_atomic(out / mask_rel, (core_mask.astype(np.uint8) * 255))
    if preview:
        core_gray = pixels[a:a + h, b:b + w]
        overlay = cv2.cvtColor(core_gray, cv2.COLOR_GRAY2BGR)
        overlay[core_mask] = (25, 130, 80)
        overlay[skeleton] = (80, 240, 230)
        _save_png_atomic(out / 'preview' / f'{tid}_source.png', core_gray)
        _save_png_atomic(out / 'preview' / f'{tid}_overlay.png', overlay)
    preview_hashes = {kind: sha256_file(out / 'preview' / f'{tid}_{kind}.png')
                      for kind in ('source', 'overlay')} if preview else {}
    return {
        'tile_id': tid, 'core_xyxy': list(core), 'read_xyxy': list(win),
        'mask_file': mask_rel, 'mask_sha256': sha256_file(out / mask_rel),
        'foreground_pixel_count': int(core_mask.sum()),
        'skeleton_pixel_count': count, 'fragment_count': len(fragments),
        'fragment_extraction_skip_reason': skip,
        'preview': preview, 'preview_sha256': preview_hashes, 'fragments': fragments,
    }


def _valid_cached_tile(rec: dict[str, Any], core, out: Path, preview: bool) -> bool:
    tid = _tile_id(core)
    if rec.get('tile_id') != tid or rec.get('core_xyxy') != list(core) or rec.get('preview') != preview:
        return False
    mask = out / 'masks' / f'{tid}.png'
    if rec.get('mask_file') != f'masks/{tid}.png' or not mask.is_file():
        return False
    if sha256_file(mask) != rec.get('mask_sha256'):
        raise ValueError('resume mask has changed: ' + tid)
    fragments = rec.get('fragments')
    if not isinstance(fragments, list) or rec.get('fragment_count') != len(fragments) or any(
        fragment.get('status') != 'UNKNOWN' or not str(fragment.get('fragment_id', '')).startswith(tid + '_f')
        for fragment in fragments):
        raise ValueError('resume fragment provenance/status invalid: ' + tid)
    if preview:
        for suffix in ('source', 'overlay'):
            path = out / 'preview' / f'{tid}_{suffix}.png'
            if not path.is_file():
                return False
            if sha256_file(path) != rec.get('preview_sha256', {}).get(suffix):
                raise ValueError('resume preview has changed: ' + tid)
    return True


def _seam_advisories(records: list[dict[str, Any]], out: Path) -> int:
    """Only observed 8-neighbor endpoints across cores, NEVER inferred joins."""
    sites: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for record in records:
        core = tuple(record['core_xyxy'])
        for frag in record['fragments']:
            for p in (frag['centerline_xy_source_px'][0], frag['centerline_xy_source_px'][-1]):
                if _edge_point(p, core):
                    key = (int(p[0]), int(p[1]))
                    sites.setdefault(key, []).append((frag['fragment_id'], record['tile_id']))
    candidate_pairs = set()
    for (x, y), items in sites.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                for f1, t1 in items:
                    for f2, t2 in sites.get((x + dx, y + dy), []):
                        if t1 != t2 and f1 != f2:
                            candidate_pairs.add(tuple(sorted((f1, f2))))
    path = out / 'seam_advisories.ndjson'
    tmp = path.with_suffix('.ndjson.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        for f1, f2 in sorted(candidate_pairs):
            stream.write(json.dumps({'fragment_ids': [f1, f2], 'status': 'UNKNOWN',
                                     'observed_pixel_adjacency_only': True,
                                     'physical_auto_join': False}) + '\n')
    tmp.chmod(0o600)
    os.replace(tmp, path)
    return len(candidate_pairs)


def _write_viewer(out: Path, records: list[dict[str, Any]], method: str, source_name: str) -> None:
    rows = []
    for rec in records:
        if not rec['preview']:
            continue
        tid = rec['tile_id']
        rows.append('<section><h2>' + html.escape(tid) + ' (x/y in original BMP)</h2>'
                    + '<img alt="original" src="preview/' + tid + '_source.png">'
                    + '<img alt="overlay machine only" src="preview/' + tid + '_overlay.png">'
                    + '<img alt="binary candidate mask" src="masks/' + tid + '.png">'
                    + '<p>Machine fragments: ' + str(rec['fragment_count']) + '</p></section>')
    page = ('<!doctype html><html lang="ru"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>PRIVATE H50Z tiled segmentation</title><style>'
            'body{font:14px system-ui;background:#121923;color:#ecf0f6;padding:20px}'
            'section{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;'
            'margin:30px 0;border-top:1px solid #758}h2,p{grid-column:1/-1}'
            'img{width:100%;image-rendering:auto}@media(max-width:700px){section{grid-template-columns:1fr}}'
            '</style><h1>PRIVATE machine-only full-BMP segmentation</h1><p>File: '
            + html.escape(source_name) + '; method: ' + html.escape(method) + '. '
            'No physical GT; mask 0 is NOT verified background; no physical joins.</p>'
            + ''.join(rows) + '</html>')
    viewer = out / 'index.html'
    viewer.write_text(page, encoding='utf-8')
    viewer.chmod(0o600)


def run_full_bmp(bmp_path: Path, out_dir: Path, cfg: FullBMPConfig = FullBMPConfig(),
                 *, checkpoint: Path | None = None, resume: bool = False) -> dict[str, Any]:
    """Resumable disk-backed extraction; output and every source pixel stay PRIVATE."""
    cfg.validate()
    bmp_path, out_dir = Path(bmp_path).resolve(), Path(out_dir).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if out_dir == repo_root or repo_root in out_dir.parents:
        raise ValueError('PRIVATE output cannot be inside source repository')
    if not bmp_path.is_file():
        raise FileNotFoundError(bmp_path)
    if out_dir.exists() and any(out_dir.iterdir()) and not resume:
        raise FileExistsError('output is not empty; resume explicitly or select a new directory')
    if cfg.method == 'azunet' and checkpoint is None:
        raise ValueError('azunet requires --checkpoint with source-locked private weights')
    if cfg.method != 'azunet' and checkpoint is not None:
        raise ValueError('checkpoint only valid for azunet')
    model = None
    checkpoint_sha = None
    if checkpoint is not None:
        from .s14_anza_frontend import load_original_az_checkpoint
        checkpoint = Path(checkpoint).resolve()
        checkpoint_sha = sha256_file(checkpoint)
        model, _ = load_original_az_checkpoint(checkpoint)
    bmp = GrayBMP8(bmp_path)
    source_sha = sha256_file(bmp_path)
    identity = {'schema': SCHEMA, 'source_bmp_sha256': source_sha,
                'source_bmp_name': bmp_path.name, 'image_shape_yx': [bmp.height, bmp.width],
                'checkpoint_sha256': checkpoint_sha, 'config': asdict(cfg)}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.chmod(0o700)
    identity_path = out_dir / 'run_identity.json'
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise ValueError('resume identity mismatch: source, weights or configuration changed')
    elif any(out_dir.iterdir()):
        raise ValueError('nonempty directory without run identity')
    else:
        _save_json_atomic(identity_path, identity)
    for folder in ('masks', 'records', 'preview'):
        (out_dir / folder).mkdir(exist_ok=True)
        (out_dir / folder).chmod(0o700)
    cores = list(core_rectangles(bmp.width, bmp.height, cfg.core_size))
    preview_ids = set(int(round(x)) for x in np.linspace(0, len(cores) - 1,
                        min(len(cores), cfg.preview_limit))) if cfg.preview_limit else set()
    records = []
    reused = 0
    for idx, core in enumerate(cores):
        tid = _tile_id(core)
        record_path = out_dir / 'records' / f'{tid}.json'
        preview = idx in preview_ids
        record = None
        if resume and record_path.is_file():
            cached = json.loads(record_path.read_text(encoding='utf-8'))
            if _valid_cached_tile(cached, core, out_dir, preview):
                record = cached
                reused += 1
        if record is None:
            record = _process_tile(bmp, core, cfg, out_dir, preview, az_model=model)
            _save_json_atomic(record_path, record)
        records.append(record)
    fragments_tmp = out_dir / 'fragments.ndjson.tmp'
    tiles_tmp = out_dir / 'tiles.ndjson.tmp'
    with fragments_tmp.open('w', encoding='utf-8') as frag_stream, \
         tiles_tmp.open('w', encoding='utf-8') as tile_stream:
        for record in records:
            for fragment in record['fragments']:
                frag_stream.write(json.dumps(fragment, ensure_ascii=False) + '\n')
            small = {key: value for key, value in record.items() if key != 'fragments'}
            tile_stream.write(json.dumps(small, ensure_ascii=False) + '\n')
    fragments_tmp.chmod(0o600)
    tiles_tmp.chmod(0o600)
    os.replace(fragments_tmp, out_dir / 'fragments.ndjson')
    os.replace(tiles_tmp, out_dir / 'tiles.ndjson')
    seam_count = _seam_advisories(records, out_dir)
    if cfg.full_mask_npy:
        temp = out_dir / 'full_candidate_mask.tmp.npy'
        full = np.lib.format.open_memmap(temp, mode='w+', dtype=np.uint8,
                                          shape=(bmp.height, bmp.width))
        for record in records:
            x0, y0, x1, y1 = record['core_xyxy']
            mask = cv2.imread(str(out_dir / record['mask_file']), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (y1 - y0, x1 - x0):
                raise ValueError('invalid mask shard during full-image export')
            full[y0:y1, x0:x1] = mask
        full.flush()
        del full
        temp.chmod(0o600)
        os.replace(temp, out_dir / 'full_candidate_mask.npy')
    if sha256_file(bmp_path) != source_sha:
        raise ValueError('source BMP changed during processing; refuse complete manifest')
    _write_viewer(out_dir, records, cfg.method, bmp_path.name)
    manifest = {
        **identity, 'status': 'COMPLETE_MACHINE_ONLY',
        'tile_count': len(records), 'tiles_reused_on_this_run': reused,
        'candidate_foreground_pixels': sum(r['foreground_pixel_count'] for r in records),
        'candidate_fragments': sum(r['fragment_count'] for r in records),
        'tiles_with_fragment_extraction_skipped': sum(r['fragment_extraction_skip_reason'] is not None for r in records),
        'seam_advisories_UNKNOWN': seam_count,
        'full_mask_file': 'full_candidate_mask.npy' if cfg.full_mask_npy else None,
        'mask_tiles_are_exact_nonoverlapping_pixel_cover': True,
        'mask_values': {'0': 'no candidate, NOT physical background GT',
                        '255': 'machine candidate, NOT physical track GT'},
        'native_physics_GT_count': 0, 'physical_auto_joins_authorized': False,
        'predictions_are_detector_only': True, 'stereo_reconstruction': False,
        'source_pixels_and_coordinates_PRIVATE': True,
        'files': ['index.html', 'run_identity.json', 'tiles.ndjson', 'fragments.ndjson',
                  'seam_advisories.ndjson'],
    }
    _save_json_atomic(out_dir / 'manifest.json', manifest)
    return manifest
