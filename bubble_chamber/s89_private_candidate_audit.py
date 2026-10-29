"""S89 read-only, private audit of S88 candidate fragments on actual grayscale BMP.

Ranks *image-space diagnostic uncertainty*, never particle probabilities or
physical track/artifact identities. The S86 input and source BMP are immutable.
Selected source tiles are stratified without manual/native ground truth.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import html
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter

from .bitmap_tiles import GrayBMP8, sha256_file
from .full_bmp_segmentation import _window
from .s87_seam_continuity import _read_source
from .s88_weak_ridges import WeakRidgeConfig, dark_ridge_candidates

SCHEMA = 'ANZA_LIRA_S89_PRIVATE_IMAGE_SPACE_AUDIT_V1'


@dataclass(frozen=True)
class AuditConfig:
    max_tiles: int = 16
    preview_tiles: int = 12
    sample_points_per_fragment: int = 128
    flank_distance_px: float = 3.0
    stability_radius_px: int = 2
    extra_grain_sigma: float = 1.5
    extra_blur_sigma: float = 0.65
    weak_contrast_review_px: float = 1.5
    unstable_support_review: float = 0.55

    def validate(self) -> None:
        if not 1 <= self.max_tiles <= 128 or not 0 <= self.preview_tiles <= 64:
            raise ValueError('invalid max_tiles or preview_tiles')
        if not 4 <= self.sample_points_per_fragment <= 4096:
            raise ValueError('invalid sample_points_per_fragment')
        if not 1 <= self.stability_radius_px <= 5:
            raise ValueError('invalid stability_radius_px')
        for name in ('flank_distance_px', 'extra_grain_sigma', 'extra_blur_sigma',
                     'weak_contrast_review_px', 'unstable_support_review'):
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError('non-finite audit parameter: ' + name)
        if not 2 <= self.flank_distance_px <= 12:
            raise ValueError('invalid flank_distance_px')
        if not 0.2 <= self.extra_grain_sigma <= 6 or not 0.2 <= self.extra_blur_sigma <= 2:
            raise ValueError('invalid noise/blur perturbation')
        if not 0 <= self.weak_contrast_review_px <= 30 or not 0 < self.unstable_support_review < 1:
            raise ValueError('invalid heuristic review thresholds')


def select_tiles(records: list[dict], count: int) -> list[dict]:
    """Represent sparse through dense *candidate* tiles without source labels.

    Order by original S86 candidate density, then deterministic evenly spaced
    quantiles. No expert marks or ground truth influence the sampled locations.
    """
    if count < 1:
        raise ValueError('count must be positive')
    ordered = sorted(records, key=lambda r: (
        r['foreground_pixel_count'] / max(1, (r['core_xyxy'][2]-r['core_xyxy'][0])
                                                 * (r['core_xyxy'][3]-r['core_xyxy'][1])),
        r['fragment_count'], r['tile_id']))
    if not ordered:
        return []
    indexes = sorted({int(round(i)) for i in np.linspace(0, len(ordered) - 1,
                                                         min(count, len(ordered)))})
    return [ordered[i] for i in indexes]


def _sample_points(points_xy: np.ndarray, limit: int) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError('invalid fragment point path')
    idx = np.unique(np.rint(np.linspace(0, len(points)-1, min(limit, len(points)))).astype(int))
    return points[idx]


def _sample_map(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return np.asarray(map_coordinates(image, [xy[:, 1], xy[:, 0]], order=1,
                                       mode='nearest'), dtype=np.float64)


def transverse_darkness(gray: np.ndarray, local_xy: np.ndarray, *, flank: float) -> dict:
    """Film pixel difference at two normal flanks; NOT calibrated energy loss."""
    path = np.asarray(local_xy, dtype=np.float64)
    tangent = np.gradient(path, axis=0)
    norms = np.linalg.norm(tangent, axis=1)
    valid = norms > 1e-9
    if not valid.any():
        raise ValueError('zero-length photometric path')
    normal = np.zeros_like(tangent)
    normal[valid, 0] = -tangent[valid, 1] / norms[valid]
    normal[valid, 1] = tangent[valid, 0] / norms[valid]
    center = _sample_map(gray, path)
    left = _sample_map(gray, path + flank * normal)
    right = _sample_map(gray, path - flank * normal)
    deficits = ((left + right) / 2 - center)[valid]
    return {
        'median_transverse_darkness_gray_levels': round(float(np.median(deficits)), 4),
        'fraction_darker_than_flanks': round(float(np.mean(deficits > 0)), 4),
        'number_of_image_samples': int(valid.sum()),
        'interpretation': 'LOCAL_PIXEL_BRIGHTNESS_NOT_DEDX_OR_PARTICLE_ID',
    }


def persistence_under_perturbations(gray: np.ndarray, score_cfg: WeakRidgeConfig,
                                    audit_cfg: AuditConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate independent image-only candidates with ORIGINAL frozen S88 gates.

    A stable scratch may remain stable; this is not a scratch-vs-track oracle.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(gray, dtype=np.uint8)
    blurred = np.uint8(np.rint(np.clip(gaussian_filter(arr.astype(np.float32),
                                             audit_cfg.extra_blur_sigma), 0, 255)))
    grainy = np.uint8(np.rint(np.clip(arr.astype(np.float32)
                    + rng.normal(0, audit_cfg.extra_grain_sigma, arr.shape), 0, 255)))
    masks = [dark_ridge_candidates(v, score_cfg)[0] for v in (blurred, grainy)]
    footprint = 2 * audit_cfg.stability_radius_px + 1
    return tuple(maximum_filter(m.astype(np.uint8), size=footprint, mode='constant') > 0
                 for m in masks)


def audit_one_fragment(fragment: dict, original_window: np.ndarray, window_xyxy: tuple,
                       perturbed_support: tuple[np.ndarray, np.ndarray],
                       cfg: AuditConfig) -> dict:
    points = _sample_points(np.asarray(fragment['centerline_xy_source_px']),
                            cfg.sample_points_per_fragment)
    local_xy = points - np.asarray(window_xyxy[:2], dtype=float)
    if (np.any(local_xy < 0) or np.any(local_xy[:, 0] >= original_window.shape[1])
            or np.any(local_xy[:, 1] >= original_window.shape[0])):
        raise ValueError('S86 fragment coordinates outside recorded read window')
    photo = transverse_darkness(original_window, local_xy,
                                 flank=cfg.flank_distance_px)
    xy_int = np.rint(local_xy).astype(np.int64)
    supports = [round(float(np.mean(m[xy_int[:, 1], xy_int[:, 0]])), 4)
                for m in perturbed_support]
    min_support = min(supports)
    flags = []
    if min_support < cfg.unstable_support_review:
        flags.append('PERTURBATION_UNSTABLE_REVIEW')
    if photo['median_transverse_darkness_gray_levels'] < cfg.weak_contrast_review_px:
        flags.append('LOW_TRANSVERSE_DARKNESS_REVIEW')
    # Straight scratches and straight particle projections cannot be separated
    # by this geometry alone: a flag is a review hint, never a veto.
    original = np.asarray(fragment['centerline_xy_source_px'], dtype=float)
    length = float(fragment['observed_length_px'])
    straightness = float(np.linalg.norm(original[-1] - original[0]) / max(length, 1e-9))
    if straightness > .995 and length > 60:
        flags.append('STRAIGHT_IMAGE_LINE_AMBIGUOUS_REVIEW')
    if fragment.get('touches_tile_boundary'):
        flags.append('TILE_EDGE_CONTINUATION_UNKNOWN')
    return {
        'fragment_id': fragment['fragment_id'],
        'tile_id': fragment['fragment_id'].split('_f')[0],
        'status': 'UNKNOWN',
        's88_score_proxy_median': fragment.get('s88_ridge_score_proxy_median'),
        'observed_length_px': fragment['observed_length_px'],
        'straightness_ratio_image_only': round(straightness, 5),
        'photometry': photo,
        'fraction_source_centerline_supported_after_mild_blur': supports[0],
        'fraction_source_centerline_supported_after_extra_grain': supports[1],
        'minimum_image_perturbation_support': min_support,
        'review_flags': flags,
        'review_priority': 'REVIEW_FIRST' if flags else 'REVIEW_LATER_NOT_VERIFIED',
        'is_particle_or_artifact_classified': False,
        'physical_auto_accept': False,
        'physical_auto_reject': False,
    }


def _write_once(path: Path, payload: bytes) -> None:
    if path.is_file():
        if path.read_bytes() == payload:
            return  # deterministic source-verified replay, no modification
        raise FileExistsError('existing output differs; use new PRIVATE directory: ' + str(path))
    if path.exists():
        raise FileExistsError('non-file output already exists: ' + str(path))
    tmp = path.with_name(path.name + '.private.tmp')
    try:
        with tmp.open('xb') as stream:
            stream.write(payload)
        tmp.chmod(0o600)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_private_png(path: Path, pixels: np.ndarray) -> None:
    success, binary = cv2.imencode('.png', pixels)
    if not success:
        raise OSError('could not encode PRIVATE preview')
    _write_once(path, binary.tobytes())


def _preview_tile(image: np.ndarray, core: tuple[int,int,int,int],
                  audited: list[dict], fragments: list[dict],
                  out: Path, tid: str) -> None:
    x0, y0, x1, y1 = core
    orig = image[:y1-y0, :x1-x0].copy()
    overlay = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    findings = {r['fragment_id']: r for r in audited}
    for f in fragments:
        item = findings[f['fragment_id']]
        pts = (np.asarray(f['centerline_xy_source_px'], dtype=np.int32)
               - np.array([x0, y0])).reshape(-1,1,2)
        color = (0, 120, 245) if item['review_priority'] == 'REVIEW_FIRST' else (95, 210, 95)
        cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=1,
                      lineType=cv2.LINE_AA)
    _write_private_png(out / (tid + '_gray.png'), orig)
    _write_private_png(out / (tid + '_diagnostics.png'), overlay)


def run(s86_directory: Path, original_bmp: Path, private_output: Path,
        cfg: AuditConfig = AuditConfig()) -> dict:
    """Produce strictly private, replayable prioritized *UNKNOWN* review samples.

    S86 contains the canonical pixel masks; this stage NEVER alters them, adds
    geometry, infers particle identity, or auto-discards candidate fragments.
    """
    cfg.validate()
    s86_directory = Path(s86_directory).resolve()
    original_bmp = Path(original_bmp).resolve()
    private_output = Path(private_output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if private_output == repo or repo in private_output.parents:
        raise ValueError('PRIVATE output must be outside public source repository')
    if private_output == s86_directory or s86_directory in private_output.parents:
        raise ValueError('PRIVATE S89 output cannot overwrite S86 source')
    if not original_bmp.is_file():
        raise FileNotFoundError(original_bmp)
    source, records, s86_manifest_sha = _read_source(s86_directory)
    if source['config']['method'] != 's88-steerable':
        raise ValueError('S89 requires source S86 method s88-steerable for consistent perturbation')
    if sha256_file(original_bmp) != source['source_bmp_sha256']:
        raise ValueError('source BMP sha256 differs from S86 provenance')
    bmp = GrayBMP8(original_bmp)
    if [bmp.height, bmp.width] != source['image_shape_yx']:
        raise ValueError('source BMP dimension mismatch')
    score_cfg = WeakRidgeConfig(low_snr=source['config']['s88_low_snr'],
                                high_snr=source['config']['s88_high_snr'],
                                min_component_pixels=source['config']['min_component_area'])
    score_cfg.validate()
    chosen = select_tiles(records, cfg.max_tiles)
    # Preview the full density range, not just the first (sparsest) tiles.
    preview_ids = {r['tile_id'] for r in select_tiles(chosen, cfg.preview_tiles)} if cfg.preview_tiles else set()
    audited_rows = []
    tile_rows = []
    all_previews = {}
    flag_count = Counter()
    for record in chosen:
        tid = record['tile_id']
        core = tuple(record['core_xyxy'])
        pixels, window = _window(bmp, core, source['config']['halo'])
        observed_read = tuple(record['read_xyxy'])
        if window != observed_read:
            raise ValueError('S86 read window changed')
        seed = int.from_bytes(sha256((source['source_bmp_sha256'] + tid
                                      + 'S89_GENERATED_PERTURBATION').encode()).digest()[:8], 'big')
        masks = persistence_under_perturbations(pixels, score_cfg, cfg, seed)
        rows = [audit_one_fragment(f, pixels, window, masks, cfg)
                for f in record['fragments']]
        for item in rows:
            flag_count.update(item['review_flags'])
        audited_rows.extend(rows)
        tile_rows.append({'tile_id': tid, 'core_xyxy_source_px': list(core),
                          'source_fragment_count': record['fragment_count'],
                          'fragment_audited_count': len(rows),
                          'review_first_count': sum(r['review_priority'] == 'REVIEW_FIRST' for r in rows),
                          'mask_candidate_pixel_count': record['foreground_pixel_count'],
                          'no_GT_or_physical_identity': True})
        if tid in preview_ids:
            off_y, off_x = core[1] - window[1], core[0] - window[0]
            h, w = core[3]-core[1], core[2]-core[0]
            all_previews[tid] = (pixels[off_y:off_y+h, off_x:off_x+w], core, rows, record['fragments'])
    if sha256_file(original_bmp) != source['source_bmp_sha256'] or sha256_file(s86_directory / 'manifest.json') != s86_manifest_sha:
        raise ValueError('original BMP or S86 provenance modified during audit')
    private_output.mkdir(parents=True, exist_ok=True)
    private_output.chmod(0o700)
    preview_dir = private_output / 'preview'
    preview_dir.mkdir(exist_ok=True)
    preview_dir.chmod(0o700)
    for tid, (pixels, core, rows, fragments) in all_previews.items():
        _preview_tile(pixels, core, rows, fragments, preview_dir, tid)
    audit_payload = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                            for row in audited_rows).encode('utf-8')
    tiles_payload = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                            for row in tile_rows).encode('utf-8')
    _write_once(private_output / 'candidate_audit.ndjson', audit_payload)
    _write_once(private_output / 'sampled_tiles.ndjson', tiles_payload)
    preview_html = ''.join('<section><h3>' + html.escape(tid) + '</h3>'
                           '<img src="preview/' + html.escape(tid) + '_gray.png" alt="original pixels">'
                           '<img src="preview/' + html.escape(tid) + '_diagnostics.png" alt="machine unknown">'
                           '</section>' for tid in all_previews)
    page = ('<!doctype html><html lang="ru"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>PRIVATE S89 image-space candidate audit</title><style>'
            'body{font:14px/1.5 system-ui;background:#121a24;color:white;max-width:1140px;margin:auto;padding:20px}'
            'section{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:22px 0}'
            'h3{grid-column:1/-1}img{width:100%} @media(max-width:700px){section{grid-template-columns:1fr}}'
            '</style><h1>PRIVATE S89: UNKNOWN machine candidates</h1>'
            '<p>Orange: image-space review first. Green: review later, NOT verified. '
            'Stability under blur/grain does NOT distinguish particle tracks from scratches. '
            'No automatic acceptance, rejection or particle identity. No real particle GT.</p>'
            + preview_html + '</html>')
    _write_once(private_output / 'index.html', page.encode('utf-8'))
    result = {
        'schema': SCHEMA, 'status': 'COMPLETE_MACHINE_ONLY',
        'input_s86_manifest_sha256': s86_manifest_sha,
        'original_BMP_sha256': source['source_bmp_sha256'],
        'source_s88_detector': {'method': source['config']['method'],
                                'low_snr': score_cfg.low_snr, 'high_snr': score_cfg.high_snr},
        'config': asdict(cfg),
        'source_total_tiles': len(records),
        'sampled_tile_count': len(chosen), 'sampled_fragment_count': len(audited_rows),
        'review_first_fragment_count': sum(r['review_priority'] == 'REVIEW_FIRST' for r in audited_rows),
        'image_space_review_flag_count': dict(sorted(flag_count.items())),
        'preview_tiles_count': len(all_previews),
        'sampling': 'DETERMINISTIC_SOURCE_CANDIDATE_DENSITY_QUANTILES_NOT_GT',
        'review_class_labels_are_heuristic_not_probabilities': True,
        'no_physics_labels_or_reference_track_masks_used': True,
        'physical_auto_accept_count': 0, 'physical_auto_reject_count': 0,
        'physics_track_GT_count': 0, 'physical_auto_joins_authorized': False,
        'source_S86_unchanged': True,
        'private_output_only': True,
        'files': ['candidate_audit.ndjson', 'sampled_tiles.ndjson', 'index.html'],
    }
    _write_once(private_output / 'report.json', (json.dumps(result, ensure_ascii=False, sort_keys=True,
                                                          indent=2) + '\n').encode('utf-8'))
    return result
