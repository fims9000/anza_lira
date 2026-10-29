"""Conservative observation-only stitching of S86 tile-boundary skeletons.

Only 8-neighbor observed pixels may be connected. No interpolation, particle
identity, calibrated confidence, stereo correspondence or physical join.
Frozen S72/S85 graph scorers and source S86 mask shards remain unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import html
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from .bitmap_tiles import GrayBMP8, sha256_file
from .full_bmp_segmentation import SCHEMA as SOURCE_SCHEMA, core_rectangles

SCHEMA = 'ANZA_LIRA_S87_SEAM_CONTINUITY_PRIVATE_V1'


def _boundary(p, box):
    x, y = p
    x0, y0, x1, y1 = box
    return x in (x0, x1 - 1) or y in (y0, y1 - 1)


def _tangent(points, side):
    """Unit vector pointing out of the observed open fragment."""
    k = min(5, len(points) - 1)
    v = points[0] - points[k] if side == 0 else points[-1] - points[-1 - k]
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-9 else None


def _read_source(root):
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('schema') != SOURCE_SCHEMA or manifest.get('status') != 'COMPLETE_MACHINE_ONLY'
            or manifest.get('physical_auto_joins_authorized') is not False
            or manifest.get('native_physics_GT_count') != 0):
        raise ValueError('expected completed, machine-only S86 manifest')
    height, width = manifest['image_shape_yx']
    cfg = manifest['config']
    expected = {f'y{y:06d}_x{x:06d}': (x, y, x1, y1)
                for x, y, x1, y1 in core_rectangles(width, height, cfg['core_size'])}
    if manifest['tile_count'] != len(expected):
        raise ValueError('source tile_count does not match exact BMP cover')
    actual = sorted((root / 'records').glob('*.json'))
    if len(actual) != len(expected) or {p.stem for p in actual} != set(expected):
        raise ValueError('source records are missing, extra or duplicated')
    records = []
    for path in actual:
        record = json.loads(path.read_text(encoding='utf-8'))
        tid = path.stem
        if record.get('tile_id') != tid or tuple(record.get('core_xyxy', [])) != expected[tid]:
            raise ValueError('bad tile identity or pixel bounds: ' + tid)
        mask_path = root / 'masks' / (tid + '.png')
        if record.get('mask_file') != 'masks/' + tid + '.png' or not mask_path.is_file():
            raise ValueError('missing S86 mask shard: ' + tid)
        if record.get('mask_sha256') != sha256_file(mask_path):
            raise ValueError('modified S86 mask shard: ' + tid)
        image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        x0, y0, x1, y1 = expected[tid]
        if image is None or image.shape != (y1 - y0, x1 - x0) or not np.isin(image, [0, 255]).all():
            raise ValueError('bad S86 binary mask: ' + tid)
        for f in record['fragments']:
            points = np.asarray(f['centerline_xy_source_px'])
            if (f.get('status') != 'UNKNOWN' or not f['fragment_id'].startswith(tid + '_f')
                    or points.ndim != 2 or points.shape[1] != 2 or len(points) < 4
                    or not np.issubdtype(points.dtype, np.integer)):
                raise ValueError('invalid source polyline: ' + tid)
            local = points - np.array([x0, y0])
            if (np.any(local < 0) or np.any(local[:, 0] >= x1 - x0)
                    or np.any(local[:, 1] >= y1 - y0)
                    or np.any(np.max(np.abs(np.diff(points, axis=0)), axis=1) > 1)
                    or np.any(np.max(np.abs(np.diff(points, axis=0)), axis=1) == 0)
                    or not np.all(image[local[:, 1], local[:, 0]] == 255)):
                raise ValueError('source polyline not supported by stored mask: ' + tid)
        records.append(record)
    return manifest, records, sha256_file(manifest_path)


def assemble(records, *, tangent_angle_deg=55.0):
    """Return image-contiguous proposals and abstentions; never physical joins."""
    if not 0 < tangent_angle_deg <= 90:
        raise ValueError('invalid tangent cutoff')
    fragments, tile_of, endpoints = {}, {}, {}
    by_position = defaultdict(list)
    for record in records:
        box, tid = record['core_xyxy'], record['tile_id']
        for f in record['fragments']:
            fid = f['fragment_id']
            if fid in fragments:
                raise ValueError('duplicate fragment ID')
            points = np.asarray(f['centerline_xy_source_px'], dtype=np.int64)
            fragments[fid], tile_of[fid] = points, tid
            if np.array_equal(points[0], points[-1]):
                continue  # Closed local image loop, no open seam endpoint.
            for side, point in [(0, points[0]), (1, points[-1])]:
                if not _boundary(point, box):
                    continue
                key = (fid, side)
                endpoints[key] = (int(point[0]), int(point[1]))
                by_position[endpoints[key]].append(key)
    # Detect every across-core 8-neighbor pair BEFORE angular filtering.
    candidates = set()
    for (x, y), keys in by_position.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if not (dx or dy):
                    continue
                for a in keys:
                    for b in by_position.get((x + dx, y + dy), ()):
                        if tile_of[a[0]] != tile_of[b[0]] and a[0] != b[0]:
                            candidates.add(tuple(sorted((a, b))))
    degrees = defaultdict(int)
    for a, b in candidates:
        degrees[a] += 1
        degrees[b] += 1
    accepted = []
    decisions = []
    threshold = math.cos(math.radians(tangent_angle_deg))
    for a, b in sorted(candidates):
        pa = np.asarray(endpoints[a], float)
        pb = np.asarray(endpoints[b], float)
        delta = pb - pa
        distance = float(np.linalg.norm(delta))
        ta, tb = _tangent(fragments[a[0]], a[1]), _tangent(fragments[b[0]], b[1])
        if degrees[a] != 1 or degrees[b] != 1:
            reason = 'multiple_adjacent_endpoints_ABSTAIN'
        elif ta is None or tb is None or (np.dot(ta, delta / distance) < threshold
                                          or np.dot(tb, -delta / distance) < threshold):
            reason = 'tangent_incompatible_ABSTAIN'
        else:
            reason = 'unique_observed_pixel_continuity'
            accepted.append((a, b))
        decisions.append({'endpoints': [[a[0], a[1]], [b[0], b[1]]],
                          'source_pixels_xy': [endpoints[a], endpoints[b]],
                          'distance_px': round(distance, 4),
                          'decision': reason, 'physical_identity': 'UNKNOWN',
                          'physical_auto_join': False})
    links = {}
    for a, b in accepted:
        links[a] = b
        links[b] = a
    # Every fragment has at most two incident edges; components are paths or cycles.
    visited = set()
    tracks = []
    skipped_cycles = 0
    for initial in sorted({f for key in links for f in [key[0]]}):
        if initial in visited:
            continue
        group, queue = set(), [initial]
        while queue:
            fid = queue.pop()
            if fid in group:
                continue
            group.add(fid)
            queue.extend(links[(fid, side)][0] for side in (0, 1) if (fid, side) in links)
        visited |= group
        terminals = sorted(f for f in group if sum((f, s) in links for s in (0, 1)) == 1)
        if len(terminals) != 2:
            skipped_cycles += 1
            continue
        fid = terminals[0]
        # Begin at the unconnected endpoint of the first segment.
        joined_side = 0 if (fid, 0) in links else 1
        ordered, joined, points = [], [], []
        while True:
            if fid in ordered:
                raise ValueError('unexpected cycle while walking stitched polyline')
            ordered.append(fid)
            piece = fragments[fid]
            oriented = piece if joined_side == 1 else piece[::-1]
            if points:
                p0, p1 = points[-1], oriented[0]
                if max(abs(int(p0[0] - p1[0])), abs(int(p0[1] - p1[1]))) != 1:
                    raise ValueError('nonadjacent seam bridge is forbidden')
                joined.append([p0.tolist(), p1.tolist()])
            points.extend(oriented)
            other = links.get((fid, joined_side))
            if other is None:
                break
            fid, arrived_side = other
            joined_side = 1 - arrived_side
        if len(ordered) != len(group):
            raise ValueError('incomplete stitched component traversal')
        array = np.asarray(points, dtype=np.int64)
        length = float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())
        identifier = 's87_' + sha256('|'.join(sorted(group)).encode()).hexdigest()[:20]
        tracks.append({'track_hypothesis_id': identifier,
                       'source_fragment_ids_in_geometric_order': ordered,
                       'points_xy_source_px': array.tolist(),
                       'seam_observed_pixel_steps_xy': joined,
                       'length_px_image_only': round(length, 3),
                       'physical_identity': 'UNKNOWN',
                       'source':'OBSERVED_BINARY_SKELETON_ONLY_NO_INTERPOLATION',
                       'is_particle_track_gt': False})
    return tracks, decisions, {
        'source_fragments': len(fragments), 'source_boundary_endpoints': len(endpoints),
        'seam_candidate_pairs': len(candidates), 'unique_tangent_compatible_adjacencies': len(accepted),
        'candidate_pairs_abstained': len(candidates) - len(accepted),
        'stitched_image_components': len(tracks), 'cycle_components_abstained': skipped_cycles,
        'fragments_represented_in_stitched_components': sum(len(t['source_fragment_ids_in_geometric_order']) for t in tracks),
        'physical_GT_count': 0, 'physical_auto_joins_authorized': False,
    }


def _write_new(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError('existing S87 artifact differs, use a new output directory: ' + str(path))
        return
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('xb') as f:
        f.write(content)
    tmp.chmod(0o600)
    os.replace(tmp, path)


def _write_private_previews(s86_dir, out, source, tracks, bmp_path, limit):
    if not 0 <= limit <= 64:
        raise ValueError('private preview limit must be in 0..64')
    if bmp_path is None or not limit or not tracks:
        return []
    if not bmp_path.is_file() or sha256_file(bmp_path) != source['source_bmp_sha256']:
        raise ValueError('original BMP does not match S86 source SHA-256')
    bitmap = GrayBMP8(bmp_path)
    if (bitmap.height, bitmap.width) != tuple(source['image_shape_yx']):
        raise ValueError('original BMP dimensions have changed')
    entries = []
    # Deterministic spatially distributed visual evidence; no cherry-picking
    # based on an unknown true label or detector confidence.
    subset = sorted(tracks, key=lambda t: (t['seam_observed_pixel_steps_xy'][0][0][1],
                                            t['seam_observed_pixel_steps_xy'][0][0][0],
                                            t['track_hypothesis_id']))
    indexes = sorted(set(int(round(v)) for v in np.linspace(0, len(subset)-1, min(limit,len(subset)))))
    (out / 'preview').mkdir(exist_ok=True)
    (out / 'preview').chmod(0o700)
    for j, idx in enumerate(indexes):
        track = subset[idx]
        seam = track['seam_observed_pixel_steps_xy'][0]
        x, y = np.mean(seam, axis=0)
        photo, loc = bitmap.crop((float(x), float(y)), size=192)
        x0, y0, x1, y1 = loc.original_xyxy
        overlay = cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
        points = np.asarray(track['points_xy_source_px'], dtype=np.int64)
        for a, b in zip(points[:-1], points[1:]):
            if (x0 <= a[0] < x1 and y0 <= a[1] < y1 and
                    x0 <= b[0] < x1 and y0 <= b[1] < y1):
                cv2.line(overlay, (int(a[0]-x0), int(a[1]-y0)),
                         (int(b[0]-x0), int(b[1]-y0)), (60, 220, 70), 1)
        for p in seam:
            cv2.circle(overlay, (int(p[0]-x0), int(p[1]-y0)), 4, (40, 100, 250), 1)
        label = f'IMAGE CONTIGUITY / PHYSICS UNKNOWN {j+1}'
        cv2.putText(overlay, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, .27,
                    (40, 220, 250), 1, cv2.LINE_AA)
        stem = f'seam_{j+1:03d}'
        for kind, value in (('original', photo), ('observed', overlay)):
            ok, blob = cv2.imencode('.png', value)
            if not ok:
                raise OSError('could not encode private BMP preview')
            _write_new(out/'preview'/f'{stem}_{kind}.png', blob.tobytes())
        entries.append({'track_hypothesis_id': track['track_hypothesis_id'],
                        'core_seam_xy_PRIVATE': seam, 'stem': stem})
    return entries


def _write_viewer(out, preview, source, metrics):
    sections = []
    for item in preview:
        identifier, stem = item['track_hypothesis_id'], item['stem']
        sections.append('<section><h3>' + html.escape(identifier) + '</h3>'
                        + '<img alt="original private BMP crop" src="preview/' + stem + '_original.png">'
                        + '<img alt="observed machine line pixels only" src="preview/' + stem + '_observed.png">'
                        + '</section>')
    if not sections:
        sections = ['<p>Для фотопроверки укажите исходный приватный BMP через --bmp. '
                    'Без него опубликованы только локальные геометрические файлы.</p>']
    page = ('<!doctype html><html lang="ru"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>PRIVATE S87 seam continuity</title><style>'
            'body{font:14px/1.5 system-ui;color:#eef;background:#111b28;max-width:1080px;'
            'margin:auto;padding:20px}section{display:grid;grid-template-columns:1fr 1fr;'
            'gap:10px;margin:24px 0;border-top:1px solid #456;padding-top:12px}'
            'h3{grid-column:1/-1}img{width:100%;image-rendering:auto}'
            '@media(max-width:600px){section{grid-template-columns:1fr}}'
            '</style><h1>Прямое наблюдение через границы тайлов</h1>'
            '<p>Source: ' + html.escape(source['source_bmp_name']) + '. '
            'All tracks: UNKNOWN. Green: observed skeleton pixels. Orange: adjacent pixel seam. '
            'NO artificial interpolation, NO confirmed physical particle identities.</p>'
            '<p>Image-connected candidate groups: ' + str(metrics['stitched_image_components'])
            + '; vetoed seam candidate pairs: ' + str(metrics['candidate_pairs_abstained'])
            + '.</p>' + ''.join(sections) + '</html>')
    _write_new(out/'index.html', page.encode('utf-8'))


def run(s86_dir: Path, private_out_dir: Path, *, tangent_angle_deg: float = 55.0,
        bmp_path: Path | None = None, preview_limit: int = 12):
    s86_dir, private_out_dir = Path(s86_dir).resolve(), Path(private_out_dir).resolve()
    if bmp_path is not None:
        bmp_path = Path(bmp_path).resolve()
    source_root = Path(__file__).resolve().parents[1]
    if private_out_dir == source_root or source_root in private_out_dir.parents:
        raise ValueError('PRIVATE output must be outside public Git repository')
    if private_out_dir == s86_dir or s86_dir in private_out_dir.parents:
        raise ValueError('S87 must not write to or inside frozen S86 output')
    if not 0 <= preview_limit <= 64:
        raise ValueError('preview_limit must be in 0..64')
    source, records, source_manifest_sha = _read_source(s86_dir)
    tracks, decisions, metrics = assemble(records, tangent_angle_deg=tangent_angle_deg)
    if bmp_path is not None:
        if not bmp_path.is_file() or sha256_file(bmp_path) != source['source_bmp_sha256']:
            raise ValueError('original BMP does not match S86 source SHA-256')
    config = {'tangent_angle_deg': tangent_angle_deg, 'preview_limit': preview_limit,
              'source_BMP_for_previews_checked': bmp_path is not None,
              'source_S86_manifest_sha256': source_manifest_sha,
              'source_S86_file_SHA256': source['source_bmp_sha256'],
              'source_S86_config': source['config']}
    result = {'schema': SCHEMA, 'source': config, 'status': 'COMPLETE_MACHINE_ONLY',
              **metrics, 'no_interpolated_bridge_pixels': True,
              'source_artifacts_unchanged': True,
              'track_hypothesis_ids_not_particle_ids': True,
              'output_files': ['seam_tracks.ndjson', 'seam_decisions.ndjson',
                               'report.json', 'index.html']}
    if private_out_dir.exists() and not private_out_dir.is_dir():
        raise FileExistsError('S87 destination is not a directory')
    private_out_dir.mkdir(parents=True, exist_ok=True)
    private_out_dir.chmod(0o700)
    for name, rows in (('seam_tracks.ndjson', tracks), ('seam_decisions.ndjson', decisions)):
        payload = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows).encode('utf-8')
        _write_new(private_out_dir / name, payload)
    private_gallery = _write_private_previews(s86_dir, private_out_dir, source, tracks,
                                               bmp_path, preview_limit)
    result['private_gallery_crop_count'] = len(private_gallery)
    _write_viewer(private_out_dir, private_gallery, source, metrics)
    _write_new(private_out_dir / 'report.json', (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8'))
    return result
