"""S117 long-gap association of 2-D bubble-chamber line fragments.

This module changes *group identity*, never observed geometry. It takes source
curves produced by S109 or S115; every bridge remains inferred and auditable.
Automatic geometry grouping is conservative; alternative links stay reviewable.
A provisional group is not a physically identified particle track.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GapConfig:
    min_gap_px: float = 9.0
    max_gap_px: float = 400.0
    min_observed_length_px: float = 18.0
    tangent_window_px: float = 18.0
    min_alignment: float = 0.80
    min_long_alignment: float = 0.91
    min_auto_alignment: float = 0.94
    min_auto_multiscale_alignment: float = 0.90
    multiscale_tangent_window_px: float = 70.0
    min_auto_support: float = 0.45
    min_auto_support_long: float = 0.58
    min_auto_score: float = 0.78
    min_auto_margin: float = 0.09
    max_auto_rival_fraction: float = 0.045
    max_auto_lateral_fraction: float = 0.18
    max_auto_endpoint_projection_px: float = 14.0
    max_blocked_fraction: float = 0.0
    max_observation_sample: int = 1500


def _unit(v):
    v = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(v))
    return None if norm < 1e-6 else v / norm


def _length(xy):
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0


def _normalize(curves, cfg):
    result = []
    ids = set()
    for curve in curves:
        name = str(curve.get('candidate_id') or curve.get('curve_id') or curve.get('tracklet_id') or '')
        if not name or name in ids:
            raise ValueError('Missing or duplicate curve identity: ' + name)
        ids.add(name)
        source = curve.get('observed_pixel_path_xy')
        if source is None:
            source = curve.get('sampled_xy')
        xy = np.asarray(source, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
            raise ValueError('Malformed two-dimensional source geometry: ' + name)
        length = _length(xy)
        flags = curve.get('sampled_source_supported')
        if flags is not None and len(flags) != len(xy):
            raise ValueError('Source support flag count mismatch: ' + name)
        result.append({'id': name, 'xy': xy, 'length': length, 'flags': flags,
                       'source_anchored': bool(curve.get('source_anchored', True)),
                       'eligible': length >= cfg.min_observed_length_px})
    return result


def _endpoint(xy, end, window):
    order = xy if end == 0 else xy[::-1]
    steps = np.linalg.norm(np.diff(order, axis=0), axis=1)
    if len(steps) == 0:
        return None
    distance = np.r_[0.0, np.cumsum(steps)]
    j = min(len(order) - 1, max(1, int(np.searchsorted(distance, window, side='left'))))
    outward = _unit(order[0] - order[j])
    return outward


def _bezier(p, q, t0, t1, count):
    d = float(np.linalg.norm(q - p))
    controls = np.stack((p, p + 0.28 * d * t0, q + 0.28 * d * t1, q))
    ts = np.linspace(0.0, 1.0, count)[:, None]
    path = ((1 - ts)**3 * controls[0] + 3 * (1 - ts)**2 * ts * controls[1]
            + 3 * (1 - ts) * ts**2 * controls[2] + ts**3 * controls[3])
    return path, controls


def _evidence_map(gray, response):
    if response is not None:
        m = np.asarray(response, np.float32)
        if m.shape != gray.shape or not np.isfinite(m).all():
            raise ValueError('Directional response must match image and be finite')
        return cv2.dilate(m, np.ones((3, 3), np.uint8)), 8.5, 'DIRECTIONAL_SOURCE_RESPONSE'
    f = np.asarray(gray, np.float32)
    # Local dark thin-ridge evidence, not a replacement for the S114 detector.
    signal = cv2.GaussianBlur(f, (0, 0), 4.0) - cv2.GaussianBlur(f, (0, 0), 1.0)
    threshold = max(2.8, float(np.percentile(signal[::5, ::5], 82)))
    return cv2.dilate(signal, np.ones((3, 3), np.uint8)), threshold, 'PHOTO_HIGH_PASS_UNCALIBRATED'


def _rival_cloud(curves, cfg):
    clouds = []
    owners = []
    for i, c in enumerate(curves):
        xy = c['xy']
        step = max(1, len(xy) // cfg.max_observation_sample)
        cloud = xy[::step]
        clouds.append(cloud)
        owners.extend([i] * len(cloud))
    if not clouds:
        return None, None
    return cKDTree(np.vstack(clouds)), np.asarray(owners, np.int32)


def _rival_fraction(tree, owners, path, ai, bi):
    if tree is None or len(owners) < 3 or len(path) < 12:
        return 0.0
    pts = path[4:-4: max(1, len(path) // 120)]
    # A true crossing may be a third track. Penalise sustained overlap, not
    # every isolated intersection or nearby independent track.
    ids = tree.query_ball_point(pts, r=3.0)
    hit = sum(any(int(owners[k]) not in (ai, bi) for k in found) for found in ids)
    return hit / max(1, len(ids))


def _intersection(a, b, tolerance=3.0):
    if len(a) < 10 or len(b) < 10:
        return False
    x = a[4:-4:max(1, len(a) // 120)]
    y = b[4:-4:max(1, len(b) // 120)]
    if not len(x) or not len(y):
        return False
    return bool(np.any(cKDTree(x).query(y)[0] < tolerance))


def associate_long_gaps(curves, gray, response=None, blocked=None,
                        decisions: Mapping[str, str] | None = None,
                        cfg: GapConfig = GapConfig()):
    """Return candidate links and provisional groups, with no hallucinated pixels.

    ``decisions`` can force an expert-verified ``same`` or exclude ``different``;
    any forced port conflict or cycle is rejected, rather than silently changed.
    Source labels are not consulted by this function.
    """
    gray = np.asarray(gray)
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError('Expected uint8 grayscale ROI')
    height, width = gray.shape
    if gray.size > 4194304:
        raise ValueError('S117 requires bounded ROIs (<= 4 Mi pixels); use tiled segmentation')
    block = np.zeros(gray.shape, dtype=bool) if blocked is None else np.asarray(blocked, bool)
    if block.shape != gray.shape:
        raise ValueError('Blocked region mask shape mismatch')
    c = _normalize(curves, cfg)
    for item in c:
        xy = item['xy']
        if (xy[:, 0].min() < 0 or xy[:, 1].min() < 0 or
                xy[:, 0].max() >= width or xy[:, 1].max() >= height):
            raise ValueError('Curve outside ROI: ' + item['id'])
    evidence, evidence_thr, evidence_origin = _evidence_map(gray, response)
    ports = []
    for i, item in enumerate(c):
        if not item['eligible'] or not item['source_anchored']:
            continue
        for end in (0, 1):
            tangent = _endpoint(item['xy'], end, cfg.tangent_window_px)
            if tangent is not None:
                flags = item['flags']
                endpoint_supported = flags is None or (
                    sum(bool(z) for z in (flags[:min(10,len(flags))] if end == 0
                                         else flags[-min(10,len(flags)):])) >= 3)
                ports.append({'i': i, 'end': end, 'xy': item['xy'][0 if end == 0 else -1],
                              'outward': tangent, 'endpoint_supported': endpoint_supported})
    proposals = []
    if len(ports) > 1:
        positions = np.stack([p['xy'] for p in ports])
        pairs = sorted(cKDTree(positions).query_pairs(cfg.max_gap_px))
        rival_tree, rival_owner = _rival_cloud(c, cfg)
        for a, b in pairs:
            pa, pb = ports[a], ports[b]
            if pa['i'] == pb['i']:
                continue
            p, q = pa['xy'], pb['xy']
            gap = float(np.linalg.norm(q-p))
            if not cfg.min_gap_px <= gap <= cfg.max_gap_px:
                continue
            axis = (q-p) / gap
            ca = float(pa['outward'] @ axis)
            cb = float(pb['outward'] @ (-axis))
            align = min(ca, cb)
            # A shallow angle over hundreds of pixels can still jump between
            # two nearly parallel tracks. Measure the *absolute* departure
            # from each endpoint tangent, not only its cosine.
            projection_px = max(abs(pa['outward'][0] * (q-p)[1] - pa['outward'][1] * (q-p)[0]),
                                abs(pb['outward'][0] * (q-p)[1] - pb['outward'][1] * (q-p)[0]))
            if align < (cfg.min_long_alignment if gap > 150 else cfg.min_alignment):
                continue
            size = max(24, min(400, int(gap)))
            path, controls = _bezier(p, q, pa['outward'], pb['outward'], size)
            lateral = np.abs(axis[0] * (path[:, 1] - p[1]) - axis[1] * (path[:, 0] - p[0])).max() / gap
            if lateral > cfg.max_auto_lateral_fraction * 2.2:
                continue
            ij = np.rint(path).astype(np.int32)
            if ((ij[:, 0] < 0) | (ij[:, 0] >= width) |
                    (ij[:, 1] < 0) | (ij[:, 1] >= height)).any():
                continue
            center = ij[3:-3]
            blocked_fraction = float(block[center[:, 1], center[:, 0]].mean())
            support = float((evidence[center[:, 1], center[:, 0]] >= evidence_thr).mean())
            rival = _rival_fraction(rival_tree, rival_owner, path, pa['i'], pb['i'])
            # A locally straight 18-pixel join can conceal a switch at an
            # observed crossing. Compare the underlying fragment directions
            # over a longer arc before assigning one provisional identity.
            ta_far = _endpoint(c[pa['i']]['xy'], pa['end'], cfg.multiscale_tangent_window_px)
            tb_far = _endpoint(c[pb['i']]['xy'], pb['end'], cfg.multiscale_tangent_window_px)
            far_align = (min(float(ta_far @ axis), float(tb_far @ (-axis)))
                         if ta_far is not None and tb_far is not None else -1.0)
            score = (.42 * align + .33 * support +
                     .15 * max(0.0, 1.0 - lateral / cfg.max_auto_lateral_fraction) +
                     .10 * max(0.0, 1.0 - gap / cfg.max_gap_px) - .18 * rival)
            ids = (c[pa['i']]['id'], c[pb['i']]['id'])
            ends = (pa['end'], pb['end'])
            # Stable across enumeration and input curve order.
            canonical = '|'.join(sorted(f'{n}:{e}' for n, e in zip(ids, ends)))
            key = 'S117_' + sha256(canonical.encode('utf-8')).hexdigest()[:16]
            proposals.append({'candidate_id': key, 'curve_ids': list(ids),
                              'endpoint_indices': list(ends),
                              'endpoint_ports': [f'{n}:{e}' for n, e in zip(ids, ends)],
                              'gap_px': round(gap, 2), 'alignment_min': round(align, 4),
                              'multiscale_alignment_min': round(far_align, 4),
                              'photo_support_fraction': round(support, 4),
                              'rival_fraction': round(rival, 4),
                              'blocked_fraction': round(blocked_fraction, 4),
                              'lateral_fraction': round(float(lateral), 4),
                              'endpoint_projection_residual_px': round(float(projection_px), 3),
                              'both_endpoints_source_supported': bool(pa['endpoint_supported'] and pb['endpoint_supported']),
                              'score': round(float(score), 5),
                              'inferred_bezier_controls_xy': np.round(controls, 2).tolist(),
                              'inferred_connector_xy': np.round(path, 2).tolist(),
                              'observed_connector_pixels': 0,
                              'connector_is_observed': False,
                              'confirmed_particle_id': None,
                              'decision': 'REVIEW_ONLY_UNRESOLVED'})
    proposals.sort(key=lambda e: (-e['score'], e['candidate_id']))
    requested = dict(decisions or {})
    if set(requested.values()) - {'same', 'different', 'uncertain'}:
        raise ValueError('Invalid decision: expected same/different/uncertain')
    byid = {e['candidate_id']: e for e in proposals}
    if set(requested) - set(byid):
        raise ValueError('Decision references unknown S117 candidate')
    rankings = {}
    for e in proposals:
        if requested.get(e['candidate_id']) != 'different':
            for port in e['endpoint_ports']:
                rankings.setdefault(port, []).append(e)
    for entries in rankings.values():
        entries.sort(key=lambda x: (-x['score'], x['candidate_id']))
    parent = list(range(len(c)))
    id_to_index = {item['id']: idx for idx, item in enumerate(c)}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    joined = []
    port_used = set()
    forced = [e for e in proposals if requested.get(e['candidate_id']) == 'same']
    candidates = forced + [e for e in proposals if e not in forced]
    for e in candidates:
        key = e['candidate_id']
        choice = requested.get(key)
        if choice == 'different':
            e['decision'] = 'EXPERT_REJECTED_DIFFERENT'
            continue
        if choice == 'uncertain':
            e['decision'] = 'EXPERT_UNCERTAIN'
            continue
        pi, pj = e['endpoint_ports']
        if pi in port_used or pj in port_used:
            if choice == 'same':
                raise ValueError('Conflicting expert same links at endpoint')
            e['decision'] = 'REVIEW_ONLY_ENDPOINT_IN_USE'
            continue
        i, j = (id_to_index[name] for name in e['curve_ids'])
        if root(i) == root(j):
            if choice == 'same':
                raise ValueError('Expert same links create a cycle')
            e['decision'] = 'REVIEW_ONLY_CYCLE'
            continue
        ranked = (rankings.get(pi, []), rankings.get(pj, []))
        best = all(v and v[0]['candidate_id'] == key for v in ranked)
        margin = min((v[0]['score'] - v[1]['score'] if len(v) > 1 else 1.0) for v in ranked)
        e['best_at_both_endpoints'] = bool(best)
        e['endpoint_score_margin'] = round(float(margin), 5)
        limit = cfg.min_auto_support_long if e['gap_px'] > 150 else cfg.min_auto_support
        projection_limit = min(cfg.max_auto_endpoint_projection_px, 6.0 + 0.025 * e['gap_px'])
        eligible = (best and margin >= cfg.min_auto_margin and
                    e['alignment_min'] >= cfg.min_auto_alignment and
                    e['multiscale_alignment_min'] >= cfg.min_auto_multiscale_alignment and
                    e['photo_support_fraction'] >= limit and
                    e['score'] >= cfg.min_auto_score and
                    e['rival_fraction'] <= cfg.max_auto_rival_fraction and
                    e['blocked_fraction'] <= cfg.max_blocked_fraction and
                    e['lateral_fraction'] <= cfg.max_auto_lateral_fraction and
                    e['endpoint_projection_residual_px'] <= projection_limit and
                    e['both_endpoints_source_supported'])
        if choice != 'same' and not eligible:
            if not best or margin < cfg.min_auto_margin:
                e['decision'] = 'REVIEW_ONLY_COMPETING_CONTINUATION'
            elif e['multiscale_alignment_min'] < cfg.min_auto_multiscale_alignment:
                e['decision'] = 'REVIEW_ONLY_MULTISCALE_DIRECTION_CHANGE'
            elif e['endpoint_projection_residual_px'] > projection_limit:
                e['decision'] = 'REVIEW_ONLY_PARALLEL_OFFSET_OR_CURVATURE'
            elif not e['both_endpoints_source_supported']:
                e['decision'] = 'REVIEW_ONLY_ENDPOINT_NOT_SOURCE_BACKED'
            elif e['blocked_fraction'] > cfg.max_blocked_fraction:
                e['decision'] = 'REVIEW_ONLY_PROTECTED_REGION'
            elif e['rival_fraction'] > cfg.max_auto_rival_fraction:
                e['decision'] = 'REVIEW_ONLY_RIVAL_TRACK'
            else:
                e['decision'] = 'REVIEW_ONLY_INSUFFICIENT_EVIDENCE'
            continue
        if choice != 'same' and any(_intersection(np.asarray(f['inferred_connector_xy']),
                                                   np.asarray(e['inferred_connector_xy'])) for f in joined):
            e['decision'] = 'REVIEW_ONLY_CROSSING_INFERRED_BRIDGES'
            continue
        port_used.update((pi, pj))
        parent[root(j)] = root(i)
        e['decision'] = ('EXPERT_CONFIRMED_SAME_LINK' if choice == 'same'
                         else 'AUTO_PROVISIONAL_GEOMETRY_GROUP_NOT_GT')
        joined.append(e)
    members = {}
    for i, t in enumerate(c):
        members.setdefault(root(i), []).append(t['id'])
    groups = []
    for names in members.values():
        sorted_names = sorted(names)
        connections = [e['candidate_id'] for e in joined if
                       set(e['curve_ids']).issubset(sorted_names)]
        groups.append({'group_id': 's117_group_' + sha256('|'.join(sorted_names).encode()).hexdigest()[:16],
                       'source_curve_ids': sorted_names,
                       'observed_components': len(sorted_names),
                       'inferred_connection_ids': sorted(connections),
                       'confirmed_particle_id': None,
                       'status': 'PROVISIONAL_GEOMETRY_NOT_PHYSICAL_ID',
                       'observations_unchanged': True})
    groups.sort(key=lambda x: (-len(x['source_curve_ids']), x['group_id']))
    return {'groups': groups, 'links': proposals,
            'report': {'source_curve_count': len(c),
                       'source_eligible_count': sum(x['eligible'] and x['source_anchored'] for x in c),
                       'candidate_link_count': len(proposals),
                       'accepted_provisional_links': sum(e['decision'] == 'AUTO_PROVISIONAL_GEOMETRY_GROUP_NOT_GT' for e in proposals),
                       'expert_accepted_links': sum(e['decision'] == 'EXPERT_CONFIRMED_SAME_LINK' for e in proposals),
                       'review_links': sum(e['decision'].startswith('REVIEW_ONLY') for e in proposals),
                       'output_group_count': len(groups),
                       'max_link_gap_px': max((e['gap_px'] for e in joined), default=None),
                       'source_evidence': evidence_origin,
                       'source_photo_pixels_used_for_group_decision': True,
                       'labels_used_in_inference': False,
                       'added_observed_pixel_count': 0,
                       'physical_identity_accuracy_known': False}}
