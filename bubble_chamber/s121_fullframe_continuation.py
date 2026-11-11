"""S121: conservative source-backed whole-frame continuations after S118.

Original pixel paths are never modified; interpolated bridges remain separate,
reviewable geometry. Automatic decisions are PROVISIONAL, not particle IDs.
No human annotations or private classifiers are used for S121 matching.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter, defaultdict
from hashlib import sha256
from math import ceil
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ContinuationConfig:
    max_gap_px: float = 400.
    min_gap_px: float = 8.
    min_anchor_length_px: float = 55.
    dedup_port_px: float = 9.
    min_local_alignment: float = .90
    min_far_alignment: float = .82
    max_lateral_px: float = 2.5
    max_rival_fraction: float = .055
    min_photo_fraction: float = .70
    min_photo_fraction_long: float = .72
    min_score: float = .83
    min_best_margin: float = .10
    tangent_window_px: float = 22.
    far_tangent_window_px: float = 78.


def _unit(v):
    n = float(np.linalg.norm(v))
    return np.asarray(v, dtype=float)/n if n > 1e-9 else None


def _tangent(xy, end, span):
    pts = xy if end == 0 else xy[::-1]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    j = min(len(pts)-1, max(1, int(np.searchsorted(arc, span))))
    return _unit(pts[0]-pts[j])


def _ports(tiles, local_groups, groups, cfg):
    membership = {key: group['global_group_id']
                  for group in groups for key in group['source_curve_keys']}
    if len(membership) != sum(len(g['source_curve_keys']) for g in groups):
        raise ValueError('Duplicate source curve membership')
    raw, cloud, cloud_groups = [], [], []
    for tile in tiles:
        tid = tile['tile_id']
        x0, y0, x1, y1 = tile['xyxy']
        occupied = set()
        for link in local_groups.get(tid, {}).get('links', []):
            if link.get('decision') in (
                    'AUTO_PROVISIONAL_GEOMETRY_GROUP_NOT_GT',
                    'EXPERT_CONFIRMED_SAME_LINK'):
                occupied.update((str(k), int(end)) for k, end in
                                zip(link['curve_ids'], link['endpoint_indices']))
        for c in tile['curves']:
            name = str(c.get('candidate_id') or c.get('curve_id'))
            key = tid+':'+name
            if key not in membership:
                raise ValueError('Source curve missing from S118 groups')
            xy = np.asarray(c['observed_pixel_path_xy'], float)
            if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
                raise ValueError('Malformed observed source path')
            if np.any(xy[:, 0] < 0) or np.any(xy[:, 0] >= x1-x0) or np.any(xy[:, 1] < 0) or np.any(xy[:, 1] >= y1-y0):
                raise ValueError('Observed curve outside source tile')
            full = xy+[x0, y0]
            arc = float(np.linalg.norm(np.diff(full, axis=0), axis=1).sum())
            stride = max(1, int(ceil(len(full)/max(4, arc/7))))
            sampled = full[::stride]
            cloud.append(sampled)
            cloud_groups.extend([membership[key]]*len(sampled))
            if arc < cfg.min_anchor_length_px:
                continue
            for end in (0, 1):
                if (name, end) in occupied:
                    continue
                tangent = _tangent(full, end, cfg.tangent_window_px)
                far = _tangent(full, end, cfg.far_tangent_window_px)
                if tangent is None or far is None:
                    continue
                raw.append({'id': key+':'+str(end), 'group': membership[key],
                            'pos': full[0 if end == 0 else -1],
                            'tangent': tangent, 'far': far, 'arc': arc})
    # Multiple overlapping tiles can observe the SAME original endpoint.
    keep, cells = [], defaultdict(list)
    for item in sorted(raw, key=lambda z: (-z['arc'], z['id'])):
        pos = item['pos']
        gx = int(pos[0]//cfg.dedup_port_px)
        gy = int(pos[1]//cfg.dedup_port_px)
        near = [keep[j] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                for j in cells.get((gx+dx, gy+dy), ())]
        if any(other['group'] == item['group'] and
               np.linalg.norm(other['pos']-pos) < cfg.dedup_port_px for other in near):
            continue
        cells[(gx, gy)].append(len(keep))
        keep.append(item)
    pts = np.vstack(cloud).astype(np.float32) if cloud else np.empty((0, 2), np.float32)
    return keep, pts, np.asarray(cloud_groups)


def _bridge(a, b, n):
    start, end = a['pos'], b['pos']
    distance = float(np.linalg.norm(end-start))
    cp = np.stack((start, start+.29*distance*a['tangent'],
                   end+.29*distance*b['tangent'], end))
    t = np.linspace(0, 1, n)[:, None]
    return ((1-t)**3*cp[0]+3*(1-t)**2*t*cp[1]+
            3*(1-t)*t**2*cp[2]+t**3*cp[3])


def _photo_support(gray, xy):
    path = xy[3:-3]
    if len(path) < 8:
        return 0., 0.
    direction = np.gradient(path, axis=0)
    norm = np.maximum(np.linalg.norm(direction, axis=1), 1e-6)
    normal = np.column_stack((-direction[:, 1]/norm, direction[:, 0]/norm))
    h, w = gray.shape

    def intensity(offset):
        pts = np.rint(path+offset*normal).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w-1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h-1)
        return gray[pts[:, 1], pts[:, 0]].astype(float)

    center = intensity(0)
    contrast = .4*((intensity(4)+intensity(-4))/2-center)+.6*(
        (intensity(8)+intensity(-8))/2-center)
    return float(np.mean(contrast > 4)), float(np.mean(contrast))


class _Union:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, k):
        while self.parent[k] != k:
            self.parent[k] = self.parent[self.parent[k]]
            k = self.parent[k]
        return k

    def link(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if a > b:
            a, b = b, a
        self.parent[b] = a
        return True


def associate_fullframe(tiles, local_groups, s118_groups, gray,
                        cfg: ContinuationConfig = ContinuationConfig()):
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError('Original grayscale uint8 BMP required')
    ports, cloud, owners = _ports(tiles, local_groups, s118_groups, cfg)
    if len(ports) < 2:
        return {'groups': list(s118_groups), 'proposals': [],
                'report': {'ports': len(ports), 'accepted_provisional_links': 0}}
    endpoints = cKDTree(np.array([p['pos'] for p in ports]))
    rival_tree = cKDTree(cloud) if len(cloud) else None
    pairs = sorted(endpoints.query_pairs(cfg.max_gap_px))
    candidates = []
    for i, j in pairs:
        a, b = ports[i], ports[j]
        if a['group'] == b['group']:
            continue
        delta = b['pos']-a['pos']
        gap = float(np.linalg.norm(delta))
        if gap < cfg.min_gap_px:
            continue
        axis = delta/gap
        alignment = min(float(a['tangent']@axis),
                        float(b['tangent']@(-axis)))
        if alignment < cfg.min_local_alignment:
            continue
        far = min(float(a['far']@axis), float(b['far']@(-axis)))
        if far < cfg.min_far_alignment:
            continue
        lateral = (abs(delta[0]*a['tangent'][1]-delta[1]*a['tangent'][0])+
                   abs(delta[0]*b['tangent'][1]-delta[1]*b['tangent'][0]))/2
        if lateral > min(cfg.max_lateral_px, 3.+.02*gap):
            continue
        path = _bridge(a, b, max(28, min(180, int(gap/2))))
        if not np.all((path[:, 0] >= 8) & (path[:, 0] < gray.shape[1]-8) &
                      (path[:, 1] >= 8) & (path[:, 1] < gray.shape[0]-8)):
            continue
        support, contrast = _photo_support(gray, path)
        rival = 0.
        if rival_tree is not None:
            samples = path[5:-5:max(1, len(path)//110)]
            near = rival_tree.query_ball_point(samples, r=3.2)
            rival = sum(any(owners[k] not in (a['group'], b['group'])
                            for k in neighbors) for neighbors in near)/max(1, len(near))
        score = .33*alignment+.16*far+.44*support+.07*(1-gap/cfg.max_gap_px)-.25*rival
        ports_ids = [a['id'], b['id']]
        candidates.append({
            'candidate_id': 's121_'+sha256('|'.join(sorted(ports_ids)).encode()).hexdigest()[:16],
            'group_ids': [a['group'], b['group']],
            'source_endpoint_ports': ports_ids, 'gap_px': round(gap, 2),
            'alignment': round(alignment, 4), 'far_alignment': round(far, 4),
            'max_lateral_residual_px': round(float(lateral), 3),
            'source_photo_support': round(support, 4),
            'photo_center_minus_flank': round(contrast, 2),
            'rival_fraction': round(rival, 4), 'score': round(score, 5),
            'inferred_xy': np.round(path, 2).tolist(),
            'source_observed_bridge_pixels': 0, 'inferred_geometry_only': True,
            'decision': 'REVIEW_ONLY_UNRESOLVED',
        })
    byport = defaultdict(list)
    for e in candidates:
        for port in e['source_endpoint_ports']:
            byport[port].append(e)
    for edges in byport.values():
        edges.sort(key=lambda z: (-z['score'], z['candidate_id']))
    chosen = []
    used = set()
    union = _Union([g['global_group_id'] for g in s118_groups])
    for e in sorted(candidates, key=lambda z: (-z['score'], z['candidate_id'])):
        a, b = e['source_endpoint_ports']
        if a in used or b in used:
            e['decision'] = 'REVIEW_ONLY_PORT_IN_USE'
            continue
        if union.find(e['group_ids'][0]) == union.find(e['group_ids'][1]):
            e['decision'] = 'REVIEW_ONLY_GROUP_CYCLE'
            continue
        ranked = [byport[a], byport[b]]
        margin = min((x[0]['score']-x[1]['score'] if len(x)>1 else 1.)
                     for x in ranked)
        mutual = all(x[0]['candidate_id'] == e['candidate_id'] for x in ranked)
        e['score_margin'] = round(float(margin), 4)
        if not mutual or margin < cfg.min_best_margin:
            e['decision'] = 'REVIEW_ONLY_COMPETING_CONTINUATION'
            continue
        photo_min = cfg.min_photo_fraction_long if e['gap_px'] > 125 else cfg.min_photo_fraction
        if (e['source_photo_support'] < photo_min or e['score'] < cfg.min_score or
                e['rival_fraction'] > cfg.max_rival_fraction):
            e['decision'] = 'REVIEW_ONLY_INSUFFICIENT_SOURCE_EVIDENCE'
            continue
        if any(np.any(cKDTree(np.asarray(f['inferred_xy'])[5:-5:3]).query(
                   np.asarray(e['inferred_xy'])[5:-5:3])[0] < 3.) for f in chosen):
            e['decision'] = 'REVIEW_ONLY_CROSSING_BRIDGES'
            continue
        e['decision'] = 'AUTO_PROVISIONAL_CONTINUATION_NOT_GT'
        chosen.append(e)
        used.update((a, b))
        union.link(*e['group_ids'])
    members, old_ids = defaultdict(list), defaultdict(list)
    for group in s118_groups:
        root = union.find(group['global_group_id'])
        members[root].extend(group['source_curve_keys'])
        old_ids[root].append(group['global_group_id'])
    groups = []
    for root in sorted(members):
        names = sorted(set(old_ids[root]))
        groups.append({
            'global_group_id': 's121_group_'+sha256('|'.join(names).encode()).hexdigest()[:16],
            'source_curve_keys': sorted(set(members[root])),
            's118_member_group_ids': names,
            'source_observations_unchanged': True, 'physical_particle_id': None,
            'identity_status': 'PHOTO_SUPPORTED_PROVISIONAL_NOT_GT',
        })
    return {'groups': groups, 'proposals': candidates,
            'report': {'ports': len(ports), 'raw_endpoint_pairs': len(pairs),
                       'geometric_source_candidates': len(candidates),
                       'accepted_provisional_links': len(chosen),
                       'review_only_links': len(candidates)-len(chosen),
                       'original_s118_groups': len(s118_groups),
                       'provisional_s121_groups': len(groups),
                       'no_observed_pixels_inferred': True,
                       'manual_labels_used': False,
                       'physical_identity_accuracy_known': False}}
