"""S118: identity-preserving stitching of S117 tile groups on one source frame.

This stage consumes *observed* local line coordinates and local provisional groups.
It never turns an interpolation into a detected source pixel or asserts a particle
identity. Overlap aliases require bidirectional measured geometry agreement.
Long cross-tile gaps are exported for review and are NOT automatically accepted
without photo evidence (which this stage deliberately does not receive).
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from collections import defaultdict
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class StitchConfig:
    alias_radius_px: float = 2.4
    alias_median_max_px: float = 1.15
    alias_p90_max_px: float = 2.3
    alias_fraction_min: float = 0.86
    alias_overlap_length_px: float = 35.0
    alias_margin: float = 0.12
    seam_band_px: float = 26.0
    review_gap_px: float = 800.0
    auto_seam_gap_px: float = 7.0
    auto_seam_alignment: float = 0.975
    auto_seam_lateral_px: float = 1.6
    auto_seam_margin: float = 0.085
    auto_nearby_rival_px: float = 4.0
    max_curve_sample: int = 2500


def _hash(*parts):
    return sha256('|'.join(parts).encode('utf8')).hexdigest()[:16]


def _arc(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _unit(vector):
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector, float) / norm if norm > 1e-8 else None


def _endpoint(points, end, window=16.):
    seq = points if end == 0 else points[::-1]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seq, axis=0), axis=1))]
    j = min(len(seq)-1, max(1, int(np.searchsorted(arc, window))))
    return seq[0], _unit(seq[0] - seq[j])


def _rect_overlap(a, b):
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def _inside(points, rect):
    return ((points[:, 0] >= rect[0]) & (points[:, 0] < rect[2]) &
            (points[:, 1] >= rect[1]) & (points[:, 1] < rect[3]))


def _boundary_exit(point, direction, rect, band):
    x, y = point
    faces = ((x-rect[0], np.array([-1.,0.])), (rect[2]-1-x, np.array([1.,0.])),
             (y-rect[1], np.array([0.,-1.])), (rect[3]-1-y, np.array([0.,1.])))
    return any(-0.5 <= d <= band and float(direction @ normal) >= 0.55 for d, normal in faces)


class _Union:
    def __init__(self, items, local_group):
        self.parent = {name: name for name in items}
        self.occupied = {name: {name.split(':',1)[0]: {local_group[name]}} for name in items}

    def root(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b, allow_local=False):
        x, y = self.root(a), self.root(b)
        if x == y:
            return 'already'
        left, right = self.occupied[x], self.occupied[y]
        if not allow_local and any(left[t] != right[t] for t in left.keys() & right.keys()):
            return 'conflicting_local_groups'
        if x > y:
            x, y = y, x
        self.parent[y] = x
        occupied = self.occupied[x]
        for tile, groups in self.occupied[y].items():
            occupied.setdefault(tile, set()).update(groups)
        del self.occupied[y]
        return 'joined'


def _validate(tiles, cfg):
    if not isinstance(tiles, list) or not tiles:
        raise ValueError('At least one tile required')
    records = {}
    groups = {}
    tile_meta = {}
    frame_id = None
    source_sha = None
    for t in tiles:
        name = t.get('tile_id')
        if not isinstance(name, str) or not name or ':' in name or name in tile_meta:
            raise ValueError('Invalid or duplicate tile ID')
        img_id, sha = t.get('source_image_id'), t.get('source_image_sha256')
        if not isinstance(img_id, str) or not img_id or not isinstance(sha, str) or len(sha) != 64:
            raise ValueError('Missing source image name or SHA256')
        if frame_id is None:
            frame_id, source_sha = img_id, sha
        if img_id != frame_id or sha != source_sha:
            raise ValueError('Tiles originate from different source images')
        rect = t.get('xyxy')
        if not isinstance(rect, list) or len(rect) != 4 or not all(isinstance(z, int) for z in rect):
            raise ValueError('Tile bounds must be integer xyxy')
        if rect[0] < 0 or rect[1] < 0 or rect[0] >= rect[2] or rect[1] >= rect[3]:
            raise ValueError('Invalid tile bounds')
        tile_meta[name] = rect
        curves = t.get('curves')
        local = t.get('s117', {})
        if not isinstance(curves, list) or not isinstance(local.get('groups'), list):
            raise ValueError('Missing S117 curves or groups')
        seen = set()
        for curve in curves:
            cid = curve.get('candidate_id') or curve.get('curve_id') or curve.get('tracklet_id')
            if not isinstance(cid, str) or not cid or ':' in cid or cid in seen:
                raise ValueError('Invalid or duplicate curve ID in tile ' + name)
            seen.add(cid)
            src = curve.get('observed_pixel_path_xy', curve.get('sampled_xy'))
            xy = np.asarray(src, dtype=float)
            if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
                raise ValueError('Malformed curve geometry')
            if not np.all(_inside(xy, [0, 0, rect[2]-rect[0], rect[3]-rect[1]])):
                raise ValueError('Curve outside local tile')
            step = max(1, len(xy) // cfg.max_curve_sample)
            full = xy[::step] + np.asarray(rect[:2], float)
            if not np.array_equal(full[-1], xy[-1] + rect[:2]):
                full = np.vstack([full, xy[-1] + rect[:2]])
            records[name+':'+cid] = {'tile': name, 'rect': rect, 'xy': full, 'arc': _arc(full)}
        once = {}
        for row in local['groups']:
            gid = row.get('group_id')
            if not isinstance(gid, str) or not gid:
                raise ValueError('Invalid local group ID')
            for cid in row.get('source_curve_ids', []):
                if cid not in seen or cid in once:
                    raise ValueError('S117 groups omit or double-assign curves')
                once[cid] = name+':'+gid
        if seen != set(once):
            raise ValueError('S117 groups omit or double-assign curves')
        groups.update({name+':'+cid: gid for cid, gid in once.items()})
    return records, groups, tile_meta, frame_id, source_sha


def _alias_metrics(a, b, rect, cfg):
    p = a['xy'][_inside(a['xy'], rect)]
    q = b['xy'][_inside(b['xy'], rect)]
    if len(p) < 4 or len(q) < 4:
        return None
    if min(_arc(p), _arc(q)) < cfg.alias_overlap_length_px:
        return None
    dp = cKDTree(q).query(p)[0]
    dq = cKDTree(p).query(q)[0]
    fraction = min(float(np.mean(dp <= cfg.alias_radius_px)),
                   float(np.mean(dq <= cfg.alias_radius_px)))
    median = max(float(np.median(dp)), float(np.median(dq)))
    p90 = max(float(np.quantile(dp,.9)), float(np.quantile(dq,.9)))
    score = fraction - .025 * median - .012 * p90
    return {'fraction': round(fraction, 5), 'median_px': round(median, 4),
            'p90_px': round(p90, 4), 'score': round(score, 5),
            'overlap_arc_min_px': round(min(_arc(p), _arc(q)), 2),
            'geometry_supported': fraction >= cfg.alias_fraction_min and
                median <= cfg.alias_median_max_px and p90 <= cfg.alias_p90_max_px}


def _segments_close(a, b, distance=3):
    # Existing accepted inferred seam links cannot cross except at a shared endpoint.
    def cross(p,q,r,s):
        d1 = q-p; d2 = s-r
        den = d1[0]*d2[1]-d1[1]*d2[0]
        if abs(den) < 1e-8:
            return False
        t = ((r-p)[0]*d2[1]-(r-p)[1]*d2[0])/den
        u = ((r-p)[0]*d1[1]-(r-p)[1]*d1[0])/den
        return .015 < t < .985 and .015 < u < .985
    return cross(np.asarray(a[0]),np.asarray(a[1]),np.asarray(b[0]),np.asarray(b[1]))


def stitch_tiles(tiles: list[dict], cfg: StitchConfig = StitchConfig(),
                 decisions: Mapping[str,str] | None = None):
    """Cross-tile provisional identity; all coordinates in FULL BMP XY frame.

    Alias = same photo-supported trace rediscovered in overlapping tiles.
    Seam connection = inferred and reversible, only tiny clean discontinuities
    may auto-join; larger gaps remain review-only without an image-scored bridge.
    """
    records, local_group, rectangles, image_id, image_sha = _validate(tiles, cfg)
    uf = _Union(records, local_group)
    by_local_group = defaultdict(list)
    for key, group in local_group.items():
        by_local_group[group].append(key)
    for members in by_local_group.values():
        for other in members[1:]:
            assert uf.union(members[0], other, allow_local=True) in ('joined', 'already')
    names = sorted(records)
    # Index source curves once; large scans have thousands of fragments and
    # should not rescan all curves for every overlapping tile pair.
    by_tile = defaultdict(list)
    bounds = {}
    for key, record in records.items():
        by_tile[record['tile']].append(key)
        xy = record['xy']
        bounds[key] = (float(xy[:,0].min()), float(xy[:,1].min()),
                       float(xy[:,0].max()), float(xy[:,1].max()))
    proposals = []
    tile_pairs = [(a,b,_rect_overlap(rectangles[a],rectangles[b]))
                  for i,a in enumerate(sorted(rectangles)) for b in sorted(rectangles)[i+1:]]
    for ta, tb, overlap in tile_pairs:
        if overlap[2] <= overlap[0] or overlap[3] <= overlap[1]:
            continue
        radius = cfg.alias_radius_px
        def overlaps(p, rect):
            return (p[2]+radius >= rect[0] and p[0]-radius < rect[2] and
                    p[3]+radius >= rect[1] and p[1]-radius < rect[3])
        a = [k for k in by_tile[ta] if overlaps(bounds[k], overlap)]
        b = [k for k in by_tile[tb] if overlaps(bounds[k], overlap)]
        for ak in a:
            pa = bounds[ak]
            for bk in b:
                pb = bounds[bk]
                if (max(pa[0],pb[0])-radius > min(pa[2],pb[2]) or
                    max(pa[1],pb[1])-radius > min(pa[3],pb[3])):
                    continue
                metrics = _alias_metrics(records[ak], records[bk], overlap, cfg)
                if metrics and metrics['geometry_supported']:
                    proposals.append({'candidate_id':'alias_'+_hash(*sorted((ak,bk))),
                                      'curves':list(sorted((ak,bk))), 'tiles':[ta,tb],
                                      'kind':'SOURCE_OBSERVATION_ALIAS','decision':'UNRESOLVED',**metrics})
    rankings = defaultdict(list)
    for e in proposals:
        for key in e['curves']:
            # Duplicates from different *other* tiles are independently valid,
            # but ambiguity between alternatives *within* an other tile is not.
            other = next(k for k in e['curves'] if k != key)
            rankings[(key,records[other]['tile'])].append(e)
    for v in rankings.values():
        v.sort(key=lambda e: (-e['score'],e['candidate_id']))
    for e in sorted(proposals,key=lambda e:(-e['score'],e['candidate_id'])):
        a, b = e['curves']
        by = [rankings[(a,records[b]['tile'])],rankings[(b,records[a]['tile'])]]
        unique = all(r[0]['candidate_id'] == e['candidate_id'] and
                     (len(r) == 1 or r[0]['score']-r[1]['score'] >= cfg.alias_margin) for r in by)
        if not unique:
            e['decision'] = 'REVIEW_ALIAS_COMPETING_OBSERVATIONS'
            continue
        outcome = uf.union(a, b)
        e['decision'] = ('AUTO_ALIAS_SAME_OBSERVED_SOURCE' if outcome=='joined' else
                         'REDUNDANT_ALIAS_SAME_GROUP' if outcome=='already' else
                         'REVIEW_ALIAS_WOULD_MERGE_DISTINCT_LOCAL_GROUPS')
    # Cross-boundary continuation candidates, not image detections.
    ports = []
    for key in names:
        obj = records[key]
        for end in (0, 1):
            point, vec = _endpoint(obj['xy'],end)
            if vec is not None and _boundary_exit(point,vec,obj['rect'],cfg.seam_band_px):
                ports.append({'key':key,'end':end,'point':point,'vec':vec,'tile':obj['tile']})
    seams = []
    port_by_id = {f"{p['key']}:{p['end']}":p for p in ports}
    port_tree = cKDTree(np.array([p['point'] for p in ports])) if ports else None
    if len(ports) > 1:
        pairs = sorted(cKDTree(np.array([x['point'] for x in ports])).query_pairs(cfg.review_gap_px))
        for ai, bi in pairs:
            a,b=ports[ai],ports[bi]
            if a['tile']==b['tile'] or uf.root(a['key'])==uf.root(b['key']):
                continue
            delta=b['point']-a['point']
            gap=float(np.linalg.norm(delta))
            if gap<=.5:
                continue
            axis=delta/gap
            align=min(float(a['vec']@axis),float(b['vec']@(-axis)))
            if align<.88:
                continue
            lateral=max(abs(a['vec'][0]*delta[1]-a['vec'][1]*delta[0]),
                        abs(b['vec'][0]*delta[1]-b['vec'][1]*delta[0]))
            if lateral>max(15.,gap*.2):
                continue
            ports_ids = [f"{a['key']}:{a['end']}", f"{b['key']}:{b['end']}"]
            score = .8*align+.2*max(0,1-gap/cfg.review_gap_px)
            seams.append({'candidate_id':'seam_'+_hash(*sorted(ports_ids)),
                          'curve_ids':[a['key'],b['key']],
                          'port_ids':ports_ids,'gap_px':round(gap,3),
                          'alignment_min':round(align,5),
                          'lateral_px':round(float(lateral),3),
                          'score':round(float(score),5),
                          'kind':'INFERRED_CROSS_TILE_GAP',
                          'connector_is_observed':False,
                          'observed_connector_pixels':0,
                          'inferred_connector_full_xy':[np.round(a['point'],2).tolist(),np.round(b['point'],2).tolist()],
                          'decision':'REVIEW_ONLY_NO_PHOTO_FOR_GAP'})
    requested=dict(decisions or {})
    if set(requested.values())-{'same','different','uncertain'}:
        raise ValueError('Invalid expert decision; use same/different/uncertain')
    if set(requested)-{e['candidate_id'] for e in seams}:
        raise ValueError('Expert decision references missing seam candidate')
    byport=defaultdict(list)
    for e in seams:
        if requested.get(e['candidate_id'])!='different':
            for p in e['port_ids']:
                byport[p].append(e)
    for v in byport.values():
        v.sort(key=lambda e:(-e['score'],e['candidate_id']))
    selected=[];used=set()
    forced=sorted((e for e in seams if requested.get(e['candidate_id'])=='same'),key=lambda e:e['candidate_id'])
    ordinary=sorted((e for e in seams if e not in forced),key=lambda e:(-e['score'],e['candidate_id']))
    for e in forced+ordinary:
        choice=requested.get(e['candidate_id'])
        if choice=='different':
            e['decision']='EXPERT_REJECTED_DIFFERENT';continue
        if choice=='uncertain':
            e['decision']='EXPERT_UNCERTAIN';continue
        a,b=e['curve_ids']
        if uf.root(a)==uf.root(b):
            e['decision']='REVIEW_CYCLE_OR_ALIAS_ALREADY_JOINED';continue
        if any(p in used for p in e['port_ids']):
            if choice=='same':raise ValueError('Expert seam links reuse endpoint')
            e['decision']='REVIEW_PORT_ALREADY_USED';continue
        ranking=[byport[p] for p in e['port_ids']]
        best=all(r and r[0]['candidate_id']==e['candidate_id'] for r in ranking)
        margin=min((r[0]['score']-r[1]['score'] if len(r)>1 else 1.) for r in ranking)
        # Long gaps cannot auto-join without photo evidence. Do not do costly
        # O(number_of_all_ports) checks for these review-only candidates.
        auto_shape=(e['gap_px']<=cfg.auto_seam_gap_px and
                    e['alignment_min']>=cfg.auto_seam_alignment and
                    e['lateral_px']<=cfg.auto_seam_lateral_px and
                    best and margin>=cfg.auto_seam_margin)
        if not auto_shape and choice!='same':
            e['decision']='REVIEW_ONLY_NO_PHOTO_FOR_GAP'
            continue
        # Rival endpoints that failed angle screening still matter. Check
        # neighbours from a spatial index, not a scan of the full frame.
        nearby_rival=False
        for port in e['port_ids']:
            target=port_by_id[port]
            for index in port_tree.query_ball_point(target['point'],r=cfg.auto_nearby_rival_px):
                alternative=ports[index]
                if alternative['tile']!=target['tile'] or alternative['key']==target['key']:
                    continue
                if local_group[alternative['key']]==local_group[target['key']]:
                    continue
                nearby_rival=True;break
            if nearby_rival:break
        if nearby_rival and choice!='same':
            e['decision']='REVIEW_ONLY_NEARBY_RIVAL_AT_SEAM'
            continue
        if any(_segments_close(e['inferred_connector_full_xy'],other['inferred_connector_full_xy']) for other in selected):
            if choice=='same':raise ValueError('Expert seam links geometrically cross')
            e['decision']='REVIEW_CROSSING_SEAM';continue
        result=uf.union(a,b)
        if result=='conflicting_local_groups':
            if choice=='same':raise ValueError('Expert seam would merge incompatible local groups')
            e['decision']='REVIEW_CONFLICTING_LOCAL_IDENTITIES';continue
        if result!='joined':
            e['decision']='REVIEW_REDUNDANT';continue
        used.update(e['port_ids']);selected.append(e)
        e['decision']='EXPERT_SAME_INFERRED_GAP' if choice=='same' else 'AUTO_TINY_SEAM_GAP_NOT_GT'
    members=defaultdict(list)
    for k in names:members[uf.root(k)].append(k)
    grouped=[]
    for component in members.values():
        component=sorted(component)
        grouped.append({'global_group_id':'s118_group_'+_hash(*component),
                        'source_curve_keys':component,
                        'source_tile_ids':sorted({records[k]['tile'] for k in component}),
                        'confirmed_particle_id':None,
                        'observations_unchanged':True,
                        'identity_status':'PROVISIONAL_SOURCE_GEOMETRY_NOT_PHYSICAL_GT'})
    grouped.sort(key=lambda x:(-len(x['source_curve_keys']),x['global_group_id']))
    return {'source_image_id':image_id,'source_image_sha256':image_sha,
            'tiles':[{'tile_id':t,'xyxy':rectangles[t]} for t in sorted(rectangles)],
            'groups':grouped,'overlap_aliases':sorted(proposals,key=lambda e:e['candidate_id']),
            'seam_candidates':sorted(seams,key=lambda e:e['candidate_id']),
            'report':{'tile_count':len(rectangles),'input_curve_instances':len(records),
                      'local_s117_groups':len(by_local_group),'global_group_count':len(grouped),
                      'confirmed_overlap_aliases':sum(e['decision']=='AUTO_ALIAS_SAME_OBSERVED_SOURCE' for e in proposals),
                      'automatic_tiny_seam_joins':sum(e['decision']=='AUTO_TINY_SEAM_GAP_NOT_GT' for e in seams),
                      'expert_seam_joins':sum(e['decision']=='EXPERT_SAME_INFERRED_GAP' for e in seams),
                      'review_only_aliases':sum(e['decision'].startswith('REVIEW') for e in proposals),
                      'review_only_gaps':sum(e['decision'].startswith('REVIEW') for e in seams),
                      'no_fabricated_observations':True,
                      'no_auto_long_gap_without_photo':True,
                      'is_physical_ground_truth':False}}
