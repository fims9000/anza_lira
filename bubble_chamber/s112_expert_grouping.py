"""Explicit human-reviewed merging of S111 group IDs via S112 bead bridges.

Automatic S112 never changes particle IDs. The groups below are a reversible
record of human decisions, separate from source-observed photography.
"""
from __future__ import annotations
from collections import defaultdict
from hashlib import sha256
from lxml import etree


def merge_reviewed_groups(s111_tracks, bridge_proposals, decisions=None):
    decisions=decisions or {}
    gids={t['global_track_id'] for t in s111_tracks}
    parent={x:x for x in gids}
    byid={b['bridge_id']:b for b in bridge_proposals}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    accepted=[];notfound=[]
    for identifier,status in sorted(decisions.items()):
        if status!='same' or identifier not in byid:continue
        b=byid[identifier]
        a,z=b['source_groups']
        if a not in gids or z not in gids:
            notfound.append(identifier)
            continue
        ra,rz=root(a),root(z)
        if ra!=rz:parent[rz]=ra
        accepted.append(identifier)
    bygroup=defaultdict(list)
    for t in s111_tracks:bygroup[root(t['global_track_id'])].append(t['global_track_id'])
    group_objects=[]
    for entries in sorted(bygroup.values(),key=lambda x:tuple(sorted(x))):
        ids=sorted(entries)
        key='review_'+sha256('|'.join(ids).encode()).hexdigest()[:12]
        included=[name for name in accepted if set(byid[name]['source_groups']).issubset(ids)]
        group_objects.append({'reviewed_track_id':key,'original_s111_global_group_ids':ids,
           'explicit_reviewed_s112_bridge_ids':included,
           'source_s109_curve_coordinates_untouched':True,
           'this_is_not_independently_verified_physical_particle_GT':True})
    return {'schema':'S112_EXPLICIT_HUMAN_BEAD_GROUP_MERGE_V1',
            'original_S111_group_count':len(gids),
            'groups_after_explicit_review':len(group_objects),
            'human_approved_bridge_count':len(accepted),
            'approved_bridge_ids':accepted,'unmatched_review_ids':notfound,
            'groups':group_objects,'any_automatic_merged_bead_bridges':False,
            'no_source_curves_ever_changed':True}


def reviewed_groups_svg(original_svg, bridges, decisions):
    """S111 SVG plus *visibly different* human-approved bridging paths."""
    root=etree.fromstring(original_svg.encode('utf-8'))
    ns='http://www.w3.org/2000/svg'
    layers=etree.SubElement(root,f'{{{ns}}}g',id='S112_ONLY_EXPLICIT_HUMAN_REVIEWED_BRIDGES')
    for bridge in bridges:
        if decisions.get(bridge['bridge_id'])!='same':continue
        spline=bridge.get('candidate_inferred_spline') or {}
        path=spline.get('svg_path')
        if not path:continue
        tag=etree.SubElement(layers,f'{{{ns}}}path',d=path,fill='none',stroke='#00c5e8',
                  **{'stroke-width':'3','stroke-dasharray':'9 6'})
        tag.set('data-bead-bridge-id',bridge['bridge_id'])
        tag.set('data-provenance','EXPERT_REVIEWED_INFERRED_NOT_PHOTOGRAPHED')
    return etree.tostring(root,encoding='unicode',xml_declaration=False)


def validate_review_provenance(review, source_sha, s111_sha):
    """Fail closed on stale or foreign-source human association decisions."""
    if review.get('schema')!='S112_EXPERT_BEAD_REVIEW_V1':
        raise ValueError('Unknown expert-review schema')
    if not source_sha or review.get('source_sha256')!=source_sha:
        raise ValueError('Expert-review source SHA256 missing or mismatched')
    if not s111_sha or review.get('s111_global_tracks_sha256')!=s111_sha:
        raise ValueError('S111 global matching revision changed; re-review association')
    return True