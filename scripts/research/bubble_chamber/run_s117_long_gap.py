"""S117 source-only long-gap grouping, offline audit and preview.

Usage: python scripts/research/bubble_chamber/run_s117_long_gap.py \
  --curves /path/to/S115_GEOMETRIC_CURVES_NOT_GT.json \
  --photo /path/to/SOURCE_GRAY.png --out /tmp/s117

S109 vector_tracks.json and S115 JSON are accepted. This tool never writes to
its inputs; accepted joins have IDs but are NOT asserted as particle ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from bubble_chamber.s117_gap_identity import associate_long_gaps


def _load_curves(obj):
    if isinstance(obj, list):
        return obj
    for key in ('curves', 'tracks', 'vector_tracks'):
        if isinstance(obj.get(key), list):
            return obj[key]
    raise ValueError('Expected curves/tracks/vector_tracks list')


def run(curves_json: Path, photo: Path, out: Path, response: Path | None = None,
        blocked: Path | None = None, decisions: Path | None = None):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite nonempty output directory')
    payload = json.loads(curves_json.read_text(encoding='utf8'))
    curves = _load_curves(payload)
    gray = cv2.imread(str(photo), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(photo)
    respond = None
    if response is not None:
        # A PNG may be a thresholded preview, NOT S114 floating-point response;
        # require .npy to keep scientific scale and avoid hidden unit changes.
        if response.suffix.lower() != '.npy':
            raise ValueError('Photometric response must be an original float32 .npy')
        respond = np.load(response, allow_pickle=False)
    protect = cv2.imread(str(blocked), cv2.IMREAD_GRAYSCALE) if blocked else None
    if blocked and protect is None:
        raise FileNotFoundError(blocked)
    expert = json.loads(decisions.read_text(encoding='utf8')) if decisions else None
    result = associate_long_gaps(curves, gray, response=respond, blocked=protect, decisions=expert)
    provenance = {'source_curves_sha256': hashlib.sha256(curves_json.read_bytes()).hexdigest(),
                  'source_photo_sha256': hashlib.sha256(photo.read_bytes()).hexdigest(),
                  'manual_GT_used': False, 'source_revision': 'S117_GEOMETRIC_HYPOTHESES'}
    result['provenance'] = provenance
    out.mkdir(parents=True, exist_ok=True)
    (out/'S117_LINKS_GROUPS_NOT_GT.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    grouped = {cid: group['group_id'] for group in result['groups'] for cid in group['source_curve_ids']}
    group_palette = {g['group_id']: i for i, g in enumerate(result['groups'])}
    for i, c in enumerate(curves):
        pts = np.asarray(c.get('observed_pixel_path_xy', c.get('sampled_xy')), dtype=float)
        pts = np.rint(pts).astype(np.int32)
        cid = c.get('candidate_id') or c.get('curve_id') or c.get('tracklet_id')
        palette_index = group_palette[grouped[cid]]
        color = tuple(int(q) for q in cv2.cvtColor(np.uint8([[[(palette_index*53+25)%180,200,245]]]),cv2.COLOR_HSV2BGR)[0,0])
        cv2.polylines(image, [pts], False, color, 1, cv2.LINE_AA)
    for e in result['links']:
        if e['decision'] not in ('AUTO_PROVISIONAL_GEOMETRY_GROUP_NOT_GT','EXPERT_CONFIRMED_SAME_LINK'):
            continue
        points = np.rint(np.asarray(e['inferred_connector_xy'])).astype(np.int32)
        for i in range(0, len(points)-1, 10):
            cv2.line(image, tuple(points[i]),tuple(points[min(i+5,len(points)-1)]),(0,150,255),2,cv2.LINE_AA)
    cv2.putText(image,'S117: same colour = provisional group; orange dashes = inferred gap',
                (12,30),cv2.FONT_HERSHEY_SIMPLEX,.58,(20,20,20),2,cv2.LINE_AA)
    cv2.imwrite(str(out/'S117_OBSERVED_VS_INFERRED_REVIEW.png'),image)
    print(json.dumps(result['report'],ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--curves',required=True,type=Path)
    p.add_argument('--photo',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--response',type=Path)
    p.add_argument('--blocked',type=Path)
    p.add_argument('--decisions',type=Path)
    a=p.parse_args()
    run(a.curves,a.photo,a.out,a.response,a.blocked,a.decisions)
