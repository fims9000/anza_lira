"""S121 whole-frame source-only provisional continuations from S118."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2
from bubble_chamber.s121_fullframe_continuation import associate_fullframe


def run(bmp: Path, run_dir: Path, out: Path):
    config = json.loads((run_dir/'RUN_CONFIG.json').read_text())
    gray = cv2.imread(str(bmp), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(bmp)
    digest = hashlib.sha256()
    with bmp.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != config['source_image_sha256'] or list(gray.shape) != config['shape_hw']:
        raise ValueError('Source BMP SHA or dimensions differ from S118')
    n = config['grid_count']
    manifest_path = run_dir/f'S118_MANIFEST_{n:05d}.json'
    manifest = json.loads(manifest_path.read_text())
    result_path = run_dir/f'S118_STITCH_{n:05d}'/'S118_STITCH_NOT_GT.json'
    s118 = json.loads(result_path.read_text())
    if len(manifest['tiles']) != n or len(s118['tiles']) != n:
        raise ValueError('S118 full requested frame coverage is incomplete')
    tiles = []
    local = {}
    for t in manifest['tiles']:
        tid = t['tile_id']
        data = json.loads((run_dir/t['curves']).read_text())
        s117 = json.loads((run_dir/t['s117']).read_text())
        tiles.append({'tile_id': tid, 'xyxy': t['xyxy'], 'curves': data['curves']})
        local[tid] = s117
    result = associate_fullframe(tiles, local, s118['groups'], gray)
    result['provenance'] = {
        'source_sha256': digest.hexdigest(),
        's118_stitch_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
        'manual_annotations_used_in_inference': False, 'is_GT': False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError('Refusing to overwrite S121 output')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result['report']


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bmp', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.bmp, a.run_dir, a.out), indent=2))
