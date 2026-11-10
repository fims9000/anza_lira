#!/usr/bin/env python3
"""Guarded S108 research pilot: train from verified sparse strokes or preview."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bubble_chamber.s107_review import verify_pack
from bubble_chamber.s108_sparse_unet import (
    S108UNet, research_pilot_train, predict_score_preview,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    train = sub.add_parser('train')
    train.add_argument('--pack-dir', type=Path, required=True)
    train.add_argument('--review-json', type=Path, required=True)
    train.add_argument('--out-dir', type=Path, required=True)
    train.add_argument('--steps', type=int, default=50)
    train.add_argument('--patch', type=int, default=192)
    train.add_argument('--batch', type=int, default=4)
    train.add_argument('--one-run-pilot-only', action='store_true',
                       help='Explicitly acknowledge H50Z 005/006 are the SAME run, not independent TEST')
    predict = sub.add_parser('predict')
    predict.add_argument('--pack-dir', type=Path, required=True)
    predict.add_argument('--model-dir', type=Path, required=True)
    predict.add_argument('--tile-id', required=True)
    predict.add_argument('--out-dir', type=Path, required=True)
    predict.add_argument('--patch', type=int, default=256)
    a = p.parse_args(argv)
    if a.mode == 'train':
        report = research_pilot_train(a.pack_dir, a.review_json, a.out_dir,
                  steps=a.steps, patch=a.patch, batch_size=a.batch,
                  one_run_pilot_only=a.one_run_pilot_only)
        print(json.dumps({k:v for k,v in report.items() if k!='training_loss_only'},indent=2))
        return

    manifest = verify_pack(a.pack_dir)
    meta = json.loads((a.model_dir/'report.json').read_text(encoding='utf-8'))
    if meta['source_manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Unverified training source manifest')
    blob = (a.model_dir/'weights.pt').read_bytes()
    if sha256(blob).hexdigest() != meta['weights_sha256']:
        raise ValueError('Model checksum mismatch')
    check = torch.load(a.model_dir/'weights.pt', map_location='cpu', weights_only=True)
    model = S108UNet(base=int(check['base']))
    model.load_state_dict(check['state_dict'], strict=True)
    tile = next((t for t in manifest['tiles'] if t['tile_id']==a.tile_id),None)
    if tile is None:
        raise ValueError('Unknown source-bound tile ID')
    gray = cv2.imread(str(a.pack_dir/tile['original_png']),cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError('Missing original 8-bit source pixels')
    logits = predict_score_preview(model, gray, patch=a.patch)
    if a.out_dir.exists():
        raise FileExistsError(a.out_dir)
    a.out_dir.mkdir(parents=True)
    np.save(a.out_dir/'uncalibrated_track_likeness_logits.npy',logits.astype(np.float16))
    lo,hi = np.percentile(logits,[2,98])
    preview = np.clip((logits-lo)/max(hi-lo,1e-4)*255,0,255).astype(np.uint8)
    if not cv2.imwrite(str(a.out_dir/'uncalibrated_rank_visual_ONLY.png'),preview):
        raise OSError('Failed to write visual preview')
    report = {'schema':'S108_UNCALIBRATED_PREDICTION_PREVIEW_V1',
              'tile_id':tile['tile_id'],'source_sha256':tile['source_sha256'],
              'model_sha256':meta['weights_sha256'],
              'model_trained_on_same_physical_run':True,
              'native_particle_segmentation_metric_available':False,
              'threshold_or_probability_calibration_done':False,
              'visual_percentile_range':[float(lo),float(hi)]}
    (a.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
