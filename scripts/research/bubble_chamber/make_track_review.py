#!/usr/bin/env python3
"""Build offline human-review HTML of Frangi thin-line proposals on H50Z tiles."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import cv2
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from bubble_chamber.track_mask_review import propose_tile_segments, review_html,convert_review_to_sparse_mask

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--tile-manifest',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--reviews',type=Path,help='Optional exported JSON from review HTML')
 a=p.parse_args()
 original=json.loads(a.tile_manifest.read_text())
 root=a.tile_manifest.parent
 tiles=[x for x in original['labelled_tiles'] if x['label']=='start']
 ims=[];props=[]
 for t in tiles:
  im=cv2.imread(str(root/t['tile_filename']),cv2.IMREAD_GRAYSCALE)
  if im is None:raise ValueError('unreadable source tile')
  ims.append(im);props.append(propose_tile_segments(im,t))
 a.out.mkdir(parents=True,exist_ok=True)
 for i,payload in enumerate(props):
  (a.out/f'proposal_{i:02}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 (a.out/'track_mask_review.html').write_text(review_html(props,ims),encoding='utf-8')
 summary={'tiles':len(props),'proposed_fragments':sum(len(x['candidates']) for x in props),
          'human_confirmed_pixels':0,'statuses':{'unreviewed':'UNKNOWN'},
          'roundtrip_schema':'bubble_human_track_centerline_review_v1'}
 if a.reviews:
  reviews=json.loads(a.reviews.read_text())
  if not isinstance(reviews,list) or len(reviews)!=len(props):raise ValueError('review set mismatch')
  stats=[]
  for i,(proposed,review) in enumerate(zip(props,reviews)):
   mask,state=convert_review_to_sparse_mask(proposed,review)
   cv2.imwrite(str(a.out/f'sparse_mask_{i:02}.png'),mask)
   stats.append(state)
  summary['review_stats']=stats
 (a.out/'review_manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
