from pathlib import Path
import json
import numpy as np
import cv2
from scripts.research.bubble_chamber.s25_real_background_anza import examples
from scripts.research.bubble_chamber.s25_real_anchor_anza import dataset


def test_real_source_split_and_synthetic_unknown_mask(tmp_path):
    # This fixture contains no real particle labels. Generated marks are kept
    # disjoint from every unknown real photo pixel.
    source=Path('/mnt/data/s25_work/private/1mH2Data/S1')
    tiles=Path('/mnt/data/s25_work/private')
    if not source.exists() or not list(tiles.glob('*.png')):return
    groups,receipt=examples(source,tiles,seed=33,per_photo=2)
    assert set(groups)=={'H50Z_030_01_005','H50Z_030_01_006'}
    assert receipt['real_human_track_pixels']==0
    assert receipt['independent_physical_run_test'] is False
    for name,items in groups.items():
        for image,mark,synthetic,base,provenance in items:
            assert image.shape==(64,64)==mark.shape==synthetic.shape
            assert set(np.unique(mark)).issubset({0,1,255})
            assert np.any(mark==255) and np.any(mark==1) and np.any(mark==0)
            assert np.all(synthetic[mark==1]==1)
            assert np.all(synthetic[mark==0]==0)
            assert provenance['photo']==name


def test_real_labels_are_anchor_categories_not_track_masks():
    base=Path('/mnt/data/s25_private')
    if not (base/'labels.json').is_file():return
    a,info=dataset(base,base/'labels.json','005')
    b,other=dataset(base,base/'labels.json','006')
    assert info['human_start_points']==6 and info['human_fiducials']==7
    assert other['human_start_points']==3 and other['human_fiducials']==8
    assert len(a)==13 and len(b)==11
    assert all(x.shape==(64,64) for x,_,_ in a+b)
