import numpy as np
import pytest
from bubble_chamber.synthetic_from_clicks import SyntheticFilmConfig, render_measured_track, frame_render_dataset


def _track(fid='070_014'):
    return {'frame_id':fid,'track_id':'e1t01','n_points':12,
            'measured_pixel_xy':{str(k):[[1000+j*5,2000+j*2+(j-6)**2] for j in range(12)] for k in (1,2,3)}}


def test_render_has_reproducible_masks_and_dynamic_ranges():
    a,y,m=render_measured_track(_track(),1,3)
    b,z,n=render_measured_track(_track(),1,3)
    assert np.array_equal(a,b) and np.array_equal(y,z)
    assert a.shape==y.shape==(128,128)
    assert a.dtype==y.dtype==np.uint8
    assert 0<y.sum()<y.size*.2
    assert set(np.unique(y)).issubset({0,1})
    assert m['source'].startswith('synthetic pixels')
    assert m==n

@pytest.mark.parametrize('bad', [SyntheticFilmConfig(size=50), SyntheticFilmConfig(min_contrast=120,max_contrast=10)])
def test_invalid_config_fails(bad):
    with pytest.raises(ValueError):
        render_measured_track(_track(),1,0,bad)


def test_not_real_label_semantics():
    a,y,meta=render_measured_track(_track(),2,1)
    assert meta['source_frame']=='070_014'
    assert 'synthetic' in meta['source']
    assert y.dtype==np.uint8


def test_frame_filter_guards_split_and_track_deterministic():
    m={'audit_schema':'visualmetro_large_audit_v1','frames':[{'frame_id':'070_014','tracks':[_track()]},{'frame_id':'070_015','tracks':[_track('070_015')]}]}
    src=list(frame_render_dataset(m,['070_014'],repeats=3))
    assert len(src)==3 and {x[2]['source_frame'] for x in src}=={'070_014'}
