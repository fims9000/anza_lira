import numpy as np,cv2,pytest
from bubble_chamber.s28_anchor_tracker import candidate_paths,evidence_from_photo,_dp_path


def test_synthetic_curved_line_anchor_tracing_and_no_true_track_claim():
    image=np.zeros((192,192),dtype='float32')
    pts=np.array([[x,round(60+.12*(x-10)+.0006*(x-10)**2)] for x in range(10,178)],dtype='int32')
    cv2.polylines(image,[pts.reshape(-1,1,2)],False,1.,2)
    x=candidate_paths(image,(10.,60.),length=140,width=20,step=2,angles=24,top_k=5)
    assert x and len(x)<=5
    assert all(q['human_confirmed_track'] is False for q in x)
    assert all(q['automatic_physical_join_allowed'] is False for q in x)
    # At least one candidate must follow the visibly anchored synthetic line.
    dist=[]
    for row in x:
        ar=np.asarray(row['path_points_original_xy'])
        # Closest-segment distance to true line.
        d=np.linalg.norm(ar[:,None,:]-pts[None,:,:],axis=-1).min(axis=1)
        dist.append((d<4).mean())
    assert max(dist)>.5


def test_dark_photo_evidence_unchanged_origin():
    g=np.full((256,256),170,'uint8')
    cv2.line(g,(12,120),(245,122),50,2)
    e=evidence_from_photo(g)
    assert e.shape==g.shape and e.max()<=1 and e.min()>=0
    az=np.ones(g.shape,'float32')*.1
    ee=evidence_from_photo(g,az)
    assert not np.array_equal(e,ee)
    with pytest.raises(ValueError):evidence_from_photo(g,np.zeros((14,14)))


def test_bad_tracker_input_fail_closed():
    with pytest.raises(ValueError):candidate_paths(np.zeros((128,128)) ,(300,22))
    with pytest.raises(ValueError):candidate_paths(np.full((160,160),np.nan),(50,50))
    with pytest.raises(ValueError):_dp_path(np.zeros((2,2)))
