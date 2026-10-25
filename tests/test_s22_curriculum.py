import numpy as np
from scripts.research.bubble_chamber.s22_gap_curriculum import REGISTERED,fit_classifier

def test_registered_split_does_not_reuse_s21_seeds():
    assert REGISTERED['split_seeds']==[20261027,20261028,20261029]
    assert not set(REGISTERED['split_seeds']) & {20261024,20261025,20261026}
    assert REGISTERED['additional_calibration_reviews']==160
    assert REGISTERED['test_gaps']==[3,5]
    assert REGISTERED['no_test_selection'] is True

def test_train_only_supervised_review_indices():
    rng=np.random.default_rng(44)
    x=rng.normal(size=(100,14))
    rows=[{'truth_same_track':i%4==0} for i in range(100)]
    indices=list(range(36))
    model=fit_classifier(rows,x,indices,seed=56)
    a=model.predict_proba(x)[:,1]
    flipped=[dict(row) for row in rows]
    for i in range(36,100):flipped[i]['truth_same_track']=not flipped[i]['truth_same_track']
    model2=fit_classifier(flipped,x,indices,seed=56)
    assert np.allclose(a,model2.predict_proba(x)[:,1])
