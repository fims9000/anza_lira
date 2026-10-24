from pathlib import Path
import importlib.util
import sys
import numpy as np

PATH=Path(__file__).resolve().parents[1]/'scripts/research/bubble_chamber/evaluate_s6.py'
spec=importlib.util.spec_from_file_location('evaluate_s6',PATH)
module=importlib.util.module_from_spec(spec)
sys.modules['evaluate_s6']=module
spec.loader.exec_module(module)

def test_counts_bootstrap_and_identity():
    a={f'f{i}':np.array([10,2,3]) for i in range(8)}
    out=module.paired_bootstrap(a,a,repeats=100)
    assert out['delta_dice']==0
    assert out['ci95']==[0.,0.]

def test_stress_preserves_dimensions_deterministically():
    x=np.full((3,32,32),220,dtype=np.uint8)
    x[:,10:22,12:17]=10
    y=module.stress_images(x)
    assert np.array_equal(y,module.stress_images(x))
    assert y.shape==x.shape and not np.array_equal(x,y)

def test_centerline_tolerance_identity():
    x=np.zeros((1,32,32),bool)
    x[:,10:26,11]=True
    assert abs(module.line_f1(x,x)['f1']-1)<1e-9
