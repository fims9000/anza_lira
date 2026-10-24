from pathlib import Path
import importlib.util,sys
import numpy as np
from bubble_chamber.s14_sparse_teacher import SparseTeacherProtocol

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/research/bubble_chamber/s14_make_synthetic_sparse_demo.py'
spec=importlib.util.spec_from_file_location('s14_demo',SCRIPT)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def test_demo_is_deterministic_and_manifest_parses(tmp_path):
    from importlib.util import spec_from_file_location,module_from_spec
    p=Path(__file__).resolve().parents[1]/'scripts/research/bubble_chamber/s14_train_sparse_anza.py'
    sp=spec_from_file_location('s14_train_for_demo',p); t=module_from_spec(sp);sys.modules[sp.name]=t;sp.loader.exec_module(t)
    a=m.generate(tmp_path/'one',seed=11)
    b=m.generate(tmp_path/'two',seed=11)
    assert a.read_text()==b.read_text()
    groups,meta=t.read_reviewed_manifest(a)
    assert meta['split_examples']=={'train':6,'val':3,'unlabeled':6}
    for im,mask,info in groups['train']:
        assert np.any(mask==1) and np.any(mask==0) and np.any(mask==255)
