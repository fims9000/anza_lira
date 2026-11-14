"""V35 tensor and full real patch digest guards."""
from __future__ import annotations

import numpy as np
import torch

from scripts.tver_v35_thebe_determinism import (
    tensor_state_hash,dataset_patch_hash,
)


def test_model_state_hash_is_stable_to_mapping_order_not_tensor_data():
    a={"b":torch.arange(4).reshape(2,2),"a":torch.ones(1)}
    b={"a":torch.ones(1),"b":torch.arange(4).reshape(2,2)}
    assert tensor_state_hash(a)==tensor_state_hash(b)
    c={"a":torch.ones(1),"b":torch.arange(4).reshape(2,2)+1}
    assert tensor_state_hash(a)!=tensor_state_hash(c)


def test_patch_hash_is_content_and_order_dependent():
    x=np.ones((4,4,3),np.float32)
    y=np.zeros((4,4),np.float32)
    patches=[(x,y)]
    first=dataset_patch_hash(patches)
    assert first==dataset_patch_hash([(x.copy(),y.copy())])
    assert first!=dataset_patch_hash([(x,y),(x,y)])
    assert first!=dataset_patch_hash([(x,y+1.)])
