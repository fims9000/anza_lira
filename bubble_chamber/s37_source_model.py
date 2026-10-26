"""Exact-source AZUNet loader for isolated offline S37 replay.

This small shim avoids importing the S14 training pipeline. It does not change
AZConv2d, AZUNet, checkpoint contents, normalization, or any model weights.
"""
from __future__ import annotations
from hashlib import sha1
from pathlib import Path
from models.azconv import AZConvConfig
from models.segmentation import AZUNet,BaselineUNet
ROOT=Path(__file__).resolve().parents[1]
EXPECTED={'models/azconv.py':'f4217ada00024927506fa8ebe520aa1f68ae5854',
          'models/segmentation.py':'0b971c117f3d56cc588ebade6991c65faceb8a2a'}
def verify_model_source(root=ROOT):
    for rel,correct in EXPECTED.items():
        data=(Path(root)/rel).read_bytes()
        actual=sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual!=correct:raise ValueError('Original AZ source mismatch: '+rel)
    return EXPECTED

def build_model(name='azunet',widths=(4,6,8,12),rules=2):
    verify_model_source()
    if name=='azunet':
        cfg=AZConvConfig(geometry_mode='local_hyperbolic',use_fuzzy=True,use_anisotropy=True)
        return AZUNet(in_channels=1,out_channels=1,num_rules=rules,widths=tuple(widths),cfg=cfg)
    if name=='unet':
        return BaselineUNet(in_channels=1,out_channels=1,widths=tuple(widths))
    raise ValueError(name)
