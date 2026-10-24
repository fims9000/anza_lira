"""ANZA-assisted U-Net: additive orientation-sensitive image preprocessor.

Does not replace, monkeypatch or edit original AZConv2d. Every model has a
matched baseline U-Net backbone; convolution-only preprocessor controls extra
capacity. The learned scale is bounded and the az module uses authentic repo.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.segmentation import BaselineUNet
from models.azconv import AZConv2d,AZConvConfig


class GuidedUNet(nn.Module):
    def __init__(self, *, variant:str='az', widths=(4,6,8,12), num_rules=2,
                 gate_init=.20):
        super().__init__()
        if variant not in ('az','conv'):
            raise ValueError('variant must be az or conv')
        if not 0<gate_init<1:raise ValueError('gate must be in (0,1)')
        self.variant=variant
        # The backbone is constructed BEFORE the extra encoder. With equal
        # initialization seed its weights match the independently created U-Net.
        self.backbone=BaselineUNet(in_channels=1,out_channels=1,widths=widths)
        if variant=='az':
            self.guide=AZConv2d(1,2,kernel_size=3,num_rules=num_rules,cfg=AZConvConfig(
                geometry_mode='local_hyperbolic',use_fuzzy=True,use_anisotropy=True,
                learn_directions=True,use_input_residual=True,residual_init=0.5))
        else:
            self.guide=nn.Conv2d(1,2,kernel_size=3,padding=1)
        self.projection=nn.Conv2d(2,1,kernel_size=1,bias=False)
        nn.init.normal_(self.projection.weight,mean=0,std=.01)
        self.gate_logit=nn.Parameter(torch.tensor(math.log(gate_init/(1-gate_init)),dtype=torch.float32))

    def forward(self,x):
        guide=self.projection(F.relu(self.guide(x)))
        # Preserve a direct photo path and avoid forcing uncertain geometry
        # evidence into every pixel; trainable gate can downweight this path.
        return self.backbone(x+torch.sigmoid(self.gate_logit)*guide)

    def learned_guide_gate(self)->float:
        return float(torch.sigmoid(self.gate_logit).detach())


def instantiate_matched(seed:int,widths=(4,6,8,12),rules=2):
    """Three capacity-controlled alternatives, all ORIGINAL repo kernels."""
    from models.segmentation import AZUNet
    torch.manual_seed(seed)
    plain=BaselineUNet(in_channels=1,out_channels=1,widths=widths)
    torch.manual_seed(seed)
    conv=GuidedUNet(variant='conv',widths=widths,num_rules=rules)
    torch.manual_seed(seed)
    az=GuidedUNet(variant='az',widths=widths,num_rules=rules)
    for p,q in zip(plain.parameters(),conv.backbone.parameters()):
        if not torch.equal(p,q):raise RuntimeError('U-Net backbones are not matched at initialization')
    for p,q in zip(plain.parameters(),az.backbone.parameters()):
        if not torch.equal(p,q):raise RuntimeError('U-Net backbones are not matched at initialization')
    return {'unet_base':plain,'conv_guided':conv,'az_guided':az}
