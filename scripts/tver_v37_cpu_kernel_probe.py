"""V37: minimal raw tensor CPU fingerprint, no Thebe data, no TEST7.

Scientifically diagnostic only: compare original AZConv-v1 / U-Net operations
on a deterministic fixed tensor across independent GitHub runner machines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.segmentation_v2 import build_comparable_model
from scripts.tver_v33_thebe_train_segmenters import PARAM_WIDTHS


def arr_sha(x):
    a=x.detach().cpu().contiguous().numpy() if isinstance(x,torch.Tensor) else np.ascontiguousarray(x)
    h=hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(tuple(a.shape)).encode())
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def state_sha(model,kind="parameter"):
    h=hashlib.sha256()
    iterator=(model.named_parameters() if kind=="parameter" else
              ((k,p.grad) for k,p in model.named_parameters()))
    for k,t in sorted(iterator):
        h.update(k.encode())
        if t is None:
            h.update(b"NONE")
        else:
            h.update(arr_sha(t).encode())
    return h.hexdigest()


def cpu_note():
    path=Path("/proc/cpuinfo")
    if not path.exists():return {"cpuinfo":"unavailable"}
    s=path.read_text(errors="replace")
    fields={}
    for key in ("model name","vendor_id","cpu family","model","stepping"):
        match=re.search(rf"^{re.escape(key)}\s*:\s*(.*)$",s,re.M)
        if match:fields[key]=match.group(1).strip()
    match=re.search(r"^flags\s*:\s*(.*)$",s,re.M)
    if match:
        flags=set(match.group(1).split())
        fields["dispatch_flags"]=[k for k in ("sse4_2","avx","avx2","avx512f","avx512bw","fma") if k in flags]
    return fields


def execute(out):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)
    case={
        "status":"V37_FIXED_SYNTHETIC_CPU_FINGERPRINT_COMPLETE",
        "no_external_seismic_data_downloaded":True,
        "no_TEST7_read":True,
        "cpu":cpu_note(),
        "python":platform.python_version(),
        "torch":torch.__version__,
        "numpy":np.__version__,
        "threads":torch.get_num_threads(),
        "interop":torch.get_num_interop_threads(),
        "mkldnn":torch.backends.mkldnn.enabled,
        "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        "models":{},
    }
    # Fixed bytes same for both runners; input only synthetic visible sample.
    yy,xx=np.indices((64,64),dtype=np.float32)
    example=np.stack([
        np.sin((xx+2*yy)/17.),
        np.cos((2*xx-yy)/23.),
        np.sin((xx-yy)/11.)],axis=0).astype(np.float32)
    truth=np.asarray(np.hypot(xx-32,yy-32)<17,np.float32)[None,None,:,:]
    x=torch.from_numpy(example[None])
    y=torch.from_numpy(truth)
    case["input_sha256"]=arr_sha(x)
    case["mask_sha256"]=arr_sha(y)
    for model_name,seed in (("unet",20263312),("anza_v1",20263311)):
        torch.manual_seed(seed)
        model=build_comparable_model(model_name,widths=PARAM_WIDTHS)
        model.train()
        initial=state_sha(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
        logits=model(x)
        loss=F.binary_cross_entropy_with_logits(logits,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients=state_sha(model,"gradient")
        forward=arr_sha(logits)
        loss_value=float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        optimizer.step()
        updated=state_sha(model)
        case["models"][model_name]={
            "seed":seed,
            "initial_parameter_tensor_SHA256":initial,
            "first_forward_SHA256":forward,
            "first_gradient_SHA256":gradients,
            "after_one_AdamW_step_SHA256":updated,
            "first_loss":loss_value,
            "n_parameters":sum(p.numel() for p in model.parameters()),
        }
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(case,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(case,indent=2,ensure_ascii=False))
    return case


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    execute(a.out)
