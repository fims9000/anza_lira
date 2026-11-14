"""Tver V25 CPU 2D/2.5D CNN baselines on exactly matched hard synthetic worlds.

These are fair *task-matched modest CNNs*, not pretrained FaultSeg3D,
published CNN benchmark weights, or trained historical ANZA-LIRA.
Official V25 hard synthetic TEST cannot be opened until CNN 12-epoch TRAIN
and CAL acceptance rules are frozen. Real source/candidate coordinates are
supplied oracle geometry in marker maps, never inferred by this network.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

IMAGE_SIZE=64
N_EPOCHS=12
BATCH=48
RATE=.001
WEIGHT_DECAY=.01
TORCH_SEEDS=(11,23)


class Tiny2p5DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(6,16,5,padding=2),
            nn.ReLU(),nn.AvgPool2d(2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),nn.AvgPool2d(2),
            nn.Conv2d(32,48,3,padding=1),
            nn.ReLU(),nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(48*8*8,96),nn.ReLU(),
            nn.Linear(96,3),
        )

    def forward(self,x):
        return self.layers(x)


def train_image_statistics(cases):
    """Observed 3-layer image normalization, strictly TRAIN cohort only."""
    signal=np.stack([
        np.stack((c["adjacent"][0],c["central"],c["adjacent"][1]))
        for c in cases])
    mu=float(signal.mean())
    sd=float(max(signal.std(),.1))
    if not np.isfinite([mu,sd]).all():
        raise ValueError("Bad TRAIN image mean/std")
    return mu,sd


def encode_observed_worlds(cases,*,mode,mean,std):
    if mode not in ("central_2d","neighbors_25d"):
        raise ValueError("Bad V25 fixed CNN information scope")
    xx=np.arange(IMAGE_SIZE,dtype=float)[None,:]
    yy=np.arange(IMAGE_SIZE,dtype=float)[:,None]
    encoded=[]
    for c in cases:
        if mode=="central_2d":
            image=np.repeat(c["central"][None,...],3,axis=0)
        else:
            image=np.stack((
                c["adjacent"][0],c["central"],c["adjacent"][1]))
        image=(image-mean)/std
        points=[c["source"],*c["candidates"]]
        cue=np.stack([
            np.exp(-.5*((xx-float(p[0]))**2+
                         (yy-float(p[1]))**2)/2.**2)
            for p in points])
        encoded.append(np.concatenate((image,cue),axis=0).astype(np.float32))
    return np.stack(encoded)


def fit_cnn_v25(cases, *, mode,seed,mean,std,output_dir):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(202625000+int(seed))
    images=encode_observed_worlds(
        cases,mode=mode,mean=mean,std=std)
    labels=np.asarray([c["label"] for c in cases],np.int64)
    if len(images)!=600 or len(set(labels))!=3:
        raise ValueError("V25 fixed TRAIN 200×3 complete labeled worlds required")
    model=Tiny2p5DCNN()
    opt=torch.optim.AdamW(
        model.parameters(),lr=RATE,weight_decay=WEIGHT_DECAY)
    xt=torch.as_tensor(images,dtype=torch.float32)
    yt=torch.as_tensor(labels,dtype=torch.int64)
    generator=torch.Generator()
    generator.manual_seed(202625100+int(seed))
    history=[]
    model.train()
    for _ in range(N_EPOCHS):
        permutation=torch.randperm(len(labels),generator=generator)
        total=0.
        for block in permutation.split(BATCH):
            yhat=model(xt[block])
            loss=F.cross_entropy(yhat,yt[block])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total+=float(loss.detach())*len(block)
        history.append(total/len(labels))
    model.eval()
    model=model.cpu()
    path=Path(output_dir)/f"v25_{mode}_seed{seed}.pt"
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),path)
    passport={
        "method":"observed_only_six_channels_three_image_or_center_and_endpoints",
        "mode":mode,"seed":int(seed),"epochs":N_EPOCHS,
        "batch_size":BATCH,"optimizer":"AdamW",
        "lr":RATE,"weight_decay":WEIGHT_DECAY,
        "n_train_worlds":len(labels),"n_independent_train_base_groups":200,
        "train_image_mean":float(mean),"train_image_std":float(std),
        "train_objective_first":float(history[0]),
        "train_objective_last":float(history[-1]),
        "checkpoint_SHA256":sha256(path.read_bytes()).hexdigest(),
        "checkpoint_path":path.name,
        "original_historical_ANZA_checkpoint":False,
        "uses_latent_counterfactual_world_id_as_input":False,
        "uses_TEST_or_CAL_labels_for_weight_fit":False,
    }
    return model,passport


@torch.no_grad()
def score_cnn_v25(model,cases,*,mode,mean,std):
    if not cases:return np.zeros((0,2))
    images=encode_observed_worlds(
        cases,mode=mode,mean=mean,std=std)
    x=torch.as_tensor(images,dtype=torch.float32)
    model.eval()
    prob=[]
    for block in x.split(64):
        prob.append(F.softmax(model(block),dim=1).cpu().numpy())
    prob=np.concatenate(prob,axis=0)
    if prob.shape!=(len(cases),3):
        raise AssertionError("Wrong native CNN 3class order")
    return np.column_stack([prob[:,0]-prob[:,2],
                            prob[:,1]-prob[:,2]])
