"""V30: actual repository five-convolution P0 architecture on real predicted pairs.

All input evidence comes from observed seismic RGB and prediction-derived
SBPP source/candidate coordinates. Label/hidden expert endpoints are read
ONLY for TRAIN supervision and CAL/TEST scoring. Prior historical weights
are not loaded. Different from V27/V28 12-feature logistic relation head.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from path_completion.pair_classifier import (
    EndpointPairClassifier,oriented_pair_crop,
)

SEEDS=(11,23)
EPOCHS=30
BATCH=32
LR=.001
WEIGHT_DECAY=.0001


def observed_pair_crop(scene,proposal,threshold):
    """Do not access hidden centerlines, labeled ends or proposal['y']."""
    rgb=np.asarray(scene["original_rgb"],np.float32)
    visible=np.asarray(scene["visible_prob"],np.float32)
    if rgb.shape!=(96,96,3) or visible.shape!=(96,96):
        raise ValueError("P0 requires genuine 96×96 observed RGB and predicted segmentation")
    a=np.asarray(proposal["source"],float)
    b=np.asarray(proposal["target"],float)
    d=float(np.linalg.norm(a-b))
    if not np.isfinite([a[0],a[1],b[0],b[1],d]).all() or d<=0:
        raise ValueError("Invalid prediction-derived endpoints")
    sample={
        "image":np.transpose(rgb,(2,0,1)),
        "visible_fault_mask":visible>=float(threshold),
    }
    pair=SimpleNamespace(first=tuple(a),second=tuple(b),distance=d)
    output=oriented_pair_crop(sample,pair)
    if output.shape!=(6,33,49) or not np.isfinite(output).all():
        raise AssertionError("Repository P0 context contract failed")
    return output


def paired_training_examples(train,threshold):
    """One negative for each positive; same scene preferred, closest geometry.

    The negative is chosen by actual TRAIN y, which is allowed for FIT.
    Group source scene identity is retained for provenance, never a feature.
    Every positive is included. No expert ends enter observed crop features.
    """
    positives=[];negatives=[]
    allneg=[]
    for scene_ix,scene in enumerate(train):
        if scene["kind"]!="LINK":continue
        for pos_ix,row in enumerate(scene["proposals"]):
            if int(row["y"])!=1:continue
            positives.append((scene_ix,pos_ix,scene,row))
        for neg_ix,row in enumerate(scene["proposals"]):
            if int(row["y"])!=0:
                continue
            allneg.append((scene_ix,neg_ix,scene,row))
    if len(positives)<8 or len(allneg)<8:
        raise RuntimeError("STOP V30 insufficient prediction-derived TRAIN pairs")
    selected=[];seen=set()
    for si,pi,scene,p in positives:
        distance=float(p["features"][0])
        ranked=sorted(
            (n for n in allneg if (n[0],n[1]) not in seen),
            key=lambda n:(
                n[0]!=si,
                abs(float(n[3]["features"][0])-distance),
                abs(float(n[3]["geometry"])-float(p["geometry"])),
                n[0],n[1]))
        if not ranked:
            raise RuntimeError("STOP V30 unable to match one unique negative per positive")
        n=ranked[0]
        seen.add((n[0],n[1]))
        selected.append((scene,p,n[2],n[3]))
    X=[];y=[];F=[];groups=[]
    for anchor,p,negative,n in selected:
        for scene,row,label in ((anchor,p,1),(negative,n,0)):
            tensor=observed_pair_crop(scene,row,threshold)
            scalar=np.asarray(row["features"],np.float64)
            if scalar.shape!=(12,) or not np.isfinite(scalar).all():
                raise AssertionError("V30 candidate numeric features mismatch")
            X.append(tensor);y.append(label);F.append(scalar)
            groups.append(int(scene["section"]))
    images=np.asarray(X,np.float32)
    labels=np.asarray(y,np.int64)
    scalar=np.asarray(F,np.float64)
    if len(images)!=2*len(positives) or labels.sum()!=len(positives):
        raise AssertionError("V30 exact 1:1 TRAIN matching broken")
    passport={
        "positive_unique_predicted_train_pairs":len(positives),
        "matched_negative_unique_predicted_train_pairs":len(positives),
        "n_train_examples":len(labels),
        "train_source_sections":sorted(set(groups)),
        "same_scene_negative_preferred":True,
        "source_candidate_endpoints_from_SBPP_only":True,
        "no_expert_endpoints_in_CNN_input":True,
        "expert_labels_only_for_TRAIN_pairs":True,
    }
    return images,labels,scalar,passport


def fit_native_p0(images,labels,*,seed,root,frontend):
    """Real P0 5-layer convolution network, train-only BCE, two initializations."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    model=EndpointPairClassifier()
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    xt=torch.as_tensor(images,dtype=torch.float32)
    yt=torch.as_tensor(labels,dtype=torch.float32)
    rng=torch.Generator().manual_seed(20263000+int(seed))
    history=[]
    for epoch in range(EPOCHS):
        indices=torch.randperm(len(labels),generator=rng)
        epoch_loss=0.
        for ix in indices.split(BATCH):
            data=xt[ix]
            truth=yt[ix]
            # Deterministic horizontal/vertical flips of the whole observed
            # context, applied equally to source mask and marker channels.
            if epoch%2==1:data=torch.flip(data,(-1,))
            if epoch%4>=2:data=torch.flip(data,(-2,))
            pred=model(data)
            loss=F.binary_cross_entropy_with_logits(pred,truth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
            epoch_loss+=float(loss.item())*len(ix)
        history.append(epoch_loss/len(labels))
    model.eval()
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    checkpoint=root/f"{frontend}_new_p0_seed{seed}.pt"
    torch.save(model.state_dict(),checkpoint)
    receipt={
        "architecture":"path_completion.pair_classifier.EndpointPairClassifier",
        "genuine_repository_five_convolution_P0":True,
        "original_historical_P0_checkpoint_used":False,
        "seed":int(seed),"n_epochs":EPOCHS,"batch_size":BATCH,
        "train_examples":len(labels),"optimizer":"AdamW",
        "learning_rate":LR,"weight_decay":WEIGHT_DECAY,
        "train_BCE_epoch1":history[0],"train_BCE_epoch30":history[-1],
        "model_parameters":sum(p.numel() for p in model.parameters()),
        "checkpoint_SHA256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint":checkpoint.name,
    }
    return model,receipt


@torch.no_grad()
def context_scores(model,scene,threshold):
    proposals=scene["proposals"]
    if not proposals:return np.empty((0,),float)
    crops=np.stack([observed_pair_crop(scene,p,threshold)
                    for p in proposals])
    model.eval()
    tensor=torch.from_numpy(crops)
    outputs=[]
    for b in tensor.split(32):
        outputs.append(torch.sigmoid(model(b)).cpu().numpy())
    values=np.concatenate(outputs).astype(float)
    if values.shape!=(len(proposals),) or not np.isfinite(values).all():
        raise AssertionError("P0 context scorer unavailable")
    return values


def fit_matched_logistic(features,labels):
    model=make_pipeline(StandardScaler(),LogisticRegression(
        max_iter=1400,C=.3,class_weight="balanced",random_state=20263000))
    model.fit(features,labels)
    return model


def score_logistic(model,scene):
    if not scene["proposals"]:return np.empty((0,),float)
    x=np.asarray([p["features"] for p in scene["proposals"]],np.float64)
    return np.asarray(model.predict_proba(x)[:,1],float)
