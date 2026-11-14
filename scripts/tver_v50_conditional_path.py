"""V50 source-conditioned CNN and ribbon-DP decoder, no oracle inference input.

The Thebe experiment is posthoc on an already-inspected CAL set. Inference
uses only gray seismic, intervened ANZA probability, intervention rectangle
and predicted SBPP source/target; expert centerlines enter TRAIN targets only.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
import torch.nn.functional as F

PATCH=48
CHANNELS=7
WIDTH=16
EPOCHS=25
BATCH=16
SEED=20261114
LR=.001
WEIGHT_DECAY=.0001
VALID_MODES=("learned_ribbon","untrained_ribbon")


class ResidualBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__()
        self.first=nn.Conv2d(WIDTH,WIDTH,3,padding=dilation,dilation=dilation)
        self.second=nn.Conv2d(WIDTH,WIDTH,3,padding=1)
        self.norm=nn.GroupNorm(4,WIDTH)

    def forward(self,x):
        return F.gelu(self.norm(self.second(F.gelu(self.first(x))))+x)


class ConditionedPathNet(nn.Module):
    """Small residual-dilated pixel path decoder with predicted-port markers."""
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(CHANNELS,WIDTH,3,padding=1),nn.GELU())
        self.layers=nn.Sequential(*[ResidualBlock(d) for d in (1,2,4,1)])
        self.head=nn.Conv2d(WIDTH,1,1)

    def forward(self,x):
        if x.ndim!=4 or x.shape[1:]!=(CHANNELS,PATCH,PATCH):
            raise ValueError("Expected Bx7x48x48 observational gap inputs")
        return self.head(self.layers(self.stem(x)))


def _subpatch(rect):
    y0,y1,x0,x1=map(int,rect)
    if not (0<=y0<y1<=96 and 0<=x0<x1<=96):
        raise ValueError("Invalid intervention rectangle")
    y=int(np.clip(round((y0+y1)/2-PATCH/2),0,96-PATCH))
    x=int(np.clip(round((x0+x1)/2-PATCH/2),0,96-PATCH))
    if not (y<=y0 and y1<=y+PATCH and x<=x0 and x1<=x+PATCH):
        raise ValueError("Gap wider than 48px model context")
    return y,x


def _prob_observations(scene):
    gray=np.asarray(scene["gray"],np.float32)
    visible=np.asarray(scene["visible_prob"],np.float32)
    if (gray.shape!=(96,96) or visible.shape!=(96,96)
        or not np.isfinite(gray).all() or not np.isfinite(visible).all()
        or np.any((visible<0)|(visible>1))):
        raise ValueError("Bad observed 96x96 seismic or ANZA probability")
    return gray,visible


def _pair_endpoints(proposal):
    a=np.asarray(proposal["source"],np.float32)
    b=np.asarray(proposal["target"],np.float32)
    if (a.shape!=(2,) or b.shape!=(2,)
        or not np.isfinite(a).all() or not np.isfinite(b).all()
        or np.linalg.norm(a-b)<=1):
        raise ValueError("Bad predicted pair endpoints")
    return a,b


def observed_features(scene,proposal):
    """Seven observation-only channels. Never access hidden label fields."""
    gray,p=_prob_observations(scene)
    a,b=_pair_endpoints(proposal)
    oy,ox=_subpatch(scene["rectangle"])
    yy,xx=np.mgrid[oy:oy+PATCH,ox:ox+PATCH].astype(np.float32)
    gap=np.zeros((PATCH,PATCH),np.float32)
    y0,y1,x0,x1=scene["rectangle"]
    gap[y0-oy:y1-oy,x0-ox:x1-ox]=1
    pts=np.stack([yy,xx],axis=-1)
    source=np.exp(-np.sum((pts-a)**2,axis=-1)/(2*3**2))
    dest=np.exp(-np.sum((pts-b)**2,axis=-1)/(2*3**2))
    delta=b-a
    t=np.clip(np.sum((pts-a)*delta,axis=-1)/max(float(delta@delta),1.),0,1)
    chord_foot=a+t[...,None]*delta
    chord=np.exp(-np.sum((pts-chord_foot)**2,axis=-1)/(2*3**2))
    observed=gray[oy:oy+PATCH,ox:ox+PATCH]
    image=ndi.gaussian_filter(gray,1.)
    gy=ndi.sobel(image,axis=0);gx=ndi.sobel(image,axis=1)
    gradient=np.sqrt(gx**2+gy**2)[oy:oy+PATCH,ox:ox+PATCH]
    gradient/=max(float(np.quantile(gradient,.95)),1e-6)
    x=np.asarray([
        np.clip(observed,0,1),p[oy:oy+PATCH,ox:ox+PATCH],gap,
        source,dest,chord,np.clip(gradient,0,1)
    ],dtype=np.float32)
    if x.shape!=(CHANNELS,PATCH,PATCH) or not np.isfinite(x).all():
        raise RuntimeError("Bad neural pathway input")
    return x,(oy,ox)


def target_inside_rectangle(scene,patch_origin):
    """TRAIN target only, never called from any inference/decoder function."""
    centerline=np.asarray(scene["hidden_centerline"],int)
    origin=np.asarray(scene["origin"],int)
    if centerline.ndim!=2 or centerline.shape[1]!=2 or origin.shape!=(2,):
        raise ValueError("Bad TRAIN expert centerline")
    oy,ox=patch_origin
    local=centerline-origin[None,:]
    mask=np.zeros((PATCH,PATCH),np.float32)
    y0,y1,x0,x1=scene["rectangle"]
    for y,x in local:
        if y0<=y<y1 and x0<=x<x1 and oy<=y<oy+PATCH and ox<=x<ox+PATCH:
            mask[y-oy,x-ox]=1
    if not mask.any():
        raise RuntimeError("TRAIN target outside intervention rectangle")
    inside=np.zeros_like(mask)
    inside[y0-oy:y1-oy,x0-ox:x1-ox]=1
    return ndi.binary_dilation(mask.astype(bool),iterations=1).astype(np.float32)*inside


def build_train_dataset(scenes):
    """One true and one hard false predicted pair per eligible TRAIN LINK gap."""
    images,targets,source_groups,kinds=[],[],[],[]
    for scene in scenes:
        if scene["kind"]!="LINK":continue
        positive=sorted((p for p in scene["proposals"] if int(p["y"])==1),
                        key=lambda p:-float(p["geometry"]))
        if not positive:continue
        negative=sorted((p for p in scene["proposals"] if int(p["y"])==0),
                        key=lambda p:-float(p["geometry"]))
        sampled=[(True,positive[0])]+([(False,negative[0])] if negative else [])
        for is_positive,proposal in sampled:
            x,patch_origin=observed_features(scene,proposal)
            y=(target_inside_rectangle(scene,patch_origin) if is_positive
               else np.zeros((PATCH,PATCH),np.float32))
            images.append(x);targets.append(y)
            source_groups.append(int(scene["section"]))
            kinds.append("positive_hidden_path" if is_positive else "hard_wrong_pair")
    if len(images)<30 or kinds.count("positive_hidden_path")<30:
        raise RuntimeError("STOP fewer than 30 real TRAIN path-positive samples")
    return np.stack(images),np.asarray(targets,np.float32),{
        "total_train_examples":len(images),
        "positive_path_examples":kinds.count("positive_hidden_path"),
        "same_scene_hard_negative_examples":kinds.count("hard_wrong_pair"),
        "source_sections":len(set(source_groups)),
        "TEST7_never_opened":True,
        "expert_hidden_line_ONLY_TRAIN_target":True,
    }


def fit_path_head(train_scenes,out_dir):
    """Train fixed 25 epochs, no CAL tuning or checkpoint selection."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    x,y,counts=build_train_dataset(train_scenes)
    tx=torch.from_numpy(x)
    ty=torch.from_numpy(y)[:,None]
    net=ConditionedPathNet()
    optimizer=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    rng=torch.Generator().manual_seed(SEED)
    losses=[]
    net.train()
    for epoch in range(EPOCHS):
        perm=torch.randperm(len(tx),generator=rng)
        total=0.
        for ids in perm.split(BATCH):
            inp=tx[ids];target=ty[ids]
            if epoch%2:
                inp=inp.flip(-1);target=target.flip(-1)
            if epoch%4>=2:
                inp=inp.flip(-2);target=target.flip(-2)
            logits=net(inp)
            valid=inp[:,2:3]
            weight=1+5*target
            bce=(F.binary_cross_entropy_with_logits(logits,target,reduction="none")
                 *valid*weight).sum()/(valid*weight).sum().clamp_min(1)
            probability=torch.sigmoid(logits)*valid
            positive=target.flatten(1).sum(dim=1)>0
            if positive.any():
                pp=probability[positive];yy=target[positive]
                dice=1-(2*(pp*yy).flatten(1).sum(dim=1)+1)/(
                    pp.flatten(1).sum(dim=1)+yy.flatten(1).sum(dim=1)+1)
                loss=bce+.3*dice.mean()
            else:
                loss=bce
            if not torch.isfinite(loss):
                raise RuntimeError("STOP neural path training loss diverged")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),3.)
            optimizer.step()
            total+=float(loss.detach())*len(ids)
        losses.append(total/len(tx))
        if (epoch+1)%5==0:
            print("V50_TRAIN_PATH_EPOCH",epoch+1,"loss",round(losses[-1],6),flush=True)
    net.eval()
    root=Path(out_dir);root.mkdir(parents=True,exist_ok=True)
    checkpoint=root/"V50_TRAIN_ONLY_CONDITIONED_PATH_HEAD.pt"
    torch.save({
        "state_dict":net.state_dict(),"model":"ConditionedPathNet",
        "data_counts":counts,"epochs":EPOCHS,"seed":SEED,
        "input_channels":CHANNELS,"patch":PATCH
    },checkpoint)
    receipt={**counts,"epochs":EPOCHS,"seed":SEED,"batch":BATCH,
        "loss_first_epoch":losses[0],"loss_final_epoch":losses[-1],
        "training_final_epoch_not_CAL_selected":True,
        "checkpoint_filename":checkpoint.name,
        "checkpoint_SHA256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model_parameters":sum(p.numel() for p in net.parameters()),
        "architecture":"7-channel source-conditioned residual CNN + ribbon DP"}
    (root/"V50_TRAIN_PATH_RECEIPT.json").write_text(
        json.dumps(receipt,indent=2)+"\n",encoding="utf8")
    return net,receipt


def path_heatmap(net,scene,proposal):
    """Inference: score pixels using observed image, not expert truth."""
    x,(oy,ox)=observed_features(scene,proposal)
    net.eval()
    with torch.no_grad():
        pred=torch.sigmoid(net(torch.from_numpy(x[None]))[0,0]).cpu().numpy()
    score=np.full((96,96),1e-4,np.float32)
    score[oy:oy+PATCH,ox:ox+PATCH]=np.clip(pred,1e-4,1-1e-4)
    return score


def ribbon_dp_route(heat,start,end,*,radius=9):
    """Bounded Viterbi along predicted endpoints with learned spatial support."""
    from scripts.tver_v49_boundary_path import dense_polyline
    image=np.asarray(heat,np.float64)
    a=np.asarray(start,float);b=np.asarray(end,float)
    if image.shape!=(96,96) or not np.isfinite(image).all():
        raise ValueError("Invalid learned heatmap")
    if a.shape!=(2,) or b.shape!=(2,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid anchor positions")
    direction=b-a
    distance=float(np.linalg.norm(direction))
    if distance<=2:
        return dense_polyline([a,b],image.shape)
    unit=direction/distance
    normal=np.array([-unit[1],unit[0]])
    count=max(16,min(96,int(math.ceil(distance*2))))
    offsets=np.arange(-radius,radius+1,dtype=float)
    base=np.linspace(a,b,count)
    points=base[:,None,:]+offsets[None,:,None]*normal
    reward=ndi.map_coordinates(image,[points[...,0],points[...,1]],
                               order=1,mode="constant",cval=1e-4)
    reward=np.log(np.clip(reward,1e-5,1))-.025*abs(offsets)[None,:]
    n=len(offsets);center=radius
    dp=np.full(n,-np.inf)
    dp[center]=reward[0,center]
    pointers=np.zeros((count,n),dtype=np.int16)
    for step in range(1,count):
        future=np.full(n,-np.inf)
        for state in range(n):
            low,high=max(0,state-2),min(n,state+3)
            prev=np.arange(low,high)
            score=dp[prev]-.20*np.abs(prev-state)
            j=int(np.argmax(score))
            future[state]=score[j]+reward[step,state]
            pointers[step,state]=prev[j]
        dp=future
    states=[center]
    for step in range(count-1,0,-1):
        states.append(int(pointers[step,states[-1]]))
    states.reverse()
    route=[a]+[p for p in points[np.arange(count),states]]+[b]
    return dense_polyline(route,image.shape)


def decode_candidate(net,scene,proposal,*,mode="learned_ribbon"):
    """Model-based full path. Does not inspect section, label or hidden line."""
    if mode not in VALID_MODES:
        raise ValueError("Unknown V50 path mode")
    from scripts.tver_v49_boundary_path import boundary_anchors
    a,b=_pair_endpoints(proposal)
    anchors=boundary_anchors(scene["visible_prob"],scene["rectangle"],a,b)
    if anchors is None:
        source,target=a,b
    else:
        source,target,_=anchors
    if mode=="learned_ribbon":
        heat=path_heatmap(net,scene,proposal)
    else:
        yy,xx=np.mgrid[:96,:96]
        p=np.stack([yy,xx],axis=-1)
        delta=b-a
        t=np.clip(np.sum((p-a)*delta,axis=-1)/max(float(delta@delta),1.),0,1)
        foot=a+t[...,None]*delta
        heat=np.exp(-np.sum((p-foot)**2,axis=-1)/(2*3**2))
    route=ribbon_dp_route(heat,source,target)
    status="observed_boundary_anchors" if anchors is not None else "predicted_endpoints_fallback"
    return route,{
        "mode":mode,"anchor_status":status,"n_path_pixels":len(route),
        "visible_only":True,"no_hidden_endpoints_used":True
    }
