"""Tver V23: paired 3D counterfactuals with EXACTLY identical central 2D pixels.

This is a new, explicitly *synthetic* 2.5D link/no-link study, not a trained
historical ANZA -> SBPP -> LIRA model and not geological ground truth.
Three worlds per base: the source is connected to candidate A, connected to B,
or disconnected. All three share exactly the same middle image, source/candidate
geometry, nuisance texture and candidate order. Neighbor views encode only
3D continuity and change under the three worlds.

The "continuity matcher" is a concrete future LIRA component prototype:
it estimates a left-source strand offset from both adjacent slices, then tests
which right-side continuation has the same offset. All scoring sees ONLY
the rendered observed image and fixed visible geometry, never latent lineage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier

H=W=64
SEED_BASE={"train":202623010,"cal":202623110,
           "test":202623210,"ood":202623310}
N_BASE={"train":200,"cal":100,"test":160,"ood":160}
WORLDS=(0,1,2)
POLICIES=("geometry","central","adjacent_straight",
          "far_sham","continuity_25d",
          "learned_2d","learned_25d")
BOOT=3000
BOOT_SEED=20261114
NO_LINK=2


def _rng(seed):
    return np.random.default_rng(int(seed))


def _hashflip(base):
    return hashlib.sha256(f"v23-order|{base}".encode()).digest()[0]&1


def _background(seed,*,ood=False):
    rng=_rng(seed+90000)
    output=[]
    for layer in range(3):
        smooth=ndi.gaussian_filter(rng.normal(size=(H,W)),sigma=2.7)
        smooth/=max(smooth.std(),1e-6)
        granular=ndi.gaussian_filter(rng.normal(size=(H,W)),sigma=.6)
        granular/=max(granular.std(),1e-6)
        array=(.055*smooth+.055*granular+
               (.10 if ood else .035)*rng.normal(size=(H,W)))
        output.append(array.astype(np.float64))
    return np.asarray(output)


def _add_ridge(image,xx,yline,width,amplitude):
    """Gaussian stripe, independent of hidden 3D track identity."""
    Y=np.arange(H,dtype=float)[:,None]
    image[:,xx]+=amplitude*np.exp(
        -.5*((Y-yline[None,:])/width)**2)


def _render(base_noise, *,A,shift_left,shift_right,width,amplitudes,
            center_y=32.,clutter=0.,seed=0,z=0,ood=False):
    image=base_noise.copy()
    xleft=np.arange(8,32,dtype=int)
    xright=np.arange(32,56,dtype=int)
    xl=(xleft-32.)/22.
    xr=(xright-32.)/22.
    # Four separate observed arms. At z=0 all are identical in the
    # three counterfactual worlds. No reference label enters rendering.
    for arm in range(2):
        sign=1. if arm==0 else -1.
        left=center_y-sign*A*xl+shift_left[arm]
        right=center_y+sign*A*xr+shift_right[arm]
        if ood and z:
            # Frozen OOD longitudinal shape families, both independent
            # of world label. At junction, perturbation is identically 0.
            if seed&1:
                shape_l=np.sin(np.pi*np.abs(xl))*.60
                shape_r=np.sin(np.pi*np.abs(xr))*.60
            else:
                shape_l=np.sin(2*np.pi*np.abs(xl))*.85
                shape_r=np.sin(2*np.pi*np.abs(xr))*.85
            left=left+z*shape_l
            right=right+z*shape_r
        _add_ridge(image,xleft,left,width,amplitudes[arm])
        _add_ridge(image,xright,right,width,amplitudes[arm+2])
    if clutter:
        rng=_rng(seed+70000+int(z)*31)
        for _ in range(7 if ood else 2):
            intercept=float(rng.uniform(2,62))
            slope=float(rng.uniform(-1.3,1.3))
            xs=np.arange(2,62)
            ys=intercept+slope*(xs-32.)
            _add_ridge(image,xs,ys,
                       width*float(rng.uniform(.7,1.9)),
                       clutter*float(rng.uniform(.3,1.)))
    return image


def render_triplet(split,index):
    if split not in SEED_BASE or not 0<=index<N_BASE[split]:
        raise ValueError("Unknown frozen V23 base-group split/index")
    base=SEED_BASE[split]+index
    rng=_rng(base)
    A=float(rng.uniform(8.5,12.))
    d=float(rng.uniform(2.8,4.8))
    width=float(rng.uniform(.65,.95))
    amps=rng.uniform(.62,1.05,size=4)
    ood=(split=="ood")
    if ood:
        amps*=float(rng.uniform(.6,1.4))
    bg=_background(base,ood=ood)
    bg_center=bg[1].copy()
    center=_render(bg_center,A=A,shift_left=(0.,0.),
                   shift_right=(0.,0.),width=width,
                   amplitudes=amps,clutter=.1 if ood else .035,
                   seed=base,z=0,ood=ood)
    # Everything in the central frame, including the nuisance clutter,
    # is reused bit-for-bit for the three opposite worlds.
    order=_hashflip(base)
    result=[]
    for world in WORLDS:
        views=[]
        for z,noise in ((-1,bg[0]),(+1,bg[2])):
            l=(z*d,-z*d)
            if world==0:
                r=(z*d,-z*d)
            elif world==1:
                r=(-z*d,z*d)
            else:
                # A genuinely different 3D source lineage with no physically
                # continuous match to the supplied source; both candidate
                # strands are foreign. Its 2D intersection is nevertheless
                # identical at z=0. Not a natural geological negative.
                ratio=float(rng.uniform(2.1,2.45))
                r=(z*d*ratio,-z*d*ratio)
            registration=float(rng.uniform(-2.4,2.4)) if ood else 0.
            image=_render(
                noise,A=A,
                shift_left=(l[0]+registration,l[1]+registration),
                shift_right=(r[0]+registration,r[1]+registration),
                width=width,amplitudes=amps,
                clutter=.26 if ood else .05,seed=base,z=z,ood=ood)
            views.append(image)
        # New unrelated matched-intensity nuisance images; generate
        # from the SAME base seed, independent of world, so a distant
        # sham cannot reveal a latent world label.
        far=(bg[0]+ndi.gaussian_filter(
            _rng(base+600).normal(size=(H,W)),.9)*.08,
             bg[2]+ndi.gaussian_filter(
            _rng(base+601).normal(size=(H,W)),.9)*.08)
        label=(2 if world==2 else world^order)
        result.append({
            "base":base,"world":world,"label":int(label),"flip":order,
            "A":A,"source":(8.,32.+A),
            "candidates":((56.,32.+A),(56.,32.-A))[::(-1 if order else 1)],
            "central":center.copy(),"adjacent":np.asarray(views),
            "far":np.asarray(far),"family":"two_lobed" if (ood and base&1)
            else ("triangle" if ood else "matched"),
            "is_OOD":ood,
        })
    assert all(np.array_equal(x["central"],result[0]["central"])
               for x in result)
    assert set(x["label"] for x in result)=={0,1,2}
    return result


def _sample(image,x,y):
    return ndi.map_coordinates(image,
        [np.asarray(y,float),np.asarray(x,float)],
        order=1,mode="nearest")


def _arm_y(x,A,arm,side):
    sign=1. if arm==0 else -1.
    if side=="left":return 32.-sign*A*(x-32.)/22.
    return 32.+sign*A*(x-32.)/22.


def _mean_along(image,x,y):
    values=_sample(image,x,y)
    return float(np.mean(values))


def _candidate_image_scores(image,A,flip):
    x=np.linspace(8,56,40)
    start=(8.,32.+A)
    candidates=[(56.,32.+A),(56.,32.-A)]
    if flip:candidates.reverse()
    return np.asarray([
        _mean_along(image,x,start[1]+
                     (c[1]-start[1])*(x-8)/48.)
        for c in candidates
    ])


def _bridge_features(imgs,A,flip,*,adaptive=True):
    # Source=top-left visible original fragment; no ground truth world.
    xleft=np.linspace(12,21,12)
    xright=np.linspace(43,52,12)
    ysource=_arm_y(xleft,A,0,"left")
    candidates=[0,1]
    if flip:candidates.reverse()
    per_slice=[];offsets=[];support=[]
    for image in imgs:
        if adaptive:
            shifts=np.linspace(-9,9,73)
            response=np.asarray([
                _mean_along(image,xleft,ysource+s) for s in shifts])
            chosen=float(shifts[np.argmax(response)])
            source_strength=float(response.max())
        else:
            chosen=0.
            source_strength=_mean_along(image,xleft,ysource)
        observed=[
            _mean_along(image,xright,
                        _arm_y(xright,A,arm,"right")+chosen)
            for arm in candidates
        ]
        per_slice.append(observed)
        offsets.append(chosen)
        support.append(source_strength)
    right=np.asarray(per_slice)
    src=np.asarray(support)
    # Link requires observed support in both neighbors. A high right
    # response without the *source* strand is insufficient.
    scores=np.minimum(right,src[:,None]).mean(axis=0)
    features=[
        scores[0],scores[1],scores[0]-scores[1],
        right[0,0],right[0,1],right[1,0],right[1,1],
        src[0],src[1],offsets[0],offsets[1],
        abs(offsets[0]+offsets[1]),
        float(np.max(scores)),float(np.min(scores)),
    ]
    return scores,np.asarray(features,float)


def visible_features(case):
    """Only images/visible geometry, no world label, index or RNG key."""
    A=case["A"];flip=case["flip"]
    g=np.asarray([-.5,-.5]) # intentionally tied geometry
    central=_candidate_image_scores(case["central"],A,flip)
    neighbor=np.mean([
        _candidate_image_scores(im,A,flip) for im in case["adjacent"]
    ],axis=0)
    far=np.mean([
        _candidate_image_scores(im,A,flip) for im in case["far"]
    ],axis=0)
    continuity,features=_bridge_features(case["adjacent"],A,flip)
    # Central features are exactly identical for matched worlds.
    central_feat=np.asarray([A,float(flip),
        central[0],central[1],central[0]-central[1],
        float(case["central"].mean()),float(case["central"].std())])
    all_feat=np.hstack((
        central_feat,neighbor,neighbor[0]-neighbor[1],
        far,far[0]-far[1],features))
    return {
        "geometry":g,"central":central,
        "adjacent_straight":neighbor,"far_sham":far,
        "continuity_25d":continuity,
        "center_features":central_feat,"all_features":all_feat,
    }


def _stream(split):
    for i in range(N_BASE[split]):
        for w in render_triplet(split,i):
            observed=visible_features(w)
            yield {
                "base":w["base"],"family":w["family"],
                "label":w["label"],"flip":w["flip"],
                "central_sha":hashlib.sha256(
                    w["central"].tobytes()).hexdigest(),
                "scores":{m:observed[m] for m in (
                    "geometry","central","adjacent_straight","far_sham",
                    "continuity_25d")},
                "center_features":observed["center_features"],
                "all_features":observed["all_features"],
            }


def _materialize(split):
    return list(_stream(split))


def _fit_learned(train):
    y=np.asarray([x["label"] for x in train],int)
    outputs={}
    for model_name,key in (("learned_2d","center_features"),
                           ("learned_25d","all_features")):
        xx=np.stack([x[key] for x in train])
        model=HistGradientBoostingClassifier(
            learning_rate=.045,max_iter=130,max_leaf_nodes=11,
            min_samples_leaf=12,l2_regularization=8.,
            random_state=20261114)
        model.fit(xx,y)
        outputs[model_name]=(model,key)
    return outputs


def _policy_scores(row,models):
    result=dict(row["scores"])
    for name,(model,key) in models.items():
        prob=model.predict_proba(row[key][None,:])[0]
        # Evaluators compare two link-class probabilities and a real
        # NO_LINK probability. Preserve source-trained 3-class mapping.
        class_to_p={int(c):float(p)
                    for c,p in zip(model.classes_,prob)}
        result[name]=np.asarray([
            class_to_p.get(0,0.)-class_to_p.get(2,0.),
            class_to_p.get(1,0.)-class_to_p.get(2,0.)])
    return result


def _score_table(rows,models):
    d={name:[] for name in POLICIES}
    for row in rows:
        scores=_policy_scores(row,models)
        for name in POLICIES:
            d[name].append(np.asarray(scores[name],float))
    return {k:np.asarray(v) for k,v in d.items()}


def _calibrate(y,score):
    n=len(y)
    positive=y!=NO_LINK;negative=~positive
    best=score.max(axis=1)
    margin=np.abs(score[:,0]-score[:,1])
    choices=np.argmax(score,axis=1)
    # Never-link is always legal. All other thresholds are CAL-derived.
    thresholds=np.unique(np.quantile(best,np.linspace(0,1,25)))
    margins=np.unique(np.quantile(margin,np.linspace(0,1,25)))
    thresholds=np.r_[np.nextafter(best.max(),np.inf),thresholds]
    margins=np.r_[np.nextafter(margin.max(),np.inf),margins]
    best_decision=(-1,-float("inf"),-float("inf"),float("inf"))
    chosen=(float(thresholds[0]),float(margins[0]))
    for t in thresholds:
        for m in margins:
            accept=(best>=t)&(margin>=m)
            wrong=int(np.sum(accept&positive&(choices!=y)))
            false=int(np.sum(accept&negative))
            if (false/max(negative.sum(),1)>.02+1e-12
                or wrong/max(positive.sum(),1)>.01+1e-12):
                continue
            correct=int(np.sum(accept&positive&(choices==y)))
            order=(correct,-false,-wrong,float(t))
            if order>best_decision:
                best_decision=order
                chosen=(float(t),float(m))
    return {
        "accept_threshold":chosen[0],"margin_threshold":chosen[1],
        "cal_correct_accepted":int(best_decision[0]),
        "cal_false_accepted":int(-best_decision[1])
            if best_decision[0]>=0 else 0,
        "cal_wrong_accepted":int(-best_decision[2])
            if best_decision[0]>=0 else 0,
        "cal_positive_n":int(positive.sum()),
        "cal_NOLINK_n":int(negative.sum()),
        "n_grid_search_pairs":int(len(thresholds)*len(margins)),
    }


def _predict(score,threshold):
    best=score.max(axis=1)
    diff=np.abs(score[:,0]-score[:,1])
    pred=np.argmax(score,axis=1)
    pred[(best<threshold["accept_threshold"])|
         (diff<threshold["margin_threshold"])]=NO_LINK
    return pred


def _metric(rows,pred):
    y=np.asarray([r["label"] for r in rows],int)
    p=y!=NO_LINK;n=~p
    return {
        "total_worlds":len(y),"positive":int(p.sum()),"NOLINK":int(n.sum()),
        "correct_accepted":int(np.sum(p&(pred==y))),
        "wrong_branch_accepted":int(np.sum(p&(pred!=y)&(pred!=NO_LINK))),
        "false_accepted":int(np.sum(n&(pred!=NO_LINK))),
        "positive_correct_accepted_rate":float(np.mean(pred[p]==y[p])),
        "wrong_branch_rate":float(np.mean(
            (pred[p]!=y[p])&(pred[p]!=NO_LINK))),
        "false_bridge_rate":float(np.mean(pred[n]!=NO_LINK)),
        "all_3way_accuracy":float(np.mean(pred==y)),
        "rejection_rate":float(np.mean(pred==NO_LINK)),
        "confusion":[[
            int(np.sum((y==a)&(pred==b))) for b in (0,1,2)
        ] for a in (0,1,2)],
    }


def _paired_base_bootstrap(rows,pred_a,pred_b):
    y=np.asarray([r["label"] for r in rows],int)
    groups=np.asarray([r["base"] for r in rows],int)
    unique=np.unique(groups)
    successes_a=[];successes_b=[]
    for grp in unique:
        loc=(groups==grp)
        assert int(loc.sum())==3
        successes_a.append(float(np.mean(pred_a[loc]==y[loc])))
        successes_b.append(float(np.mean(pred_b[loc]==y[loc])))
    d=np.asarray(successes_a)-np.asarray(successes_b)
    rng=_rng(BOOT_SEED)
    ix=rng.integers(0,len(d),size=(BOOT,len(d)))
    ci=np.quantile(d[ix].mean(axis=1),[.025,.975])
    return {"delta_3way_accuracy":float(d.mean()),
            "base_group_cluster_CI95":list(map(float,ci)),
            "N_independent_base_groups":len(unique),"bootstrap_reps":BOOT}


def _semantic_pair_audit(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r["base"]].append(r)
    for grp,cases in grouped.items():
        if len(cases)!=3 or set(z["label"] for z in cases)!={0,1,2}:
            raise AssertionError("V23 3-world matched group identity broken")
        if len({c["central_sha"] for c in cases})!=1:
            raise AssertionError("Central image differs between counterfactual worlds")
        if any(not np.array_equal(c["center_features"],
                                  cases[0]["center_features"])
               for c in cases):
            raise AssertionError("2D central/geometry group not observationally identical")
    return {"independent_base_groups":len(grouped),
            "all_3_labels_per_base":True,
            "bitwise_identical_central_pixels":True,
            "identical_visible_2D_features":True}


def execute(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # TRAIN and CAL exclusively opened before thresholds/model lock.
    train=_materialize("train")
    cal=_materialize("cal")
    models=_fit_learned(train)
    cal_score=_score_table(cal,models)
    cal_y=np.asarray([r["label"] for r in cal],int)
    thresholds={name:_calibrate(cal_y,cal_score[name])
                for name in POLICIES}
    config={
        "status":"TRAIN_CAL_FROZEN_BEFORE_ANY_TEST_OPENED",
        "protocol":"TV_V23_25D_PAIRED_COUNTERFACTUAL_THREE_WORLDS",
        "split_seeds":SEED_BASE,"split_base_groups":N_BASE,
        "policies":POLICIES,
        "cal_thresholds":thresholds,
        "train_pair_audit":_semantic_pair_audit(train),
        "cal_pair_audit":_semantic_pair_audit(cal),
        "n_train_y_labels":len(train),"n_CAL_y_labels":len(cal),
        "hyperparameters_TEST_tuned":False,
        "actual_ANZA_or_original_LIRA_checkpoint_loaded":False,
        "geometry_oracle_source_and_two_candidates_supplied":True,
        "already_seen_CRACKS_or_V16_used_to_train_new_weights":False,
        "synthetic_not_real_geological_ground_truth":True,
    }
    frozen=out/"v23_train_cal_frozen.json"
    frozen.write_text(json.dumps(config,indent=2)+"\n")
    frozen_sha=hashlib.sha256(frozen.read_bytes()).hexdigest()
    # Open each TEST stream once, with all policies and CAL decisions frozen.
    reports={};case_rows=[]
    for split in ("test","ood"):
        rows=_materialize(split)
        audit=_semantic_pair_audit(rows)
        table=_score_table(rows,models)
        by_policy={}
        predictions={}
        for name in POLICIES:
            pred=_predict(table[name],thresholds[name])
            predictions[name]=pred
            by_policy[name]=_metric(rows,pred)
        contrasts={
            k:_paired_base_bootstrap(
                rows,predictions["continuity_25d"],predictions[k])
            for k in ("geometry","central","adjacent_straight",
                      "far_sham","learned_2d","learned_25d")
        }
        # The center model/geometry have literally identical observations
        # for all 3 worlds, hence exactly 1/3 any deterministic 3-way
        # decision, regardless of rejected/accepted threshold.
        for central_policy in ("geometry","central","learned_2d"):
            if not np.isclose(
                by_policy[central_policy]["all_3way_accuracy"],
                1/3,atol=1e-12,rtol=0):
                raise AssertionError("V23 exact 2D Bayes ambiguity violated")
        reports[split]={
            "status":"EXECUTED_FROZEN_IID" if split=="test"
                     else "EXECUTED_FROZEN_OOD",
            "base_group_pair_audit":audit,
            "policy_results":by_policy,
            "continuity_minus_controls_cluster_bootstrap":contrasts,
            "OOD_admissibility":
                (None if split=="test" else
                 (by_policy["continuity_25d"]["false_bridge_rate"]<=.02 and
                  by_policy["continuity_25d"]["wrong_branch_rate"]<=.01 and
                  by_policy["continuity_25d"]["positive_correct_accepted_rate"]>=.15 and
                  contrasts["learned_25d"]["base_group_cluster_CI95"][0]>0)),
        }
        for i,r in enumerate(rows):
            item={"split":split,"base":r["base"],
                  "family":r["family"],"label":r["label"],
                  "central_sha256":r["central_sha"]}
            for name in POLICIES:
                item[name+"_prediction"]=int(predictions[name][i])
                item[name+"_score0"]=float(table[name][i,0])
                item[name+"_score1"]=float(table[name][i,1])
            case_rows.append(item)
    case_path=out/"v23_test_case_scores.csv"
    with case_path.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(case_rows[0]))
        writer.writeheader();writer.writerows(case_rows)
    output={
        "protocol":"TV_V23_25D_PAIRED_COUNTERFACTUAL_THREE_WORLDS",
        "status":"V23_SYNTHETIC_EXECUTED_AUDITED",
        "train_cal_frozen_SHA256":frozen_sha,
        "train_cal_frozen_before_test_opened":True,
        "report":reports,
        "test_case_scores_SHA256":hashlib.sha256(
            case_path.read_bytes()).hexdigest(),
        "not_trained_historical_ANZA_LIRA":True,
        "new_algorithm_only_prototype_25D_link_verification":True,
        "not_natural_geological_validation":True,
        "not_state_of_the_art_seismic_segmentation_comparison":True,
    }
    (out/"v23_results.json").write_text(
        json.dumps(output,indent=2)+"\n")
    print(json.dumps({
        "status":output["status"],
        "IID":{k:{m:v for m,v in vals.items()
                   if m in ("positive_correct_accepted_rate",
                            "false_bridge_rate","wrong_branch_rate",
                            "all_3way_accuracy")}
                for k,vals in reports["test"]["policy_results"].items()},
        "OOD":{k:{m:v for m,v in vals.items()
                   if m in ("positive_correct_accepted_rate",
                            "false_bridge_rate","wrong_branch_rate",
                            "all_3way_accuracy")}
                for k,vals in reports["ood"]["policy_results"].items()},
        "OOD_strict_pass":reports["ood"]["OOD_admissibility"],
    },indent=2))
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    execute(args.out)


if __name__=="__main__":
    main()
