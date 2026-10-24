#!/usr/bin/env python3
"""Audit S18 checkpoint receipts and run paired seed+frame synthetic comparisons.

Requires `index.json` containing only relative artifact paths. This script
NEVER accesses the private original visualMetro measurements or real H50Z.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from hashlib import sha256
import argparse
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def dice(counts) -> float:
    tp,fp,fn=np.asarray(counts,dtype=np.float64)
    return float(2*tp/max(1,2*tp+fp+fn))


def per_frame_counts(prob,truth,groups,threshold):
    """Compute pixel confusion counts without treating pixels as independent."""
    out=defaultdict(lambda:np.zeros(3,np.int64))
    if len(prob)!=len(truth) or len(prob)!=len(groups):
        raise ValueError('prediction, mask and frame arrays misaligned')
    for p,y,g in zip(prob,truth,groups):
        target=y.astype(bool);pred=p>=threshold
        out[str(g)] += np.array([int((pred&target).sum()),int((pred&~target).sum()),int((~pred&target).sum())])
    return dict(out)


def hierarchical_frame_seed_bootstrap(a:dict[int,dict],b:dict[int,dict],*,seed=18,draws=5000):
    """Bootstrap WHOLE source frames and initialization seeds, preserving pairs."""
    seeds=sorted(a)
    if sorted(b)!=seeds or len(seeds)<2:raise ValueError('needs >=2 paired seeds')
    frames=sorted(a[seeds[0]])
    for s in seeds:
        if sorted(a[s])!=frames or sorted(b[s])!=frames:
            raise ValueError('frame IDs differ across arms/initializations')
    av=np.array([[a[s][f] for f in frames] for s in seeds],dtype=np.int64)
    bv=np.array([[b[s][f] for f in frames] for s in seeds],dtype=np.int64)
    rng=np.random.default_rng(seed)
    seed_draw=rng.integers(0,len(seeds),size=(draws,len(seeds)))
    frame_draw=rng.integers(0,len(frames),size=(draws,len(frames)))
    # Common sampled seed/frame indices for both models => paired comparison.
    ac=av[seed_draw[:,:,None],frame_draw[:,None,:],:].sum((1,2))
    bc=bv[seed_draw[:,:,None],frame_draw[:,None,:],:].sum((1,2))
    da=2*ac[:,0]/np.maximum(1,2*ac[:,0]+ac[:,1]+ac[:,2])
    db=2*bc[:,0]/np.maximum(1,2*bc[:,0]+bc[:,1]+bc[:,2])
    deltas=db-da
    observed=dice(bv.sum((0,1)))-dice(av.sum((0,1)))
    per_seed=[{'seed':s,'delta_dice':dice(bv[i].sum(0))-dice(av[i].sum(0))} for i,s in enumerate(seeds)]
    return {'pooled_delta_dice':float(observed),
            'mean_seed_delta_dice':float(np.mean([x['delta_dice'] for x in per_seed])),
            'paired_ci95_seed_and_frame':[float(x) for x in np.quantile(deltas,[.025,.975])],
            'per_seed_delta':per_seed, 'seeds':seeds,'test_unique_source_frames':len(frames),
            'bootstrap_draws':draws,
            'conditional_limit':'all frames from ONE film 070, no independent camera/run replication'}


def verify_and_load(index: Path):
    base=index.parent.resolve()
    data=json.loads(index.read_text())
    if data.get('schema')!='s18_run_index_v1':raise ValueError('wrong index schema')
    results={};fingerprint=None;reference=None;train_hashes={}
    for e in data['entries']:
        path=(base/e['relative_dir']).resolve()
        if not path.is_relative_to(base):raise ValueError('unsafe index path')
        report=json.loads((path/'metrics.json').read_text())
        p=np.load(path/'heldout_probabilities.npz',allow_pickle=False)
        if len(report['history'])!=data['expected_epochs'] or report['train_examples']!=data['expected_train_samples']:
            raise ValueError('training budget mismatch')
        if (report['seed'],report['architecture'],report['train_arm'])!=(e['seed'],e['architecture'],e['arm']):
            raise ValueError('run identity mismatch')
        if abs(float(report['threshold_selected_on_synthetic_dense_validation'])-float(p['threshold']))>1e-9:
            raise ValueError('threshold mismatch')
        test=(p['test_truth'],p['test_groups']);book=p['book_truth']
        if reference is None:
            reference=(test[0].copy(),test[1].copy(),book.copy())
        else:
            if any(not np.array_equal(a,b) for a,b in zip(reference,(test[0],test[1],book))):
                raise ValueError('heldout test mask/group differs across arms')
        threshold=float(p['threshold'])
        for label,key in [('test_measured','test_prob'),('test_measured_stress','test_stress_prob'),('test_book','book_prob')]:
            y=p['book_truth'] if key=='book_prob' else p['test_truth']
            cc=per_frame_counts(p[key],y,np.arange(len(y)),threshold)
            val=dice(np.array(list(cc.values())).sum(0))
            if abs(val-report['metrics'][label]['dice'])>2e-6:
                raise ValueError(f'evaluation parity lost: {label} {e}')
        hash_model=sha256((path/'best.pt').read_bytes()).hexdigest()
        expected_hash=e.get('checkpoint_sha256')
        if expected_hash and hash_model!=expected_hash:raise ValueError('checkpoint bytes changed')
        name=(e['architecture'],e['arm'],e['seed'])
        if name in results:raise ValueError('duplicate model run')
        results[name]={'record':report,'prob':p,'checkpoint_sha256':hash_model}
    names=sorted(set((k[0],k[1]) for k in results))
    for arch,arm in names:
        seeds=sorted(s for a,b,s in results if a==arch and b==arm)
        if seeds!=data['expected_seeds']:raise ValueError(f'incomplete seed coverage: {arch}/{arm}: {seeds}')
    if len(results)!=data['expected_models']:raise ValueError('missing run')
    return results,reference,data


def analyze(index:Path,out_dir:Path):
    loaded,reference,idx=verify_and_load(index)
    out_dir.mkdir(parents=True,exist_ok=True)
    arms=sorted({a for _m,a,_s in loaded})
    arches=sorted({m for m,_a,_s in loaded})
    mean={}
    for model in arches:
        mean[model]={}
        for arm in arms:
            recs=[loaded[(model,arm,s)]['record'] for s in idx['expected_seeds']]
            keys={'clean_dice':('test_measured','dice'),
                  'stress_dice':('test_measured_stress','dice'),
                  'book_test_dice':('test_book','dice'),
                  'centerline_f1':('test_measured_topology','macro_centerline_f1'),
                  'cldice':('test_measured_topology','macro_cldice'),
                  'stress_centerline_f1':('test_stress_topology','macro_centerline_f1')}
            means={metric:round(float(np.mean([r['metrics'][path][field] for r in recs])),6)
                   for metric,(path,field) in keys.items()}
            means['n_model_seeds']=len(recs)
            means['model_parameters']=recs[0]['model_parameters']
            means['sparse_positive_clicks']=recs[0]['train_positive_sparse_clicks']
            means['sparse_negative_clicks']=recs[0]['train_negative_sparse_clicks']
            means['per_seed']=[{'seed':r['seed'],'clean_dice':r['metrics']['test_measured']['dice'],
                                 'stress_dice':r['metrics']['test_measured_stress']['dice'],
                                 'best_epoch':r['best_epoch'],
                                 'validation_threshold':r['threshold_selected_on_synthetic_dense_validation']}
                                for r in recs]
            mean[model][arm]=means
    # Source frames are not independent runs, and nuisance noise seeds were not replicated.
    def frame_dict(model,arm,kind):
        dic={}
        for s in idx['expected_seeds']:
            m=loaded[(model,arm,s)]['prob']
            dic[s]=per_frame_counts(m[kind],m['test_truth'],m['test_groups'],float(m['threshold']))
        return dic
    comparisons=[]
    for model in arches:
        for arm in arms:
            if arm=='A_measured_only':continue
            for kind in ('test_prob','test_stress_prob'):
                a=frame_dict(model,'A_measured_only',kind);b=frame_dict(model,arm,kind)
                res=hierarchical_frame_seed_bootstrap(a,b,seed=20261025,draws=4000)
                comparisons.append({'model':model,'comparison':arm+' - A_measured_only',
                                    'condition':'stress' if kind=='test_stress_prob' else 'clean',**res})
    for arm in arms:
        for kind in ('test_prob','test_stress_prob'):
            a=frame_dict('unet',arm,kind);b=frame_dict('azunet',arm,kind)
            res=hierarchical_frame_seed_bootstrap(a,b,seed=20261026,draws=4000)
            comparisons.append({'model':'azunet minus unet','comparison':arm,
                                'condition':'stress' if kind=='test_stress_prob' else 'clean',**res})
    for comp in ('C_measured_book_photo_style','E_book_generic_robust_augmentation'):
        if comp not in arms:continue
        # Additional control: C vs E and C vs D, not only each vs A.
    for comparison,base,alt in [('H50Z style vs generic robust augmentation','E_book_generic_robust_augmentation','C_measured_book_photo_style'),
                                ('H50Z style vs generic histogram match','D_measured_book_synthetic_style','C_measured_book_photo_style')]:
        if base not in arms or alt not in arms:continue
        for model in arches:
            for kind in ('test_prob','test_stress_prob'):
                res=hierarchical_frame_seed_bootstrap(frame_dict(model,base,kind),frame_dict(model,alt,kind),seed=20261028,draws=4000)
                comparisons.append({'model':model,'comparison':comparison,
                                    'condition':'stress' if kind=='test_stress_prob' else 'clean',**res})
    document={'schema':'s18_paired_seed_frame_comparison_v1','source':'real visualMetro measured track shapes + simulated pixels',
              'index_sha256':sha256(index.read_bytes()).hexdigest(),
              'original_models_unchanged':True,'total_trained_model_runs':len(loaded),
              'test_samples':len(reference[0]),'test_unique_frames':len(set(reference[1].tolist())),
              'train_seed_count':len(idx['expected_seeds']),
              'mean_by_model_and_arm':mean,'paired_comparisons':comparisons,
              'scientific_limitations':['Synthetic pixels/labels from film 070 expert click geometry only; no real film mask GT',
                 'Seed and frame bootstrap conditional on ONE film and 3 inits',
                 'Sparse annotation counts differ slightly between measured and book renderers even though 120 images and optimizer steps are equal',
                 'Threshold uses DENSE SYNTHETIC validation GT (not available for small real human annotation)',
                 'Style C uses unlabeled H50Z imagery, not human labels or independent photogrammetry',
                 'Style E is post-hoc, selected after initial C result, exploratory only',
                 'No false-merge/particle-ID metric without independently labeled instances',
                 'No external multi-run/medium transfer demonstrated']}
    (out_dir/'S18_metrics_and_intervals.json').write_text(json.dumps(document,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'n_runs':len(loaded),'means':{m:{a:{k:round(v[k],4) for k in ('clean_dice','stress_dice','centerline_f1')}
                                                         for a,v in rows.items()} for m,rows in mean.items()},
                      'interesting_comparisons':[{'model':x['model'],'comparison':x['comparison'],'condition':x['condition'],
                          'delta':round(x['mean_seed_delta_dice'],5),'ci95':[round(y,5) for y in x['paired_ci95_seed_and_frame']]}
                             for x in comparisons if x['model']=='azunet' and x['condition']=='stress']},indent=2,ensure_ascii=False))
    return document


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();analyze(a.index,a.out_dir)

if __name__=='__main__':main()
