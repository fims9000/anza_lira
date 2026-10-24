#!/usr/bin/env python3
"""S18 paired *synthetic-pixel* ablation of Belyaev prior for original AZUNet.

A: 120 measured-track geometry renderings (no book).
B: the SAME first 96 measured-track renderings + 24 book hadroidae.
C: B with book TRAIN images quantile-styled from UNLABELED H50Z only.

All arms use equal counts, epochs, minibatches, source-locked original code,
identical held-out measured geometry frames, and validation on a shared set
of only measured-geometry synthetic pixels. No real-BMP GT or physical PID.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.large_relation_study import frame_partition
from bubble_chamber.neural_metrics import aggregate_topology
from scripts.research.bubble_chamber.s17_mix_real_measured_shapes import _choose_measured
from scripts.research.bubble_chamber.s17_make_book_prior_dataset import make_example
from scripts.research.bubble_chamber.s17_match_unlabeled_photo_style import grayscale_quantile_match
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model, verify_model_source


@dataclass(frozen=True)
class Config:
    source_split_seed: int = 20261025
    examples_train: int = 120
    shared_measured: int = 96
    examples_val: int = 24
    examples_test: int = 24
    book_test: int = 8
    size: int = 64
    epochs: int = 8
    batch_size: int = 8
    lr: float = 0.0008
    widths: tuple[int, ...] = (4, 6, 8, 12)
    rules: int = 2
    seed: int = 20261024
    num_threads: int = 2
    # Extra background clicks come ONLY from generated GT, never real photos.
    extra_simulated_bg_points: int = 24
    stressful_contrast: float = 0.42
    stressful_noise_std: float = 12.0


@dataclass
class Sample:
    image: np.ndarray
    sparse: np.ndarray
    truth: np.ndarray
    group: str
    source_kind: str
    ident: str


def sha256_images(samples: list[Sample]) -> str:
    h = sha256()
    for s in samples:
        h.update(s.image.tobytes())
        h.update(s.sparse.tobytes())
        h.update(s.truth.tobytes())
        h.update(s.group.encode())
        h.update(s.source_kind.encode())
    return h.hexdigest()


def background_clicks(sparse: np.ndarray, truth: np.ndarray, *, seed: int, count: int) -> np.ndarray:
    """Simulation-only: synthetic truth is *never* available for real film."""
    if sparse.shape != truth.shape or sparse.dtype != np.uint8 or not set(np.unique(sparse)).issubset({0, 1, 255}):
        raise ValueError('sparse labels invalid')
    if count < 0:
        raise ValueError('negative bg point budget')
    mask = sparse.copy()
    candidates = np.argwhere((mask == 255) & (truth == 0))
    if count and len(candidates):
        rng = np.random.default_rng(seed)
        new = candidates[rng.choice(len(candidates), min(count, len(candidates)), replace=False)]
        mask[new[:, 0], new[:, 1]] = 0
    return mask


def _sample(im: np.ndarray, sparse: np.ndarray, truth: np.ndarray, *, group: str, kind: str, ident: str, cfg: Config) -> Sample:
    if any(x.shape != (cfg.size, cfg.size) for x in (im, sparse, truth)):
        raise ValueError('shape mismatch')
    clean = background_clicks(sparse, truth, seed=int.from_bytes(sha256(ident.encode()).digest()[:8], 'little'), count=cfg.extra_simulated_bg_points)
    if not (np.any(clean == 1) and np.any(clean == 0)):
        raise ValueError('both synthetic sparse classes required')
    return Sample(im, clean, truth.astype(np.uint8), str(group), kind, ident)


def make_dataset(measurements: Path, cfg: Config, *, style_crops: Path | None) -> tuple[dict[str, list[Sample]], dict]:
    manifest = audited_large_manifest(Path(measurements))
    splits = frame_partition([frame['frame_id'] for frame in manifest['frames']], cfg.source_split_seed)
    expected = {'train': cfg.examples_train, 'val': cfg.examples_val, 'test': cfg.examples_test}
    measured = {}
    for split, count in expected.items():
        measured[split] = []
        for i, (image, mask, dense, meta) in enumerate(_choose_measured(manifest, splits[split], count, cfg.size,
                                                                          cfg.source_split_seed, split)):
            measured[split].append(_sample(image, mask, dense, group=str(meta['frame_id']), kind='measured_070_synthetic_pixels',
                                           ident=f'measured:{split}:{i}:{meta["frame_id"]}:{meta["track_id"]}', cfg=cfg))
    assert len(measured['train']) == cfg.examples_train
    assert 0 < cfg.shared_measured < cfg.examples_train
    assert len(set(s.group for s in measured['train']) & set(s.group for s in measured['val'])) == 0
    assert len(set(s.group for s in measured['test']) & set(s.group for s in measured['val'])) == 0
    assert len(set(s.group for s in measured['train']) & set(s.group for s in measured['test'])) == 0
    book_train = []
    for i in range(cfg.examples_train - cfg.shared_measured):
        rng = np.random.default_rng(int.from_bytes(sha256(f'S18:book:train:{cfg.source_split_seed}:{i}'.encode()).digest()[:8], 'little'))
        medium = 'liquid_hydrogen' if i % 2 == 0 else 'liquid_propane'
        image, truth, sparse, _full, _meta = make_example(cfg.size, rng, medium)
        book_train.append(_sample(image, sparse, truth, group=f'S18_SYNTH_BOOK_train_{i}', kind='book_hadroide_simulated',
                                  ident=f'book_train_{i}', cfg=cfg))
    book_test = []
    for i in range(cfg.book_test):
        rng = np.random.default_rng(int.from_bytes(sha256(f'S18:book:test:{cfg.source_split_seed}:{i}'.encode()).digest()[:8], 'little'))
        image, truth, sparse, _full, _meta = make_example(cfg.size, rng, 'liquid_hydrogen' if i % 2 == 0 else 'liquid_propane')
        book_test.append(_sample(image, sparse, truth, group=f'S18_SYNTH_BOOK_test_{i}', kind='book_hadroide_simulated',
                                 ident=f'book_test_{i}', cfg=cfg))
    styled_book = []
    styles = []
    if style_crops is not None:
        for fp in sorted(style_crops.glob('*.png')):
            im = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if im is not None and im.dtype == np.uint8:
                styles.append((im, sha256(fp.read_bytes()).hexdigest()))
        if not styles:
            raise ValueError('no readable unlabeled real styles')
        for i, s in enumerate(book_train):
            reference = styles[i % len(styles)][0]
            img = grayscale_quantile_match(s.image, reference)
            styled_book.append(Sample(img, s.sparse.copy(), s.truth.copy(), s.group, 'book_hadroide_styled_real_unlabeled', s.ident))
    data = {'A_measured_only': measured['train'], 'B_measured_plus_book': measured['train'][:cfg.shared_measured] + book_train}
    # D isolates generic histogram matching from the use of genuine unlabeled H50Z.
    # References come ONLY from measured-geometry TRAIN images, never held-out frames.
    synthetic_styled_book = []
    for i,s in enumerate(book_train):
        train_reference = measured['train'][i % cfg.shared_measured].image
        img = grayscale_quantile_match(s.image, train_reference)
        synthetic_styled_book.append(Sample(img, s.sparse.copy(), s.truth.copy(), s.group,
                                           'book_hadroide_train_synthetic_histogram', s.ident))
    data['D_measured_book_synthetic_style'] = measured['train'][:cfg.shared_measured] + synthetic_styled_book
    # E is a POST-HOC stress-augmentation control (exploratory, not pre-registered).
    # It uses no H50Z intensity statistics and changes ONLY book train pixels.
    generic_robust_book = []
    for i, s in enumerate(book_train):
        rng = np.random.default_rng(int.from_bytes(sha256(f'S18:generic_stress:{cfg.source_split_seed}:{i}'.encode()).digest()[:8], 'little'))
        center = float(np.median(s.image))
        scale = float(rng.uniform(.35, .85))
        sigma = float(rng.uniform(4.0, 16.0))
        v = center + (s.image.astype('float32')-center)*scale + rng.normal(0,sigma,s.image.shape)
        photo = np.uint8(np.clip(np.rint(v),0,255))
        generic_robust_book.append(Sample(photo,s.sparse.copy(),s.truth.copy(),s.group,
                                          'book_hadroide_generic_noise_contrast_no_real_refs',s.ident))
    data['E_book_generic_robust_augmentation'] = measured['train'][:cfg.shared_measured] + generic_robust_book
    if styles:
        data['C_measured_book_photo_style'] = measured['train'][:cfg.shared_measured] + styled_book
    for arm, examples in data.items():
        if len(examples) != cfg.examples_train: raise RuntimeError('unequal train sample count '+arm)
    data.update({'validation_measured': measured['val'], 'test_measured': measured['test'], 'test_book_only':book_test})
    summary = {'source_manifest_sha256':manifest['provenance_sha256'], 'group_split_counts': {k:len(v) for k,v in splits.items()},
               'source_only_one_film':True,'real_film_images_used_for_training':False,
               'real_reference_style_unlabeled_sha256':[x[1] for x in styles],
               'train_arms':{k:{'count':len(v),'hash':sha256_images(v),
                                'measured_count':sum(s.source_kind.startswith('measured') for s in v),
                                'book_count':sum(s.source_kind.startswith('book') for s in v)}
                             for k,v in data.items() if k.startswith(('A_', 'B_', 'C_', 'D_', 'E_'))},
               'validation_sha256':sha256_images(data['validation_measured']),
               'test_sha256':sha256_images(data['test_measured']),
               'test_frame_count':len(set(s.group for s in measured['test'])),
               'test_book_sha256':sha256_images(book_test),
               'pixel_label_origin':'all synthetic; measured visualMetro track shapes only; real photos style only'}
    return data, summary


def as_tensor(samples: list[Sample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = np.stack([s.image for s in samples])
    y = np.stack([s.sparse for s in samples])
    t = np.stack([s.truth for s in samples])
    return torch.from_numpy(((255 - x).astype('float32') / 255)[:,None]), torch.from_numpy(y.astype(np.int64)[:,None]), torch.from_numpy(t.astype('float32')[:,None])


def sparse_supervised_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    fg = labels == 1
    bg = labels == 0
    if not fg.any() or not bg.any(): raise ValueError('missing sparse class')
    known = fg | bg
    pw = (bg.sum().float() / fg.sum().float()).clamp(1.0, 15.0)
    point_weight = torch.where(fg, pw, torch.ones_like(logits))
    return F.binary_cross_entropy_with_logits(logits[known],labels[known].float(),weight=point_weight[known])


def counts(prob: np.ndarray, truth: np.ndarray, thr: float):
    pred = prob >= thr; target = truth.astype(bool)
    return np.array([int((pred & target).sum()), int((pred & ~target).sum()),int((~pred & target).sum())], np.int64)


def make_metrics(c: np.ndarray) -> dict:
    tp, fp, fn = (int(x) for x in c)
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(1,tp+fp), 'recall':tp/max(1,tp+fn),
            'dice':2*tp/max(1,2*tp+fp+fn)}


def stress_images(x: torch.Tensor, cfg: Config, *, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    value = x.numpy()[:,0]
    gray = 1 - value
    # Deterministic low-contrast and additional noise; labels unchanged.
    noisy = np.clip(.5 + (gray - .5)*cfg.stressful_contrast + rng.normal(0,cfg.stressful_noise_std/255.0,gray.shape), 0, 1)
    return torch.from_numpy((1 - noisy).astype('float32')[:,None])


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, batch_size:int = 16) -> np.ndarray:
    model.eval(); out = []
    for start in range(0,len(x),batch_size):
        out.append(model(x[start:start+batch_size]).sigmoid().numpy()[:,0])
    return np.concatenate(out,axis=0)


def choose_threshold(prob: np.ndarray, truth: np.ndarray) -> float:
    """Synthetic validation dense GT is an *oracle*, not low-label field calibration."""
    grid = np.arange(.15,.91,.05)
    best = (-1., .5)
    for t in grid:
        c = counts(prob,truth,t)
        d = make_metrics(c)['dice']
        if d > best[0]:best=(d,float(t))
    return round(best[1],4)


def paired_frame_bootstrap(a:np.ndarray,b:np.ndarray,y:np.ndarray,groups:list[str],thr_a:float,thr_b:float,
                           *, seed:int, draws:int=2000)->dict:
    """Pair and resample WHOLE source frames; not pixels or simulated draws."""
    frames = sorted(set(groups)); by = {f:np.flatnonzero(np.asarray(groups) == f) for f in frames}
    def m(p:np.ndarray,t:float,index):return counts(p[index],y[index],t)
    total_a=counts(a,y,thr_a);total_b=counts(b,y,thr_b)
    delta=make_metrics(total_b)['dice']-make_metrics(total_a)['dice']
    rng=np.random.default_rng(seed); ds=[]
    for _ in range(draws):
        keys=rng.choice(frames,size=len(frames),replace=True)
        ix=np.concatenate([by[f] for f in keys]);ca=m(a,thr_a,ix);cb=m(b,thr_b,ix)
        ds.append(make_metrics(cb)['dice']-make_metrics(ca)['dice'])
    return {'delta_dice':float(delta),'ci95_frame_bootstrap':[float(q) for q in np.quantile(ds,[.025,.975])],
            'source_frames':len(frames), 'bootstrap_draws':draws}


def atomic_torch_save(payload,path:Path):
    tmp=path.with_suffix('.tmp'); torch.save(payload,tmp);tmp.replace(path)


def run_arm(arm:str, arch:str, train: list[Sample], val: list[Sample], test:list[Sample], book_test:list[Sample],
            *, cfg:Config, out:Path)->dict:
    """Equal budgets for every arm; independent seed per (arm,architecture)."""
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed);random.seed(cfg.seed)
    model = build_model(arch,widths=cfg.widths,rules=cfg.rules)
    optimiser = torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=.0001)
    tx,tl,_ = as_tensor(train)
    vx,vl,vy = as_tensor(val)
    sx,_,sy = as_tensor(test)
    bx,_,by = as_tensor(book_test)
    best = float('inf'); history=[]; start=time.monotonic()
    for epoch in range(cfg.epochs):
        model.train(); losses=[]
        rng=np.random.default_rng(cfg.seed+epoch)
        indexes=rng.permutation(len(train))
        for offset in range(0,len(indexes),cfg.batch_size):
            idx=indexes[offset:offset+cfg.batch_size]
            optimiser.zero_grad(set_to_none=True)
            logits=model(tx[idx]);loss=sparse_supervised_loss(logits,tl[idx])
            if not torch.isfinite(loss):raise RuntimeError(f'{arm}/{arch}: nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimiser.step();losses.append(float(loss.detach()))
        model.eval();v_losses=[]
        with torch.no_grad():
            for offset in range(0,len(val),cfg.batch_size):
                logits=model(vx[offset:offset+cfg.batch_size]);
                v_losses.append(float(sparse_supervised_loss(logits,vl[offset:offset+cfg.batch_size]).detach()))
        val_bce=float(np.mean(v_losses))
        row={'epoch':epoch+1,'train_sparse_bce':float(np.mean(losses)),'val_sparse_bce':val_bce,
             'optimizer_steps':len(losses)}
        history.append(row)
        payload={'weights':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                 'epoch':epoch+1,'model':arch,'arm':arm,'config':asdict(cfg),
                 'synthetic_source_not_real_pixel_labels':True,'history':history}
        atomic_torch_save(payload,out/'last.pt')
        if val_bce < best:
            best=val_bce;atomic_torch_save(payload,out/'best.pt')
        print('TRAIN',arm,arch,'seed',cfg.seed,'epoch',epoch+1,'sparse_BCE',round(val_bce,5),flush=True)
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ck['weights']);model.eval()
    val_prob=predict(model,vx);threshold=choose_threshold(val_prob,vy.numpy()[:,0])
    pred=predict(model,sx)
    stress=predict(model,stress_images(sx,cfg,seed=cfg.source_split_seed+551))
    book=predict(model,bx)
    truth=sy.numpy()[:,0].astype('uint8')
    book_truth=by.numpy()[:,0].astype('uint8')
    clean_mask=pred>=threshold;stress_mask=stress>=threshold
    metrics={'test_measured':make_metrics(counts(pred,truth,threshold)),
             'test_measured_stress':make_metrics(counts(stress,truth,threshold)),
             'test_book':make_metrics(counts(book,book_truth,threshold)),
             'test_measured_topology':aggregate_topology(clean_mask,truth),
             'test_stress_topology':aggregate_topology(stress_mask,truth),
             'test_predicted_positive_fraction':float(clean_mask.mean()),
             'test_truth_positive_fraction':float(truth.mean()),
             'test_stress_predicted_positive_fraction':float(stress_mask.mean())}
    result={'architecture':arch,'train_arm':arm,'seed':cfg.seed,'train_examples':len(train),
            'train_positive_sparse_clicks':sum(int((s.sparse==1).sum()) for s in train),
            'train_negative_sparse_clicks':sum(int((s.sparse==0).sum()) for s in train),
            'model_parameters':sum(p.numel() for p in model.parameters()),
            'best_epoch':ck['epoch'],'threshold_selected_on_synthetic_dense_validation':threshold,
            'warning':'synthetic dense val oracle used ONLY for threshold; not available with small real pixel labels',
            'elapsed_seconds':round(time.monotonic()-start,2),'history':history,'metrics':metrics,
            'source_only_one_film':True,'real_bmp_pixel_metrics':None}
    (out/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    np.savez_compressed(out/'heldout_probabilities.npz',test_prob=pred.astype(np.float32),
                        test_stress_prob=stress.astype(np.float32),test_truth=truth,
                        test_groups=np.array([s.group for s in test]),
                        book_prob=book.astype(np.float32),book_truth=book_truth,
                        threshold=np.array(threshold))
    return result


def experiment(measurements:Path,out_dir:Path,*,cfg:Config,models:tuple[str,...]=('azunet','unet'),
               arms:tuple[str,...]=('A_measured_only','B_measured_plus_book'),style_crops:Path|None=None):
    verify_model_source()
    torch.set_num_threads(cfg.num_threads)
    d,summary=make_dataset(measurements,cfg,style_crops=style_crops)
    if set(arms)-set(d):raise ValueError('requested arm unavailable')
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/'dataset_audit.json').write_text(json.dumps({'config':asdict(cfg),'summary':summary},indent=2)+'\n')
    results=[]
    for arch in models:
        for arm in arms:
            p=out_dir/f'{arch}_{arm}_seed{cfg.seed}'
            results.append(run_arm(arm,arch,d[arm],d['validation_measured'],d['test_measured'],d['test_book_only'],cfg=cfg,out=p))
    comparative=[]
    for arch in models:
        baseline=next((r for r in results if r['architecture']==arch and r['train_arm']=='A_measured_only'),None)
        if baseline is None:continue
        ap=np.load(out_dir/f'{arch}_A_measured_only_seed{cfg.seed}'/'heldout_probabilities.npz')
        for arm in arms:
            if arm=='A_measured_only':continue
            target=next(r for r in results if r['architecture']==arch and r['train_arm']==arm)
            bp=np.load(out_dir/f'{arch}_{arm}_seed{cfg.seed}'/'heldout_probabilities.npz')
            if not np.array_equal(ap['test_groups'],bp['test_groups']) or not np.array_equal(ap['test_truth'],bp['test_truth']):
                raise RuntimeError('paired test not identical')
            for kind in ['test_prob','test_stress_prob']:
                aa=ap[kind].astype('float32'); bb=bp[kind].astype('float32')
                r=paired_frame_bootstrap(aa,bb,ap['test_truth'],ap['test_groups'].tolist(),
                                         baseline['threshold_selected_on_synthetic_dense_validation'],
                                         target['threshold_selected_on_synthetic_dense_validation'],seed=cfg.seed+31)
                comparative.append({'architecture':arch,'compare':arm+' minus A_measured_only',
                                    'condition':'synthetic_stress' if kind=='test_stress_prob' else 'synthetic_normal',**r})
    report={'schema':'s18_book_ablation_v1','research_question':'Does adding Belyaev hadroida priors improve original AZUNet on held-out measured expert track SHAPES with synthetic image pixels?',
            'comparison_design':'120 examples in all arms; same 24 measured-frame validation and 24 measured-frame test samples; same initialization, hyperparams and train steps',
            'source_models_git_blob':verify_model_source(),'dataset':summary,'configuration':asdict(cfg),
            'arms':results,'paired_frame_differences':comparative,
            'limitations':['No real photographic ground truth or verified track IDs','All measured geometry from single film 070',
                            'Sparse 0/1/255 training clicks generated by simulator (not expert)',
                            'Threshold tuned using dense synthetic validation truth (synthetic-only diagnostic oracle)',
                            'One seed is preliminary; multiple train seeds needed; repeated synthetic samples are not independent films']}
    (out_dir/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('S18_RESULTS',json.dumps({'results':[{ 'arm':r['train_arm'],'model':r['architecture'],'dice':r['metrics']['test_measured']['dice'],
                                    'stress_dice':r['metrics']['test_measured_stress']['dice'],
                                    'centerline_f1':r['metrics']['test_measured_topology']['macro_centerline_f1']} for r in results],
                                    'paired':comparative},ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--source-split-seed',type=int,default=20261025)
    p.add_argument('--size',type=int,default=64)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--train',type=int,default=120)
    p.add_argument('--shared-measured',type=int,default=96)
    p.add_argument('--val',type=int,default=24)
    p.add_argument('--test',type=int,default=24)
    p.add_argument('--book-test',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--widths',nargs=4,type=int,default=[4,6,8,12])
    p.add_argument('--rules',type=int,default=2)
    p.add_argument('--arms',nargs='+',default=['A_measured_only','B_measured_plus_book'])
    p.add_argument('--models',nargs='+',default=['azunet','unet'])
    p.add_argument('--style-crops',type=Path)
    args=p.parse_args()
    cfg=Config(seed=args.seed,source_split_seed=args.source_split_seed,size=args.size,
               epochs=args.epochs,batch_size=args.batch_size,examples_train=args.train,
               shared_measured=args.shared_measured,examples_val=args.val,
               examples_test=args.test,book_test=args.book_test,num_threads=args.threads,
               widths=tuple(args.widths),rules=args.rules)
    experiment(args.measurements,args.out_dir,cfg=cfg,models=tuple(args.models),arms=tuple(args.arms),style_crops=args.style_crops)

if __name__=='__main__':main()
