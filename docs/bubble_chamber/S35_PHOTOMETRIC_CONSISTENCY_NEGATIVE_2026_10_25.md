# S35 — Is photometric consistency the missing mathematics in source-locked AZUNet?

**Executed 25 October 2026. Outcome: technically correct, but no material accuracy gain.** Original `models/azconv.py` and `models/segmentation.py` are unchanged. The tests never assume that model-drafted H50Z tracks are ground truth.

## Motivation and fixed hypothesis

S33 had established a substantial same-run shift in H50Z image brightness (source 005 mean gray 182.6, source 006 215.6). A **fit-free** quantile transform and a local Gaussian-residual transform improved synthetic line/scratch ranking, but neither produced human-certified image segmentation. S34's orientation loss improved only two of four artificial-control seeds. S35 therefore tested a different *a priori* mechanism: whether a photo-normalized AZUNet should be explicitly constrained to give the same probability map under independent exposure perturbations.

Let `u = normalize(I)` and `v = normalize(clip(a I + b))` be aligned, differently exposed 64×64 observations. Let `M=1` denote S31 provisional positive marks, `M=0` provisional likely-background marks and `M=255` original, entirely unknown photo pixels; let `w` be strictly zero on all unknown pixels. We use the unchanged S32 bounded confidence-BCE:

\[
\mathcal L_{\mathrm{sup}}=\tfrac12[\operatorname{BCE}_{w}(f_\theta(u),M)+\operatorname{BCE}_{w}(f_\theta(v),M)].
\]

The S35 experimental arm adds detached, UNLABELED consistency on the *same pixels* only:

\[
\mathcal L_{\mathrm{S35}}=\mathcal L_{\mathrm{sup}}+\lambda\,\frac{\sum_{i\in\Omega}\big[\sigma(f_\theta(v)_i)-\mathrm{sg}(\sigma(f_\theta(u)_i))\big]^2}{|\Omega|},\quad
\Omega=\{i:M_i=255,\,|\sigma(f_\theta(u)_i)-0.5|\ge0.08\},\quad \lambda=0.10.
\]

The baseline receives **exactly the same two exposed views, identical supervised loss, crops, initialization, batch order, optimizer, train epochs and two forward passes**. The only difference is the extra unlabeled term. Epoch 1 has no consistency for either arm; epoch 2 enables it only for the experiment. Its teacher view is detached. This does **not** manufacture expert labels or authorize physical joins. The training checkpoint is fixed to epoch 2 in both arms, never selected on the test photograph.

## Original architecture and source data

| Source / protocol | Fixed value |
|---|---|
| Architectures | Original source-locked AZUNet and BaselineUNet, widths 4/6/8/12, original AZConv rules=2 |
| Source annotation | 39 authentic H50Z crops, S31 0/1/255 **machine-only** weak masks |
| Supervision | S32 confidence BCE, 5 positive-centered and 5 negative-centered windows per crop; 64×64 |
| Training | Two epochs, AdamW lr=0.0005, batch 4, train seeds 20261025/20261026 |
| Input for AZUNet | S33 quantile-normalized, original AZConv unchanged |
| Input for U-Net | S33 local-normalized (strong preexisting comparator) |
| Photo separation | Train source 005, evaluate 006; separately train 006, evaluate 005 |
| Injections | 4 reserved synthetic seeds **per direction**, 48 scenes per seed, 384 new synthetic scene overlays overall |
| Exposure controls | Identity, seeded fixed `bright` and `dark`, same control scenes for both arms |
| Physical independence | 005 and 006 are from **one** acquisition / same physical run |

Only artificially injected polyline and scratch pixels, not original native photograph tracks, were eligible for injected-control AUROC/AUPRC. Synthetic shape templates came from **a separate original visualMetro set** and are not 2D physical correspondence to the H50Z photos. Four seeds each reuse the same source photo backgrounds, so they do not establish independent-film confidence.

## Results: identity perturbation, holdout photograph (mean over 4 fixed seeds)

| Train → test | Architecture | Matched two-view baseline AUROC | Additional consistency AUROC | Paired Δ AUROC |
|---|---|---:|---:|---:|
| 005 → 006 | Original AZUNet / quantile | 0.62563 | **0.62666** | **+0.00102** |
| 005 → 006 | Baseline U-Net / local | 0.63187 | 0.63167 | −0.00020 |
| 006 → 005 | Original AZUNet / quantile | 0.60788 | **0.60903** | **+0.00115** |
| 006 → 005 | Baseline U-Net / local | 0.59409 | **0.59454** | +0.00045 |

For AZUNet forward, the four *paired synthetic-seed* deltas were `[+0.00186, +0.00115, +0.00169, −0.00061]`; for AZUNet reverse, `[-0.00001, +0.00231, +0.00038, +0.00190]`. Matched bootstrap over just four seeds gave a forward CI approximately `[−0.00004,+0.00178]`, reverse `[+0.00019,+0.00211]`, but these are **NOT** independent physical samples and the tiny effects do not justify a claim of native-real trajectory performance. Under bright/dark transformations the average effects remained around the same small size; all twelve directional/illumination comparisons are retained in the JSON receipts.

### Why was the extra term ineffective?

At the last training epoch, `λ × consistency_MSE / labelled_weak_supervised_BCE` ranged from roughly **0.00008 to 0.00196** across seeds, architectures and directions. Thus the new term was just **0.008%–0.196%** of the numerical supervised objective; source-independent quantile/local normalization had already canceled much of the affine exposure variance, so there was little signal to regularize. Increasing λ *after inspecting these test results* would make a different, post-hoc hypothesis and would need fresh reserved test seeds. S35 does not cherry-pick a new coefficient.

## New diagnosis on 39 real photo crops: exposure clipping

We sampled 8 independently located 64×64 windows per verified real photo crop (168 patches of 005, 144 of 006). A fixed seeded affine exposure perturbation produces the following average input-map absolute change, scale `[0,1]`:

| Real photo | Raw input | Quantile | Local residual | After-perturbation saturated 0/255 pixels |
|---|---:|---:|---:|---:|
| 005 | 0.0973 | 0.0649 | **0.0268** | 13.8% |
| 006 | 0.1053 | **0.1114 (worse than raw)** | **0.0488** | **24.0%** |

This is an **input transformation diagnostic**, independent of track labels, not a measured real segmentation score. Why it matters: if `I' = aI+b` is a strictly affine exposure transform, quantiles satisfy `Q_p(I')=aQ_p(I)+b` for `a>0`; therefore `z=(Q_0.9-I)/(Q_0.9-Q_0.1)` is invariant (ignoring rounding). Real/simulated clipping `I' = clip(aI+b,0,255)` destroys that relation. Photo 006 is brighter, so seeded exposure perturbations saturated many more pixels. This is a **specific mathematical failure mode** of quantile normalization. Local Gaussian residual is also not exactly invariant under clipping, but showed less average input shift here. The original AZConv2d is not responsible for this algebraic limitation; any CNN fed a distorted input is affected.

## Decision and next falsifiable experiment

**Do not enable photo-consistency regularization as default**; a +0.001 AUROC delta on synthetic overlays is not meaningful evidence of better physical reconstruction. Keep the S33 quantile-AZUNet and local-U-Net controls and the classical curvature-constrained candidate tracker. Prioritize a separate predeclared **saturation-aware normalization** protocol and assess it on *new* simulated seeds, rather than choosing thresholds on S35 test sets. In parallel, manually correct S31/S32/S33 pseudo-mask lines and fragment-link labels; current source contains **zero human-certified real trace centerlines or relation truth**. Model outputs must stay `MACHINE_DRAFT_UNREVIEWED` / `PROVISIONAL_REVIEW_REQUIRED`.

### Reproduction

All scripts are source-only public; the original real photo crops, 0/1/255 weak masks and model weights are kept in a private archive.

```
PYTHONPATH=. python scripts/research/bubble_chamber/s35_train_consistency.py --manifest /private/S31_source_weak_masks/weak_manifest.json --tiles /private/S31_source_photos_derived --out /private/s35_models --train-photo 005
PYTHONPATH=. python scripts/research/bubble_chamber/s35_train_consistency.py --manifest /private/S31_source_weak_masks/weak_manifest.json --tiles /private/S31_source_photos_derived --out /private/s35_reverse_models --train-photo 006
PYTHONPATH=. python scripts/research/bubble_chamber/s35_eval_consistency.py --tiles /private/S31_source_photos_derived --measurements /private/1mH2Data/S1 --models /private/s35_models --train-photo 005 --out /private/S35_forward_test.json
PYTHONPATH=. python scripts/research/bubble_chamber/s35_eval_consistency.py --tiles /private/S31_source_photos_derived --measurements /private/1mH2Data/S1 --models /private/s35_reverse_models --train-photo 006 --out /private/S35_reverse_test.json
PYTHONPATH=. python -m pytest -q tests/test_s35_photometric_consistency.py
```

No original BMPs, real photo crops, private manual corrections or model checkpoints are committed to public GitHub. The unchanged S33 source-specific preprocessing and S32 bounded confidence loss remain the baselines.
