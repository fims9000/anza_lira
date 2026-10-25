# S36: Is exposure saturation the reason AZUNet transfers poorly?

**Date:** 25 October 2026. **Verdict:** a useful negative/mixed ablation; **do not adopt as the default**. Original source-locked `AZConv2d` and `AZUNet` are unchanged. This study improves artificial control metrics in one source→target direction but **worsens** the other direction. It is not ground-truth physical particle-track recognition.

## Mathematical problem inherited from S33/S35

The successful S33 quantile input is

\[
 z_q(I)=\mathrm{clip}\bigl[(Q_{0.9}(I)-I)/(Q_{0.9}(I)-Q_{0.1}(I)),0,1\bigr].
\]

For an invertible positive affine exposure `I'=aI+b` the quantiles transform exactly as `Q_p(I')=aQ_p(I)+b`, so `z_q(I')=z_q(I)`, up to rounding. However exposure clipping **destroys this property**: `clip(aI+b,0,255)` is not invertible, its upper quantiles can collapse, and many bright pixels become indistinguishable. In S35, 144 verified real H50Z-006 patches were randomly re-exposed; an average 24% of their pixels reached 0 or 255 and the quantile-normalized map changed *more* than the raw normalized input (mean absolute differences 0.1114 vs 0.1053). A local Gaussian-background residual was more stable (0.0488). These were **unlabeled input diagnostics**, not native physical particle recognition.

## S36 hypothesis and fixed input rule

Create a **single-channel input only**, leaving the original AZConv and U-Net code unchanged:

\[
 s=\operatorname{mean}_{i}\,\mathbb{1}[I_i\le1\;\mathrm{or}\;I_i\ge254],\quad
 \alpha=\mathrm{clip}\bigl[(s-0.02)/0.18,0,1\bigr],\quad
 z_{36}=(1-\alpha)z_q+\alpha z_{local}.
\]

`z_local` is the original S33 Gaussian-residual normalizer. We chose the transition at 2%→20% clipped pixels based on the preceding *unlabeled* photo-shift diagnosis, not by optimizing on S36 test labels. The mixing coefficient is a deterministic source-image statistic **not a track probability, physical density or certified confidence**. It cannot recover pixels lost to clipping. We compare against **matched source-locked AZUNet/S35 quantile** with the same training seed, original model, two augmented views, S31 machine masks, optimizer, learning rate, batch budget and two fixed epochs, and retain S35 **local U-Net** as strong external architecture control. Training photo 005 is evaluated only on 006 and vice versa.

S36 test seeds were fixed before first evaluation: `20261049–20261052` for 005→006 and `20261053–20261056` for 006→005. Each seed has 48 artificial measured-shape trace and scratch overlays on authentic H50Z backgrounds. Four exposure modes per scene: unchanged, bright, dark, and **highlight_clip** `clip(1.18 I + 15,0,255)`. All 384 unique source scenes and four exposure variants share the same *one physical run*, so synthetic seeds are not independent physical media.

## Fixed synthetic-injection test outcomes, AUROC on generated track/scratch pixels

| Train → test | Photo mode | Quantile original AZUNet (S35) | Saturation-blended AZUNet (S36) | Local U-Net control (S35) | Δ AZUNet |
|---|---|---:|---:|---:|---:|
| 005→006 | Identity | 0.6376 | **0.6633** | 0.6535 | **+0.0257** |
| 005→006 | Bright | 0.6368 | **0.6648** | 0.6535 | +0.0280 |
| 005→006 | Dark | 0.6344 | **0.6653** | 0.6523 | +0.0310 |
| 005→006 | Highlight clipped | **0.6962** | 0.6945 | 0.6473 | −0.0017 |
| 006→005 | Identity | **0.6696** | 0.6648 | 0.6432 | **−0.0048** |
| 006→005 | Bright | **0.6693** | 0.6655 | 0.6427 | −0.0038 |
| 006→005 | Dark | 0.6717 | **0.6735** | 0.6411 | +0.0018 |
| 006→005 | Highlight clipped | 0.6650 | **0.6651** | 0.6299 | +0.0001 |

The 005→006 identity gains for four new synthetic seeds were `[+0.03487,+0.00054,+0.07224,−0.00504]` (3/4 wins). For 006→005 `[+0.02850,−0.03485,−0.00619,−0.00655]` (1/4 wins). A seed bootstrap CI on the two four-seed averages spans zero (forward approximately `[-0.00225,+0.05432]`; reverse `[-0.02768,+0.01974]`); these are **synthetic-seed uncertainty descriptors, not an independent film/run statistical guarantee**. Under severe highlight clipping, the new branch does not robustly outperform the original quantile AZUNet in either direction.

## Post-hoc ablation: training versus inference

**After** the above S36 tests were inspected, we ran a clearly marked **post-hoc diagnostic on the SAME test seeds** using already trained weights, without fitting new parameters. The two extra switches isolate training from inference:

- `quantile-trained AZUNet` tested with `saturation-blended input` (inference change only);
- `saturation-blend-trained AZUNet` tested with `quantile input` (training change only).

| Source → target | Exposure | Inference switch only Δ AUROC | Training switch only Δ | Both Δ |
|---|---|---:|---:|---:|
| 005→006 | identity | −0.00247 | **+0.02750** | +0.02565 |
| 005→006 | highlight_clip | **−0.02763** | +0.00242 | −0.00172 |
| 006→005 | identity | +0.00028 | **−0.00503** | −0.00477 |
| 006→005 | highlight_clip | +0.00202 | +0.00348 | +0.00006 |

**Interpretation:** most apparent forward improvement comes from *how the model was trained on altered views*, not from changing the inference input normalization, which can make the bright clipped test **worse**. This is a specific falsification of the naive proposition “blending local normalization always fixes saturation”. These post-hoc deltas cannot be used as a new heldout evaluation or for selecting a blend threshold; they only help diagnose the observed experiment.

## Research recommendation

- **Retain original S33 quantile AZUNet and local U-Net as the reference controls.** Do not publish S36 as proven superior; it is direction-sensitive on artificial controls.
- Study true sources of domain mismatch: only two physical H50Z photos of the same run; S31 provisional positives cover 13/39 photo crops and many negatives are *assumed*, not expert-confirmed. A different filming exposure can change the source class balance and produce pseudo-label error, which extra preprocessing cannot solve.
- Next genuine scientific gate: an initial manually corrected subset of S31 133 provisional fragments and 28 pair questions, retaining all unreviewed as UNKNOWN; then a physics-review layer and a different physical run. Test false track joins and continuity, not just artificial pixel AUROC.
- No current model output authorizes physically automatic joins, momentum, charge, particle identity or 3D reconstruction.

## Code and private reproduction

New modules: `bubble_chamber/s36_saturation_normalization.py`, scripts `s36_train_saturation.py`, `s36_eval_saturation.py`, `s36_posthoc_component_audit.py`, `s36_collect_report.py` and source-only contract tests. With S35 source-locked model checkpoints, S31 photo crops/masks and preserved visualMetro data, reproducible experiments run in **both** photo directions. All private photos, machine masks, original visualMetro coordinates and model weights are excluded from the public repository; the full private experiment package is archived separately.
