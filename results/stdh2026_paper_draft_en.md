# STDH-2026 Paper 3 Draft (finalized metrics)

Paper title: **Cross-Domain Evaluation of Anisotropic Fuzzy Local Convolution for Thin-Structure Segmentation**

## Positioning (non-overlap)

- Paper 1: theoretical operator properties.
- Paper 2: single-domain medical benchmark.
- Paper 3 (this paper): cross-domain transfer behavior of one local mechanism.

Domains used here:
1. Roads_HF (small GIS roads),
2. SpaceNet3/GlobalScaleRoad (larger GIS roads),
3. HRF_SegPlus (medical vessels).

## Abstract (ready)

This paper presents a cross-domain study of anisotropic fuzzy local convolution for thin-structure segmentation. We evaluate one local aggregation mechanism on three domains with different visual statistics: a small road dataset (Roads_HF), a larger satellite road dataset (SpaceNet3/GlobalScaleRoad), and a medical vessel dataset (HRF_SegPlus). The method is integrated into a U-Net style pipeline and compared against a matched baseline. We report overlap and structure-aware metrics and analyze precision-recall operating regimes across domains. The results show that with domain-specific calibration, anisotropic fuzzy aggregation improves overlap on HRF and SpaceNet3 while remaining near parity on Roads_HF. This supports practical reuse of one geometric local mechanism across heterogeneous imaging tasks.

Keywords: thin-structure segmentation, anisotropic fuzzy local convolution, vessel segmentation, road segmentation, cross-domain evaluation

## I. Introduction

Thin elongated structures appear in both medical and remote-sensing imaging, but local ambiguity and weak contrast make segmentation unstable. Standard isotropic local aggregation often loses weak branches and breaks continuity. We evaluate anisotropic fuzzy local aggregation as a transferable local mechanism and study how its operating regime changes across domains.

## II. Method (compact)

The model uses a U-Net style encoder-decoder with anisotropic fuzzy local aggregation blocks. Local compatibility combines directional geometry and fuzzy local agreement, followed by normalization in the local neighborhood. Baseline and AZ variants are trained under matched dataset protocols.

### II-A. Geometry Visualization Protocol (core contribution of Paper 3)

To avoid purely qualitative overlays, we visualize internal AZ geometry using explicit per-pixel quantities from AZ snapshots.

For each valid pixel `p`:
- fuzzy memberships `mu_r(p)`, `r=1..R`;
- dominant regime `r*(p) = argmax_r mu_r(p)`;
- regime confidence `c(p) = max_r mu_r(p)`.

Directional field:
- if `theta_map` is available:
  `theta(p) = theta_map[r*(p), p]`;
- otherwise:
  `theta(p) = atan2(u_{r*,y}, u_{r*,x})`.

Signed anisotropy contribution:
`g(p) = tanh(log(sigma_u/sigma_s)) * c(p)`, clipped to `[-1, 1]`.

This gives three interpretable maps:
1. direction `theta(p)`,
2. anisotropy contribution `g(p)`,
3. confidence `c(p)`.

### II-B. Figure construction used in this paper

We build one 4-panel figure per case:
1. `Input + GT`,
2. `error-centric Baseline vs AZ difference`,
3. `AZ direction arrows`,
4. `anisotropy strength map`.

Error-centric difference map (relative to GT):
- green: AZ fixed baseline false negative,
- blue: AZ removed baseline false positive,
- orange: AZ added false positive,
- red: AZ introduced new false negative.

Direction arrows are drawn on object support
`M_obj = (AZ_pred OR GT) AND valid_mask`,
then skeletonized and pruned (small components removed), so arrows follow elongated structures rather than background texture.

Anisotropy strength map uses `|g(p)|` with robust in-object normalization
(10th to 95th percentile) and a colorblind-friendly blue-to-orange ramp.

## III. Datasets and Protocol

- **Roads_HF**: 31 image/mask pairs, split `train 19 / val 6 / test 6`, resolution `1280x720`.
- **SpaceNet3 prepared**: split `train 179 / val 39 / in-domain-test 39`.
- **HRF_SegPlus**: `train 30 / test 15`, nominal image size `500x500`.

Common setup:
- optimizer: Adam,
- threshold selection: validation sweep,
- reported metrics: Dice, IoU, Precision, Recall, clDice.

## IV. Results

### A. Main table

| Dataset | Model | Dice | IoU | Precision | Recall | clDice |
|---|---|---:|---:|---:|---:|---:|
| Roads_HF | Baseline | 0.9740 | 0.9493 | 0.9704 | 0.9777 | 0.9147 |
| Roads_HF | AZ-Thesis | 0.9727 | 0.9468 | 0.9710 | 0.9744 | 0.9164 |
| SpaceNet3 | Baseline | 0.5466 | 0.3761 | 0.5429 | 0.5503 | 0.6138 |
| SpaceNet3 | AZ-Thesis (recovered) | 0.5914 | 0.4198 | 0.5955 | 0.5873 | 0.6802 |
| HRF_SegPlus | Baseline | 0.6458 | 0.4769 | 0.6165 | 0.6780 | 0.5361 |
| HRF_SegPlus | AZ-Thesis | 0.6822 | 0.5177 | 0.7103 | 0.6562 | 0.5140 |

### B. Delta (AZ - Baseline)

- Roads_HF: Dice `-0.0014`, IoU `-0.0026`, Precision `+0.0006`, Recall `-0.0033`, clDice `+0.0017`.
- SpaceNet3 (recovered): Dice `+0.0448`, IoU `+0.0437`, Precision `+0.0525`, Recall `+0.0370`, clDice `+0.0664`.
- HRF_SegPlus: Dice `+0.0364`, IoU `+0.0408`, Precision `+0.0938`, Recall `-0.0219`, clDice `-0.0220`.

## V. Discussion

The same geometric local prior is transferable, but the best operating regime is domain-dependent. SpaceNet3 illustrates this clearly: an initial AZ setup was conservative and underperformed baseline; after calibration of AZ depth/mix and residual contribution, AZ surpassed baseline on overlap and structure metrics. This indicates that deployment should include lightweight domain-specific calibration rather than a fixed universal setting.

From the visualization side, the same runs show that AZ improvements are spatially concentrated on thin elongated fragments where directional agreement is high and anisotropy strength is non-zero. This supports the interpretation that gains are linked to geometric local aggregation rather than threshold-only effects.

## VI. Conclusion

Cross-domain evaluation shows that anisotropic fuzzy local aggregation is practically reusable across heterogeneous thin-structure segmentation tasks. With calibrated settings, AZ improves overlap on HRF and SpaceNet3, while remaining near parity on Roads_HF.

## Run references (final)

- Roads_HF baseline: `results/article3_roads_hf_s42_e25_gpu_20260428_123652_baseline/metrics.json`
- Roads_HF AZ: `results/article3_roads_hf_s42_e25_gpu_20260428_123652_az_thesis/metrics.json`
- SpaceNet3 baseline: `results/article3_spacenet_s42_e12_gpu_20260428_123652_baseline/metrics.json`
- SpaceNet3 AZ recovered: `results/article3_spacenet_recover_azthesis_continue_s42_e12_20260428_135128/metrics.json`
- HRF baseline: `results/article3_hrf_baseline_s42_e40/metrics.json`
- HRF AZ: `results/article3_hrf_final_s42_e40/metrics.json`

Figure assets for this paper:
- Main geometry figure: `results/a3_final_package/final_article3/figures/geometry_clean_global_roads_spacenet3_paris_img0087.png`
- Case-analysis report: `results/a3_final_package/final_article3/figures/spacenet_v3_advantage_report_ru.md`
