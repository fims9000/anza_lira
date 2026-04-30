# STDH-2026 Paper 3 Draft (Full Text)

## Title
**Cross-Domain Evaluation of Anisotropic Fuzzy Local Convolution for Thin-Structure Segmentation**

## Abstract
Thin elongated structures are difficult to segment in both medical and remote-sensing images because they combine weak local contrast, branching topology, and local ambiguity. This paper presents an evaluation of anisotropic fuzzy local convolution (AZ) used as a local aggregation mechanism inside a U-Net style segmentation pipeline. The main claim is not universal superiority on every metric, but geometry-aware behavior that can be inspected through model-native orientation and anisotropy diagnostics. We report quantitative results on SpaceNet3/GlobalScaleRoad and HRF_SegPlus, and use Roads_HF only as an auxiliary qualitative sanity check because its baseline performance is already saturated. The proposed visualization protocol links local error corrections to explicit internal states of the AZ layer. This makes the method more interpretable and helps identify both successful and failure regimes.

**Keywords:** thin-structure segmentation, anisotropic fuzzy local convolution, retinal vessels, road extraction, cross-domain evaluation, geometric interpretability

## I. Introduction
Segmentation of thin structures remains unstable when the target class is long, narrow, branching, and partially low-contrast. This pattern appears in both digital healthcare (vessels) and geospatial vision (roads). Standard isotropic local aggregation often treats longitudinal and transverse neighborhoods too similarly, which may reduce continuity and damage weak branches.

The method studied in this work introduces anisotropic fuzzy local aggregation. Its local compatibility combines directional geometry and fuzzy local agreement, so that neighborhood contributions are weighted by both spatial orientation and regime confidence. This mechanism has been analyzed in our previous works from theoretical and single-domain perspectives. The purpose of the current paper is different: we evaluate whether one geometric local mechanism is reusable across heterogeneous domains and how its operating regime should be interpreted.

The paper makes three contributions:
1. A matched U-Net vs AZ comparison on representative thin-structure datasets, with Roads_HF treated as a qualitative saturated-baseline case rather than as primary evidence.
2. A model-native geometry visualization protocol that connects segmentation changes to internal AZ directional states.
3. A transparent discussion of operating regimes, including precision-recall trade-offs, clDice degradation on HRF, and limitations requiring larger ablation studies.

## II. Method

### A. AZ local aggregation
For pixel \(p\), neighborhood point \(q\), and regime \(r \in \{1,\dots,R\}\), local compatibility is:

\[
w_r(p,q)=\mu_r(p)\mu_r(q)\exp\left(-\frac{d_u^2}{\sigma_u^2}-\frac{d_s^2}{\sigma_s^2}\right),
\]

where \(d_u=\langle q-p,u_r\rangle\), \(d_s=\langle q-p,s_r\rangle\), \(\mu_r\in[0,1]\) is fuzzy membership, and \((u_r,s_r)\) defines local directional axes. Weighted local aggregation is followed by normalization and feature mixing in a U-Net style encoder-decoder pipeline.

### B. Why we visualize orientation axis, not signed flow
Because the kernel uses squared projections \(d_u^2,d_s^2\), it is invariant to sign flip of direction:

\[
u_r \equiv -u_r,\quad \theta \equiv \theta+\pi.
\]

Therefore, the current model identifies orientation axis (angle modulo \(\pi\)), not one-way polarity (“forward/backward flow”). For this reason, the correct interpretation for panel-level geometry visualization is axis glyphs aligned with model \(\theta\), not one-way arrows with semantic flow direction.

### C. Geometry states used for interpretation
From the AZ snapshot for valid pixel \(p\):
- dominant regime \(r^*(p)=\arg\max_r \mu_r(p)\),
- confidence \(c(p)=\max_r \mu_r(p)\),
- model orientation \(\theta(p)\) from `theta_map`,
- signed anisotropy gain:
\[
g(p)=\tanh(\log(\sigma_u/\sigma_s))\cdot c(p), \quad g\in[-1,1].
\]

Interpretation panels:
1. Input + GT overlay.
2. Error-centric baseline vs AZ difference map.
3. Model orientation axis map on skeletonized object support.
4. Anisotropy strength map using robust normalization of \(|g(p)|\).

### D. Architecture correction applied
To keep interpretation and training consistent, `az_thesis` defaults were corrected:
- geometry mode: `local_hyperbolic`,
- learnable directions: `learn_directions=True`,
- anti-collapse regularization: `direction_collapse` enabled (small positive weight by default or explicit config).

Backward compatibility for older checkpoints (historical fixed-cat geometry) is preserved by loader fallback.

## III. Experimental Setup

### A. Datasets
1. **SpaceNet3 prepared (GlobalScaleRoad split)**: train 179 / val 39 / in-domain-test 39.
2. **HRF_SegPlus**: train 30 / test 15; nominal size around \(500\times500\).
3. **Roads_HF**: 31 image/mask pairs; split train/val/test = 19/6/6; used only as a qualitative saturated-baseline check.

### B. Protocol
- Baseline: U-Net style model without AZ block.
- Proposed: AZ-Thesis variant with anisotropic fuzzy local aggregation.
- Optimizer: Adam.
- Threshold policy: validation sweep (Dice-based selection).
- Main metrics: Dice, IoU, Precision, Recall, clDice.

## IV. Results

### A. Main quantitative table

| Dataset | Model | Dice | IoU | Precision | Recall | clDice |
|---|---|---:|---:|---:|---:|---:|
| SpaceNet3 | Baseline | 0.5466 | 0.3761 | 0.5429 | 0.5503 | 0.6138 |
| SpaceNet3 | AZ-Thesis (recovered) | 0.5914 | 0.4198 | 0.5955 | 0.5873 | 0.6802 |
| HRF_SegPlus | Baseline | 0.6458 | 0.4769 | 0.6165 | 0.6780 | 0.5361 |
| HRF_SegPlus | AZ-Thesis | 0.6822 | 0.5177 | 0.7103 | 0.6562 | 0.5140 |

Roads_HF is omitted from the main quantitative table because the baseline is already near saturation (Dice about 0.974), and the AZ variant remains within noise-level parity. We keep this dataset only for qualitative visualization and sanity checking.

### B. Delta AZ - Baseline
- **SpaceNet3:** Dice +0.0448, IoU +0.0437, Precision +0.0525, Recall +0.0370, clDice +0.0664.
- **HRF_SegPlus:** Dice +0.0364, IoU +0.0408, Precision +0.0938, Recall -0.0219, clDice -0.0220.

### C. Per-image SpaceNet3 statistics
For the reproducible SpaceNet3 split, we additionally computed per-image metrics over 39 test tiles. This avoids relying only on aggregate pixel-level scores.

| Metric | Baseline mean ± std | AZ mean ± std | Delta mean ± std |
|---|---:|---:|---:|
| Dice | 0.4931 ± 0.2246 | 0.5248 ± 0.2001 | +0.0318 ± 0.1238 |
| IoU | 0.3538 ± 0.1854 | 0.3790 ± 0.1776 | +0.0252 ± 0.0987 |
| Precision | 0.4958 ± 0.2112 | 0.5071 ± 0.2055 | +0.0113 ± 0.1618 |
| Recall | 0.5112 ± 0.2598 | 0.5956 ± 0.2317 | +0.0844 ± 0.1833 |

The per-image table confirms the positive trend on SpaceNet3, but also shows substantial tile-to-tile variance. Therefore, the result should be interpreted as promising but not yet statistically conclusive.

### D. Component interpretation / analytical ablation
The current submission does not include a full retrained ablation grid. However, the role of the two components can be analyzed directly from the local weight:

- Without the fuzzy factor, \(w_r(p,q)\) reduces to a purely geometric anisotropic kernel. This keeps directional selectivity but loses the ability to down-weight neighbors with weak structural agreement.
- Without anisotropy, setting \(\sigma_u=\sigma_s\) collapses the kernel to an isotropic fuzzy local aggregation. This keeps soft membership weighting but removes longitudinal/transverse discrimination.
- The full AZ layer combines both terms, so local evidence is selected both by orientation and by fuzzy structural agreement.

A full retrained ablation with `baseline`, `baseline + topology loss`, `AZ without fuzzy`, `AZ without anisotropy`, and `AZ full` is required for a journal-level extension.

### E. Regime-count scalability
The current reproducible SpaceNet3 runs use a fixed number of fuzzy regimes. We do not claim that this value is optimal. A systematic sweep over \(R=1,2,4,8,16\) is required to characterize scalability and rule redundancy. This is listed as a required extension rather than a completed result.

## V. Discussion

### A. Cross-domain behavior
The same AZ mechanism does not produce identical gains across domains. This is expected because target geometry, contrast profile, and annotation style differ. The strongest quantitative gain is observed on SpaceNet3 after calibration, indicating that the mechanism can improve both overlap and structure-sensitive behavior when configured to the domain.

### B. Structure-aware interpretation
The geometry protocol shows where AZ corrections happen and how they align with model orientation and anisotropy activity. Instead of purely qualitative “pretty masks,” the analysis links error corrections to explicit internal quantities \((\theta, g, c)\). This is especially useful in reviewer dialogue because improvements become mechanistically explainable.

### C. clDice and recall trade-off
On HRF_SegPlus, AZ improves Dice, IoU, and Precision, but Recall and clDice decrease. This suggests that the layer can suppress faint vessel continuations while reducing false positives. The effect is important and should not be hidden: it indicates that the current loss is not sufficiently topology-preserving for weak vessels. A natural correction is to include a topology-preserving term such as clDice loss or skeleton recall regularization in future training.

### D. Limitations
We acknowledge that the current evaluation is limited to datasets of modest size, the baseline comparison omits attention-based and vessel-specific architectures, and no fully retrained component ablation has been performed. The present study should therefore be read as a geometry-interpretability and proof-of-concept evaluation rather than as a complete state-of-the-art benchmark.

Additional limitations are:
1. Current formulation models orientation axis, not signed directional flow.
2. Trade-offs between Precision and Recall can vary by domain.
3. Full statistical significance analysis over many seeds remains future work.
4. The influence of the number of fuzzy regimes \(R\) has not yet been systematically evaluated.

### E. Future work
Future work will address:
1. comparison with Attention U-Net, U-Net++, nnU-Net, and at least one transformer-based segmentation model;
2. evaluation on additional medical datasets such as DRIVE, CHASE_DB1, and STARE;
3. full ablation over fuzzy-only, anisotropy-only, and full AZ variants;
4. sensitivity analysis over the number of regimes \(R=1,2,4,8,16\);
5. topology-aware training with clDice or skeleton-based losses.

### F. References to prioritize
The final manuscript should avoid overemphasizing dynamical-systems terminology for the longitudinal/transverse split. The directional part should instead be positioned through standard image-processing references such as steerable filters and anisotropic diffusion.

## VI. Conclusion
Anisotropic fuzzy local convolution is a practical local mechanism for thin-structure segmentation when domain calibration is applied. The current evidence is strongest for geometry-aware interpretability and for selected calibrated cases rather than for universal superiority. The proposed model-native visualization protocol provides interpretable evidence of how local anisotropic behavior influences segmentation outcomes and where it fails. This makes AZ a useful candidate for further study in robust elongated-structure analysis.

## VII. Reproducibility Notes
- Public repository:

`https://github.com/fims9000/anza_lira`

- Final geometry figure command:

`python scripts/export_geometry_clean_article_figure.py --results-dir results --run article3_spacenet_sprint_v3_recover --baseline-run article3_spacenet_sprint_v3_baseline --sample-index 13 --output-dir results/a3_final_package/final_article3/figures --device cpu`
Recommended main sample index for the paper body: `30` (`spacenet3_paris_img0417`).

- Final figure used in paper:

`results/a3_final_package/final_article3/figures/geometry_clean_global_roads_spacenet3_paris_img0417.png`

## VIII. Run References
Public-repo reproducible SpaceNet3 materials:
- SpaceNet3 baseline: `results/article3_spacenet_sprint_v3_baseline/metrics.json`
- SpaceNet3 AZ recovered: `results/article3_spacenet_sprint_v3_recover/metrics.json`
- direction-diversity probe: `results/article3_spacenet_v3_dirlearn_probe_s42_e10/metrics.json`
- visual bundle: `results/a3_final_package/final_article3/article3_final_visual_bundle_ru.md`

Medical HRF and Roads_HF values in this draft are retained as local experimental results. For a final public submission, either include the corresponding reproducible artifacts or move those numbers to supplementary/local notes.
