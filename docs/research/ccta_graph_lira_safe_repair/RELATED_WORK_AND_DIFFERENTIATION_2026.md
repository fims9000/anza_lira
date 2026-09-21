# Related work and differentiation for risk-controlled coronary connectivity repair

Date: 2026-09-21
Branch: `research/ccta-graph-lira-safe-repair`

## Scope

This note is a research-positioning checkpoint, not a claim that every cited method has been reproduced.

The current ANZA-LIRA / Graph-LIRA line targets a different problem from ordinary vessel segmentation:

> given an already segmented but possibly fragmented coronary tree, decide which local fragments should be structurally connected, which junction relation is plausible, and when the system should abstain instead of forcing a repair.

The intended contribution is therefore a **risk-controlled structural repair layer** rather than another segmentation backbone.

## 1. Segmentation-level topology preservation

### clDice

Shit et al., CVPR 2021 introduced clDice / soft-clDice for tubular segmentation. The loss uses overlap between segmentation and skeletons and is designed to preserve connected topology for 2-D and 3-D tubular structures.

Reference:
- S. Shit et al., "clDice - A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation", CVPR 2021.
- DOI: 10.1109/CVPR46437.2021.01629
- https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html

Relevance to us:
- strong baseline for preventing connectivity loss during segmentation;
- evaluates topology at mask/skeleton level;
- does not by itself solve ambiguous post-hoc fragment identity at a bifurcation;
- does not provide an explicit selective "do not repair" policy.

### Skeleton Recall Loss

Kirchhoff et al., ECCV 2024 proposed Skeleton Recall Loss. Ground-truth skeletonization is precomputed and the network is encouraged to recover the tubed skeleton, avoiding expensive differentiable skeletonization. The paper reports >90% reduction of topology-loss computational overhead relative to heavier alternatives and supports multi-class thin structures.

Reference:
- Y. Kirchhoff et al., "Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures", ECCV 2024.
- DOI: 10.1007/978-3-031-72980-5_13
- https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9904_ECCV_2024_paper.php
- code: https://github.com/MIC-DKFZ/Skeleton-Recall

Relevance:
- useful future segmentation-loss baseline;
- particularly attractive if we later retrain the segmentation stage because it is inexpensive and multi-class capable;
- again, it addresses connectivity preservation during segmentation rather than risk-controlled reconnection of already ambiguous fragments.

### Persistent-homology topology losses

Persistent-homology approaches penalize discrepancies in topological invariants / persistent features. More recent work adds spatial information to reduce ambiguous topological matching.

Examples:
- N. Byrne et al., persistent-homology topological loss for multiclass CMR segmentation, IEEE TMI 2023, DOI 10.1109/TMI.2022.3203309.
- B. Wen et al., "Topology-Preserving Image Segmentation with Spatial-Aware Persistent Feature Matching", ICCV Workshops 2025 / arXiv:2412.02076.

Relevance:
- useful if we later optimize the segmentation itself for Betti-number / loop / component structure;
- expensive and indirect for the current pair/junction identity question;
- topology equality can still be compatible with the wrong anatomical branch identity, which is exactly the class of error Graph-LIRA is intended to expose.

### HarmonySeg

Huang et al., ICCV 2025 combine multiscale feature fusion, vesselness information, and a growth-suppression balanced topology-preserving loss for tubular structures.

Reference:
- Y. Huang et al., "HarmonySeg: Tubular Structure Segmentation with Deep-Shallow Feature Fusion and Growth-Suppression Balanced Loss", ICCV 2025.
- https://openaccess.thecvf.com/content/ICCV2025/html/Huang_HarmonySeg_Tubular_Structure_Segmentation_with_Deep-Shallow_Feature_Fusion_and_Growth-Suppression_ICCV_2025_paper.html

Relevance:
- reinforces the value of coupling image evidence with explicit structural priors;
- the "growth versus suppression" idea is conceptually close to our repair-vs-abstain balance;
- however the decision is still embedded in segmentation training rather than an explicit graph repair policy.

## 2. Explicit vessel reconnection / reconstruction

This family is the most important comparison for Graph-LIRA.

### OGMC: Optimal Geometric Matching Connection

Zhu et al. propose a two-stage vessel framework: an edge-aware segmentation network followed by an Optimal Geometric Matching Connection model. The reconnection stage uses local curve geometry to filter broken endpoints, a global connection ordering, and geodesic repair under a Riemannian metric.

Reference:
- Y. Zhu, Y. Qiao, Q. Zhou, X. Yang, "Edge morphology attention mechanism and optimal geometric matching connection model for vascular segmentation", Biomedical Signal Processing and Control 99 (2025) 106849.
- DOI: 10.1016/j.bspc.2024.106849.

Important similarity:
- separates segmentation from explicit connectivity repair;
- uses endpoint geometry;
- reasons about candidate pairing and connection ordering;
- uses a geodesic-like path only after candidate connection decisions.

Important difference:
- our central endpoint decision is learned / image-conditioned and then constrained globally by Graph-LIRA;
- our system is explicitly allowed to abstain;
- our principal safety metric is false structural repair under a coverage/risk tradeoff, rather than forcing an optimally connected topology.

This is probably the closest conceptual geometry baseline for the repair stage.

### CorSegRec

Qiu et al. propose a topology-preserving three-stage framework specifically for coronary artery extraction:

1. segmentation with a centerline-enhanced loss;
2. centerline reconnection with a regularized walk that combines distance, centerline-classifier probability, and directional cosine similarity;
3. reconstruction of missing vessels with an implicit neural representation.

Reported segmentation results include Dice 88.53% on ASOCA and 85.07% on PDSCA, with HD 1.07 mm and 1.63 mm respectively.

Reference:
- "A topology-preserving three-stage framework for fully-connected coronary artery extraction", Medical Image Analysis, 2025.
- PubMed PMID 40239457.
- https://pubmed.ncbi.nlm.nih.gov/40239457/
- repository announced at https://github.com/YH-Qiu/CorSegRec

This is the most important coronary-specific external baseline for our current paper direction.

Key distinction:
- CorSegRec is designed to obtain a fully connected tree;
- Graph-LIRA is deliberately **not** required to connect everything;
- our scientific objective is: maximize correct repair at a controlled false-link budget, with uncertain relations left unresolved.

That distinction needs to remain explicit in any paper. "Fully connected" is not automatically "anatomically correct".

### Image-to-graph methods

Vesselformer (Prabhakar et al., MIDL 2024) directly predicts 3-D vessel graph structure from images and adds explicit radius information. It targets complete graph prediction rather than segmentation -> skeletonization -> heuristic pruning.

Reference:
- C. Prabhakar et al., "Vesselformer: Towards Complete 3D Vessel Graph Generation from Images", MIDL 2024, PMLR 227:320-331.
- https://proceedings.mlr.press/v227/prabhakar24a.html

A separate coronary-tree line uses CNN tracking followed by GCN refinement for automatic coronary tree extraction and labeling from CTA.

Reference:
- "Graph neural networks for automatic extraction and labeling of the coronary artery tree in CT angiography", 2024.
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11095121/

These are relevant later if we decide to replace the candidate graph with an image-to-graph model. They are not the immediate experiment because they would change several components at once and destroy the clean ablation of CT evidence versus geometry.

## 3. Why the current Graph-LIRA formulation remains defensible

The literature suggests three broad strategies:

```text
A. prevent disconnections during segmentation
B. force / optimize reconnection after segmentation
C. predict the vessel graph directly
```

Our line is intentionally a fourth formulation:

```text
D. selective structural repair

candidate graph
    -> evidence for relation identity
    -> global compatibility
    -> calibrated uncertainty
    -> repair only when evidence is sufficient
    -> otherwise abstain
```

The distinction is clinically and scientifically useful because a wrong connection at LAD/LCX or a high-degree bifurcation can be more damaging than leaving a gap for manual review.

This is why our key result should not be "all trees are connected" and should not be only Dice / clDice.

## 4. Primary evaluation question

The publication-level question should be:

> At a fixed false structural-repair budget, how many repair-needed scenes can be resolved correctly?

Report at minimum:

- repair-needed exact;
- false structural repair;
- incomplete / abstained;
- coverage;
- false among accepted;
- exact among accepted;
- risk-coverage curve;
- patient-cluster confidence intervals;
- LAD / LCX / high-degree-junction strata.

Segmentation Dice and clDice remain supporting metrics, not the principal outcome for the repair layer.

## 5. External baselines to implement

The next proper comparison matrix should contain methods from different families while keeping the same controlled candidate scenes.

### B0 — no repair

Never connect a gap.

Purpose:
- lower-bound coverage;
- zero forced false links.

### B1 — nearest endpoint

Choose the closest candidate endpoint within the same candidate generator.

Purpose:
- trivial geometric baseline.

### B2 — distance + direction

Score:

`s = -z(distance) + w1 * axis_alignment_1 + w2 * axis_alignment_2 + w3 * tangent_consistency`

Weights selected on validation only.

Purpose:
- reproduces the classical local-geometry family.

### B3 — OGMC-style geometry

Approximate the externally described OGMC decision family with:

- endpoint touching / tangent continuity;
- curvature penalty;
- cycle prohibition;
- global one-to-one matching / priority ordering.

Do not call it a reproduction unless the original code and exact equations are used. Label it **OGMC-style geometric baseline**.

### B4 — CorSegRec-style regularized walk score

Use the same candidate graph and combine:

- endpoint distance;
- directional cosine;
- image/centerline probability along the proposed path.

Again, unless exact source code is reproduced, label it **regularized-walk-style baseline**, not CorSegRec reproduction.

### B5 — radial CCTA relation evidence

Current strongest image baseline.

### B6 — geometry + radial CCTA

Current strongest local hybrid.

### B7 — Graph-LIRA geometry-only

Frozen canonical graph baseline.

### B8 — CT-conditioned Graph-LIRA + abstention

Target method.

### B9 — segmentation topology-loss baselines

Only when retraining segmentation:

- Dice/CE;
- Dice/CE + clDice;
- Dice/CE + Skeleton Recall.

These answer a different question and should be reported separately from the post-hoc repair comparison.

## 6. ANZA position

ANZA remains meaningful as a compact local encoder / geometry representation, but only after the expanded patient calibration experiment.

The clean future ablation is:

```text
same patients
same pair/junction scenes
same Graph-LIRA
same validation threshold selection
same tau / consistency gates

radial hand-crafted image features
vs
compact conventional CNN
vs
compact ANZA encoder
```

Do not combine a new ANZA encoder with a new graph policy and a new split in one experiment.

## 7. Transformer / Mamba position

The existing sequence-statistics pilot already showed that a Transformer over aggressively compressed cross-section statistics did not beat radial 2.5-D.

Therefore sequence modeling should be tested only as:

```text
full local cross-section
    -> compact image encoder (CNN or ANZA)
    -> token sequence along vessel
    -> Transformer / Mamba
```

not:

```text
mean/std/max
    -> Transformer
```

Sequence context is aimed at long-range branch consistency, not basic local lumen detection.

## 8. Immediate experiment ordering

Current ordering should remain:

1. finish the 28-patient matched-CCTA calibration experiment;
2. patient-cluster uncertainty and heterogeneity;
3. insert the calibrated radial image evidence into frozen Graph-LIRA;
4. compare risk-coverage against geometry-only Graph-LIRA;
5. add OGMC-style and regularized-walk-style baselines;
6. only then test compact ANZA versus conventional CNN;
7. only then test encoded cross-section sequence context;
8. only after that consider modifying the underlying segmentation loss with Skeleton Recall / clDice / persistent-homology approaches.

This ordering isolates which component actually improves structural correctness.
