# Tver V33 — Thebe expert seismic field, real-image TRAIN/CAL protocol

**Frozen 14 November 2026 before any Thebe TEST data or model outcome is opened.** V31 on CRACKS F3 failed source feasibility (22 TRAIN eligible, 0 CAL); do not tune its already seen TEST. Thebe is a different geological field, but all its files belong to one Thebe survey, so its sections may be correlated.

## Official source

Harvard Dataverse DOI doi:10.7910/DVN/YBYGBK, Thebe Gas Field (Australia). V32 verified exact array dimensions in published archives.
TRAIN: seistrain1.npz ID 4862642 MD5 534e78ab34bb0ece32ca0a9b69ee99cd, faulttrain1.npz ID 4862484 MD5 0b393c61283986c82c804172147dcde9.
VALID: seisval1.npz ID 4863099 MD5 518d2979a1f90d535ebc2a44f615970f, faultval1.npz ID 4862498 MD5 22059e5e5155bff0f3e39e142d47c032.
Reserved untouched TEST: seistest7.npz ID 4863124 MD5 05c004c280b36f6bc6b34723949cc47d; faulttest7.npz ID 4862486 MD5 1491cda6751d2541c8ecf0326e0454b8. DO NOT DOWNLOAD OR DECODE TEST IN V33.
Each TRAIN and VALID image block is float64 [100,3174,1537]; paired expert fault label is bool [100,3174,1537]. All verified with original Dataverse MD5. A fixed non-outcome-selective sample (indices 0,10,...,90) confirmed long unbranched expert paths in TRAIN and VALID; no independent TEST opened.

## Frozen source section partitions

- TRAIN: all 100 sections in official train1.
- Segmentation topology CAL: val1 indices 0..29. If runtime costs require, use fixed every-third subset indices 0,3,..,27 and disclose.
- Exclusion buffer: val1 indices 30..39.
- Relation and risk CAL: val1 indices 40..99 (60 source sections).
- Reserved official TEST block test7 remains unopened until model/segmentation/P0 and CAL thresholds are frozen, to be used in a separate run.

## Reproducible method

1. Extract simple expert centerline segments length >=80px from bool mask. At most two deterministic disjoint traces/section. Hide the middle 12 points from **predicted** segmentation probability only; keep observed amplitude image. Expert endpoints only choose intervention and label/evaluate pairs, NEVER inference features for source-port/target/candidate.
2. Train new genuine AZConv ANZA-v1 and equal-architecture U-Net on exactly the same 600 deterministic real TRAIN 64x64 patches, batch1, AdamW lr .002 weight_decay .0001, masked weighted BCE + 0.2 Dice. Treat unmarked pixels as operational background, not certain natural negatives. Train-only amplitude clip at 1% and 99% from fixed seed 20263377 sample, replicate grayscale to three channels. Same preprocessing for ANZA and U-Net. Original historical checkpoints absent.
3. Select segmentation threshold from fixed 0.35..0.90 by steps .05 using only TRAIN candidate counts and topology CAL (val1 indices 0..29), with actual repository SBPP V3-B hard/soft candidates. Require >=4 correct predicted TRAIN pairs and >=1 topology-CAL correct proposal, else STOP. Do not use TEST source or predicted scores to tune.
4. New relation heads use *predicted TRAIN candidate pairs only*. Fit original five-convolution P0 architecture with independent seeds 11 and 23, 30 epochs, one hard matched negative per true predicted positive. Fit fair matched 12-feature logistic regression, full-TRAIN 12-feature logistic with its larger negative budget explicitly disclosed, and geometry-only baseline.
5. Relation CAL on val1 40..99: choose threshold in {0,.05,...,1,1.01} maximizing correct accepted positives under one-sided 95% Clopper–Pearson *section-level* error upper bounds <=0.1 separately for wrong accepted LINK and synthetic candidate-absent NO_LINK acceptance. Require >=29 CAL source sections with candidate-eligible synthetic gaps; else STOP. Adjacent sections within one Thebe survey are correlated, so this binomial assumption is conditional, not a geological natural-world guarantee.
6. Freeze SHA256 of source manifest, AZConv/U-Net, P0 weights, normalization, topology threshold and relation CAL selection. V33 MUST NOT open TEST. All false bridge claims restricted to synthetic worlds.

## Publication boundaries

These are newly trained real-image models and artificial missing-segmentation gaps inside expert annotated lines. Binary expert masks do not encode instance identity across natural disconnections. No claim that historical P0 weights have been recovered, ANZA outperforms U-Net, or a deployed system meets 10% natural geological false-link risk. Full independent TEST to follow only if all fixed gates pass.