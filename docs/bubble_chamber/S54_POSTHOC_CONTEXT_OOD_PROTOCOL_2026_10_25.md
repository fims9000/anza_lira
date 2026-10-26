# S54 — post-hoc source-context and OOD audit of S53 simulator scores

**Status:** predeclared before running the S54 audit, but it was motivated by observed native H50Z score saturation in S53. It is therefore a *diagnostic*, NOT an independent confirmatory test.

S53 was fitted on 192x192 native-background patches containing artificial strokes. Its first application to actual H50Z machine lines used whole 512x512 proposal crops, changing the context on which several S43 Hessian-quantile features depend. The first native application also gave wildly different median simulator scores on 005 versus 006. This is a potential covariate/feature-context shift, not evidence of different particle populations.

## Fixed audit

1. For each of 567 unchanged original S39 lines, find a lossless **192x192 pixel crop** entirely inside its 512x512 H50Z tile with at least 12px border to the centerline. No resizing, histogram equalization, re-labeling, or use of a matching photo for segmentation. If a candidate cannot fit, return `UNKNOWN_UNSCORABLE_CONTEXT` without a numerical score, retaining original line and all S40–S52 warnings.
2. Apply the pre-existing S53 logistic models trained on *the opposite* H50Z photo with synthetic training strokes. No refit on the real candidates.
3. From **the original S53 source-photo calibration set only**, establish 99th-percentile max-absolute-standardized-feature novelty threshold for both S43 and S43+S53. Native pixels and their (nonexistent) physics labels are not used in threshold selection. A novelty check is only a rough simulator-support diagnostic, not a statistical guarantee.
4. Quantify how many real hypotheses cannot fit a 192x192 patch, how many exceed source simulator support, and the paired difference between 512- and 192-context *simulator-origin* scores. Never call either score a native track probability. Preserve all graph proposals and refuse physical automatic joins.
5. All outputs with pixel coordinates and visual overlays remain private. Commit only aggregate counts and test code to Git.

The native-photo background is one physical H50Z run; training and evaluation on different photographs from this one run do **not** establish independent physical generalization. No physics accuracy, mass, 3D reconstruction, or native track identification is claimed.
