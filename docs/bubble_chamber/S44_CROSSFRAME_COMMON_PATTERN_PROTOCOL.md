# S44: rigid 2D fiducial alignment and stationary-pattern diagnostic

Status: exploratory, since the two H50Z scans share a physical run and the
source package does not certify that their particle events are independent.
Neither the meaning of film-frame filenames nor native track labels may be
inferred from appearance alone.

Use original 8-bit uncompressed BMP images without downscaling. Fit 005->006
similarity transform using only f1-f6 fiducial point correspondences. Keep the
cross-label f8(005)->f7(006) proximity as a *diagnostic*, not as training
calibration. Fail closed if any fiducial fit has too large residual. Align
source 512x512 crop and exactly verify S39 source pixel SHA-256. Resample a
counterpart region in the other BMP, without staging either source image in Git.

Measure local multiscale Hessian ridge support for every one of 567 native
machine paths and its registered counterpart. Compare to a deterministic
within-window shifted-path control. Use identical support-map thresholds and
2px tolerance in both images. If the global 95th percentile is zero, use the 30th percentile among strictly positive ridge responses (never treat the entire zero map as a ridge). Report known fiducial arm support as positive
registration/structure controls. A high common-pattern response is evidence of
persistence of 2D image structure, not proof that it is a scratch; a repeated
physical line, fiducial, scanner artifact and chance overlap can all produce it.

No machine candidate can be deleted, treated as real ground truth, or used to
auto-join; all results are advisory. Affine coefficients and per-pixel
positions remain private; source code, tests, and aggregated counts only are
eligible for public Git. No physical classification or 3D reconstruction.
