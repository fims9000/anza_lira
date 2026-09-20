# Scan 953 alignment audit

Date: 2026-09-20.

Status: **BLOCKED for CT + anatomical-label training with the currently available original ImageCAS volume.**

## Inputs

ImageCAS-X anatomical package:

- `953.coronary.nii.gz`
- `953.coronary_left_centerline.vtk`
- `953.coronary_right_centerline.vtk`
- `953.coronary_surface.vtk`

Candidate original ImageCAS package:

- `ct(1).nii.gz`
- `coronary_artery(1).nii.gz`

The external ImageCAS mapping places scan 953 at `BDMAP_00015590`. The local candidate files match the public BDMAP_00015590 LFS SHA256 values:

- CT: `043e9679675ad4167d6ada17a0b631e70380aef811e11d444c58e71540dd61c5`
- binary coronary mask: `76ca99c7d983eac7be89f8049d5adc46a0043ddf6d3f610c6be0e0c79a5a48f3`

## Internal consistency of the ImageCAS-X annotation

The VTK centerlines and `953.coronary.nii.gz` are internally aligned.

After the expected LPS -> RAS conversion, every sampled centerline point is within 1 mm of the multi-label mask:

- left coronary centerline: median distance about `0.126 mm`, 100% within 1 mm;
- right coronary centerline: median distance about `0.121 mm`, 100% within 1 mm.

Therefore the uploaded ImageCAS-X anatomical package itself is coherent.

## Candidate original CT / binary-mask geometry

The files do **not** share a directly usable voxel grid:

- ImageCAS binary mask / CT: `512 x 512 x 221`;
- ImageCAS-X anatomical mask: `512 x 512 x 223`;
- nominal spacing is the same: `0.318359375 x 0.318359375 x 0.5 mm`;
- the sform orientations / origins differ.

The world-coordinate-derived voxel transform from ImageCAS-X mask coordinates to the candidate original mask is approximately:

```text
x_orig = x_x - 34.552
y_orig = y_x + 15.706
z_orig = 208 - z_x
```

A local integer search around that transform failed badly:

- best exact-mask Dice: `0.0176`;
- only `3.65%` of transformed ImageCAS-X mask voxels lie within 1 mm of the candidate binary mask;
- only `2.67%` of candidate original-mask voxels lie within 1 mm of the transformed ImageCAS-X mask.

Centerline-to-candidate-mask coverage is likewise incompatible:

- left centerline weighted coverage within 1 mm: about `2.26%`;
- right centerline weighted coverage within 1 mm: `0%`.

An additional unconstrained rigid ICP check did not rescue the alignment. For the candidate BDMAP_00015590 binary mask, the best sampled surface fit still had:

- median nearest-surface distance about `6.05 mm`;
- p90 about `34.98 mm`;
- only about `9.6%` within 1 mm.

This is far too poor to treat the current CT and ImageCAS-X branch labels as matched observations.

## Consequence

Do **not** train or evaluate a CT + LAD/LCX/D1/... model by indexing `ct(1).nii.gz` with the ImageCAS-X labels.

The evidence supports one of these possibilities:

1. ImageCAS-X uses a different source/reconstruction of scan 953;
2. the anatomical package was generated after a transformation/crop not recoverable from the available NIfTI headers alone;
3. another mapping artifact is required by ImageCAS-X.

The audit does not distinguish these explanations.

## Next data requirement

We need either:

- the CT volume distributed inside the ImageCAS-X package for scan 953, if present; or
- an official transform / source-volume mapping supplied with ImageCAS-X.

Until then, branch-aware geometry experiments on scan 953 are valid, but CT-conditioned experiments using the candidate original ImageCAS volume are not.

Machine outputs:

- `results/ccta_graph_lira_safe_repair/2026-09-20/alignment_953_report.json`
- `results/ccta_graph_lira_safe_repair/2026-09-20/alignment_953_centerline_segment_coverage.csv`
