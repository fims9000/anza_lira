# Scan 953 alignment audit

Date: 2026-09-20.

Status: **the previously downloaded BDMAP candidate is rejected as a matched scan-953 source. CT + anatomical-label training remains blocked until the actual original ImageCAS scan 953 volume is available.**

## Inputs

ImageCAS-X anatomical package:

- `953.coronary.nii.gz`
- `953.coronary_left_centerline.vtk`
- `953.coronary_right_centerline.vtk`
- `953.coronary_surface.vtk`

Previously tested third-party ImageCAS mirror candidate:

- `ct(1).nii.gz`
- `coronary_artery(1).nii.gz`

Those local candidate files match the public `BDMAP_00015590` LFS SHA256 values:

- CT: `043e9679675ad4167d6ada17a0b631e70380aef811e11d444c58e71540dd61c5`
- binary coronary mask: `76ca99c7d983eac7be89f8049d5adc46a0043ddf6d3f610c6be0e0c79a5a48f3`

## Source correction

The assumption that the 953rd row of a third-party BDMAP mirror corresponds to ImageCAS-X scan ID 953 is **not an authoritative ImageCAS-X mapping**.

The official ImageCAS-X repository expects the original CCTA volume for a case as:

```text
volumes/<scan_id>.img.nii.gz
```

and the ImageCAS-X paper/data description states that patient IDs are retained from the original ImageCAS cohort.

Therefore the required source for these labels is the original ImageCAS case with scan ID `953` — operationally `953.img.nii.gz` in the ImageCAS-X benchmark layout — not `BDMAP_00015590` merely because it appeared at row/index 953 in an unrelated mirror list.

Official references:

- https://github.com/kitbransby/ImageCAS-X
- https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas

This changes the interpretation of the failed alignment below: it falsifies our provisional BDMAP indexing shortcut; it does not show that ImageCAS-X requires an unknown nonlinear transform.

## Internal consistency of the ImageCAS-X annotation

The VTK centerlines and `953.coronary.nii.gz` are internally aligned.

After the expected LPS -> RAS conversion, every sampled centerline point is within 1 mm of the multi-label mask:

- left coronary centerline: median distance about `0.126 mm`, 100% within 1 mm;
- right coronary centerline: median distance about `0.121 mm`, 100% within 1 mm.

Therefore the uploaded ImageCAS-X anatomical package itself is coherent.

## Rejected BDMAP candidate

The BDMAP candidate and ImageCAS-X labels do **not** represent a usable matched pair.

The tested files have:

- BDMAP binary mask / CT: `512 x 512 x 221`;
- ImageCAS-X anatomical mask: `512 x 512 x 223`;
- nominal spacing `0.318359375 x 0.318359375 x 0.5 mm`;
- different NIfTI world transforms.

The header-derived candidate mapping followed by a local integer search produced:

- best exact-mask Dice: `0.0176`;
- only `3.65%` of transformed ImageCAS-X vessel voxels within 1 mm of the BDMAP candidate mask;
- only `2.67%` of BDMAP candidate-mask voxels within 1 mm of the transformed ImageCAS-X mask.

Centerline-to-candidate-mask coverage was likewise incompatible:

- left centerline coverage within 1 mm: about `2.26%`;
- right centerline coverage within 1 mm: `0%`.

An additional free rigid ICP surface check did not rescue the match:

- median nearest-surface distance about `6.05 mm`;
- p90 about `34.98 mm`;
- about `9.6%` of sampled surface points within 1 mm.

## Consequence

Do **not** train or evaluate a CT + LAD/LCX/D1/... model by indexing `ct(1).nii.gz` with the ImageCAS-X scan-953 labels.

The valid next data target is the original ImageCAS volume for case `953` itself.

Once `953.img.nii.gz` is available, the alignment check becomes much simpler:

1. load `953.img.nii.gz` and `953.coronary.nii.gz`;
2. compare their image geometry/header;
3. verify the ImageCAS-X centerlines against the multi-label mask;
4. sample HU values only after the matched geometry is confirmed.

## Machine outputs

- `results/ccta_graph_lira_safe_repair/2026-09-20/alignment_953_report.json`
- `results/ccta_graph_lira_safe_repair/2026-09-20/alignment_953_centerline_segment_coverage.csv`
- archived audit source: `scripts/research/ccta_graph_lira_safe_repair/audit_953_alignment_fast.py.gz.b64`
