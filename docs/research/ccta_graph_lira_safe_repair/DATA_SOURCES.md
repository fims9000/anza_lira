# Data sources for continuing the CCTA / Graph-LIRA research

Snapshot: 2026-09-20.

This file exists so the research can be resumed without reconstructing dataset provenance from chat history.

## A. Original ImageCAS dataset — CT + binary coronary masks

Scientific dataset:

**ImageCAS: A Large-Scale Dataset and Benchmark for Coronary Artery Segmentation Based on Computed Tomography Angiography Images**

Official project repository:

https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT

Official Kaggle download:

https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas

Paper:

https://doi.org/10.1016/j.compmedimag.2023.102287

Convenient Hugging Face mirror used for the exact four exploratory cases:

https://huggingface.co/datasets/jethro682/imagecas

The uploaded/local files were verified by SHA256 against this mirror and have the following exact source identities.

| Local exploratory name | Exact source ID | CT SHA256 | Mask SHA256 |
|---|---|---|---|
| `ct(1).nii.gz` + `coronary_artery(1).nii.gz` | `BDMAP_00015590` | `043e9679675ad4167d6ada17a0b631e70380aef811e11d444c58e71540dd61c5` | `76ca99c7d983eac7be89f8049d5adc46a0043ddf6d3f610c6be0e0c79a5a48f3` |
| `ct(2).nii.gz` + `coronary_artery(2).nii.gz` | `BDMAP_00015593` | `b3171d03d9615a2cc78c2daa4bced85841602a9903814aed6230dacc7bd5f288` | `26342951d73904e4ba4b142d070a08286e4c53274fe10734b25a89e29128abad` |
| `ct.nii.gz` + `coronary_artery.nii.gz` | `BDMAP_00015594` | `193c935a0f9ac21605f61e6f0a284c934d9306b80d56f1d34c68306d1e625c7c` | `1b5f03de1d66fd71f2684720f67f76e0311b392dced0be40f3642cd3bb91061d` |
| `ct(3).nii.gz` + `coronary_artery(3).nii.gz` | `BDMAP_00015597` | `d3464af3836359120663fe123326db15e8731bb3588a8214ec9230f57431b1f9` | `e91ab5efd8293407638d1fb5bc6e21336f26e16e3688acfebf344701a114785a` |

Direct mirror downloads:

### BDMAP_00015590

CT:
https://huggingface.co/datasets/jethro682/imagecas/resolve/main/BDMAP_00015590/ct.nii.gz?download=true

Binary coronary mask:
https://huggingface.co/datasets/jethro682/imagecas/resolve/main/BDMAP_00015590/segmentations/coronary_artery.nii.gz?download=true

### BDMAP_00015593

CT:
https://huggingface.co/datasets/jethro682/imagecas/resolve/main/BDMAP_00015593/ct.nii.gz?download=true

Binary coronary mask:
https://huggingface.co/datasets/jethro682/imagecas/resolve/main/BDMAP_00015593/segmentations/coronary_artery.nii.gz?download=true

### BDMAP_00015594

CT:
https://huggingface.co/datasets/jethro682/imagecas/resolve/main/BDMAP_00015594/ct.nii.gz?download=true

Binary coronary mask:
https://huggingface.co/datasets/jethro682/imagecas/resolve/main/BDMAP_00015594/segmentations/coronary_artery.nii.gz?download=true

### BDMAP_00015597

CT:
https://huggingface.co/datasets/jethro682/imagecas/resolve/main/BDMAP_00015597/ct.nii.gz?download=true

Binary coronary mask:
https://huggingface.co/datasets/jethro682/imagecas/resolve/main/BDMAP_00015597/segmentations/coronary_artery.nii.gz?download=true

## Critical ID correction

The mirror also contains an `ImageCAS_ID.txt` whose row positions happen to place:

- row 953 -> `BDMAP_00015590`;
- row 956 -> `BDMAP_00015593`;
- row 957 -> `BDMAP_00015594`;
- row 960 -> `BDMAP_00015597`.

Earlier exploratory scripts used the row numbers `953/956/957/960` as convenient case labels.

**Those row numbers are not valid evidence that the BDMAP volumes are ImageCAS-X scans 953/956/957/960.**

The attempted pairing of `BDMAP_00015590` with ImageCAS-X anatomical scan `953` was explicitly audited and rejected. Keep the BDMAP source IDs above as the canonical identity of the four image-only pilot cases.

The historical CSV is kept untouched for provenance. The corrected-ID copy is:

`results/ccta_graph_lira_safe_repair/2026-09-20/sequence_cross_patient_bdmap_ids.csv`

## B. ImageCAS-X — anatomical branch labels / centerlines

Repository:

https://github.com/kitbransby/ImageCAS-X

Project / data website:

https://kitbransby.github.io/ImageCAS-X/

ImageCAS-X adds multi-label coronary segment masks, centerlines and surfaces to 800 scans from ImageCAS. It is the source used for the branch-aware / Graph-LIRA experiments on labelled scans such as 921 and 953.

Expected layout from the project:

```text
volumes/<scan_id>.img.nii.gz
segmentations/<scan_id>.coronary.nii.gz
centerlines/<scan_id>.coronary_left_centerline.vtk
centerlines/<scan_id>.coronary_right_centerline.vtk
surfaces/<scan_id>.coronary_mesh.vtk
```

For a real CT + anatomical-label experiment, use an ImageCAS-X scan ID together with the **same original ImageCAS scan**, not a row-indexed BDMAP guess.

## Practical resume rule

Use the four BDMAP cases for image-representation prototyping where only CT + binary lumen masks are required.

Use ImageCAS-X 921 / 953 for branch-aware geometry, pair/junction identity, Graph-LIRA and topology experiments.

Do not merge the two sources at patient level until a true matching original ImageCAS volume for an ImageCAS-X scan ID is obtained and verified geometrically.
