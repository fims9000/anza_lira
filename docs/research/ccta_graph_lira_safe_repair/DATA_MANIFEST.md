# Local data manifest for CCTA Graph-LIRA research

Snapshot date: 2026-09-20.

Raw data are intentionally **not** committed to the public repository. This manifest records the local filenames, byte sizes and SHA256 checksums used by the exploratory work.

## Original ImageCAS CCTA / binary masks available locally

| Local file | Bytes | SHA256 | Working role |
|---|---:|---|---|
| `ct.nii.gz` | 92597375 | `193c935a0f9ac21605f61e6f0a284c934d9306b80d56f1d34c68306d1e625c7c` | case used in earlier four-patient pilot |
| `coronary_artery.nii.gz` | 339388 | `1b5f03de1d66fd71f2684720f67f76e0311b392dced0be40f3642cd3bb91061d` | binary coronary mask |
| `ct(1).nii.gz` | 79447503 | `043e9679675ad4167d6ada17a0b631e70380aef811e11d444c58e71540dd61c5` | candidate CT corresponding to mapped scan 953 / BDMAP_00015590 |
| `coronary_artery(1).nii.gz` | 291709 | `76ca99c7d983eac7be89f8049d5adc46a0043ddf6d3f610c6be0e0c79a5a48f3` | corresponding binary mask candidate |
| `ct(2).nii.gz` | 101790885 | `b3171d03d9615a2cc78c2daa4bced85841602a9903814aed6230dacc7bd5f288` | four-patient pilot |
| `coronary_artery(2).nii.gz` | 362788 | `26342951d73904e4ba4b142d070a08286e4c53274fe10734b25a89e29128abad` | binary mask |
| `ct(3).nii.gz` | 104435887 | `d3464af3836359120663fe123326db15e8731bb3588a8214ec9230f57431b1f9` | four-patient pilot |
| `coronary_artery(3).nii.gz` | 382178 | `e91ab5efd8293407638d1fb5bc6e21336f26e16e3688acfebf344701a114785a` | binary mask |

## ImageCAS-X anatomical annotations available locally

### Scan 921

| Local file | Bytes | SHA256 |
|---|---:|---|
| `921.coronary.nii.gz` | 70678 | `d92a4c7002c2eab75c68783074bc70c19fe290721c305d63ee7cf562ed75fbe2` |
| `921.coronary_left_centerline.vtk` | 42607 | `817f8489fc55ff9132044d8cf281ca46debc341b562e492f567ed77252792ebd` |
| `921.coronary_right_centerline.vtk` | 7816 | `ebfbff568fc8e6c5387c3055a63012a9ea1a207233b15219235b6db5076e1c5d` |
| `921.coronary_surface.vtk` | 4369794 | `d7fe6cf27d7d6ce2c47d8a539a0989a0f367680f55b3d17f65604642a01491d7` |

### Scan 953

| Local file | Bytes | SHA256 |
|---|---:|---|
| `953.coronary.nii.gz` | 78121 | `096f8cdeabea56b1c729ec0dc73b7bc59369d23333a4899d802859ddbb3335a4` |
| `953.coronary_left_centerline.vtk` | 51726 | `feb540978f82f59d871f0daa02c702dbde9302343b7befb7c5ac25e4d154a1b0` |
| `953.coronary_right_centerline.vtk` | 21380 | `07a6e7311e02052c896c5e7ce4810a65f48709b8540ffb1d0a851cfd1d609468` |
| `953.coronary_surface.vtk` | 6484486 | `908f5560be554a34278281f8b8e2cbad1ed990a0d0a918ef7fd2db2960b09ed4` |

The 953 centerlines contain anatomical names including LM, LAD, LCX, D1, OM1, OM2, IM, RCA, R-PDA, R-PLA and `Other`, with explicit branch/start/end markers.

## Split / metadata files

| File | SHA256 |
|---|---|
| `Descriptors.xlsx` | `966d8b198e60e974fbb5a4ea57559479437c3c780fecd05590a397332b72ea34` |
| `train.txt` | `59e94c99e4209a0a9f5d20d9904825f42f7e45fddcbb77851bb40689c90db481` |
| `val.txt` | `8c3aff5d62ad607b52d9904b782606217f2188038cd9b318888c6282b5a373ab` |
| `test.txt` | `451e8ba262a0e1b7f301b8855c52bc787b879375d1f3ed973c74c8851ebd6b05` |
| `exclude.txt` | `2dcff7ee38cad09c19603025cbc2824c63d17839048bd5aba71adfd7c4ac3cc6` |

## Critical alignment issue to resolve before CT + branch-label training

The external mapping file associates scan `953` with `BDMAP_00015590`.

However, the current local files are not directly voxel-identical:

- `953.coronary.nii.gz`: `512 x 512 x 223`, spacing `(0.318359375, 0.318359375, 0.5)`;
- candidate `ct(1).nii.gz`: `512 x 512 x 221`, same nominal spacing;
- their Q/S form transforms have different origins/orientations.

Therefore the branch labels and the CT **must not** be combined by raw voxel index. A registration / coordinate-frame audit is the next required step. The already available binary mask `coronary_artery(1).nii.gz` can be used as an intermediate geometry target to verify the transform.
