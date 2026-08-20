# ANZA-FS H3 negative result

The frozen `ANZA_FS_H3_V1` seed-41 experiment is closed with status
`STOP_ANZA_FS_NO_PRACTICAL_STRUCTURAL_GAIN`.

StressBench V6-HARD was frozen before training with 16 hard strata, disjoint
train/calibration/development/confirm RNG streams, and 1024 positive plus 1024
negative structural events across calibration and development. Confirm,
CRACKS, H4, continuation, and expert data stayed closed.

At calibration-selected `BranchRecall >= 0.95`, development results were:

- F1 old GenericAniso: 22/512 false bridges, FBR 0.04297, Dice 0.89054;
- F2 FreeFoliation: 30/512 false bridges, FBR 0.05859, Dice 0.90286;
- F3 ANZA-FS: 32/512 false bridges, FBR 0.06250, Dice 0.88304.

Thus F3/F1 FBR ratio is 1.4545, not the required <=0.70, while Dice delta is
-0.00750, below the -0.005 non-inferiority bound. F3 also fails the causal F2
control: FBR ratio 1.0667, Dice delta -0.01982, and matched-Dice fragmentation
ratio 1.4730. The paired F3-F1 false-bridge delta is +0.01953 with 95% CI
[+0.00781, +0.03125].

The five-lobe foliation computation therefore did not create the intended
anti-bridge shift. Per the frozen stop rule, do not create another local ANZA
kernel family, do not run seeds 42/43, and do not open H4, confirm, CRACKS, or
expert evaluation from this branch.

Canonical evidence is under `results/anza_fs/h3/`; checkpoints remain outside
Git under `_wip_backups/anza_lira/anza_fs_h3_checkpoints`.
