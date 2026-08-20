# ANZA-LIRA CRACKS Structural Stability V1 SS0/SS1

This line is separate from and does not rewrite any frozen ANZA/LIRA negative
result. SS0 used the existing local CRACKS extraction without downloads and
froze a rank-based split over 393 common valid nonexpert sections: 220 train,
three 10-section buffers, 40 calibration, 50 development, and 53 confirm.
The split SHA-256 is
`43a3fb7716d5ff9e56c7da9a78f2127c20f8d13ba27d7e5576ac493176045671`.

The 40 expert files were historically evaluated in Setting A. They therefore
cannot support an untouched-expert claim. SS0/SS1 hashed their provenance but
did not decode expert label pixels. White remains unknown with zero supervision
weight. The frozen H0 is historical `t1_unet_s41`, checkpoint SHA-256
`b2a1115981902620f1b731eaee5a0f4dad6393ae714996726bdaba87dcd3e5f9`,
with its historical threshold `0.7`; it was not retrained.

SS1 evaluated all 40 calibration sections under clean plus five frozen
perturbation families at three severities, producing 640 section-condition
rows. All outputs were finite and deterministic, transformed labels retained
the exact CRACKS palette, and every warp satisfied the frozen bounds. Across
the realized warps, determinant ranged from `0.9054647` to `1.1114702` and
maximum condition number was `1.1214253`. Old STOP artifact hashes were
unchanged. The validator reports `SS_S1_PASS`.

Frozen-H0 performance is diagnostic only and was not used to select or remove
families. Mean clean Dice/clDice were `0.7527/0.8036`; severity-3 temporal
bandlimit was the strongest observed stress (`0.6065/0.6370`). This is not an
ANZA mechanism result and no degradation gate was applied.

No B0/B1/B2/B3 model, seed 42/43 replication, LIRA, development, confirm, or
new training was opened in SS0/SS1. SS1 implementation validity authorizes the
separate future SS2/SS3 seed-41 causal comparison, but that comparison has not
been run.
