# ANZA-KIR IR2 frozen result

## Result

- Frozen K2 remains `STOP_ANZA_KS_FEATURE_NOT_TRANSFERRED`; its source hash was
  unchanged throughout KIR.
- The first base-only hard construction had PairError 0.049 and a bounded
  contrast repair produced 0.096. Both invalid manifests were preserved. The
  final pre-residual construction used the same bottom-20% selector and reached
  exactly 0.100 on 2,000 dev-hard scenes from a 50,000-scene candidate pool.
- IR1 trained one common supervised evidence/orientation/segmentation base.
  IR2 froze that base and gave R0--R3 exactly 7,985 trainable parameters each.
- Seed-41 dev-hard PairError was R0 0.0920, R1 0.0825, R2 0.0835, and R3
  0.0810. R3 improved R0 by 0.011 absolute (about 12% relative), below the
  frozen 30% gate. Against CatRaw the reduction was only about 3%, below 15%.
- R3 preserved natural Dice (0.78852 vs R0 0.78820), but clDice improved only
  0.00024 and fragmentation worsened (0.61442 vs 0.59245).

## Boundary

Final status is `STOP_ANZA_LOCAL_SYMBOLIC_ARCHITECTURE`. Do not add another Cat
matrix, entropy feature, partition, or local convolution family. Do not open
seeds 42/43, controlled unfreezing, confirm, CRACKS, or expert under KIR. The
controlled K1 patch feature result remains valid, but it did not yield the
required incremental dense residual value.
