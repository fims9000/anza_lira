# Structural Reachability Phase-A stop (2026-08-18)

- The frozen zero-training probe used 120 descriptor-matched CRACKS crowd
  validation pairs from 73 sections and three completed T1 ANZA seeds.
- A0 foreground probability reached mean TPR 0.169444 at FPR <= 0.05; A4 full
  frozen geometry reached 0.086111.
- The paired section-bootstrap A4-A0 TPR delta was -0.083333 with 95% CI
  [-0.220340, 0.031609]. The predeclared minimum meaningful effect was 0.107152.
- Low-FPR partial AUC also decreased by 0.025000, with CI crossing zero.
- Gate A therefore ended as
  `STOP_ARCHITECTURAL_ANZA_NO_CAUSAL_GEOMETRY_GAIN`. Do not run Phase B,
  capacity-matched heads, new CRACKS training, completion, or expert evaluation
  under this protocol.
- The top-20 prior-classifier confuser subset remains secondary and cannot rescue
  the failed primary gate. Expert data was not accessed and no training ran.
