# V55 full-section ANZA-LIRA segmentation recovery: frozen exploratory protocol

Date: 2026-11-22. Parent source branch: `research/tver-v54-trace-candidate-recall-20261122`.

## Why this experiment

The executed V54 full-section V33 ANZA audit showed that the old model's score field is poorly selective away from its trained expert-selected 96×96 gap windows: lowering the probability threshold can cover most annotated faults only by labelling most of the seismic section as fault. This is a segmentation representation problem, not a solved continuation task or merely a problem of choosing an acceptance threshold.

This study replaces neither the original V33 checkpoint nor any V53/V54 saved result.

## Frozen matched architectural comparison

- Real provider-checked Thebe TRAIN1 images/expert masks only for model fitting. Full VAL1 images/expert masks are **previously inspected exploratory assessment**; no independent TEST7 source is opened. One field with correlated sections is not independent geological generalization.
- Train two instances of existing standard `build_comparable_model("unet", widths=(8,16,32,48))` with the same capacity, optimization, sampled centers, augmentation and loss.
- **2-D** sees the central slice repeated in all three input channels. **2.5-D** sees the immediately preceding, central and following TRAIN/VAL seismic slices in the same spatial coordinates. Train only on center indices 1..98 to avoid cross-split context.
- Fixed `seed=20261122`, `steps=800`, `batch_size=6`, `patch_size=64`, positive-centered patch fraction `0.65`, identical horizontal/vertical flips, AdamW `lr=.001`, weight decay `.0001`, weighted BCE (positive pixel weight `1+14*y`) plus `0.2` soft Dice. Final weights, no CAL checkpoint selection and no threshold tuning to labels.
- Image clipping and normalization reuses original V33 TRAIN-derived 1%/99% amplitude percentiles and mean/std. Do not use per-VAL statistics or expert labels for normalization/inference.
- Full 3174×1537 sections are inferred by overlapping 96×96 tiles with training-only scaling and raised-cosine blending. Pixel output is compared against partial expert annotations only after prediction.
- The predeclared assessment sections are TRAIN1 index 20 and previously inspected VAL1 indices 43 and 77. Both trained methods see the same slices.

## Evaluation and stop rules

The primary diagnostic is **annotated centerline and graph-fragment coverage at equal predicted foreground area budgets** of 1%, 2%, 5% and 10% per section. Also report:
- complete descriptive coverage–foreground curves,
- annotation-adjacent predicted pixel fraction (NOT population precision; unmarked faults may be real),
- number of predicted and reference fragments, and whether predicted segmentation is degenerate,
- seeded repeatability, checkpoint SHA256, TRAIN-only preprocessing receipt, and per-section outputs.

The V33 checkpoint and its source output are historical controls only. Any advantage of 2.5-D over the matched 2-D at equal budget is a within-field exploratory result, not proof of generalization. No TEST7 evaluation, synthetic-positive extrapolation, natural fault-instance precision, 95% recall claim or SOTA claim is permitted from this study alone.

If models produce dense near-constant masks, a failure receipt is a valid outcome. Do not select the most flattering threshold or increase training epochs after looking at VAL. A future independent confirm field and expert link-identity annotation are prerequisites for claims of nearly complete natural fault recovery and safe automatic linkage.

## Methodological boundary

Detecting all plausible trace pixels, deciding which fragments belong to one fault, and reconstructing a path with known correct endpoints are separate tasks. V55 addresses the first. The original V54 generator and V52 path decoder remain separate until the new candidate distribution is evaluated and relation/risk models are fitted using train-only data.
