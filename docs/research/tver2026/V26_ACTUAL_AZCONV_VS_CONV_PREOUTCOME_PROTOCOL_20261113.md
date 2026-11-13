# V26: original AZConv2d vs ordinary convolution on autonomous CRACKS pipeline (pre-outcome protocol, 13 Nov 2026)

This experiment is defined **after V23–V25 expert results have been inspected**. All outcomes on the same 40 CRACKS expert sections are exploratory; no blind generalization or geologic fault-instance identity will be claimed. V25 hybrid was 31/44 vs raw 29/44 with false-orange rate 7.108% vs 6.990% and grouped CI includes zero.

## Frozen hypothesis
Replacing the first standard convolution in an otherwise identical small CNN detector with the repository's **actual trainable AZConv2d** may improve the dense predicted mask and hence candidate reachability and topology. Must compare with an ordinary convolution front end trained on **identical crowd-only annotations**; a new learned model is **not** the absent historical ANZA checkpoint, and the deterministic link rule is not the absent historical trained LIRA classifier.

## Architecture and training
- Compare `small_conv`: Conv2d(1,8,3,padding1); `small_az`: original `models.azconv.AZConv2d(1,8,3,num_rules=2,cfg=AZConvConfig(geometry_mode='local_hyperbolic'))`; both followed by identical groupnorm/ReLU + dilated conv 8→8 dilation2 and dilation4 + 1x1 logits. No other architectural changes.
- All training and CAL sections disjoint from expert evaluation: crowd-only TRAIN ids 340,342,...392; crowd-only CAL ids 311,313,...327; preexisting V23 crowd consensus BLUE≥3/ORANGE≥3 only, ignore GREEN and WHITE.
- Crop size 80, batch4, 180 Adam updates, seed 20261113, lr 0.002, gradient clip2, per-patch explicit-class weighted BCE, no per-test optimization. Use CPU portable torch, 2 threads, parameter/log receipts.
- Calibrate each model's threshold from predefined (0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85) on crowd CAL only by max F1 subject to explicit precision≥0.60 (or max F1 if no eligible). Report threshold and crowd CAL curve.
- Original V23 HGB is a *different stronger feature-engineered baseline*; train once with exact V23 procedure and evaluate the same sections and metrics, without changing its calibration rule.

## Complete pipeline and evaluation
- On 40 CRACKS expert sections, get dense probabilities from each trained model, derive masks/trace endpoints and candidate lists without any expert annotation inputs, run raw segmentation, V23 fixed terminal-context links, and frozen V25 `region_hybrid` candidate + NO_LINK + paths.
- **Crucial sequencing:** expert masks are read ONLY after all three networks' image-only inferred probabilities, candidate link decisions and completed masks are calculated for the given section.
- Frozen metrics for each model and policy: recall on expert certain BLUE within 2px, erroneously predicted ORANGE fraction, connected expert-line anchor pairs out of a fixed eligible denominator (40 sections, likely 44 paths), six remaining links with both anchors detected, direct candidate upper bound (evaluation only) and accepted connection count. Section-cluster bootstrap for AZ−ordinary CNN and AZ−HGB. No geological false-pair identity, no natural fault FPR, no second volume. One seed means exploratory evidence of one fit only.
- If AZConv does not improve under the same training budget and same frozen policy, state that explicitly. No tuning new architecture or threshold by consulting the 40 opened expert slices.

## Stop rule
The original scientific claim (superior genuine ANZA–LIRA recovering natural physical fault links) remains **not validated** unless: independent external survey with physical LINK/REJECT/AMBIGUOUS object identities; verified model checkpoints and original operator matches (or new fully retrained and honestly named system); untouched validation split; demonstrated positive statistical improvement under controlled false-link risk.
