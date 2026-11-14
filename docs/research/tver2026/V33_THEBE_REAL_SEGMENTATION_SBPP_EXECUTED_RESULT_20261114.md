# Tver V33 Thebe: real external expert-image segmentation and SBPP topology, executed exploratory stage

## Independent same-protocol repeat shows important nonreproducibility (added after the first V33 report)

Successful second V33 [run 43602262755](https://github.com/fims9000/anza_lira/actions/runs/43602262755), artifact 14117265277, used the same official MD5-checked Thebe sources, identical normalization values, 600 training patch counts (427 positive-centered / 173 background-centered) and model seeds. Yet both original parameter-state file SHA256 and source-level candidate outcomes changed. The second ANZA .65 generated correct predicted SBPP candidates in **6/20 SEG-CAL controlled gaps** (rather than first 14/20), with 13/40 TRAIN gaps (rather than 25/40); second U-Net produced **0/20 SEG-CAL** and 0/40 TRAIN (rather than first 4/20 CAL, 1/40 TRAIN). ANZA remained the better candidate supplier in *these two* exploratory repeats, but the magnitude is unstable.

Second original weight SHA256: ANZA `eb84c75dac85a6953ad847c04bd29e4817566b2676bd502afec822bf4ea60b0b`, U-Net `a299c84be4f8f96c437aaf18d85a9574badeb554a62494e2244f75ff628c5602`. The code was the same for source training; later workflow/diagnostic changes did not intentionally change the training recipe. Source normalization and patch *count* equality do not by themselves prove bitwise patch identity, and the root cause (thread scheduling, backend numerical variance or other unobserved environment factor) remains **unconfirmed**. Do not assert it was caused by MKLDNN without an independent isolation test.

**Revised claim boundary:** the 14/20 vs 4/20 table above is the first-run result, not a reproducible rate estimate. A second run yielded 6/20 vs 0/20. No heldout test or multi-seed confidence interval has established a stable ANZA advantage. V34 deliberately uses exact first-run frozen ANZA .65 weights for a **selected-frontend exploratory** relation study, NOT to promote V33's candidate-score ranking to a confirmed theorem. V35 separately checks bitwise re-training with one CPU thread and MKLDNN disabled on original Thebe TRAIN/SEG-CAL, with full raw tensor and patch hashes.


**2026-11-14, successful GitHub Actions [43601925275](https://github.com/fims9000/anza_lira/actions/runs/43601925275), artifact 14116892059; original source data and all model states SHA-checked.** External official Thebe (Australia) TRAIN1/VAL1 datasets, separate from CRACKS F3. No Thebe TEST7 pixels, model outcomes or labels opened. Original pretrained historical ANZA/LIRA checkpoints not used.

Real source: verified MD5 original seistrain1.npz 534e78ab34bb0ece32ca0a9b69ee99cd; seisval1.npz 518d2979a1f90d535ebc2a44f615970f; faulttrain1.npz 0b393c61283986c82c804172147dcde9; faultval1.npz 22059e5e5155bff0f3e39e142d47c032. Every source array [100,3174,1537], segmentation inputs float32 mmap converted from float64 official amplitudes with TRAIN-only clipping, bool expert labels. No physical instance correspondence across natural disconnections. Synthetic intervention only.

Both segmentation backbones were newly fitted for exactly 600 identical 64x64 TRAIN patches using fixed seeds and equal AdamW steps/optimizer/loss and similar parameter count. Local 12-point artificial expert-selected gap rectangles were erased from the model-predicted probability only, with real observed seismic pixels unchanged. **Source and candidate endpoints were predicted by actual repository SBPP V3-B**, not fed from expert labels. Topology threshold grid fixed 0.35–0.90 in 0.05 increments; selected by TRAIN candidate feasibility and only SEG-CAL original `val1` indices 0,3,...,27 (20 controlled gaps from ten sections), with 10-section VAL buffer 30–39 and future REL-CAL VAL 40–99 untouched.

| V33 quantity | ANZA-v1 real AZConv | matched standard U-Net |
|---|---:|---:|
| Model parameters | 14,194 | 14,481 |
| Last 150 TRAIN weighted BCE+Dice average | 0.76732 | 0.78166 |
| Best feasible topology threshold | **0.65** | **NONE** |
| TRAIN controlled positive gaps at shown comparison threshold | 40 | 40 |
| TRAIN gaps with correct predicted SBPP candidate | 25/40 at 0.65 | 1/40 at 0.60 |
| TRAIN correct predicted candidate PAIRS (may be several/gap) | **71** | **1** |
| SEG-CAL correct-candidate gaps | **14/20** at 0.65 | **4/20** at 0.60 |
| SEG-CAL positive gaps with predicted source ports | 18/20 | 6/20 |
| SEG-CAL predicted candidate proposals (all classes) | 833 | 247 |
| SEG-CAL true candidate PAIRS | 45 | 10 |
| SEG-CAL dense-mask rejects | 2/20 | 13/20 |
| Original policy eligibility for pair-head fitting | PASS | **STOP (<4 positive TRAIN pairs)** |

**Matched case-level diagnostic at above CAL thresholds:** both frontends offered a correct candidate in 3/20 artificial gaps, only ANZA in 11, only U-Net in 1, and neither in 5. These are **related gaps from only ten same-field source sections**, and the threshold for each frontend was chosen on the same SEG-CAL set. Neither paired significance nor independent heldout superiority is claimed.

Official checkpoint SHA256: ANZA ba0c3b8060fd0e3cb1c15f6c1598b94a977677c2b075663b5414207df382c25e; U-Net 93f91fbc24e90b8ed02346fcab3c535f2b4348397d1ae1be9bbd09e400f9ef6b. Exact weights, source normalization and per-case output are saved in artifact 14116892059. The first V33 segmentation run failed only because read-only mmap masks were passed to a writable Cython skeletonizer; a single-frame-copy fix and regression test preceded this successful source run. Historical V27–V30 results remain separate.

**Scientific conclusion:** A new genuine AZConv model can expose candidate continuations on an external geological field while a similarly sized/fit U-Net fails the original minimum TRAIN candidate feasibility gate. This is an exploratory *SEGMENTATION→PREDICTED-CANDIDATE* partial result, not completed ANZA→SBPP→LIRA, because no relation CAL or external TEST7 was run. Claims of general ANZA superiority, safety, physical fault identity or SOTA are unsupported. Continue ANZA-only predicted relation feasibility as a distinctly labeled exploratory follow-up; U-Net remains a fixed failed frontend rather than being relaxed or oracle-augmented.