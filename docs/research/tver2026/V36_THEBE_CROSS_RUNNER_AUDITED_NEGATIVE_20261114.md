# V36: cross-runner Thebe model reproducibility — partially repeatable, original V35 reference **NOT** reproduced
**14 November 2026, post-V33 diagnostics, TRAIN1/VAL1 only; external TEST7 unopened.**

Original V35 [CI 43603004021](https://github.com/fims9000/anza_lira/actions/runs/43603004021) compared two consecutive same-seed TRAIN fits for original 14,481-parameter U-Net and 14,194-parameter ANZA, each with 600 deterministic original Thebe patches. Both within-run repeats yielded byte-identical sorted tensor state SHA and identical pixel CAL scores. This proves **within one V35 job only**, not cross-job stability.

V36 explicitly re-executed the same V35 module on two separate later GitHub-hosted runners:
- [CI 43612287488](https://github.com/fims9000/anza_lira/actions/runs/43612287488), independent raw audit status **FAILED** against V35.
- [CI 43612301607](https://github.com/fims9000/anza_lira/actions/runs/43612301607), independent raw audit status **FAILED** against V35.
- Both new V36 runner outputs **exactly agree with each other** on raw model SHA, first/last 150-step losses and all stored SEG-CAL pixel metrics; two repeats within *each* runner also agree. But they differ from V35 on both models. Do **not** characterize this as every independent run being different — V36 runner A and B are identical.

## Cross-run raw tensor outcome (SHA256)

| Experiment/run | U-Net raw state (prefix) | ANZA raw state (prefix) | Same-run 2 repeats |
|---|---|---|---|
| V35 original, 43603004021 | \`bce789e17a5b...\` | \`a2984002f2e4...\` | yes |
| V36 runner A, 43612287488 | \`54bde07a24cc...\` | \`db905dee46d7...\` | yes |
| V36 runner B, 43612301607 | \`54bde07a24cc...\` | \`db905dee46d7...\` | yes |

Both V36 audits report all **9 source/software settings identical** to V35: TRAIN patch SHA256 \`3750ef24fb16a53bf4252b742c2bf16d8217cb74df19f5319d55c5ff2bf7ba1e\`, SEG-CAL patch SHA256 \`8805c37393d7f63145f5fbb39b7bbcff72ea47a200fb2df2e6b0875c9525eab7\`, same train-only normalization, CPU thread count, interop thread count, MKLDNN off, deterministic PyTorch algorithms, torch \`2.5.1+cpu\`, NumPy \`2.4.6\`. In both new runners, the original model state, first+last loss aggregates and full pixel CAL score receipts diverged from V35. V35 -> V36 model differences **are not explainable by different training data or these recorded environment flags**. Other runtime / processor / floating-point dispatch differences remain possible; **cause is unknown**. Further diagnostic evidence would need initialization-only tensor checksums, CPU instruction fingerprints and parameter-state checks after the first optimization step, with a fixed source fixture.

This experiment is a **reproducibility failure of the original V35 reference**, not an assessment of fault-segmentation superiority. It does not re-execute original V33 selection on a new test; no external Thebe test7 access or reference identity was established.

## Scientific consequence
V33 segment topology was unstable (ANZA 14/20 vs 6/20, U-Net 4/20 vs 0/20 across earlier runs). V35 did not eliminate all potential cross-run nondeterminism. V34 trained P0 on **one frozen first V33 checkpoint** and all five methods yielded **zero correctly accepted** relations on 120 CAL positive artificial gaps (strict section-level risk gate). There is no foundation for a general architectural ANZA win, natural geological instance preservation, deployed risk ceiling, or a successful end-to-end reconstruction.

**TEST7 must stay sealed** until a new independently calibrated method achieves nonzero accepted correct-link coverage under its predeclared risk criterion on distinct source sections. Future training fixes require their own protocol and cannot retroactively 'correct' V33/V34 already observed results.
