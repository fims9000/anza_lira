# ANZA method-repair forensic audit

This cycle does not reinterpret or overwrite the frozen deadline result.  The
deadline experiment remains a negative result: the full mode-resolved model did
not improve the three primary structural/segmentation measures.

The implementation audit establishes the following root-cause hypotheses before
new model code is introduced:

1. The frozen v2a path creates `mu * V` states and fuses them as `sum(mu * z)`,
   producing `sum(mu^2) * V` before transport updates.  Frozen v2b adds two
   persistent half states and produces `0.5 * sum(mu^2) * V`.
2. The frozen route readout sums source and destination mode dimensions (and all
   half dimensions for v2b) before branch supervision.  Therefore it cannot
   supervise mode identity.
3. Replay occurs every third real sample with outer weight 0.25.  Its synthetic
   objective contains unit-weight visible segmentation but only 0.2 route loss,
   giving effective per-real-step weights approximately 0.0833 and 0.0167.
4. Persistent positive/negative half states impose a polarity on an axial fault
   trace where `theta` and `theta + pi` are equivalent.
5. `junction_score` is emitted as a diagnostic, but `use_junction` and `use_cone`
   do not alter the frozen operator forward path.
6. The frozen comparable v2 network installs the router in encoder stages 1, 2,
   and 3 even though ambiguity is spatially sparse.

CRACKS officially defines orange (certain no-fault), blue (certain fault), and
green (uncertain fault).  The paper says orange is not used in its experiments
and combines certain/uncertain fault.  Neither the paper nor official repository
establishes a semantic meaning for white pixels.  Therefore white remains
`NOT_ESTABLISHED`; the historical `paper_like` target is preserved as an
explicitly inferred baseline and may only be compared with predeclared
geometry-tolerant sensitivity targets.

Machine evidence is written to `results/method_repair/audit/baseline.json`.
No expert masks, checkpoints, or frozen result artifacts are read by model
selection code in this audit.
