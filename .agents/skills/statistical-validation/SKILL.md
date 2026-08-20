---
name: statistical-validation
description: Aggregate GeoCrack metrics by source image and compute reproducible cluster-bootstrap uncertainty.
---

# Statistical Validation

Use master-spec sections 26–27. Aggregate patches within `source_image_id`, then
combine the three seeds. Use 2000 source-level cluster-bootstrap replicates for
AZ-minus-baseline deltas and 95% intervals. Keep missing comparisons incomplete;
if an interval crosses zero, state that the advantage is not established. Build
CSV tables only from stored machine-readable metrics.

For CRACKS, the independent unit is the full seismic `section_id`. Setting A
uses paired section bootstrap across the same 40 expert sections and combines
the three main seeds only after retaining their run identity. Settings B and C
use each fold's held-out section rows; because every expert section is test in
exactly one fold, resample those section rows rather than fold means or pixels.
