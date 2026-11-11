# CRACKS v2: real-domain ANZA-LIRA evaluation

Official dataset: https://zenodo.org/records/13926822 (DOI 10.5281/zenodo.13926822). The two ZIP archives together are about 114.4 MB; they belong under `data/cracks/raw/`, never in Git history. Pin and verify their published MD5 checksums. Verify image and expert mask section IDs before constructing interventions.

Protocol: create artificial gaps inside confidently annotated, branch-free expert lines while preserving the original seismic image. Keep the full expert mask and hidden path out of *all* inference inputs. Use spatially blocked splits and distinct training, calibration, and test data. Report candidate availability, accepted true links, false links on truly negative controls, successful recovered paths, and model provenance. Do not use a held-out test to select thresholds or architectures.

The historical ANZA, SBPP V3-B, and LIRA checkpoints are not part of this repository checkout. A newly trained CRACKS pair model is a *different* model; the 6-channel synthetic model is not compatible with the real-domain 8-channel pair representation. If a checkpoint or frozen calibration is missing, issue a STOP receipt and do not publish purported end-to-end metrics. Explicitly distinguish any oracle-mask diagnostic from a fully predicted end-to-end evaluation.

The dataset's expert masks are semantic color annotations, not instance identity labels for naturally disconnected faults. Only locally unambiguous, deliberately masked single lines can have known positive correspondence. Distinct disconnected components must not automatically be treated as negative fault identities.
