# Pre-registered hard-anatomy targets before expanded CCTA evaluation

Date: 2026-09-21  
Branch: `research/ccta-graph-lira-safe-repair`

This note freezes the anatomical error targets **before** the 28-patient radial CCTA results are available.

The purpose is to prevent post-hoc cherry-picking of whichever vessel names happen to improve after CT is added.

## Baseline used to define the targets

The reference is the frozen 28-case geometry HGB relation baseline at the validation-selected FPR <= 5% operating point.

Held-out test overall:

- recall: 63/167 = 37.72%
- false positives: 2/167 = 1.20%
- precision: 96.92%
- AUROC: 0.9685

Candidate pair top-1 is already 95.21%, so the remaining difficulty is selective relation acceptance rather than candidate generation.

## Positive-branch recall targets

For branches with at least 8 held-out positive relations, geometry-HGB recall is:

| branch | correct / n | recall |
|---|---:|---:|
| OM1 | 1 / 9 | 11.11% |
| IM | 2 / 11 | 18.18% |
| D2 | 2 / 8 | 25.00% |
| LAD | 3 / 11 | 27.27% |
| OM2 | 4 / 13 | 30.77% |
| R-PLA | 12 / 36 | 33.33% |
| LCX | 9 / 21 | 42.86% |
| R-PDA | 7 / 16 | 43.75% |
| RCA | 9 / 19 | 47.37% |
| D1 | 14 / 22 | 63.64% |

The pre-registered **hard positive anatomy set** is therefore:

`OM1, IM, D2, LAD, OM2, R-PLA`

These are the branches with at least 8 examples and baseline recall below the overall 37.72% test recall.

The most clinically / structurally relevant subgroup for the current hypothesis is LAD plus left-system side branches (OM/IM), because geometry can remain locally plausible while image evidence may distinguish whether a proposed corridor actually contains contrast-enhanced lumen.

## False-link targets

The only geometry-HGB negative-pair groups with observed false acceptance are:

- `LCX|OM2`: 1/8 false;
- `R-PDA|RCA`: 1/16 false.

These two pair families are frozen as **observed false-link sentinels**.

In addition, anatomically adjacent branch pairs with enough examples are kept as monitoring strata even if geometry-HGB made zero observed errors:

- `D1|LAD`: 0/20;
- `D2|LAD`: 0/20;
- `LCX|OM1`: 0/24;
- `LCX|LM`: 0/12;
- `R-PLA|RCA`: 0/21.

CT is not considered safer merely because pooled FPR is low. Any new false links in these adjacent-branch strata must be inspected explicitly.

## Evaluation rule after CT is available

For radial CT and geometry+radial CT, report the exact same strata without changing group definitions.

A useful CT result should show one or both of:

1. improved true-relation acceptance on the hard positive anatomy set without increasing pooled or sentinel false-link risk;
2. reduced false acceptance in sentinel adjacent-branch groups at comparable coverage.

Do not create new "interesting" subgroups after seeing CT results unless they are labeled exploratory.

## Why this matters for ANZA

If simple radial CCTA improves the pre-registered hard anatomy groups, then the next ANZA/CNN ablation should be evaluated specifically on whether a learned local encoder adds value beyond that signal.

If radial CCTA fails specifically on these groups, then a richer cross-sectional representation has a concrete target rather than a vague goal of "better features".
