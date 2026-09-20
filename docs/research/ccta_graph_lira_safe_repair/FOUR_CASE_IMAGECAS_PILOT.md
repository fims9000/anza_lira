# Four-case ImageCAS image-context pilot

This note preserves the image-only work that used four CT + binary coronary-mask pairs before the branch-aware Graph-LIRA benchmark was available.

## Exact cases

The files are exact Hugging Face mirror objects from `jethro682/imagecas`:

- `BDMAP_00015590`
- `BDMAP_00015593`
- `BDMAP_00015594`
- `BDMAP_00015597`

See `DATA_SOURCES.md` for official ImageCAS links, direct mirror downloads and hashes.

## Important legacy-label correction

The exploratory scripts called these cases `953`, `956`, `957`, `960` because those were row positions in the mirror's `ImageCAS_ID.txt`.

Correct source mapping:

| legacy script label | canonical source ID |
|---:|---|
| 953 | BDMAP_00015590 |
| 956 | BDMAP_00015593 |
| 957 | BDMAP_00015594 |
| 960 | BDMAP_00015597 |

These labels must **not** be interpreted as ImageCAS-X patient IDs.

## Work that is worth preserving

### 1. Candidate-pair image representation comparison

The four-case pilot tested branch-disjoint / cross-case candidate representations including:

- explicit geometry;
- 5-plane / stacked 2.5-D context;
- radial multi-plane 2.5-D context;
- candidate-aligned 3-D tube context.

The exploratory cross-patient probe found the image representation signal to be real, with radial 2.5-D and candidate-aligned 3-D tube around mean AUROC `~0.959`, while simpler geometry / stacked 2.5-D were around `~0.95`.

These approximate values are retained only as an exploratory checkpoint because the original exact linear-probe artifact was not archived in this Git branch. They should be re-run before paper use.

### 2. Sequence-context pilot

A second pilot converted each proposed connection into a sequence of compressed cross-section tokens.

Exact archived results:

- sequence CNN: mean AUROC `0.927414`, median `0.937662`;
- sequence Transformer: mean AUROC `0.929432`, median `0.947155`;
- mean FPR around `0.244` for both.

Therefore the experiment did **not** show that a Transformer solves the continuation problem. The likely failure mode is aggressive compression of every cross-section into hand-crafted statistics.

Machine artifacts:

- `results/ccta_graph_lira_safe_repair/2026-09-20/sequence_summary.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/sequence_cross_patient.csv` — historical labels retained
- `results/ccta_graph_lira_safe_repair/2026-09-20/sequence_cross_patient_bdmap_ids.csv` — corrected source IDs

## What this pilot established for the current research direction

Do not discard this work when continuing Graph-LIRA.

It established that:

1. local CT context contains useful pair-level information beyond a single geometric score;
2. preserving 2-D/3-D spatial structure matters;
3. a generic Transformer over strongly compressed tokens is not automatically better;
4. the next sequence model should use full local cross-section image patches -> CNN/ANZA encoder -> token sequence -> temporal/attention aggregation;
5. image evidence should be evaluated inside the same joint Graph-LIRA + uncertainty layer rather than as a separate final system.

## Next image-model experiment once matched anatomical CT exists

Keep the structural optimizer fixed and compare:

```text
geometry-only
vs radial 2.5-D
vs candidate-aligned 3-D tube
vs full cross-section CNN encoder -> sequence model
vs ANZA encoder -> sequence model
```

The target metric is reduction of confident wrong-branch / incomplete decisions in the frozen hard LAD/LCX / high-degree junction strata, with risk-coverage reported after perturbation consistency.
