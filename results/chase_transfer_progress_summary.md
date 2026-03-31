# CHASE Transfer Progress Summary

Current best direction: `FIVES pretrain -> CHASE fine-tune`.

## Key runs

| Run | Raw Dice | Flips-TTA Dice | Notes |
| --- | ---: | ---: | --- |
| `chase_baseline_dice80_p512_fg07` | 0.7307 | 0.7603 | strongest baseline on CHASE |
| `chase_az_thesis_from_fives_probe_ft20` | 0.7213 | 0.7535 | first transfer win over older thesis runs |
| `chase_az_thesis_from_fives_continue_dice_ft20` | 0.7317 | 0.7547 | first raw thesis run above baseline |
| `chase_az_thesis_from_fives16_probe_ft20` | 0.7270 | 0.7620 | best TTA thesis run, now above baseline |

## Main takeaways

- Short `FIVES` pretraining already gave the first substantial CHASE jump.
- A small CHASE continuation stage improved raw Dice slightly above the baseline.
- Longer `FIVES` pretraining did **not** produce the best raw CHASE score, but it produced the best `flips-TTA` CHASE score.
- Current best thesis run on CHASE under the article-style evaluation protocol is:
  - `results/chase_az_thesis_from_fives16_probe_ft20/metrics.json`
  - `flips-TTA Dice = 0.7619655610`

## Comparison against baseline

- Baseline `flips-TTA Dice`: `0.7603185798`
- Best thesis `flips-TTA Dice`: `0.7619655610`
- Margin: `+0.0016469812`

## Article assets

- Updated CHASE exports:
  - `article_assets/exports_chase_transfer/chase_baseline_dice80_p512_fg07`
  - `article_assets/exports_chase_transfer/chase_az_thesis_from_fives16_probe_ft20`
- Updated figures:
  - `article_assets/final_figures/figure2_chase_examples.drawio`
  - `article_assets/final_figures/figure2_chase_examples.png`
  - `article_assets/final_figures/figure3_chase_xai.drawio`
  - `article_assets/final_figures/figure3_chase_xai.png`

## Best showcase sample

Sample-wise raw comparison (`results/chase_baseline_vs_fives16_samplewise.json`) shows the strongest full-image gain on:

- `Image_11L`
- baseline Dice: `0.6763`
- thesis Dice: `0.7263`
- delta: `+0.0500`
