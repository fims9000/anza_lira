# CHASE Multi-Seed Comparison Summary

Seeds: 41, 42, 43

## Raw test Dice

- Transfer AZ-Thesis (`FIVES16 -> CHASE continuation`): 0.7283 +- 0.0066
- Baseline U-Net: 0.7228 +- 0.0224
- Margin (transfer - baseline): +0.0055

## Flips-TTA test Dice

- Transfer AZ-Thesis: 0.7596 +- 0.0055
- Baseline U-Net: 0.7469 +- 0.0260
- Margin (transfer - baseline): +0.0127

## Notes

- Transfer AZ-Thesis has a higher mean Dice than the baseline in both raw and `flips-TTA` evaluation.
- Transfer AZ-Thesis is markedly more stable than the baseline in raw evaluation on this 3-seed CHASE slice.
- The remaining variance is mostly driven by `seed42`; `seed41` and `seed43` are already tightly clustered after `flips-TTA`.
