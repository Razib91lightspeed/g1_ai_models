# Walking obstacle training 1

## Experiment

PPO navigation-policy training in a simplified 2D
kinematic environment using 64 parallel environments.

This is NOT a Unitree G1 Isaac Lab physics training
result and does not establish physical robot performance.

## Training results

- Completed: 2026-11-04T08:00:00.015163+02:00
- Device: cuda
- PPO updates: 23,467
- Completed episodes: 1,482,381
- Successful episodes: 1,393,673
- Recorded collisions: 81,484
- Other episode outcomes: 7,224
- Aggregate training success: 94.02%
- Aggregate collision rate: 5.50%

The overall totals combine curriculum stages 0 through 3.
Stage 3 includes simulated moving obstacles.

Training success is not independent evaluation success.
The model has not yet been validated on unseen scenarios.

## Archive contents

- results/metrics.csv: training history
- results/summary.json: final cumulative results
- results/training.log: console training output
- checkpoints/latest.pt: saved PPO checkpoint
- source/: training script and launcher
- manifest.json: provenance and SHA-256 checksums
- evaluation/: future independent test results

## Reproducibility

Source files are copies of the completed experiment.
The saved checkpoint is preserved without modification.

External runtime dependencies, environment versions,
random-number states and hardware configuration may
still affect exact reproducibility.

## Safety

The archived model is not approved for deployment
to the physical Unitree G1.
