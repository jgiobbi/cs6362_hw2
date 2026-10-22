# Phase 3, round 2: seed noise, random pixels, past interpolation

Round 1 (`train_v2.py`, `results_phase3_r1_sweep/`, `results.json`) was not modified. This folder was produced by `train_r2.py`,
which imports round-1 models, data pipeline, label corruption, evaluation and constants unchanged. Numbers below are generated from the
per-run JSONs in `runs/`; none are hardcoded. Paper reference values are my readings by eye from Fig. 1 (about +-0.02) and are used only as
dashed lines and a table column.

## What changed vs round 1
- Goal 1 (done before these runs): `param_check.py` showed all four parameter counts match Table 1 (Inception only when BN gamma/beta are
  excluded; PyTorch's total is 3,424 higher). AlexNet: conv 5x5x64 x2, padding 2, max-pool 3x3 stride 2 padding 1, LRN after each pool, flatten 3136.
- Goal 2: seeds [0, 1, 2, 3, 4] at the round-1 winning settings ({'inception': 'd0.95_default', 'alexnet': 'd0.99_default', 'mlp': 'd0.99_default'}), p in [0.0, 0.4, 0.8, 1.0]. Init, data order and label corruption all follow the seed.
  Seed 0 is a fresh re-run, compared to the round-1 JSON to separate GPU nondeterminism from seed noise. Std is the sample std (ddof=1); with
  n = 5 it is a rough noise indication, not a confidence interval.
- Goal 4: for p in [0.4, 0.8], runs continue 50 epochs after the stop rule first triggers (same LR schedule). Goal-2 statistics use the first-trigger
  epoch, so they match the round-1 definition. A second arm (Inception, seed 0) holds the LR at its stop-epoch value for the continuation.
- Goal 3: Inception, true labels, 64 epochs, three input variants, seeds [0, 1]; loss logged every 50 steps. Plus a non-training
  linear probe on per-image channel mean/std. For seed 0, `random_pixels_shared` and `gaussian` use the same input seeds as round-1 panel (a).

## Results at a glance (auto-generated from the run files; interpretation is yours to add)
- Goal 2, inception: p=1 slowdown 3.25 +/- 0.18 (n=5); paper approx. 3.6. p=0.4 test error 0.442 +/- 0.015 vs paper ~0.63.
- Goal 2, alexnet: p=1 slowdown 2.29 +/- 0.15 (n=5); paper approx. 1.5. p=0.4 test error 0.490 +/- 0.007 vs paper ~0.69.
- Goal 2, mlp: p=1 slowdown 1.96 +/- 0.04 (n=5); paper approx. 2.0. p=0.4 test error 0.671 +/- 0.003 vs paper ~0.76.
- Goal 4, seeds inception p=0.4 (n=5): test error change over the next 50 epochs -0.0045 +/- 0.0078 (train err at +50: 0.0000 +/- 0.0000).
- Goal 4, seeds inception p=0.8 (n=5): test error change over the next 50 epochs -0.0006 +/- 0.0030 (train err at +50: 0.0000 +/- 0.0000).
- Goal 4, seeds alexnet p=0.4 (n=5): test error change over the next 50 epochs -0.0118 +/- 0.0041 (train err at +50: 0.0000 +/- 0.0000).
- Goal 4, seeds alexnet p=0.8 (n=5): test error change over the next 50 epochs -0.0050 +/- 0.0037 (train err at +50: 0.0000 +/- 0.0000).
- Goal 4, seeds mlp p=0.4 (n=5): test error change over the next 50 epochs 0.0008 +/- 0.0021 (train err at +50: 0.0000 +/- 0.0000).
- Goal 4, seeds mlp p=0.8 (n=5): test error change over the next 50 epochs -0.0005 +/- 0.0018 (train err at +50: 0.0000 +/- 0.0000).
- Goal 4, lrhold inception p=0.4 (n=1): test error change over the next 50 epochs -0.0234 (train err at +50: 0.0000).
- Goal 4, lrhold inception p=0.8 (n=1): test error change over the next 50 epochs -0.0020 (train err at +50: 0.0000).
- Goal 3, random_pixels_shared: final test error 0.629 (+0.271 below 0.9 (class signal retained)); first-epoch mean loss 1.984 (ln 10 = 2.303).
- Goal 3, random_pixels_flat: final test error 0.855 (+0.045 below 0.9 (class signal retained)); first-epoch mean loss 2.262 (ln 10 = 2.303).
- Goal 3, gaussian: final test error 0.898 (within 0.02 of 0.9 (no usable class signal)); first-epoch mean loss 2.334 (ln 10 = 2.303).
- Goal 3 probe, true_images: test error 0.755 (+0.145 below 0.9 (class signal retained)).
- Goal 3 probe, random_pixels_shared: test error 0.755 (+0.145 below 0.9 (class signal retained)).
- Goal 3 probe, random_pixels_flat: test error 0.894 (within 0.02 of 0.9 (no usable class signal)).
- Goal 3 probe, gaussian: test error 0.897 (within 0.02 of 0.9 (no usable class signal)).

## Still unexplained (static list; edit after reading the numbers)
- The ~1.2x step-scale gap in panel (a) between round 1 and the paper is untouched by this round.
- AlexNet's panel (b) slowdown (round 1: 2.2-3.7x vs ~1.55x in the paper) is only addressed here through the seed-noise estimate.
- The paper's concave panel (c) (round 1 tracked the line e0(1-p) + 0.9p): goal 4 tests only whether later stopping under the same or a held LR changes it.
- Fig. 1c's AlexNet p=0 test error (~0.30 by eye) differs from Table 1 (23.9%); not investigated.

## Files
`runs/` per-run JSON | `epoch_log_labels.csv`, `epoch_log_pixels.csv`, `loss_windows_pixels.csv` | `summary.md` (all tables), `summary_goal2_seeds.csv` |
`fig_goal2_seeds.png`, `fig_goal3_pixels.png`, `fig_goal4_extension.png` | goal 1 is a table (param_check.py output), not a figure
