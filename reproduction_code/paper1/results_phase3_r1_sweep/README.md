# Phase 3, round 1 (sweep): what changed vs. the Phase 1 baseline, and why

Phase 1 (`reproduce_figure1.py`, `results.json`) is untouched. Everything here was produced by `train_v2.py` into this folder.
No values are hardcoded: every number in the tables and figures comes from the per-run JSON files in `runs/`. The only
paper-derived constants are the reference values used for the selection score (see below).

## Problems found in the baseline run
1. It stopped at online loss < 0.1, not at train error ~ 0 (Inception ended at 3-6% train error; paper: 100%).
2. Panel (b) for AlexNet and MLP disagreed with the paper; MLP runs at p >= 0.6 hit the epoch cap.
3. Inception p=0 test error was higher than the paper's Table 1 value.
Hypotheses (not established by the baseline run): the stopping criterion, the x0.95-per-epoch LR decay, and the weight init.

## What this round does
- **Stop rule**: eval-mode train error <= 0.001 AND online epoch-mean loss < 0.1, or the epoch cap
  (150 for decay 0.95, 300 for 0.99 / 0.999). The loss clause is kept so the baseline's loss<0.1 threshold stays computable.
- **Sweep** over LR decay per epoch {0.95, 0.99, 0.999} x init {default, glorot}. Everything else is as in Phase 1: SGD momentum
  0.9, batch 128, lr 0.1 (Inception) / 0.01 (AlexNet, MLP), no weight decay / dropout / augmentation, same data
  preprocessing and the same label-corruption seeds. AlexNet and MLP run all 6 corruption levels in every setting; Inception
  runs only at decay 0.95 (both inits) and p in {0, .4, .8, 1} to save compute, because its baseline slowdown already looked
  paper-like. `d0.95_default` is the control (new stop rule, old decay and init).
- **Init**: `default` is PyTorch's default. `glorot` is Glorot-uniform weights with zero biases on Conv/Linear layers, my guess
  at the TensorFlow default the authors probably used (BatchNorm unchanged). This guess is unverified.
- **Per-epoch logging** of online loss, eval-mode train error, test error, lr, step (`runs/*.json`, `epoch_log.csv`), so every
  panel can be recomputed without retraining.
- **Time to overfit**: first epoch with train accuracy >= 99%, >= 99.9%, or online loss < 0.1, each normalized by the same run
  family's own p=0. Not-reached runs are plotted at a lower bound (epochs run / base) with a red x.
- **Panel (a)** is read from the Phase 1 `results.json` (training for that panel is unchanged), not re-run.

## Selection rule (fixed in code before any results existed)
For AlexNet and MLP in each complete setting, score = mean over the two models of
`|p=0 test err - paper Table 1| / 5 percentage points  +  |ln(p=1 slowdown at acc99 / paper slowdown)| / ln(1.25)`.
For a not-reached (censored) p=1 run only `max(0, ln(bound / paper))` is used, since a lower bound can prove over-slowness
but not under-slowness. The paper slowdowns (Inception 3.6, AlexNet 1.5, MLP 2.0) are my **approximate readings by eye from
Fig. 1b**, so they carry real uncertainty, and the unit weights are arbitrary. Inception is excluded from the score because
it only exists at decay 0.95. `sweep_table.md` shows all settings side by side, not just the winner.

**Disclosure:** picking the setting closest to the target figure is tuning to the target. The report should present the whole
sweep and say so.

## Seeds stage
`python train_v2.py --stage seeds` trains seeds [1, 2] for AlexNet and MLP at the winning setting only (model init, data order
and label corruption all follow the seed). `figure_winner_seeds.png` shows mean +/- std over seeds. With n = 3 the std is a
rough indication of noise, not a confidence interval.

## Caveats
- Settings other than the winner have one seed, so small gaps between them may be noise (not measured until the seeds stage).
- The sweep varies decay and init, but not batch size or the stop threshold; (b) is recomputed at three thresholds offline.
- Eval-mode train error uses BatchNorm running statistics for Inception, which can differ slightly from train-mode behaviour.
- Inception is only compared across inits at decay 0.95, so its sensitivity to decay is untested.

## Files
`runs/` per-run JSON | `epoch_log.csv` | `summary.csv`, `summary.md` | `sweep_table.md` (side-by-side + scores) |
`figure1_winner.png`, `figure1_specfaithful.png`, `figure_sweep_bc.png`, `figure_sweep_thresholds.png`,
`figure_winner_seeds.png` (after the seeds stage)
