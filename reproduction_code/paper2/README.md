# Reproducing Figure 3 (F1, C1, C2) of "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima" (Keskar et al., ICLR 2017)

PyTorch reproduction of the 1-D parametric interpolation plots between a small-batch (SB) and a large-batch (LB) minimizer:
loss and accuracy of `f(α·x_LB + (1−α)·x_SB)` for α ∈ [−1, 2], on train and test data.

| Panel | Network | Data |
|---|---|---|
| (a) F1 | fully connected, 5×512 + BN | MNIST |
| (c) C1 | shallow conv (AlexNet-like) | CIFAR-10 |
| (d) C2 | deep conv (VGG-like) | CIFAR-10 |

## Repo layout

| Path | Purpose |
|---|---|
| `fig3.py` | Everything: data, models, training, interpolation, plotting (CLI, see below) |
| `fig3_train.sbatch` | SLURM array, 6 jobs = {F1, C1, C2} × {SB, LB}. Edit account/partition/env placeholders |
| `fig3_post.sbatch` | SLURM job: interpolation + plot (run after training) |
| `run_fig3.sbatch` | *[describe: the script actually used on ACCRE]* |
| `outputs/` | Checkpoints (`<net>_<reg>.pt`), per-run logs (`<net>_<reg>_log.json`), curves (`curves_<net>.npz`), figure (`figure3_F1_C1_C2.png/.pdf`) |
| `fig3_reproduction.ipynb` | *[add if included]* notebook that runs the interpolation and plot stages and prints the training summary |

## Usage

```bash
pip install torch torchvision matplotlib numpy
python fig3.py download --data-dir data              # MNIST + CIFAR-10
python fig3.py train --net F1 --reg SB               # one run; repeat for F1/C1/C2 x SB/LB (or: sbatch fig3_train.sbatch)
python fig3.py interp --out outputs --data-dir data  # evaluate the alpha path for all nets
python fig3.py plot --out outputs                    # writes outputs/figure3_F1_C1_C2.png/.pdf
python fig3.py all                                   # all of the above, sequentially
```

- Training is **resumable**: a checkpoint is written every 5 epochs, and rerunning a command continues from it. Finished runs are skipped unless `--force` is passed.
- `SMOKE=1 python fig3.py all` runs on tiny **synthetic** data for 2 epochs. It tests the pipeline only and writes to `smoke_out/`. It is not a reproduction.
- Per-epoch training logs (running loss, full train/test loss and accuracy every 5 epochs) go to stdout and `outputs/*_log.json`. The final train/test accuracy of every SB and LB run is printed at the end of each run.

## Setup and assumptions

The paper fixes: Adam, SB batch 256, LB batch = 10% of the training set, the architectures in App. B, and the α range. It does not fix the following, which are our choices:

| Item | Choice |
|---|---|
| Adam hyperparameters | lr 1e-3, default betas/eps, no decay or schedule, identical for SB and LB |
| Init | PyTorch default; **SB and LB share the same x0** (rebuilt from a seed on CPU) |
| Epochs | Fixed budgets (SB/LB): F1 100/100, C1 150/300, C2 100/300. No early stopping |
| BatchNorm at interpolated weights | Parameters interpolated linearly; BN running stats **recomputed on the training set at every α** (dropout off); evaluation in eval mode |
| BN at end of training | Running stats recomputed on the training set before final metrics, so endpoints are not distorted by lagging stats |
| α grid | 61 points, step 0.05 |
| Data | Pixels in [0,1], no augmentation |
| Layer details | "same" conv padding; C1 pool kernel 3, stride 2, padding 1; dropout drop prob 0.5; output layer has no BN |
| Trials | One SB/LB pair per network (the paper reports 5 trials and plots one pair) |

## Results (final epoch, BN recomputed)

| Net | Train acc SB / LB | Test acc SB / LB |
|---|---|---|
| F1 | 100 / 100 | 98.84 / 98.51 |
| C1 | 99.68 / 99.79 | 69.89 / 65.97 |
| C2 | 99.89 / 100 | 86.96 / 84.78 |

SB generalizes better than LB on all three networks. The C1 and C2 interpolation curves follow the paper's structure. F1 matches up to α≈1; beyond that its tail differs from the paper's (loss peaks near 4 rather than continuing to rise, accuracy partially recovers). We suspect BN handling at extrapolated weights, which is untested here.

## Notes

- Numbers come from a single seed on an NVIDIA RTX A6000 (PyTorch 2.6, CUDA). GPU training is not bitwise deterministic.
- `outputs/` contains the model checkpoints (~130 MB total) so the interpolation and plot stages can be rerun without retraining.
- **Phase 3 snapshot:** this is the version of the code before consulting any external implementation.
