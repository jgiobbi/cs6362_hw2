# β-VAE Figure 7 A/B reproduction (PyTorch)

Reproduces panels A and B of Fig. 7 from Higgins et al., *β-VAE: Learning Basic Visual Concepts with a Constrained
Variational Framework* (ICLR 2017): a β-VAE (β = 4) and a standard VAE (β = 1) trained on the 2D-shapes dataset
(the public dSprites release). Both models are trained from scratch, and every heatmap, curve and image in the figure is
computed from the trained models. Nothing is hard-coded.

## Files

| File | Purpose |
|---|---|
| `beta_vae_fig7_AB.ipynb` | Notebook version: config, data loading, training, analysis, figure. Includes a table of every assumption made where the paper is silent. |
| `train_fig7.py` | Same code as the notebook, as a headless command-line script (for cluster/Slurm runs). |

The script was generated from the notebook's code cells, so the two implement the same model and analysis.

## Requirements

`torch`, `numpy`, `pandas`, `matplotlib` (≥ 3.4, for `fig.subfigures`). The notebook also needs `jupyterlab` and `ipykernel`.
A GPU is strongly recommended. The default run is about 345k steps per model: about 23 s/epoch on an RTX A6000
(≈ 12 min per model), but hours per epoch on a single CPU core.

## Data

Not included. Download the dSprites file `dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz` from DeepMind's public
dSprites dataset repository. The code uses only `imgs` (737,280 × 64 × 64, binary) and `latents_classes`.

## Run

```bash
python train_fig7.py \
    --dsprites-path /path/to/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz \
    --out-dir fig7_outputs --epochs 30 --batch-size 64 --seeds 0
```

Flags: `--dsprites-path` (required), `--out-dir`, `--epochs` (30), `--batch-size` (64), `--seeds` (0; the first seed is
the one in the main figure), `--print-latent-kl-every` (5).

Notebook: set `DSPRITES_PATH` in the first code cell (or as an environment variable; `OUT_DIR`, `EPOCHS`,
`BATCH_SIZE` can also be overridden) and run all cells.

## Outputs (in `--out-dir`)

- `fig7_AB.png/.pdf`: the reproduced Fig. 7 A/B (5 lowest-variance latents + 1 unused latent per model)
- `fig7_AB_all_latents.png/.pdf`: supplementary figure with all 10 latents per model
- `fig7_AB_seed{s}.png/.pdf`: one extra figure per additional seed, if several are given
- `train_log_{model}_seed{s}.csv`: per-epoch objective, reconstruction, KL, and KL per latent
- `final_summary.json`: final full-dataset reconstruction loss, total KL, KL and learnt variance per latent
- `{model}_seed{s}.pt`: model weights (the script does not reload them, so an interrupted run restarts from scratch)

Stdout shows the same per-epoch log, a final summary per model, and a table of how much of each latent's variance is
explained by each generative factor. Use these to confirm that training happened.

## Code overview

1. **Load and check data.** Verifies shapes, binary values, that the data is a complete Cartesian product of the factors,
   and, from image centroids, which of posX/posY is the image row (so heatmaps align with the images).
2. **Model.** `VAE`: 4096 → FC1200 → FC1200 (ReLU) → μ, log σ² (10 each) → FC1200 ×3 (tanh) → 4096 Bernoulli logits
   (paper Table 1). `vae_terms` returns the per-image reconstruction term (Bernoulli NLL summed over pixels) and the
   per-latent analytic KL; the loss is `recon + β · ΣKL`, averaged over the batch.
3. **Training.** `train_model` uses Adagrad (lr 1e-2), reshuffles every epoch, and logs per epoch.
   `evaluate_full` re-encodes the whole dataset to get final metrics, posterior means and the mean posterior variance
   of each latent.
4. **Analysis.** `panel_data` computes, from the posterior means of all 737,280 images:
   - position heatmaps (averaged over shape, scale, rotation),
   - per-shape scale and rotation curves (averaged over the other factors),
   - traversals (the other latents fixed at the posterior mean of a fixed seed image, the traversed latent set to
     +3, +1.5, 0, −1.5, −3),
   - each latent's learnt variance, and a variance-explained label per latent.
5. **Plotting.** `draw_panel` / `make_figure` lay out the two panels, with latents ordered by ascending learnt variance.

## Main assumptions where the paper is silent

Batch size 64, 30 epochs, a single seed, PyTorch default initialisation, summed (not mean) loss, tanh on the decoder's
hidden layers with a linear logit output, "learnt variance" = mean posterior σ² over the dataset, traversal seed image =
centred oval at mid scale, and automatic (not hand-written) column labels. The full table with reasons is in the
notebook's first cell.

## Status and caveats

- The β=4 model learns the structure shown in the paper: one latent each for pos. X, pos. Y and scale, two for rotation,
  and five unused latents at variance ≈ 0.99. The β=1 model is visibly more entangled.
- Numerically it differs from the paper: the learnt variances are much smaller (e.g. position ≈ 0.002 vs 0.08) and the
  reconstructions are sharper. The cause was not tested; no ablations were run.
- One seed per model cannot show how much of the β=4 vs β=1 difference is due to seed variation.
- The mapping of dSprites shape classes 0/1/2 to square/oval/heart (the line colours) was taken from the dataset
  documentation and not checked against the file.
