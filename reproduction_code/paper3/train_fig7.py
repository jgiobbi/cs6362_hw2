"""Reproduce Figure 7 A/B of "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
(Higgins et al., ICLR 2017): beta-VAE (beta=4) vs VAE (beta=1) trained on dSprites.

Script version of beta_vae_fig7_AB.ipynb (same code). Assumptions where the paper is silent are listed in that
notebook's first cell. Example:

    python train_fig7.py --dsprites-path /path/to/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz \
        --out-dir fig7_outputs --epochs 30 --batch-size 64 --seeds 0
"""
import os, time, math, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import argparse, matplotlib
matplotlib.use("Agg")   # headless (cluster)
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description="Reproduce Fig. 7 A/B of the beta-VAE paper (beta=4 vs beta=1 on dSprites).")
ap.add_argument("--dsprites-path", required=True, help="path to the dSprites .npz")
ap.add_argument("--out-dir", default="fig7_outputs")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="first seed is the one in the main figure")
ap.add_argument("--print-latent-kl-every", type=int, default=5)
args = ap.parse_args()

DSPRITES_PATH = args.dsprites_path
OUT_DIR = Path(args.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS     = {"beta_vae": 4.0, "vae": 1.0}        # name -> beta
SEEDS      = args.seeds
EPOCHS     = args.epochs
BATCH_SIZE = args.batch_size
LR         = 1e-2                                 # Adagrad, paper Table 1
N_LATENTS, HIDDEN = 10, 1200
ACTIVE_KL_NATS = 0.1                              # a latent counts as "active" above this mean KL
PRINT_LATENT_KL_EVERY = args.print_latent_kl_every
TRAVERSAL_VALUES = [3.0, 1.5, 0.0, -1.5, -3.0]    # top row -> bottom row

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
print(f"device: {DEVICE} | torch {torch.__version__} | epochs {EPOCHS} | batch {BATCH_SIZE} | seeds {SEEDS}")

path = Path(DSPRITES_PATH)
assert path.is_file(), f"dSprites file not found: {path}  (set DSPRITES_PATH)"
t0 = time.time()
with np.load(path, allow_pickle=False) as d:          # only imgs + latents_classes are read (no pickled metadata)
    imgs, classes = d["imgs"], d["latents_classes"]
N = imgs.shape[0]
assert imgs.shape[1:] == (64, 64) and classes.shape == (N, 6), (imgs.shape, classes.shape)
assert imgs.max() <= 1, "expected binary {0,1} images"

# latents_classes columns: [color, shape, scale, orientation, posX, posY]
sizes = tuple(int(s) for s in classes.max(0) + 1)
assert sizes[0] == 1 and sizes[1] == 3 and int(np.prod(sizes)) == N, sizes
if (N, sizes) != (737280, (1, 3, 6, 40, 32, 32)):
    print("WARNING: not the standard dSprites grid:", N, sizes)
flat = np.ravel_multi_index(tuple(classes.T.astype(np.int64)), sizes)      # canonical (C-order) index of every image
assert np.unique(flat).size == N, "dataset is not a complete Cartesian product of the factors"

# Which of posX / posY is the image row, and in which direction? Measure it from image centroids.
rng = np.random.default_rng(0)
sub = rng.choice(N, size=min(4000, N), replace=False)
w = imgs[sub].astype(np.float32)
yy, xx = np.mgrid[0:64, 0:64]
mass = np.maximum(w.sum((1, 2)), 1.0)
cy, cx = (w * yy).sum((1, 2)) / mass, (w * xx).sum((1, 2)) / mass
cs = classes[sub].astype(np.float64)
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
corr_row = {4: corr(cy, cs[:, 4]), 5: corr(cy, cs[:, 5])}
ROW_AX = max(corr_row, key=lambda k: abs(corr_row[k])); COL_AX = 9 - ROW_AX      # 4 = posX, 5 = posY
ROW_SIGN, corr_col = np.sign(corr_row[ROW_AX]), corr(cx, cs[:, COL_AX]); COL_SIGN = np.sign(corr_col)
assert abs(corr_row[ROW_AX]) > 0.9 and abs(corr_col) > 0.9, (corr_row, corr_col)

print(f"loaded {N:,} images, grid sizes {sizes} in {time.time()-t0:.1f}s")
print(f"image row <-> {'posY' if ROW_AX == 5 else 'posX'} (corr {corr_row[ROW_AX]:+.3f}), "
      f"image col <-> {'posX' if COL_AX == 4 else 'posY'} (corr {corr_col:+.3f})")

X_cpu = torch.from_numpy(imgs.reshape(N, -1))        # uint8 (N, 4096), shares memory with `imgs`
X_store = X_cpu
if DEVICE != "cpu":
    try:
        X_store = X_cpu.to(DEVICE)                   # whole dataset on the accelerator (3 GB for dSprites)
    except RuntimeError as e:
        print("could not move data to device, keeping it on CPU:", str(e)[:80])
print("data on:", X_store.device)

class VAE(nn.Module):
    # Paper Table 1 (2D shapes): 4096 -> FC1200 -> FC1200 (ReLU) -> 10 latents -> FC1200 x3 (tanh) -> 4096 (Bernoulli)
    def __init__(self, n_latents=N_LATENTS, hidden=HIDDEN):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4096, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu, self.logvar = nn.Linear(hidden, n_latents), nn.Linear(hidden, n_latents)
        self.dec = nn.Sequential(nn.Linear(n_latents, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 4096))   # Bernoulli logits
    def encode(self, x):
        h = self.enc(x); return self.mu(h), self.logvar(h)
    def decode(self, z):
        return self.dec(z)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar

def vae_terms(logits, x, mu, logvar):
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1)   # (B,)   -log p(x|z), summed over pixels
    kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)                          # (B, K) KL(q(z_j|x) || N(0,1)) per latent
    return recon, kl_dim

def train_model(beta, seed, tag):
    torch.manual_seed(seed); np.random.seed(seed)
    model = VAE().to(DEVICE)
    opt = torch.optim.Adagrad(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)
    hist = []
    for ep in range(1, EPOCHS + 1):
        t0 = time.time(); model.train()
        perm = torch.randperm(N, generator=g).to(X_store.device)
        rec_sum = torch.zeros((), device=DEVICE); kl_sum = torch.zeros(N_LATENTS, device=DEVICE)
        for step, i in enumerate(range(0, N, BATCH_SIZE)):
            xb = X_store[perm[i:i + BATCH_SIZE]].to(DEVICE).float()
            logits, mu, logvar = model(xb)
            recon, kl_dim = vae_terms(logits, xb, mu, logvar)
            loss = (recon + beta * kl_dim.sum(1)).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            rec_sum += recon.detach().sum(); kl_sum += kl_dim.detach().sum(0)
            if step == 200: assert torch.isfinite(loss), "loss is not finite"
        rec, kl_vec = rec_sum.item() / N, (kl_sum / N).cpu().numpy()
        assert np.isfinite(rec) and np.isfinite(kl_vec).all(), "non-finite loss"
        row = dict(epoch=ep, recon=rec, kl=float(kl_vec.sum()), objective=rec + beta * float(kl_vec.sum()),
                   seconds=time.time() - t0, **{f"kl_z{j+1}": float(v) for j, v in enumerate(kl_vec)})
        hist.append(row)
        print(f"[{tag}] epoch {ep:3d}/{EPOCHS} | objective {row['objective']:9.2f} | recon {rec:8.2f} | "
              f"KL {row['kl']:7.3f} | {row['seconds']:6.1f}s")
        if ep % PRINT_LATENT_KL_EVERY == 0 or ep == EPOCHS:
            print("      KL per latent (train-time avg):", np.array2string(kl_vec, precision=2, suppress_small=True, max_line_width=200))
    return model, pd.DataFrame(hist)

@torch.no_grad()
def evaluate_full(model, chunk=8192):
    # Whole dataset, one posterior sample per image (fixed RNG) for the recon term; mu/var kept for the analysis.
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    mus, lvs, rec = [], [], torch.zeros((), device=DEVICE)
    for i in range(0, N, chunk):
        xb = X_store[i:i + chunk].to(DEVICE).float()
        mu, lv = model.encode(xb)
        z = mu + torch.randn(mu.shape, generator=gen).to(DEVICE) * torch.exp(0.5 * lv)
        rec += F.binary_cross_entropy_with_logits(model.decode(z), xb, reduction="sum")
        mus.append(mu.cpu()); lvs.append(lv.cpu())
    mu, lv = torch.cat(mus).numpy(), torch.cat(lvs).numpy()
    kl_dim = (0.5 * (mu ** 2 + np.exp(lv) - lv - 1.0)).mean(0)
    return dict(mu=mu, var=np.exp(lv).mean(0), kl_per_latent=kl_dim, kl=float(kl_dim.sum()), recon=rec.item() / N)

RUNS = {}    # (name, seed) -> dict(model, hist, ev, beta)
for seed in SEEDS:
    for name, beta in MODELS.items():
        print(f"\n===== training {name} (beta={beta:g}), seed {seed}: {EPOCHS} epochs x {math.ceil(N/BATCH_SIZE)} steps =====")
        model, hist = train_model(beta, seed, tag=f"{name} b={beta:g} s={seed}")
        hist.to_csv(OUT_DIR / f"train_log_{name}_seed{seed}.csv", index=False)
        torch.save(model.state_dict(), OUT_DIR / f"{name}_seed{seed}.pt")
        RUNS[(name, seed)] = dict(model=model, hist=hist, ev=evaluate_full(model), beta=beta)

summary = {}
for (name, seed), r in RUNS.items():
    ev, h = r["ev"], r["hist"]
    n_act = int((ev["kl_per_latent"] > ACTIVE_KL_NATS).sum())
    print(f"\n=== {name} (beta={r['beta']:g}), seed {seed} ===")
    print(f"first-epoch recon {h.recon.iloc[0]:.2f} -> final-epoch recon {h.recon.iloc[-1]:.2f}   |   "
          f"first-epoch KL {h.kl.iloc[0]:.3f} -> final-epoch KL {h.kl.iloc[-1]:.3f}")
    print(f"final (full dataset, 1 posterior sample/img):  reconstruction loss = {ev['recon']:.2f} nats/img   total KL = {ev['kl']:.3f} nats/img   "
          f"active latents (KL>{ACTIVE_KL_NATS}) = {n_act}")
    df = pd.DataFrame({"latent": [f"z{j+1}" for j in range(N_LATENTS)], "KL": ev["kl_per_latent"], "learnt_var": ev["var"]})
    print(df.sort_values("learnt_var").round(4).to_string(index=False))
    summary[f"{name}_seed{seed}"] = dict(beta=r["beta"], recon=ev["recon"], kl=ev["kl"], active_latents=n_act,
                                         kl_per_latent=[float(v) for v in ev["kl_per_latent"]],
                                         learnt_var=[float(v) for v in ev["var"]])
json.dump(summary, open(OUT_DIR / "final_summary.json", "w"), indent=2)

SHAPE_NAMES, LINE_COLORS = ["square", "oval", "heart"], ["tab:red", "tab:green", "tab:blue"]   # classes 0/1/2
FACTOR_LABEL = {"shape": "shape", "scale": "scale", "rotation": "rotation", "posX": "pos. X", "posY": "pos. Y"}

# Seed image for traversals: oval, middle scale, orientation index 5/40, centred. Same for both models.
seed_cls = (0, 1, sizes[2] // 2, sizes[3] // 8, sizes[4] // 2, sizes[5] // 2)
seed_file_idx = int(np.where(flat == np.ravel_multi_index(seed_cls, sizes))[0][0])
x_seed = X_store[seed_file_idx][None].to(DEVICE).float()
print("traversal seed image classes [color,shape,scale,orient,posX,posY] =", seed_cls)

def to_image_coords(a):
    # a: (posX, posY, ...) -> (image row, image col, ...) with indices increasing along image axes
    if ROW_AX == 5: a = np.swapaxes(a, 0, 1)
    if ROW_SIGN < 0: a = np.flip(a, 0)
    if COL_SIGN < 0: a = np.flip(a, 1)
    return a

def variance_explained(g):
    # g: (shape, scale, rot, posX, posY, K) posterior means. Share of Var(mu_j) explained by each factor
    # (shape alone; every other factor *beyond* shape, interactions with shape included). Balanced design -> equal weights.
    K = g.shape[-1]
    total = g.reshape(-1, K).var(0) + 1e-12
    v_shape = g.mean(axis=(1, 2, 3, 4)).var(0)
    out = {"shape": v_shape / total}
    for nm, ax in [("scale", 1), ("rotation", 2), ("posX", 3), ("posY", 4)]:
        cell = g.mean(axis=tuple(a for a in range(1, 5) if a != ax))              # (shape, factor, K)
        out[nm] = (cell.reshape(-1, K).var(0) - v_shape) / total
    return pd.DataFrame(out)

@torch.no_grad()
def traverse(model, k):
    model.eval()
    mu0, _ = model.encode(x_seed)
    z = mu0.repeat(len(TRAVERSAL_VALUES), 1)
    z[:, k] = torch.tensor(TRAVERSAL_VALUES, device=z.device)
    return torch.sigmoid(model.decode(z)).reshape(-1, 64, 64).cpu().numpy()

def panel_data(run):
    model, ev, K = run["model"], run["ev"], N_LATENTS
    grid = np.empty((N, K), np.float32); grid[flat] = ev["mu"]
    g = grid.reshape(sizes + (K,))[0]                                   # (shape, scale, rot, posX, posY, K)
    shares = variance_explained(g)
    active = ev["kl_per_latent"] > ACTIVE_KL_NATS
    labels = []
    for k in range(K):
        if not active[k]: labels.append("")
        else:
            f = shares.loc[k].idxmax(); labels.append(FACTOR_LABEL[f] if shares.loc[k, f] > 0.5 else "mixed")
    return dict(heat=to_image_coords(g.mean(axis=(0, 1, 2))),           # (row, col, K)   mean over shape, scale, rot
                scale_curve=g.mean(axis=(2, 3, 4)),                      # (shape, scale, K) mean over rot, positions
                rot_curve=g.mean(axis=(1, 3, 4)),                        # (shape, rot, K)   mean over scale, positions
                trav=np.stack([traverse(model, k) for k in range(K)]),   # (K, n_values, 64, 64)
                var=ev["var"], kl=ev["kl_per_latent"], active=active, labels=labels, shares=shares,
                order=np.argsort(ev["var"]))

PANELS = {key: panel_data(r) for key, r in RUNS.items()}
for (name, seed), P in PANELS.items():
    print(f"\n{name} seed {seed}: share of Var(mu_j) explained by each factor (rows sorted by learnt variance)")
    t = P["shares"].copy(); t.insert(0, "learnt_var", P["var"]); t.insert(1, "KL", P["kl"]); t.insert(2, "label", P["labels"])
    t.index = [f"z{k+1}" for k in range(N_LATENTS)]
    print(t.iloc[P["order"]].round(3).to_string())

def _frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.6)

def draw_panel(sf, letter, title, P, cols):
    nC = len(cols)
    gs = sf.add_gridspec(9, nC, height_ratios=[1, 1, 1, 0.3, 1, 1, 1, 1, 1], hspace=0.07, wspace=0.07,
                         left=0.13, right=0.99, top=0.87, bottom=0.10)
    L_heat = max(float(np.abs(P["heat"][..., cols]).max()), 1e-6)
    L_sc   = 1.1 * max(float(np.abs(P["scale_curve"][..., cols]).max()), 1e-6)
    L_rot  = 1.1 * max(float(np.abs(P["rot_curve"][..., cols]).max()), 1e-6)
    axes = {}
    for c, k in enumerate(cols):
        ttl = f"$z_{{{k+1}}}$" + ("" if P["active"][k] else "\nunused")
        a = sf.add_subplot(gs[0, c]); axes[0, c] = a
        a.imshow(P["heat"][..., k], cmap="jet", vmin=-L_heat, vmax=L_heat, interpolation="nearest"); a.set_title(ttl, fontsize=9); _frame(a)
        for r, (key, L) in enumerate([("scale_curve", L_sc), ("rot_curve", L_rot)], start=1):
            a = sf.add_subplot(gs[r, c]); axes[r, c] = a
            for s_i in range(sizes[1]): a.plot(P[key][s_i, :, k], color=LINE_COLORS[s_i], lw=1)
            a.axhline(0, color="0.85", lw=0.5, zorder=0); a.set_ylim(-L, L); a.margins(x=0); _frame(a)
        for j, v in enumerate(TRAVERSAL_VALUES):
            a = sf.add_subplot(gs[4 + j, c]); axes[4 + j, c] = a
            a.imshow(P["trav"][k, j], cmap="gray", vmin=0, vmax=1); _frame(a)
            if c == 0 and j in (0, len(TRAVERSAL_VALUES) // 2, len(TRAVERSAL_VALUES) - 1):
                a.set_ylabel("0" if v == 0 else f"{v:+g}", rotation=0, ha="right", va="center", fontsize=8)
            if j == len(TRAVERSAL_VALUES) - 1:
                a.set_xlabel(f"{P['var'][k]:.2f}\n{P['labels'][k]}", fontsize=8, labelpad=2)
    for r, nm in enumerate(["position", "scale", "rotation"]): axes[r, 0].set_ylabel(nm, fontsize=8)
    y0, y1 = axes[2, 0].get_position().y0, axes[0, 0].get_position().y1
    sf.text(0.035, (y0 + y1) / 2, "mean latent response", rotation=90, ha="center", va="center", fontsize=9, fontweight="bold")
    y0, y1 = axes[8, 0].get_position().y0, axes[4, 0].get_position().y1
    sf.text(0.035, (y0 + y1) / 2, "single latent mean traversal", rotation=90, ha="center", va="center", fontsize=9, fontweight="bold")
    sf.text(0.035, axes[8, 0].get_position().y0 - 0.05, "learnt\nvariance", ha="center", va="top", fontsize=7)
    sf.suptitle(title, fontsize=12, y=0.985)
    sf.text(0.01, 0.985, letter, fontsize=16, fontweight="bold", va="top")

def make_figure(seed, fname, all_latents=False):
    Pb, Pv = PANELS[("beta_vae", seed)], PANELS[("vae", seed)]
    pick = lambda P: list(P["order"]) if all_latents else list(P["order"][:5]) + [int(P["order"][-1])]
    nC = N_LATENTS if all_latents else 6
    fig = plt.figure(figsize=(2 * (0.95 * nC + 1.0), 7.6))
    sfA, sfB = fig.subfigures(1, 2, wspace=0.02)
    draw_panel(sfA, "A", f"β-VAE (β = {MODELS['beta_vae']:g}), seed {seed}", Pb, pick(Pb))
    draw_panel(sfB, "B", f"VAE (β = {MODELS['vae']:g}), seed {seed}", Pv, pick(Pv))
    fig.text(0.5, 0.005, "rows 2-3 line colours: square = red, oval = green, heart = blue", ha="center", fontsize=9)
    fig.savefig(OUT_DIR / f"{fname}.png", dpi=200); fig.savefig(OUT_DIR / f"{fname}.pdf")
    return fig

fig = make_figure(SEEDS[0], "fig7_AB"); plt.close(fig)

fig_all = make_figure(SEEDS[0], "fig7_AB_all_latents", all_latents=True); plt.close(fig_all)
for s in SEEDS[1:]:
    plt.close(make_figure(s, f"fig7_AB_seed{s}"))
print("saved:", *sorted(p.name for p in OUT_DIR.iterdir()), sep="\n  ")
