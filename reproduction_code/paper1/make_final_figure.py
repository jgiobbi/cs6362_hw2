#!/usr/bin/env python
"""Final report figure for the Zhang et al. (ICLR 2017) Figure 1 reproduction. NO TRAINING, torch-free.

Reads (read-only) existing results and writes only to --out (default final_figure/):
  panel (a): round-1 `results.json` curves (true labels, random labels, shuffled pixels) +
             round-2 random-pixels variants and Gaussian (mean over seeds, 250-step bins)
  panel (b): time to overfit (first epoch with train acc >= 99%, relative to the same seed's p=0), mean +/- std
             over round-2 seeds at the round-1 winning settings; hollow markers = single-seed round-1 runs (p=0.2, 0.6)
  panel (c): test error at the stop-rule epoch, mean +/- std over seeds; same hollow markers
  dashed thin lines = the paper's curves READ BY EYE from its Figure 1 (about +-0.02 in (c), about +-0.1 in (b)); they are
  reference only, clearly labelled, and can be turned off with --no-paper-overlay.

Usage
  python make_final_figure.py
  python make_final_figure.py --r2-dir results_phase3_r2 --round1-dir results_phase3_r1_sweep --baseline-results results.json
"""
import argparse, os, json, math, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SETTING = {"inception": "d0.95_default", "alexnet": "d0.99_default", "mlp": "d0.99_default"}   # round-1 winners
MODELS = ["inception", "alexnet", "mlp"]
STY = {"inception": ("Inception", "k", "s"), "alexnet": ("AlexNet", "g", "o"), "mlp": ("MLP 1x512", "b", "*")}
PS_MAIN, PS_R1_EXTRA = [0.0, 0.4, 0.8, 1.0], [0.2, 0.6]
BIN = 5                                   # 5 x 50-step windows = 250-step bins (same resolution as round-1 panel (a))

# ---- paper curves READ BY EYE from Fig. 1 of the paper (reference only, never used as data) ----
PAPER_P = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
PAPER_B = {"inception": [1.0, 1.9, 2.4, 2.7, 3.1, 3.6], "alexnet": [1.0, 1.1, 1.23, 1.33, 1.42, 1.55],
           "mlp": [1.0, 1.42, 1.68, 1.93, 1.96, 1.98]}
PAPER_C = {"inception": [0.13, 0.40, 0.63, 0.78, 0.88, 0.90], "alexnet": [0.30, 0.52, 0.69, 0.82, 0.88, 0.90],
           "mlp": [0.50, 0.66, 0.76, 0.86, 0.88, 0.90]}


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2-dir", default="results_phase3_r2")
    ap.add_argument("--round1-dir", default="results_phase3_r1_sweep")
    ap.add_argument("--baseline-results", default="results.json")
    ap.add_argument("--out", default="final_figure")
    ap.add_argument("--no-paper-overlay", action="store_true")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- loading
def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def load_seed_runs(r2dir):
    L = {}
    for f in sorted(glob.glob(os.path.join(r2dir, "runs", "seeds__*.json"))):
        r = json.load(open(f)); L[(r["model"], round(r["p"], 1), r["seed"])] = r
    return L


def load_pixel_runs(r2dir):
    P = {}
    for f in sorted(glob.glob(os.path.join(r2dir, "runs", "pixels__*.json"))):
        r = json.load(open(f)); P[(r["variant"], r["seed"])] = r
    return P


def load_round1(r1dir, model, p):
    return load_json(os.path.join(r1dir, "runs", f"{SETTING[model]}__{model}__p{p:.1f}.json"))


# ---------------------------------------------------------------- metrics (same definitions as round 2)
def first_epoch_acc99(run):
    for e, tr in enumerate(run["train_err"], 1):
        if tr <= 0.01 + 1e-12: return e
    return None


def te_at_stop(run):
    se = run.get("stop_epoch")
    return run["test_err"][se - 1] if se else run["test_err"][-1]


def slowdown(base, r):
    """(time relative to base, censored?). Censored -> lower bound epochs_run / base_epoch."""
    if base is None or r is None: return float("nan"), False
    b = first_epoch_acc99(base)
    if b is None: return float("nan"), False
    e = first_epoch_acc99(r)
    return (r["epochs"] / b, True) if e is None else (e / b, False)


def ms(vals):
    v = np.array([x for x in vals if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(v) == 0: return float("nan"), float("nan"), 0
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0), len(v)


def bin_windows(r, k=BIN):
    s, l = np.array(r["win_steps"]), np.array(r["win_loss"])
    n = (len(l) // k) * k
    return s[:n].reshape(-1, k)[:, -1], l[:n].reshape(-1, k).mean(1)


# ---------------------------------------------------------------- panels
def panel_a(ax, baseline, P, data_rows):
    ref = load_json(baseline)
    if ref:
        for key, (lab, c, mk) in {"A|true_labels": ("true labels", "b", "s"), "A|random_labels": ("random labels", "r", "o"),
                                  "A|shuffled_pixels": ("shuffled pixels", "g", "*")}.items():
            if key in ref:
                ax.plot(np.array(ref[key]["log_steps"]) / 1000, ref[key]["log_loss"], color=c, marker=mk, markevery=4, ms=4, label=lab)
                data_rows += [[lab, "round1", s, l] for s, l in zip(ref[key]["log_steps"], ref[key]["log_loss"])]
    else:
        print("WARNING: baseline results.json not found; true/random/shuffled curves skipped")
    variants = [("random_pixels_flat", "random pixels (flattened C*H*W perm.)", "m", None, "-"),
                ("gaussian", "gaussian", "k", "D", "-"),
                ("random_pixels_shared", "random pixels (same perm. across channels)", "m", None, ":")]
    for v, lab, c, mk, ls in variants:
        runs = [P[(vv, s)] for (vv, s) in sorted(P) if vv == v]
        if not runs: continue
        binned = [bin_windows(r) for r in runs]
        n = min(len(b[1]) for b in binned)
        x, y = binned[0][0][:n], np.mean([b[1][:n] for b in binned], 0)
        ax.plot(x / 1000, y, color=c, ls=ls, marker=mk, markevery=4, ms=4, lw=1.4 if ls == "-" else 1.2, alpha=1.0 if ls == "-" else 0.7,
                label=f"{lab} (n={len(runs)})")
        data_rows += [[lab, f"mean of {len(runs)} seeds", int(a), float(b)] for a, b in zip(x, y)]
    ax.set(xlabel="thousand steps", ylabel="average_loss", title="(a) learning curves")
    ax.legend(fontsize=7)


def paper_overlay(ax, table, off):
    if off: return
    for m in MODELS:
        ax.plot(PAPER_P, table[m], color=STY[m][1], ls="--", lw=1, alpha=0.5)
    ax.plot([], [], color="gray", ls="--", lw=1, label="paper (read by eye)")


def panels_bc(ax_b, ax_c, L, r1dir, off, rows):
    seeds = sorted({s for (_, _, s) in L})
    for m in MODELS:
        lab, col, mk = STY[m]
        pb, pc = [], []
        for p in PS_MAIN:
            ss = [s for s in seeds if (m, p, s) in L]
            sl = [slowdown(L.get((m, 0.0, s)), L[(m, p, s)]) for s in ss]
            te = [te_at_stop(L[(m, p, s)]) for s in ss]
            (mb, sb, nb), (mc, sc, nc) = ms([a for a, _ in sl]), ms(te)
            ncens = sum(c for _, c in sl)
            pb.append((p, mb, sb)); pc.append((p, mc, sc))
            rows.append([m, p, nc, mb, sb, mc, sc, ncens])
        ax_b.errorbar([a for a, _, _ in pb], [b for _, b, _ in pb], yerr=[c for _, _, c in pb], color=col, marker=mk, capsize=3,
                      label=f"{lab} ({SETTING[m]}, n={len(seeds)})")
        ax_c.errorbar([a for a, _, _ in pc], [b for _, b, _ in pc], yerr=[c for _, _, c in pc], color=col, marker=mk, capsize=3, label=lab)
        base1 = load_round1(r1dir, m, 0.0)
        extra = [(p, load_round1(r1dir, m, p)) for p in PS_R1_EXTRA]
        extra = [(p, r) for p, r in extra if r is not None and base1 is not None]
        if extra:
            ax_b.plot([p for p, _ in extra], [slowdown(base1, r)[0] for p, r in extra], "o", mfc="none", mec=col, ms=8)
            ax_c.plot([p for p, _ in extra], [te_at_stop(r) for p, r in extra], "o", mfc="none", mec=col, ms=8)
            rows += [[m, p, 1, slowdown(base1, r)[0], "", te_at_stop(r), "", "round1 single seed"] for p, r in extra]
    paper_overlay(ax_b, PAPER_B, off); paper_overlay(ax_c, PAPER_C, off)
    ax_b.plot([], [], "o", mfc="none", mec="gray", label="round 1, single seed")
    ax_c.plot([], [], "o", mfc="none", mec="gray", label="round 1, single seed")
    ax_c.axhline(0.9, color="r", ls="--")
    ax_b.set(xlabel="label corruption", ylabel="time to overfit (acc >= 99%, rel. to p=0)", title="(b) convergence slowdown")
    ax_c.set(xlabel="label corruption", ylabel="test_error", ylim=(0, 1), title="(c) generalization error growth")
    ax_b.legend(fontsize=7); ax_c.legend(fontsize=7, loc="lower right")


def main(argv=None):
    a = parse(argv)
    os.makedirs(a.out, exist_ok=True)
    L, P = load_seed_runs(a.r2_dir), load_pixel_runs(a.r2_dir)
    if not L: raise SystemExit(f"no seed runs found in {a.r2_dir}/runs")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    rows_a, rows_bc = [], []
    panel_a(ax[0], a.baseline_results, P, rows_a)
    panels_bc(ax[1], ax[2], L, a.round1_dir, a.no_paper_overlay, rows_bc)
    seeds = sorted({s for (_, _, s) in L})
    fig.text(0.5, 0.005, f"Ours: Inception (decay 0.95) / AlexNet, MLP (decay 0.99), default init, SGD momentum 0.9, no regularization. (b),(c): mean +/- std over "
             f"{len(seeds)} seeds; hollow = single-seed round-1 runs. Dashed = paper curves read by eye (reference only).",
             ha="center", fontsize=8)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(a.out, f"figure1_final.{ext}"), dpi=220)
    plt.close(fig)
    with open(os.path.join(a.out, "figure1_final_data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel_bc: model", "p", "n_seeds", "slowdown_mean", "slowdown_std", "test_err_mean", "test_err_std", "note"]); w.writerows(rows_bc)
        w.writerow([]); w.writerow(["panel_a: curve", "source", "step", "mean_loss"]); w.writerows(rows_a)
    with open(os.path.join(a.out, "caption.md"), "w") as f:
        f.write(f"""**Figure 1 reproduction (PyTorch).** (a) Training loss of the small Inception on CIFAR10: true labels, random labels and shuffled pixels
(single runs from the Phase 1 baseline), and random-pixel and Gaussian inputs with true labels (mean over {len(sorted({s for (_, s) in P}))} seeds, {BIN * 50}-step bins).
Random pixels are shown for two permutation variants: the same permutation across channels (colour statistics survive) and a permutation of the flattened
C*H*W vector. (b) Time to overfit, defined as the first epoch with train accuracy >= 99%, relative to the same seed's p=0 run. (c) Test error at the epoch where
train error <= 0.1% and online loss < 0.1. (b),(c): mean +/- sample std over {len(seeds)} seeds (Inception: decay 0.95; AlexNet, MLP: decay 0.99; PyTorch default init;
SGD momentum 0.9, lr 0.1 / 0.01, batch 128, no weight decay, dropout or augmentation). Hollow markers are single-seed round-1 runs at p=0.2, 0.6.
Dashed lines are the paper's curves read by eye from its Figure 1 (about +-0.02 in (c), about +-0.1 in (b)); they are for visual reference only.
""")
    print("saved:", ", ".join(sorted(os.listdir(a.out))), "->", a.out)


if __name__ == "__main__":
    main()
