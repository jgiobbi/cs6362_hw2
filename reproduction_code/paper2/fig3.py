#!/usr/bin/env python
"""
Reproduce Fig. 3 (a) F1, (c) C1, (d) C2 of Keskar et al., ICLR 2017 ("On Large-Batch Training...").

Stages (each a subcommand; all state lives in --out):
  download            fetch MNIST / CIFAR-10 into --data-dir (run on a node with internet)
  train  --net N --reg SB|LB   train one model, resumable (checkpoint every CKPT_EVERY epochs)
  interp --net N      evaluate f(a*x_LB + (1-a)*x_SB) for a in [-1,2] on full train/test sets
  plot                make the 3-panel figure from saved curves
  all                 train (6 runs, sequential) -> interp -> plot

Assumptions (paper is silent): Adam lr=1e-3 (default betas/eps, no decay/schedule), identical for SB/LB;
PyTorch default init; SB and LB share the same x0 (derived from SEED on CPU, so parallel jobs agree);
FIXED epoch budgets (see BUDGET) instead of a plateau rule; pixels in [0,1], no augmentation;
BN running stats are recomputed (train-set pass, dropout off) (a) at the end of training, so endpoint
metrics are not distorted by lagging running stats, and (b) at every interpolation point.
Set SMOKE=1 to run on tiny synthetic data (pipeline test only, writes to smoke_out).
"""
import argparse, copy, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

SMOKE = os.environ.get("SMOKE") == "1"
SEED, LR, SB_BATCH, LB_FRAC = 0, 1e-3, (32 if SMOKE else 256), 0.10
EVAL_EVERY, CKPT_EVERY = (1, 1) if SMOKE else (5, 5)
N_ALPHA = 7 if SMOKE else 61
NET_IDX = {"F1": 0, "C1": 1, "C2": 2}
# fixed epoch budgets; LB gets more because it makes 25-50x fewer updates per epoch
BUDGET = {("F1", "SB"): 100, ("F1", "LB"): 100,
          ("C1", "SB"): 150, ("C1", "LB"): 300,
          ("C2", "SB"): 100, ("C2", "LB"): 300}
if SMOKE: BUDGET = {k: 2 for k in BUDGET}
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------- data
def load_data(name, data_dir):
    if SMOKE:
        g = torch.Generator().manual_seed(123)
        shape = (1, 28, 28) if name == "F1" else (3, 32, 32)
        W = torch.randn(int(np.prod(shape)), 10, generator=g)
        def mk(n):
            x = torch.rand(n, *shape, generator=g); return x, (x.flatten(1) @ W).argmax(1)
        (xtr, ytr), (xte, yte) = mk(500), mk(200)
    else:
        from torchvision import datasets
        D = datasets.MNIST if name == "F1" else datasets.CIFAR10
        tr, te = D(data_dir, train=True, download=True), D(data_dir, train=False, download=True)
        if name == "F1": f = lambda d: (d.data.float().div(255).unsqueeze(1), d.targets.long())
        else: f = lambda d: (torch.from_numpy(d.data).permute(0, 3, 1, 2).float().div(255), torch.tensor(d.targets).long())
        (xtr, ytr), (xte, yte) = f(tr), f(te)
    return [t.to(dev) for t in (xtr, ytr, xte, yte)]


# ---------------------------------------------------------------- models (App. B)
def make_F1():
    L, d = [nn.Flatten()], 784
    for _ in range(5): L += [nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU()]; d = 512
    return nn.Sequential(*L, nn.Linear(512, 10))

def make_C1():
    L, c = [], 3
    for _ in range(2):
        L += [nn.Conv2d(c, 64, 5, stride=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, padding=1)]; c = 64
    L += [nn.Flatten(), nn.Linear(256, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(0.5),
          nn.Linear(384, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.5), nn.Linear(192, 10)]
    return nn.Sequential(*L)

def make_C2():
    L, c = [], 3
    for width, n in [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]:
        for _ in range(n): L += [nn.Conv2d(c, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU()]; c = width
        L.append(nn.MaxPool2d(2))
    L += [nn.Flatten(), nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 10)]
    return nn.Sequential(*L)

MAKE = {"F1": make_F1, "C1": make_C1, "C2": make_C2}

def init_model(name):
    """Shared x0 for SB and LB: built on CPU from a fixed seed so every job reproduces it exactly."""
    torch.manual_seed(SEED + NET_IDX[name]); return MAKE[name]().to(dev)


# ---------------------------------------------------------------- eval / BN
@torch.no_grad()
def evaluate(model, X, Y, bs=1000):
    model.eval(); loss, correct = 0.0, 0
    for i in range(0, len(X), bs):
        out = model(X[i:i+bs]); loss += F.cross_entropy(out, Y[i:i+bs], reduction="sum").item()
        correct += (out.argmax(1) == Y[i:i+bs]).sum().item()
    return loss / len(X), 100.0 * correct / len(X)

@torch.no_grad()
def recompute_bn(model, X, bs=1000):
    """Re-estimate BN running stats (cumulative average) on the training set; dropout stays off."""
    model.eval()
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns: m.reset_running_stats(); m.momentum = None; m.train()
    for i in range(0, len(X), bs): model(X[i:i+bs])
    for m in bns: m.momentum = 0.1; m.eval()


# ---------------------------------------------------------------- train (resumable)
def paths(out, net, reg): return f"{out}/{net}_{reg}.pt", f"{out}/{net}_{reg}_partial.pt"

def train_one(net, reg, out, data_dir, epochs=None, force=False, crash_after=None):
    final_p, part_p = paths(out, net, reg)
    if os.path.exists(final_p) and not force:
        print(f"[{net}|{reg}] already done ({final_p}); use --force to retrain"); return
    Xtr, Ytr, Xte, Yte = load_data(net, data_dir); N = len(Xtr)
    batch = SB_BATCH if reg == "SB" else int(LB_FRAC * N)
    epochs = epochs or BUDGET[(net, reg)]; tag = f"{net}|{reg}"
    model = init_model(net)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    g = torch.Generator().manual_seed(SEED + NET_IDX[net] + (0 if reg == "SB" else 7919))
    torch.manual_seed(SEED + 1000 * NET_IDX[net] + (0 if reg == "SB" else 1))  # dropout RNG
    log, start, elapsed = [], 1, 0.0
    if os.path.exists(part_p) and not force:
        s = torch.load(part_p, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); g.set_state(s["gen"])
        torch.set_rng_state(s["rng_cpu"].cpu())
        if dev == "cuda" and s.get("rng_cuda") is not None: torch.cuda.set_rng_state(s["rng_cuda"].cpu())
        log, start, elapsed = s["log"], s["epoch"] + 1, s["elapsed"]
        print(f"[{tag}] RESUMED from epoch {s['epoch']}", flush=True)
    print(f"[{tag}] N={N} batch={batch} epochs={epochs} steps/epoch={-(-N // batch)} device={dev}", flush=True)
    t0 = time.time() - elapsed
    for ep in range(start, epochs + 1):
        model.train(); perm = torch.randperm(N, generator=g).to(dev); run, seen = 0.0, 0
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            if len(idx) < 2: continue                       # BN needs >1 sample
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item() * len(idx); seen += len(idx)
        rec = {"epoch": ep, "run_train_loss": run / seen}
        if ep % EVAL_EVERY == 0 or ep == epochs:
            rec["train_loss"], rec["train_acc"] = evaluate(model, Xtr, Ytr)
            rec["test_loss"], rec["test_acc"] = evaluate(model, Xte, Yte)
            print(f"[{tag}] ep {ep:3d} | running loss {rec['run_train_loss']:.4f} | train loss {rec['train_loss']:.4f} "
                  f"acc {rec['train_acc']:.2f} | test loss {rec['test_loss']:.4f} acc {rec['test_acc']:.2f} | {time.time()-t0:.0f}s", flush=True)
        log.append(rec)
        if ep % CKPT_EVERY == 0 and ep < epochs:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": g.get_state(), "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state() if dev == "cuda" else None,
                        "log": log, "epoch": ep, "elapsed": time.time() - t0}, part_p + ".tmp")
            os.replace(part_p + ".tmp", part_p)
        if crash_after and ep == crash_after: raise RuntimeError(f"simulated crash after epoch {ep}")
    raw = dict(zip(["train_loss", "train_acc"], evaluate(model, Xtr, Ytr)))
    raw.update(dict(zip(["test_loss", "test_acc"], evaluate(model, Xte, Yte))))
    recompute_bn(model, Xtr)                                  # consistent with interpolation-time BN handling
    final = {"epochs_run": epochs, "batch": batch, "raw_bn": raw}
    final.update(dict(zip(["train_loss", "train_acc"], evaluate(model, Xtr, Ytr))))
    final.update(dict(zip(["test_loss", "test_acc"], evaluate(model, Xte, Yte))))
    print(f"[{tag}] FINAL epochs={epochs} batch={batch}\n"
          f"    running-BN : train acc {raw['train_acc']:.2f}% loss {raw['train_loss']:.4f} | test acc {raw['test_acc']:.2f}% loss {raw['test_loss']:.4f}\n"
          f"    recomputed : train acc {final['train_acc']:.2f}% loss {final['train_loss']:.4f} | test acc {final['test_acc']:.2f}% loss {final['test_loss']:.4f}", flush=True)
    torch.save({"state": model.state_dict(), "log": log, "final": final}, final_p)
    json.dump({"net": net, "reg": reg, "log": log, "final": final}, open(f"{out}/{net}_{reg}_log.json", "w"), indent=1)
    if os.path.exists(part_p): os.remove(part_p)


# ---------------------------------------------------------------- interpolation
def interp_net(net, out, data_dir, n_alpha=None):
    n_alpha = n_alpha or N_ALPHA; alphas = np.linspace(-1, 2, n_alpha)
    S = {r: torch.load(paths(out, net, r)[0], map_location=dev, weights_only=False) for r in ("SB", "LB")}
    sd_s, sd_l = S["SB"]["state"], S["LB"]["state"]
    Xtr, Ytr, Xte, Yte = load_data(net, data_dir); m = MAKE[net]().to(dev)
    res = {k: [] for k in ["train_loss", "train_acc", "test_loss", "test_acc"]}; t0 = time.time()
    for j, a in enumerate(alphas):
        sd = {k: (a * sd_l[k] + (1 - a) * sd_s[k]) if sd_s[k].dtype.is_floating_point else sd_l[k] for k in sd_s}  # f(a*x_l+(1-a)*x_s)
        m.load_state_dict(sd); recompute_bn(m, Xtr)
        for k, v in zip(res, (*evaluate(m, Xtr, Ytr), *evaluate(m, Xte, Yte))): res[k].append(v)
        if j % 10 == 0: print(f"[{net}] alpha {a:+.2f} train loss {res['train_loss'][-1]:.3f} test acc {res['test_acc'][-1]:.2f} ({time.time()-t0:.0f}s)", flush=True)
    res = {k: np.array(v) for k, v in res.items()}
    np.savez(f"{out}/curves_{net}.npz", alpha=alphas, **res)
    for a_t, r in [(0.0, "SB"), (1.0, "LB")]:                 # endpoint sanity check vs. trained models
        i = int(np.argmin(np.abs(alphas - a_t))); f = S[r]["final"]
        print(f"[{net}] endpoint alpha={alphas[i]:.2f} ({r}): path train acc {res['train_acc'][i]:.2f} test acc {res['test_acc'][i]:.2f}"
              f" | trained model train acc {f['train_acc']:.2f} test acc {f['test_acc']:.2f}")
        if abs(res["train_acc"][i] - f["train_acc"]) > 1.0: print("   WARNING: endpoint mismatch >1pt")


# ---------------------------------------------------------------- plot
def plot(out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    panels = [("F1", "(a) $F_1$"), ("C1", "(c) $C_1$"), ("C2", "(d) $C_2$")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, (net, title) in zip(axes, panels):
        d = np.load(f"{out}/curves_{net}.npz"); a = d["alpha"]
        ax.plot(a, d["train_loss"], "b-", label="Train"); ax.plot(a, d["test_loss"], "b--", label="Test")
        ax.set_xlabel("alpha"); ax.set_ylabel("Cross Entropy", color="b"); ax.set_xlim(-1, 2); ax.set_ylim(bottom=0)
        ax.tick_params(axis="y", colors="b"); ax.grid(True, ls=":"); ax.legend(loc="upper left")
        ax2 = ax.twinx(); ax2.plot(a, d["train_acc"], "r-"); ax2.plot(a, d["test_acc"], "r--")
        ax2.set_ylabel("Accuracy", color="r"); ax2.set_ylim(0, 100); ax2.tick_params(axis="y", colors="r"); ax.set_title(title, y=-0.28)
    fig.suptitle("SYNTHETIC DATA - PIPELINE TEST ONLY" if SMOKE else
                 "solid: train, dashed: test; alpha=0 is SB minimizer, alpha=1 is LB minimizer", fontsize=9)
    fig.tight_layout(); fn = f"{out}/figure3_F1_C1_C2"
    fig.savefig(fn + ".png", dpi=200, bbox_inches="tight"); fig.savefig(fn + ".pdf", bbox_inches="tight"); print("saved", fn + ".png/.pdf")


def summary(out, nets):
    print(f"\n{'net':4s} {'reg':3s} {'batch':>6s} {'epochs':>6s} {'train acc':>10s} {'test acc':>9s} {'train loss':>10s} {'test loss':>9s}  (BN recomputed)")
    for n in nets:
        for r in ("SB", "LB"):
            p = paths(out, n, r)[0]
            if os.path.exists(p):
                f = torch.load(p, map_location="cpu", weights_only=False)["final"]
                print(f"{n:4s} {r:3s} {f['batch']:6d} {f['epochs_run']:6d} {f['train_acc']:9.2f}% {f['test_acc']:8.2f}% {f['train_loss']:10.4f} {f['test_loss']:9.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["download", "train", "interp", "plot", "all"])
    ap.add_argument("--net", choices=list(NET_IDX)); ap.add_argument("--reg", choices=["SB", "LB"])
    ap.add_argument("--out", default="smoke_out" if SMOKE else "outputs"); ap.add_argument("--data-dir", default="data")
    ap.add_argument("--epochs", type=int); ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-alpha", type=int); ap.add_argument("--crash-after", type=int, help=argparse.SUPPRESS)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    print(f"device={dev} torch={torch.__version__} SMOKE={SMOKE}")
    if SMOKE: print("*** SYNTHETIC DATA, PIPELINE TEST ONLY ***")
    nets = [a.net] if a.net else list(NET_IDX)
    if a.cmd == "download":
        from torchvision import datasets
        datasets.MNIST(a.data_dir, download=True); datasets.CIFAR10(a.data_dir, download=True)
    elif a.cmd == "train":
        assert a.net and a.reg, "--net and --reg required"
        train_one(a.net, a.reg, a.out, a.data_dir, a.epochs, a.force, a.crash_after)
    elif a.cmd == "interp":
        for n in nets: interp_net(n, a.out, a.data_dir, a.n_alpha)
    elif a.cmd == "plot": plot(a.out)
    else:
        for n in nets:
            for r in ("SB", "LB"): train_one(n, r, a.out, a.data_dir, a.epochs, a.force)
        for n in nets: interp_net(n, a.out, a.data_dir, a.n_alpha)
        plot(a.out); summary(a.out, nets)
