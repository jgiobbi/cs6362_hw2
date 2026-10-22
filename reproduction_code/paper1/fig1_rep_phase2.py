#!/usr/bin/env python
"""Reproduce Figure 1 of Zhang et al. (ICLR 2017), "Understanding deep learning requires rethinking generalization".

Same logic as reproduce_figure1.ipynb (see its first cell for the full list of assumptions).
Usage:
    python reproduce_figure1.py --download-only   # on a login node (compute nodes may lack internet)
    python reproduce_figure1.py                   # on a GPU node; resumable via results.json
    python reproduce_figure1.py --plot-only       # re-make the figure from results.json
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--download-only", action="store_true")
parser.add_argument("--plot-only", action="store_true")
parser.add_argument("--allow-cpu", action="store_true", help="override the refusal to run on CPU")
args = parser.parse_args()

import matplotlib
matplotlib.use("Agg")
import os, json, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# ---------------- configuration ----------------
SEED          = 0
BATCH         = 128
MOMENTUM      = 0.9
LR            = {"inception": 0.1, "alexnet": 0.01, "mlp": 0.01}
LR_DECAY      = 0.95          # per epoch
LOSS_THRESH   = 0.1           # convergence criterion on epoch-mean online loss
EPOCHS_A      = 64            # 64 * 391 = 25,024 steps
MAX_EPOCHS_BC = 150           # budget for panels (b),(c); runs not converging are flagged censored
LOG_EVERY     = 250           # steps per logged loss point (panel a)
CORRUPTIONS   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MODELS_BC     = ["inception", "alexnet", "mlp"]
RESULTS_PATH  = "results.json"
FIG_PATH      = "figure1_reproduction.png"

# Pipeline test only (random tensors, tiny). Leave unset for the real experiment.
SMOKE = os.environ.get("REPRO_SMOKE") == "1"
if SMOKE:
    EPOCHS_A, MAX_EPOCHS_BC, LOG_EVERY, CORRUPTIONS = 2, 3, 2, [0.0, 1.0]

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("device:", device, "| SMOKE:", SMOKE, flush=True)
if device == "cpu" and not (SMOKE or args.allow_cpu or args.download_only or args.plot_only):
    raise SystemExit("No GPU found. Refusing to run on CPU (would take days). Use --allow-cpu to override.")

# ---------------- data ----------------
def per_image_standardization(x):
    n = x[0].numel()
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(1.0 / math.sqrt(n))
    return (x - m) / s

def load_cifar10(root="./data"):
    if SMOKE:
        g = torch.Generator().manual_seed(0)
        mk = lambda n: (torch.rand(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g))
        (xtr, ytr), (xte, yte) = mk(512), mk(128)
    else:
        tr = torchvision.datasets.CIFAR10(root, train=True,  download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        f = lambda d: torch.from_numpy(d.data).permute(0, 3, 1, 2).float() / 255.0
        xtr, ytr, xte, yte = f(tr), torch.tensor(tr.targets), f(te), torch.tensor(te.targets)
    crop = lambda x: x[:, :, 2:30, 2:30]               # center 28x28
    return (per_image_standardization(crop(xtr)), ytr,
            per_image_standardization(crop(xte)), yte)

if args.download_only:
    load_cifar10(); print("CIFAR10 downloaded."); raise SystemExit(0)
XTR, YTR, XTE, YTE = (None,) * 4 if args.plot_only else load_cifar10()
if XTR is not None:
    print(XTR.shape, XTE.shape, XTR.mean().item(), XTR.std().item())

def corrupt_labels(y, p, seed):
    g = torch.Generator().manual_seed(seed)
    y = y.clone()
    mask = torch.rand(len(y), generator=g) < p
    y[mask] = torch.randint(0, 10, (int(mask.sum()),), generator=g)
    return y

def randomize_inputs(x, mode, seed):
    # returns a (fixed) transformed copy of x
    g = torch.Generator().manual_seed(seed)
    N, C, H, W = x.shape
    if mode == "shuffled_pixels":                       # one permutation for the whole dataset
        perm = torch.randperm(H * W, generator=g)
        return x.reshape(N, C, H * W)[:, :, perm].reshape(N, C, H, W)
    if mode == "random_pixels":                         # independent permutation per image
        idx = torch.argsort(torch.rand(N, H * W, generator=g), dim=1)[:, None, :].expand(N, C, H * W)
        return torch.gather(x.reshape(N, C, H * W), 2, idx).reshape(N, C, H, W)
    if mode == "gaussian":                              # matching global mean / std
        return torch.randn(x.shape, generator=g) * x.std() + x.mean()
    raise ValueError(mode)

# ---------------- models ----------------
class ConvModule(nn.Sequential):
    def __init__(s, cin, c, k, stride):
        super().__init__(nn.Conv2d(cin, c, k, stride, padding=k // 2), nn.BatchNorm2d(c), nn.ReLU(inplace=True))

class InceptionModule(nn.Module):
    def __init__(s, cin, c1, c3):
        super().__init__()
        s.b1, s.b3 = ConvModule(cin, c1, 1, 1), ConvModule(cin, c3, 3, 1)
    def forward(s, x): return torch.cat([s.b1(x), s.b3(x)], 1)

class DownsampleModule(nn.Module):
    def __init__(s, cin, c3):
        super().__init__()
        s.conv, s.pool = ConvModule(cin, c3, 3, 2), nn.MaxPool2d(3, 2, padding=1)
    def forward(s, x): return torch.cat([s.conv(x), s.pool(x)], 1)

class SmallInception(nn.Module):
    def __init__(s):
        super().__init__()
        L, ch = [ConvModule(3, 96, 3, 1)], 96
        def inc(c1, c3):
            nonlocal ch; L.append(InceptionModule(ch, c1, c3)); ch = c1 + c3
        def down(c3):
            nonlocal ch; L.append(DownsampleModule(ch, c3)); ch += c3
        inc(32, 32); inc(32, 48); down(80)
        inc(112, 48); inc(96, 64); inc(80, 80); inc(48, 96); down(96)
        inc(176, 160); inc(176, 160)
        L += [nn.AvgPool2d(7), nn.Flatten(), nn.Linear(ch, 10)]      # 28 -> 14 -> 7, global mean pool
        s.net = nn.Sequential(*L)
    def forward(s, x): return s.net(x)

def small_alexnet():
    lrn = lambda: nn.LocalResponseNorm(9, alpha=1e-3, beta=0.75, k=1.0)
    return nn.Sequential(
        nn.Conv2d(3, 64, 5, padding=2),  nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, padding=1), lrn(),
        nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, padding=1), lrn(),
        nn.Flatten(), nn.Linear(64 * 7 * 7, 384), nn.ReLU(inplace=True),
        nn.Linear(384, 192), nn.ReLU(inplace=True), nn.Linear(192, 10))

def mlp_1x512():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 28 * 28, 512), nn.ReLU(inplace=True), nn.Linear(512, 10))

def make_model(name):
    return {"inception": SmallInception, "alexnet": small_alexnet, "mlp": mlp_1x512}[name]()

for n in MODELS_BC:
    print(n, sum(p.numel() for p in make_model(n).parameters()))   # paper Table 1: 1,649,402 / 1,387,786 / 1,209,866

# ---------------- training ----------------
@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval(); wrong = 0
    for i in range(0, len(x), bs):
        wrong += (model(x[i:i + bs]).argmax(1) != y[i:i + bs]).sum().item()
    return wrong / len(x)

def train_run(model_name, xtr, ytr, xte, yte, max_epochs, stop_at_loss=None, log_every=None, seed=SEED):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = [t.to(device) for t in (xtr, ytr, xte, yte)]
    model = make_model(model_name).to(device)
    opt   = torch.optim.SGD(model.parameters(), lr=LR[model_name], momentum=MOMENTUM)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=LR_DECAY)
    N, steps = len(xtr), 0
    log_steps, log_loss, win = [], [], torch.zeros((), device=device)
    converged_epoch = steps_to_converge = None
    for ep in range(max_epochs):
        model.train()
        perm, ep_sum = torch.randperm(N, device=device), torch.zeros((), device=device)
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            steps += 1
            ep_sum += loss.detach() * len(idx)
            if log_every:
                win += loss.detach()
                if steps % log_every == 0:
                    log_steps.append(steps); log_loss.append(win.item() / log_every); win.zero_()
        sched.step()
        ep_loss = ep_sum.item() / N
        print(f"  [{model_name}] epoch {ep+1}/{max_epochs} step {steps} loss {ep_loss:.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if stop_at_loss is not None and ep_loss < stop_at_loss:
            converged_epoch, steps_to_converge = ep + 1, steps
            break
    return dict(model=model_name, steps=steps, epochs=ep + 1, converged_epoch=converged_epoch,
                steps_to_converge=steps_to_converge, final_epoch_loss=ep_loss,
                train_err=evaluate(model, xtr, ytr), test_err=evaluate(model, xte, yte),
                log_steps=log_steps, log_loss=log_loss)

def load_results():
    return json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
def save_results(R): json.dump(R, open(RESULTS_PATH, "w"))

# ---------------- panel (a): Inception, 5 settings, fixed length ----------------
SETTINGS_A = ["true_labels", "random_labels", "shuffled_pixels", "random_pixels", "gaussian"]
R = load_results()
for k, setting in enumerate([] if args.plot_only else SETTINGS_A):
    key = f"A|{setting}"
    if key in R: continue
    t0 = time.time()
    xtr, ytr = XTR, YTR
    if setting == "random_labels":
        ytr = corrupt_labels(YTR, 1.0, seed=SEED + 100 + k)
    elif setting != "true_labels":
        xtr = randomize_inputs(XTR, setting, seed=SEED + 100 + k)
    xte = XTE if setting in ("true_labels", "random_labels") else randomize_inputs(XTE, setting, seed=SEED + 200 + k)
    R[key] = train_run("inception", xtr, ytr, xte, YTE, EPOCHS_A, stop_at_loss=None, log_every=LOG_EVERY)
    save_results(R)
    print(f"{key}: final train err {R[key]['train_err']:.4f} | {time.time()-t0:.0f}s", flush=True)

# ---------------- panels (b), (c): 3 models x label corruption ----------------
R = load_results()
for m in MODELS_BC:
    for p in CORRUPTIONS:
        key = f"BC|{m}|{p}"
        if key in R or args.plot_only: continue
        t0 = time.time()
        ytr = YTR if p == 0 else corrupt_labels(YTR, p, seed=SEED + 1000 + int(p * 100))
        R[key] = train_run(m, XTR, ytr, XTE, YTE, MAX_EPOCHS_BC, stop_at_loss=LOSS_THRESH)
        save_results(R)
        r = R[key]
        print(f"{key}: converged_epoch={r['converged_epoch']} steps={r['steps']} train_err={r['train_err']:.4f} "
              f"test_err={r['test_err']:.4f} | {time.time()-t0:.0f}s", flush=True)

# ---------------- plotting ----------------
R = load_results()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# (a)
sty = {"true_labels": ("true labels", "b", "s"), "random_labels": ("random labels", "r", "o"),
       "shuffled_pixels": ("shuffled pixels", "g", "*"), "random_pixels": ("random pixels", "m", None),
       "gaussian": ("gaussian", "k", "D")}
for s, (lab, c, mk) in sty.items():
    r = R[f"A|{s}"]
    ax[0].plot(np.array(r["log_steps"]) / 1000, r["log_loss"], color=c, marker=mk, markevery=4, ms=4, label=lab)
ax[0].set(xlabel="thousand steps", ylabel="average_loss", title="(a) learning curves")
ax[0].legend()

# (b), (c)
msty = {"inception": ("Inception", "k", "s"), "alexnet": ("AlexNet", "g", "o"), "mlp": ("MLP 1x512", "b", "*")}
for m, (lab, c, mk) in msty.items():
    ps = [p for p in CORRUPTIONS if f"BC|{m}|{p}" in R]
    base = R[f"BC|{m}|0.0"]
    base_t = base["steps_to_converge"] or base["steps"]
    rel, cens = [], []
    for p in ps:
        r = R[f"BC|{m}|{p}"]
        cens.append(r["steps_to_converge"] is None)
        rel.append((r["steps_to_converge"] or r["steps"]) / base_t)     # censored -> lower bound
    ax[1].plot(ps, rel, color=c, marker=mk, label=lab)
    for p, v, cz in zip(ps, rel, cens):
        if cz: ax[1].plot([p], [v], "x", color="r", ms=10)
    ax[2].plot(ps, [R[f"BC|{m}|{p}"]["test_err"] for p in ps], color=c, marker=mk, label=lab)
ax[1].set(xlabel="label corruption", ylabel="time to overfit (relative to p=0)", title="(b) convergence slowdown")
ax[1].legend(title="red x = not converged (lower bound)")
ax[2].axhline(0.9, color="r", ls="--")
ax[2].set(xlabel="label corruption", ylabel="test_error", ylim=(0, 1), title="(c) generalization error growth")
ax[2].legend()
plt.tight_layout(); plt.savefig(FIG_PATH, dpi=200)
print("saved", FIG_PATH)

print("\nconvergence summary (b,c):")
for m in MODELS_BC:
    for p in CORRUPTIONS:
        r = R.get(f"BC|{m}|{p}")
        if r: print(f"{m:10s} p={p:.1f} conv_epoch={r['converged_epoch']} train_err={r['train_err']:.4f} test_err={r['test_err']:.4f}")