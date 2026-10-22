#!/usr/bin/env python
"""Phase 3, round 1 (sweep): reproduce Figure 1 of Zhang et al. (ICLR 2017).

Standalone: does NOT import or modify reproduce_figure1.py, and writes only to --out
(default results_phase3_r1_sweep/). The Phase 1 results file is only READ (panel (a)).

Sweep (everything else fixed: SGD momentum 0.9, batch 128, lr 0.1 Inception / 0.01 AlexNet+MLP,
no weight decay / dropout / augmentation):
  LR decay per epoch in {0.95, 0.99, 0.999}  x  init in {default (PyTorch), glorot (TF-style guess)}
  AlexNet, MLP : all 6 corruption levels, every setting
  Inception    : decay 0.95 only, both inits, p in {0, .4, .8, 1.0}
Stop rule: eval-mode train error <= 0.1% AND online epoch-mean loss < 0.1, or the epoch cap.

SELECTION RULE (fixed in code before any results exist; see score_settings): the winner is the setting whose
AlexNet+MLP are closest to the paper on (i) p=0 test error vs Table 1 and (ii) the p=1 slowdown at threshold
acc99. Choosing a setting by closeness to the target figure is tuning to the target; the report must show the
whole sweep, not just the winner.

Stages
  python train_v2.py                     # --stage sweep (default): train every missing run, then aggregate
  python train_v2.py --stage seeds       # extra seeds (1,2) for AlexNet+MLP at the winning setting only
  python train_v2.py --shard i/n         # run every nth job (parallel workers; one JSON per run)
  python train_v2.py --plot-only         # aggregate/plot from saved runs without training
"""
import argparse, os, json, math, time, csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision

# ---------------------------------------------------------------- configuration
SEED, BATCH, MOMENTUM = 0, 128, 0.9
LR = {"inception": 0.1, "alexnet": 0.01, "mlp": 0.01}
STOP_ERR = 0.001      # stop when eval-mode train error <= 0.1% ...
LOSS_THRESH = 0.1     # ... AND online epoch-mean loss < 0.1 (so all 3 thresholds below are reachable)
THRESHOLDS = {"acc99": ("acc", 0.99), "acc99.9": ("acc", 0.999), "loss0.1": ("loss", 0.1)}
PRIMARY = "acc99"
PS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
PS_INC = [0.0, 0.4, 0.8, 1.0]
DECAYS, INITS = [0.95, 0.99, 0.999], ["default", "glorot"]
CAP = {0.95: 150, 0.99: 300, 0.999: 300}
SEEDS_EXTRA = [1, 2]
COST_W = {"inception": 10, "alexnet": 2, "mlp": 1}        # only used to balance --shard

# Paper reference values. Used ONLY for the selection score and tables; never plotted as data.
PAPER_P0_TEST_ERR = {"inception": 1 - 0.8575, "alexnet": 1 - 0.7607, "mlp": 1 - 0.5051}   # Table 1 (no aug, no wd)
PAPER_SLOWDOWN_P1 = {"inception": 3.6, "alexnet": 1.5, "mlp": 2.0}   # APPROXIMATE, read by eye from Fig. 1b
SCORE_ERR_UNIT = 0.05            # 5 percentage points of p=0 test error = 1 score unit (arbitrary scale)
SCORE_SLOW_UNIT = math.log(1.25)  # a 25% slowdown-ratio mismatch = 1 score unit (arbitrary scale)

SETTINGS_A = ["true_labels", "random_labels", "shuffled_pixels", "random_pixels", "gaussian"]
STY_A = {"true_labels": ("true labels", "b", "s"), "random_labels": ("random labels", "r", "o"),
         "shuffled_pixels": ("shuffled pixels", "g", "*"), "random_pixels": ("random pixels", "m", None),
         "gaussian": ("gaussian", "k", "D")}
STY_M = {"inception": ("Inception", "k", "s"), "alexnet": ("AlexNet", "g", "o"), "mlp": ("MLP 1x512", "b", "*")}


def cname(decay, init): return f"d{decay}_{init}"


def build_conditions():
    conds = {}
    for d in DECAYS:
        for i in INITS:
            models = {"alexnet": PS, "mlp": PS}
            if d == 0.95: models = {"inception": PS_INC, **models}
            conds[cname(d, i)] = dict(decay=d, init=i, cap=CAP[d], models=models, seed=SEED)
    return conds


CONDITIONS = build_conditions()
ARGS = None
device = "cpu"


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_phase3_r1_sweep")
    ap.add_argument("--baseline-results", default="results.json",
                    help="Phase 1 results.json, read-only, used only for panel (a)")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--stage", default="sweep", choices=["sweep", "seeds"])
    ap.add_argument("--shard", default="0/1", help="i/n: run jobs with index %% n == i")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny pipeline test on random tensors")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- data (same as Phase 1)
def per_image_standardization(x):
    n = x[0].numel()
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(1.0 / math.sqrt(n))
    return (x - m) / s


def load_cifar10():
    if ARGS.smoke:
        g = torch.Generator().manual_seed(0)
        mk = lambda n: (torch.rand(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g))
        (xtr, ytr), (xte, yte) = mk(512), mk(128)
    else:
        tr = torchvision.datasets.CIFAR10(ARGS.data_root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(ARGS.data_root, train=False, download=True)
        f = lambda d: torch.from_numpy(d.data).permute(0, 3, 1, 2).float() / 255.0
        xtr, ytr, xte, yte = f(tr), torch.tensor(tr.targets), f(te), torch.tensor(te.targets)
    crop = lambda x: x[:, :, 2:30, 2:30]
    return (per_image_standardization(crop(xtr)), ytr, per_image_standardization(crop(xte)), yte)


def corrupt_labels(y, p, seed):
    g = torch.Generator().manual_seed(seed)
    y = y.clone()
    mask = torch.rand(len(y), generator=g) < p
    y[mask] = torch.randint(0, 10, (int(mask.sum()),), generator=g)
    return y


# ---------------------------------------------------------------- models (same as Phase 1) + init option
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
        L += [nn.AvgPool2d(7), nn.Flatten(), nn.Linear(ch, 10)]
        s.net = nn.Sequential(*L)

    def forward(s, x): return s.net(x)


def small_alexnet():
    lrn = lambda: nn.LocalResponseNorm(9, alpha=1e-3, beta=0.75, k=1.0)
    return nn.Sequential(
        nn.Conv2d(3, 64, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, padding=1), lrn(),
        nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, padding=1), lrn(),
        nn.Flatten(), nn.Linear(64 * 7 * 7, 384), nn.ReLU(inplace=True),
        nn.Linear(384, 192), nn.ReLU(inplace=True), nn.Linear(192, 10))


def mlp_1x512():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 28 * 28, 512), nn.ReLU(inplace=True), nn.Linear(512, 10))


def make_model(name, init="default"):
    """init='default': PyTorch defaults (no extra RNG use, identical to Phase 1).
    init='glorot': Glorot-uniform weights + zero biases on Conv/Linear (my guess at the TF default); BN unchanged."""
    model = {"inception": SmallInception, "alexnet": small_alexnet, "mlp": mlp_1x512}[name]()
    if init == "glorot":
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    elif init != "default":
        raise ValueError(init)
    return model


# ---------------------------------------------------------------- training
@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval(); wrong = 0
    for i in range(0, len(x), bs):
        wrong += (model(x[i:i + bs]).argmax(1) != y[i:i + bs]).sum().item()
    return wrong / len(x)


def train_run(model_name, init, xtr, ytr, xte, yte, decay, cap, seed):
    """Per-epoch logging; stops when eval-mode train err <= STOP_ERR and online loss < LOSS_THRESH, or at cap."""
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = [t.to(device) for t in (xtr, ytr, xte, yte)]
    model = make_model(model_name, init).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=LR[model_name], momentum=MOMENTUM)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay)
    N, steps, spe = len(xtr), 0, math.ceil(len(xtr) / BATCH)
    log = {k: [] for k in ("step", "lr", "online_loss", "train_err", "test_err")}
    stop_reason = "cap"
    for ep in range(cap):
        model.train()
        perm, ep_sum = torch.randperm(N, device=device), torch.zeros((), device=device)
        lr_used = opt.param_groups[0]["lr"]
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            steps += 1
            ep_sum += loss.detach() * len(idx)
        sched.step()
        ep_loss = ep_sum.item() / N
        tr_err, te_err = evaluate(model, xtr, ytr), evaluate(model, xte, yte)
        for k, v in zip(log, (steps, lr_used, ep_loss, tr_err, te_err)): log[k].append(v)
        print(f"  [{model_name}] ep {ep+1}/{cap} step {steps} loss {ep_loss:.4f} "
              f"train_err {tr_err:.4f} test_err {te_err:.4f} lr {lr_used:.2e}", flush=True)
        if tr_err <= STOP_ERR + 1e-12 and ep_loss < LOSS_THRESH:
            stop_reason = "converged"
            break
    return dict(model=model_name, init=init, decay=decay, cap=cap, seed=seed, batch=BATCH, steps_per_epoch=spe,
                epochs=len(log["step"]), hit_cap=(stop_reason == "cap"), stop_reason=stop_reason,
                final_train_err=log["train_err"][-1], final_test_err=log["test_err"][-1], **log)


def run_path(cond, model, p):
    return os.path.join(ARGS.out, "runs", f"{cond}__{model}__p{p:.1f}.json")


def build_jobs(conds):
    jobs = [(c, m, p) for c in conds for m, ps in CONDITIONS[c]["models"].items() for p in ps]
    return sorted(jobs, key=lambda j: -COST_W[j[1]] * CONDITIONS[j[0]]["cap"])   # deterministic, cost-balanced


def train_all(conds):
    global device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, flush=True)
    if device == "cpu" and not (ARGS.smoke or ARGS.allow_cpu):
        raise SystemExit("No GPU found. Refusing to run on CPU. Use --allow-cpu to override.")
    torch.backends.cudnn.benchmark = True
    XTR, YTR, XTE, YTE = load_cifar10()
    i, n = map(int, ARGS.shard.split("/"))
    for j, (c, m, p) in enumerate(build_jobs(conds)):
        if j % n != i: continue
        path = run_path(c, m, p)
        if os.path.exists(path):
            print(f"skip {c}/{m}/p={p} (exists)", flush=True); continue
        cfg, t0 = CONDITIONS[c], time.time()
        print(f"=== {c} | {m} | p={p} | decay={cfg['decay']} | init={cfg['init']} | cap={cfg['cap']} | seed={cfg['seed']}",
              flush=True)
        ytr = YTR if p == 0 else corrupt_labels(YTR, p, seed=cfg["seed"] + 1000 + int(round(p * 100)))
        r = train_run(m, cfg["init"], XTR, ytr, XTE, YTE, cfg["decay"], cfg["cap"], cfg["seed"])
        r.update(cond=c, p=p, wall_seconds=time.time() - t0)
        tmp = path + ".tmp"
        json.dump(r, open(tmp, "w")); os.replace(tmp, path)
        print(f"=== done {c}/{m}/p={p}: epochs={r['epochs']} reason={r['stop_reason']} "
              f"train_err={r['final_train_err']:.4f} test_err={r['final_test_err']:.4f} "
              f"| {r['wall_seconds']:.0f}s", flush=True)


# ---------------------------------------------------------------- analysis
def load_runs():
    runs, d = {}, os.path.join(ARGS.out, "runs")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                r = json.load(open(os.path.join(d, f)))
                runs[(r["cond"], r["model"], round(r["p"], 1))] = r
    return runs


def first_epoch(run, key):
    """1-based first epoch meeting threshold `key`, else None."""
    kind, thr = THRESHOLDS[key]
    for e, (tr, ls) in enumerate(zip(run["train_err"], run["online_loss"]), 1):
        ok = (tr <= (1 - thr) + 1e-12) if kind == "acc" else (ls < thr)
        if ok: return e
    return None


def rel_time(runs, cond, model, p, key):
    """(time relative to this cond/model's own p=0, censored?). Censored -> lower bound epochs_run / base."""
    base, r = runs.get((cond, model, 0.0)), runs.get((cond, model, round(p, 1)))
    if base is None or r is None: return float("nan"), False
    b = first_epoch(base, key)
    if b is None: return float("nan"), False
    e = first_epoch(r, key)
    if e is None: return r["epochs"] / b, True       # constant steps/epoch, so epoch ratio == step ratio
    return e / b, False


def is_seed_cond(c): return "+s" in c
def sweep_conds(runs): return [c for c in CONDITIONS if not is_seed_cond(c) and any(k[0] == c for k in runs)]
def cond_sort_key(c): return (list(CONDITIONS).index(c), "") if c in CONDITIONS else (len(CONDITIONS), c)


def score_settings(runs):
    """Fixed selection rule. For AlexNet and MLP of each complete setting:
         err_term  = |p=0 test err - paper Table 1| / SCORE_ERR_UNIT
         slow_term = |ln(slowdown_p1 / paper)| / SCORE_SLOW_UNIT   (censored run: max(0, ln(bound/paper)) only,
                     because a lower bound can only prove over-slowness)
       setting score = mean over {alexnet, mlp} of (err_term + slow_term). Lower is better. Inception is excluded
       because it only exists for decay 0.95; it is reported separately."""
    out = []
    for c in sweep_conds(runs):
        if not all((c, m, p) in runs for m in ("alexnet", "mlp") for p in PS): continue
        comps, tot = {}, 0.0
        for m in ("alexnet", "mlp"):
            et = abs(runs[(c, m, 0.0)]["final_test_err"] - PAPER_P0_TEST_ERR[m]) / SCORE_ERR_UNIT
            sl, cz = rel_time(runs, c, m, 1.0, PRIMARY)
            if math.isnan(sl): st = float("inf")
            elif cz: st = max(0.0, math.log(sl / PAPER_SLOWDOWN_P1[m])) / SCORE_SLOW_UNIT
            else: st = abs(math.log(sl / PAPER_SLOWDOWN_P1[m])) / SCORE_SLOW_UNIT
            comps[m] = dict(err_term=et, slow=sl, censored=cz, slow_term=st)
            tot += (et + st) / 2
        out.append(dict(cond=c, score=tot, comps=comps))
    return sorted(out, key=lambda r: r["score"])


def parse_cond(c):
    cfg = CONDITIONS[c]
    return cfg["decay"], cfg["init"]


def plot_b(ax, spec, runs, key, label_cond=True):
    for m, cond in spec.items():
        lab, c, mk = STY_M[m]
        ps = sorted(p for (cc, mm, p) in runs if cc == cond and mm == m)
        if not ps: continue
        vals = [rel_time(runs, cond, m, p, key) for p in ps]
        ax.plot(ps, [v for v, _ in vals], color=c, marker=mk, label=f"{lab} ({cond})" if label_cond else lab)
        for p, (v, cz) in zip(ps, vals):
            if cz and not math.isnan(v): ax.plot([p], [v], "x", color="r", ms=10)
    ax.set(xlabel="label corruption", ylabel="time to overfit (rel. to p=0)")
    ax.set_title(f"(b) {key}", fontsize=10)
    ax.legend(fontsize=7, title="red x = not reached (lower bound)", title_fontsize=7)


def plot_c(ax, spec, runs, label_cond=True):
    for m, cond in spec.items():
        lab, c, mk = STY_M[m]
        ps = sorted(p for (cc, mm, p) in runs if cc == cond and mm == m)
        if not ps: continue
        ax.plot(ps, [runs[(cond, m, p)]["final_test_err"] for p in ps], color=c, marker=mk,
                label=f"{lab} ({cond})" if label_cond else lab)
        for p in ps:
            if runs[(cond, m, p)]["hit_cap"]:
                ax.plot([p], [runs[(cond, m, p)]["final_test_err"]], "o", mfc="none", mec="r", ms=10)
    ax.axhline(0.9, color="r", ls="--")
    ax.set(xlabel="label corruption", ylabel="test_error", ylim=(0, 1), title="(c) generalization error growth")
    ax.legend(fontsize=7, title="red ring = hit epoch cap", title_fontsize=7)


def plot_a(ax):
    path = ARGS.baseline_results
    if not os.path.exists(path):
        ax.text(0.5, 0.5, f"{path} not found\n(panel (a) skipped)", ha="center", transform=ax.transAxes); return False
    R = json.load(open(path))
    for s in SETTINGS_A:
        r = R.get(f"A|{s}")
        if r is None: continue
        lab, c, mk = STY_A[s]
        ax.plot(np.array(r["log_steps"]) / 1000, r["log_loss"], color=c, marker=mk, markevery=4, ms=4, label=lab)
    ax.set(xlabel="thousand steps", ylabel="average_loss", title="(a) learning curves (Phase 1 runs, read-only)")
    ax.legend(); return True


def setting_spec(runs, c):
    return {m: c for m in STY_M if any(k[0] == c and k[1] == m for k in runs)}


def make_main(runs, spec, fname):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    plot_a(ax[0]); plot_b(ax[1], spec, runs, PRIMARY); plot_c(ax[2], spec, runs)
    plt.tight_layout(); plt.savefig(os.path.join(ARGS.out, fname), dpi=200); plt.close(fig)


def make_grids(runs):
    conds = sorted(sweep_conds(runs), key=cond_sort_key)
    if not conds: return
    for fname, ncols in (("figure_sweep_bc.png", 2), ("figure_sweep_thresholds.png", 3)):
        fig, ax = plt.subplots(len(conds), ncols, figsize=(5 * ncols, 3.6 * len(conds)), squeeze=False)
        for r, c in enumerate(conds):
            spec = setting_spec(runs, c)
            if ncols == 2:
                plot_b(ax[r][0], spec, runs, PRIMARY, label_cond=False); plot_c(ax[r][1], spec, runs, label_cond=False)
                ax[r][0].set_title(f"{c}\n(b) {PRIMARY}", fontsize=9)
            else:
                for k, key in enumerate(THRESHOLDS):
                    plot_b(ax[r][k], spec, runs, key, label_cond=False)
                    ax[r][k].set_title(f"{c}\n(b) {key}", fontsize=9)
        plt.tight_layout(); plt.savefig(os.path.join(ARGS.out, fname), dpi=150); plt.close(fig)


def make_seed_figure(runs, winner):
    conds = [winner] + sorted(c for c in {k[0] for k in runs} if c.startswith(winner + "+s"))
    if len(conds) < 2: return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for m in ("alexnet", "mlp"):
        lab, col, mk = STY_M[m]
        bv, cv = [], []
        for p in PS:
            b = [rel_time(runs, c, m, p, PRIMARY)[0] for c in conds if (c, m, p) in runs]
            e = [runs[(c, m, p)]["final_test_err"] for c in conds if (c, m, p) in runs]
            bv.append((np.nanmean(b), np.nanstd(b))); cv.append((np.mean(e), np.std(e)))
        ax[0].errorbar(PS, [v[0] for v in bv], yerr=[v[1] for v in bv], color=col, marker=mk, capsize=3, label=lab)
        ax[1].errorbar(PS, [v[0] for v in cv], yerr=[v[1] for v in cv], color=col, marker=mk, capsize=3, label=lab)
    ax[0].set(xlabel="label corruption", ylabel="time to overfit (rel. to own p=0)",
              title=f"(b) {PRIMARY}, mean +/- std over {len(conds)} seeds\n{winner}")
    ax[1].set(xlabel="label corruption", ylabel="test_error", ylim=(0, 1), title="(c) mean +/- std over seeds")
    ax[1].axhline(0.9, color="r", ls="--")
    for a in ax: a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(ARGS.out, "figure_winner_seeds.png"), dpi=200); plt.close(fig)


def fmt(v, nd=4):
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def md_table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def write_tables(runs, scores):
    keys = sorted(runs, key=lambda k: (cond_sort_key(k[0]), list(STY_M).index(k[1]), k[2]))
    head = ["cond", "model", "p", "decay", "init", "seed", "epochs", "hit_cap", "final_train_err", "final_test_err"]
    for t in THRESHOLDS: head += [f"{t}_epoch", f"{t}_rel", f"{t}_censored"]
    rows = []
    for (c, m, p) in keys:
        r = runs[(c, m, p)]
        row = [c, m, f"{p:.1f}", r["decay"], r["init"], r["seed"], r["epochs"], r["hit_cap"],
               fmt(r["final_train_err"]), fmt(r["final_test_err"])]
        for t in THRESHOLDS:
            e = first_epoch(r, t); rel, cz = rel_time(runs, c, m, p, t)
            row += [fmt(e), fmt(rel, 2), cz]
        rows.append(row)
    with open(os.path.join(ARGS.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(rows)
    open(os.path.join(ARGS.out, "summary.md"), "w").write("\n".join(md_table(head, rows)) + "\n")

    # side-by-side sweep table: one row per (setting, model)
    sh = ["setting", "model", "p0_test_err", "paper_p0", "p0_epochs"] + [f"p1_slow_{t}" for t in THRESHOLDS] + \
         ["p1_hit_cap", "p1_train_err", "p1_test_err"]
    srows = []
    for c in sorted(sweep_conds(runs), key=cond_sort_key):
        for m in STY_M:
            if (c, m, 0.0) not in runs or (c, m, 1.0) not in runs: continue
            r0, r1 = runs[(c, m, 0.0)], runs[(c, m, 1.0)]
            row = [c, m, fmt(r0["final_test_err"]), fmt(PAPER_P0_TEST_ERR[m]), r0["epochs"]]
            for t in THRESHOLDS:
                v, cz = rel_time(runs, c, m, 1.0, t)
                row.append(fmt(v, 2) + (">=" if cz else ""))
            row += [r1["hit_cap"], fmt(r1["final_train_err"]), fmt(r1["final_test_err"])]
            srows.append(row)
    md = ["## Sweep table (slowdowns normalized by each run's own p=0; '>=' = not reached, lower bound)", ""]
    md += md_table(sh, srows)
    md += ["", f"Paper approx. p=1 slowdown (by eye from Fig. 1b, used only for scoring): {PAPER_SLOWDOWN_P1}", "",
           "## Selection scores (AlexNet + MLP; lower = closer to paper; arbitrary units, see score_settings)", ""]
    if scores:
        hd = ["rank", "setting", "score"] + [f"{m}_{k}" for m in ("alexnet", "mlp") for k in ("err_term", "slow_p1", "slow_term")]
        md += md_table(hd, [[i + 1, s["cond"], fmt(s["score"], 3)] + [
            (fmt(s["comps"][m]["err_term"], 3) if k == "err_term" else
             fmt(s["comps"][m]["slow"], 2) + (">=" if s["comps"][m]["censored"] else "") if k == "slow_p1" else
             fmt(s["comps"][m]["slow_term"], 3)) for m in ("alexnet", "mlp") for k in ("err_term", "slow_p1", "slow_term")]
            for i, s in enumerate(scores)])
    else:
        md.append("No setting has all AlexNet/MLP runs finished yet.")
    open(os.path.join(ARGS.out, "sweep_table.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def write_epoch_csv(runs):
    with open(os.path.join(ARGS.out, "epoch_log.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cond", "model", "p", "epoch", "step", "lr", "online_loss", "train_err", "test_err"])
        for (c, m, p), r in runs.items():
            for i in range(r["epochs"]):
                w.writerow([c, m, p, i + 1, r["step"][i], r["lr"][i], r["online_loss"][i], r["train_err"][i], r["test_err"][i]])


README = """# Phase 3, round 1 (sweep): what changed vs. the Phase 1 baseline, and why

Phase 1 (`reproduce_figure1.py`, `results.json`) is untouched. Everything here was produced by `train_v2.py` into this folder.
No values are hardcoded: every number in the tables and figures comes from the per-run JSON files in `runs/`. The only
paper-derived constants are the reference values used for the selection score (see below).

## Problems found in the baseline run
1. It stopped at online loss < 0.1, not at train error ~ 0 (Inception ended at 3-6% train error; paper: 100%).
2. Panel (b) for AlexNet and MLP disagreed with the paper; MLP runs at p >= 0.6 hit the epoch cap.
3. Inception p=0 test error was higher than the paper's Table 1 value.
Hypotheses (not established by the baseline run): the stopping criterion, the x0.95-per-epoch LR decay, and the weight init.

## What this round does
- **Stop rule**: eval-mode train error <= {stop_err} AND online epoch-mean loss < {loss_thresh}, or the epoch cap
  (150 for decay 0.95, 300 for 0.99 / 0.999). The loss clause is kept so the baseline's loss<0.1 threshold stays computable.
- **Sweep** over LR decay per epoch {{0.95, 0.99, 0.999}} x init {{default, glorot}}. Everything else is as in Phase 1: SGD momentum
  {momentum}, batch {batch}, lr 0.1 (Inception) / 0.01 (AlexNet, MLP), no weight decay / dropout / augmentation, same data
  preprocessing and the same label-corruption seeds. AlexNet and MLP run all 6 corruption levels in every setting; Inception
  runs only at decay 0.95 (both inits) and p in {{0, .4, .8, 1}} to save compute, because its baseline slowdown already looked
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
`python train_v2.py --stage seeds` trains seeds {seeds} for AlexNet and MLP at the winning setting only (model init, data order
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
"""


def aggregate():
    runs = load_runs()
    if not runs:
        print("no runs found in", os.path.join(ARGS.out, "runs")); return
    scores = score_settings(runs)
    write_epoch_csv(runs)
    write_tables(runs, scores)
    spec0 = {m: cname(0.95, "default") for m in STY_M}
    make_main(runs, spec0, "figure1_specfaithful.png")
    make_grids(runs)
    if scores:
        w = scores[0]["cond"]
        d, i = parse_cond(w)
        spec = {m: w for m in ("alexnet", "mlp")}
        spec = {"inception": cname(0.95, i), **spec}          # Inception only exists at decay 0.95 (same init as winner)
        make_main(runs, spec, "figure1_winner.png")
        make_seed_figure(runs, w)
        print(f"\nWINNER by fixed selection rule: {w}")
    open(os.path.join(ARGS.out, "README.md"), "w").write(README.format(
        stop_err=STOP_ERR, loss_thresh=LOSS_THRESH, momentum=MOMENTUM, batch=BATCH, seeds=SEEDS_EXTRA))
    print("saved figures, tables, README to", ARGS.out)


def register_seed_conds(winner):
    base, names = CONDITIONS[winner], []
    for s in SEEDS_EXTRA:
        name = f"{winner}+s{s}"
        CONDITIONS[name] = dict(decay=base["decay"], init=base["init"], cap=base["cap"],
                                models={"alexnet": PS, "mlp": PS}, seed=s)
        names.append(name)
    return names


def main(argv=None):
    global ARGS
    ARGS = parse(argv)
    if ARGS.smoke:                                    # pipeline test only
        for c in CONDITIONS.values():
            c["cap"] = 2
            c["models"] = {m: [0.0, 1.0] for m in c["models"]}
        globals()["PS"] = [0.0, 1.0]
    os.makedirs(os.path.join(ARGS.out, "runs"), exist_ok=True)
    if not ARGS.plot_only:
        if ARGS.stage == "sweep":
            train_all(list(CONDITIONS))
        else:
            scores = score_settings(load_runs())
            if not scores:
                raise SystemExit("Sweep is incomplete: no setting has all AlexNet/MLP runs, cannot pick a winner.")
            names = register_seed_conds(scores[0]["cond"])
            print("winner:", scores[0]["cond"], "-> seed conditions:", names, flush=True)
            train_all(names)
    aggregate()


if __name__ == "__main__":
    main()
