#!/usr/bin/env python
"""Phase 3, round 2: reproduce Figure 1 of Zhang et al. (ICLR 2017), diagnostics round.

Standalone entry point, but it IMPORTS train_v2.py (round 1, unmodified) for the models, data pipeline,
label corruption, evaluate() and the round-1 constants (batch, momentum, lr, stop rule, decays, caps), so
parity with round 1 holds by construction. It never writes outside --out (default results_phase3_r2/) and only
READS results_phase3_r1_sweep/ and results.json.

Jobs (all independent, one JSON per run, resumable, shardable)
  seeds  (goals 2+4): seeds 0-4 x {Inception d0.95, AlexNet d0.99, MLP d0.99; default init} x p in {0,.4,.8,1}.
                      For p in {.4,.8} the run continues 50 epochs AFTER the stop rule first triggers (same LR
                      schedule), logging train/test error every epoch. Goal-2 statistics are read at the first
                      trigger epoch, which is identical to the round-1 definition.
  lrhold (goal 4b)  : Inception, p in {.4,.8}, seed 0: as above but the LR is held at its stop-epoch value
                      during the 50 extra epochs.
  pixels (goal 3)   : Inception, true labels, 64 epochs, inputs = random pixels with the SAME permutation across
                      channels (round-1 variant) | permutation of the flattened C*H*W vector | Gaussian; seeds 0,1.
  probe  (goal 3)   : non-training linear probe on per-image channel mean/std features for the same inputs.

Usage
  python train_r2.py --shard i/n          # train every missing run assigned to this worker
  python train_r2.py --plot-only          # aggregate: tables, figures, README (after all shards finish)
  python train_r2.py --smoke --out /tmp/r2_smoke   # tiny pipeline test on random tensors
"""
import argparse, os, json, math, time, csv, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, ".")
import train_v2 as v2          # round-1 script, not modified

# ---------------------------------------------------------------- configuration
BATCH, MOMENTUM, LR = v2.BATCH, v2.MOMENTUM, v2.LR
STOP_ERR, LOSS_THRESH = v2.STOP_ERR, v2.LOSS_THRESH            # eval-mode train err <= 0.1% AND online loss < 0.1
SETTING = {"inception": "d0.95_default", "alexnet": "d0.99_default", "mlp": "d0.99_default"}   # round-1 winners
SEEDS = [0, 1, 2, 3, 4]
PS = [0.0, 0.4, 0.8, 1.0]
EXT_PS, EXT_EPOCHS = [0.4, 0.8], 50
PIX_SEEDS, PIX_EPOCHS, LOG_EVERY = [0, 1], 64, 50              # 64 epochs = round-1 panel (a) length
PIXEL_DECAY_COND = "d0.95_default"
COST_W = {"inception": 8.0, "alexnet": 1.5, "mlp": 0.3}        # only used to balance --shard
MODEL_ORDER = ["inception", "alexnet", "mlp"]
STY = v2.STY_M
# Paper readings BY EYE from Fig. 1c (+-0.02), used only as dashed reference lines / table columns, never as data
PAPER_C_BYEYE = {"inception": {0.4: 0.63, 0.8: 0.88}, "alexnet": {0.4: 0.69, 0.8: 0.88}, "mlp": {0.4: 0.76, 0.8: 0.88}}
ARGS = None
device = "cpu"


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_phase3_r2")
    ap.add_argument("--round1-dir", default="results_phase3_r1_sweep", help="read-only")
    ap.add_argument("--baseline-results", default="results.json", help="Phase 1 results.json, read-only (reference curves)")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--shard", default="0/1", help="i/n: run jobs with index %% n == i")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- input variants for goal 3
def permute_shared(x, seed):
    """Round-1 'random pixels': independent permutation of the HxW positions per image, SAME across channels."""
    g = torch.Generator().manual_seed(seed)
    N, C, H, W = x.shape
    idx = torch.argsort(torch.rand(N, H * W, generator=g), dim=1)[:, None, :].expand(N, C, H * W)
    return torch.gather(x.reshape(N, C, H * W), 2, idx).reshape(N, C, H, W)


def permute_flat(x, seed):
    """Independent permutation of the flattened C*H*W vector per image (mixes channels and positions)."""
    g = torch.Generator().manual_seed(seed)
    N, C, H, W = x.shape
    idx = torch.argsort(torch.rand(N, C * H * W, generator=g), dim=1)
    return torch.gather(x.reshape(N, C * H * W), 1, idx).reshape(N, C, H, W)


def gaussian_like(x, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(x.shape, generator=g) * x.std() + x.mean()


# name -> (transform, k). Input seed = run_seed + 100 + k (train) / + 200 + k (test): for seed 0 'shared' and
# 'gaussian' reproduce the round-1 panel-(a) inputs exactly (k = index in round-1 SETTINGS_A: 3 and 4).
PIXEL_VARIANTS = {"random_pixels_shared": (permute_shared, 3), "random_pixels_flat": (permute_flat, 3),
                  "gaussian": (gaussian_like, 4)}


def make_pixel_inputs(variant, seed, XTR, XTE):
    fn, k = PIXEL_VARIANTS[variant]
    return fn(XTR, seed + 100 + k), fn(XTE, seed + 200 + k)


# ---------------------------------------------------------------- training
def train_labels(model_name, xtr, ytr, xte, yte, decay, cap, seed, ext, hold_lr):
    """Round-1 training loop + optional continuation of `ext` epochs after the stop rule first triggers.
    hold_lr=True freezes the LR at its stop-epoch value for the continuation."""
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = [t.to(device) for t in (xtr, ytr, xte, yte)]
    model = v2.make_model(model_name, "default").to(device)
    opt = torch.optim.SGD(model.parameters(), lr=LR[model_name], momentum=MOMENTUM)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay)
    N, steps, spe = len(xtr), 0, math.ceil(len(xtr) / BATCH)
    log = {k: [] for k in ("step", "lr", "online_loss", "train_err", "test_err")}
    stop_epoch, end_epoch, frozen, ep = None, cap, False, 0
    while ep < end_epoch:
        model.train()
        perm, ep_sum = torch.randperm(N, device=device), torch.zeros((), device=device)
        lr_used = opt.param_groups[0]["lr"]
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            steps += 1
            ep_sum += loss.detach() * len(idx)
        if not frozen: sched.step()
        ep_loss = ep_sum.item() / N
        tr_err, te_err = v2.evaluate(model, xtr, ytr), v2.evaluate(model, xte, yte)
        for k, v in zip(log, (steps, lr_used, ep_loss, tr_err, te_err)): log[k].append(v)
        ep += 1
        tag = "" if stop_epoch is None else f" (+{ep - stop_epoch} past stop)"
        print(f"  [{model_name}] ep {ep}/{end_epoch} step {steps} loss {ep_loss:.4f} train_err {tr_err:.4f} "
              f"test_err {te_err:.4f} lr {lr_used:.2e}{tag}", flush=True)
        if stop_epoch is None and tr_err <= STOP_ERR + 1e-12 and ep_loss < LOSS_THRESH:
            stop_epoch = ep
            if ext > 0:
                end_epoch = ep + ext
                if hold_lr:
                    frozen = True
                    for g in opt.param_groups: g["lr"] = lr_used
            else:
                break
    return dict(model=model_name, decay=decay, cap=cap, seed=seed, ext=ext, hold_lr=hold_lr, batch=BATCH,
                steps_per_epoch=spe, epochs=len(log["step"]), stop_epoch=stop_epoch, hit_cap=(stop_epoch is None), **log)


def train_pixels(xtr, ytr, xte, yte, decay, epochs, seed):
    """Inception on (randomized) inputs with true labels, fixed length. Loss logged every LOG_EVERY steps."""
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = [t.to(device) for t in (xtr, ytr, xte, yte)]
    model = v2.make_model("inception", "default").to(device)
    opt = torch.optim.SGD(model.parameters(), lr=LR["inception"], momentum=MOMENTUM)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay)
    N, steps = len(xtr), 0
    win, win_steps, win_loss = torch.zeros((), device=device), [], []
    epoch_loss, test_err = [], []
    for ep in range(epochs):
        model.train()
        perm, ep_sum = torch.randperm(N, device=device), torch.zeros((), device=device)
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            steps += 1
            ep_sum += loss.detach() * len(idx)
            win += loss.detach()
            if steps % LOG_EVERY == 0:
                win_steps.append(steps); win_loss.append(win.item() / LOG_EVERY); win.zero_()
        sched.step()
        epoch_loss.append(ep_sum.item() / N)
        test_err.append(v2.evaluate(model, xte, yte))
        print(f"  [pixels] ep {ep+1}/{epochs} step {steps} loss {epoch_loss[-1]:.4f} test_err {test_err[-1]:.4f}", flush=True)
    return dict(seed=seed, decay=decay, epochs=epochs, steps=steps, log_every=LOG_EVERY, win_steps=win_steps,
                win_loss=win_loss, epoch_loss=epoch_loss, test_err=test_err,
                final_train_err=v2.evaluate(model, xtr, ytr), final_test_err=test_err[-1],
                first_epoch_loss=epoch_loss[0])


def run_probe(XTR, YTR, XTE, YTE):
    """Non-training diagnostic: softmax regression on per-image per-channel mean and std (6 features)."""
    feats = lambda x: torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], 1)
    sets = {"true_images": (XTR, XTE)}
    for v in PIXEL_VARIANTS: sets[v] = make_pixel_inputs(v, 0, XTR, XTE)
    out = {}
    for name, (xa, xb) in sets.items():
        fa, fb = feats(xa), feats(xb)
        mu, sd = fa.mean(0), fa.std(0) + 1e-8
        fa, fb = (fa - mu) / sd, (fb - mu) / sd
        W = torch.zeros(fa.shape[1], 10, requires_grad=True); b = torch.zeros(10, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); l = F.cross_entropy(fa @ W + b, YTR); l.backward(); return l
        opt.step(closure)
        with torch.no_grad():
            err = lambda f, y: (f @ W + b).argmax(1).ne(y).float().mean().item()
            out[name] = dict(train_err=err(fa, YTR), test_err=err(fb, YTE))
        print(f"  [probe] {name}: train_err {out[name]['train_err']:.4f} test_err {out[name]['test_err']:.4f}", flush=True)
    return out


# ---------------------------------------------------------------- jobs
def runs_dir(): return os.path.join(ARGS.out, "runs")


def build_jobs():
    jobs = []
    for m in MODEL_ORDER:
        for p in PS:
            for s in SEEDS:
                ext = EXT_EPOCHS if p in EXT_PS else 0
                jobs.append(dict(kind="labels", cond="seeds", model=m, p=p, seed=s, ext=ext, hold=False,
                                 id=f"seeds__{m}__p{p:.1f}__s{s}", cost=COST_W[m] * (60 + ext)))
    for p in EXT_PS:
        jobs.append(dict(kind="labels", cond="lrhold", model="inception", p=p, seed=0, ext=EXT_EPOCHS, hold=True,
                         id=f"lrhold__inception__p{p:.1f}__s0", cost=COST_W["inception"] * (60 + EXT_EPOCHS)))
    for v in PIXEL_VARIANTS:
        for s in PIX_SEEDS:
            jobs.append(dict(kind="pixels", variant=v, seed=s, id=f"pixels__{v}__s{s}", cost=COST_W["inception"] * PIX_EPOCHS))
    jobs.append(dict(kind="probe", id="probe", cost=1.0))
    return sorted(jobs, key=lambda j: (-j["cost"], j["id"]))


def train_all():
    global device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, flush=True)
    if device == "cpu" and not (ARGS.smoke or ARGS.allow_cpu):
        raise SystemExit("No GPU found. Refusing to run on CPU. Use --allow-cpu to override.")
    torch.backends.cudnn.benchmark = True
    v2.ARGS = argparse.Namespace(smoke=ARGS.smoke, data_root=ARGS.data_root)
    XTR, YTR, XTE, YTE = v2.load_cifar10()
    i, n = map(int, ARGS.shard.split("/"))
    for j, job in enumerate(build_jobs()):
        if j % n != i: continue
        path = os.path.join(runs_dir(), job["id"] + ".json")
        if os.path.exists(path):
            print(f"skip {job['id']} (exists)", flush=True); continue
        t0 = time.time()
        print(f"=== {job['id']}", flush=True)
        if job["kind"] == "labels":
            cond_key = SETTING[job["model"]]
            cfg = v2.CONDITIONS[cond_key]
            p, s = job["p"], job["seed"]
            ytr = YTR if p == 0 else v2.corrupt_labels(YTR, p, seed=s + 1000 + int(round(p * 100)))
            r = train_labels(job["model"], XTR, ytr, XTE, YTE, cfg["decay"], cfg["cap"], s, job["ext"], job["hold"])
            r.update(kind="labels", cond=job["cond"], p=p, setting=cond_key)
        elif job["kind"] == "pixels":
            xtr, xte = make_pixel_inputs(job["variant"], job["seed"], XTR, XTE)
            r = train_pixels(xtr, YTR, xte, YTE, v2.CONDITIONS[PIXEL_DECAY_COND]["decay"], PIX_EPOCHS, job["seed"])
            r.update(kind="pixels", variant=job["variant"])
        else:
            r = dict(kind="probe", results=run_probe(XTR, YTR, XTE, YTE))
        r["wall_seconds"] = time.time() - t0
        tmp = path + ".tmp"
        json.dump(r, open(tmp, "w")); os.replace(tmp, path)
        print(f"=== done {job['id']} | {r['wall_seconds']:.0f}s", flush=True)


# ---------------------------------------------------------------- analysis helpers
def load_all():
    L, P, probe, d = {}, {}, None, runs_dir()
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json"): continue
            r = json.load(open(os.path.join(d, f)))
            if r["kind"] == "labels": L[(r["cond"], r["model"], round(r["p"], 1), r["seed"])] = r
            elif r["kind"] == "pixels": P[(r["variant"], r["seed"])] = r
            elif r["kind"] == "probe": probe = r["results"]
    return L, P, probe


def load_round1(model, p):
    f = os.path.join(ARGS.round1_dir, "runs", f"{SETTING[model]}__{model}__p{p:.1f}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def first_epoch_acc99(run):
    for e, tr in enumerate(run["train_err"], 1):
        if tr <= 0.01 + 1e-12: return e
    return None


def te_at_stop(r):
    return r["test_err"][r["stop_epoch"] - 1] if r.get("stop_epoch") else r["test_err"][-1]


def stop_epoch_of(r):
    if "stop_epoch" in r: return r["stop_epoch"]
    return r["epochs"] if r.get("stop_reason") == "converged" else None      # round-1 format


def slowdown(base, r):
    """(rel time at acc>=99% vs base, censored?). Censored -> lower bound epochs_run / base_epoch."""
    if base is None or r is None: return float("nan"), False
    b = first_epoch_acc99(base)
    if b is None: return float("nan"), False
    e = first_epoch_acc99(r)
    return (r["epochs"] / b, True) if e is None else (e / b, False)


def ms(vals):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(v) == 0: return float("nan"), float("nan"), 0
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan")), len(v)


def fm(m, s, nd=3):
    return "nan" if math.isnan(m) else (f"{m:.{nd}f} +/- {s:.{nd}f}" if not math.isnan(s) else f"{m:.{nd}f}")


def md_table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def seed_list(L, model, p, cond="seeds"):
    return sorted(s for (c, m, pp, s) in L if c == cond and m == model and pp == p)


def write_csv(path, head, rows):
    with open(os.path.join(ARGS.out, path), "w", newline="") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(rows)


# ---------------------------------------------------------------- goal 2: seed noise
def goal2(L):
    head = ["model", "p", "n_seeds", "stop_epoch", "n_hit_cap", "test_err_at_stop", "linear e0(1-p)+0.9p",
            "paper_by_eye", "slowdown_acc99 (>= if any censored)", "n_censored"]
    rows, md_rows = [], []
    stats = {}
    for m in MODEL_ORDER:
        for p in PS:
            seeds = seed_list(L, m, p)
            if not seeds: continue
            te = [te_at_stop(L[("seeds", m, p, s)]) for s in seeds]
            lin = [L[("seeds", m, 0.0, s)]["test_err"][(L[("seeds", m, 0.0, s)]["stop_epoch"] or L[("seeds", m, 0.0, s)]["epochs"]) - 1]
                   * (1 - p) + 0.9 * p for s in seeds if ("seeds", m, 0.0, s) in L]
            sd = [slowdown(L.get(("seeds", m, 0.0, s)), L[("seeds", m, p, s)]) for s in seeds]
            stops = [L[("seeds", m, p, s)]["stop_epoch"] for s in seeds]
            ncap = sum(L[("seeds", m, p, s)]["hit_cap"] for s in seeds)
            ncens = sum(c for _, c in sd)
            mt, st, n = ms(te); ml, sl, _ = ms(lin); msd, ssd, _ = ms([v for v, _ in sd]); mst, sst, _ = ms(stops)
            stats[(m, p)] = dict(te=(mt, st), slow=(msd, ssd), n=n, ncens=ncens)
            paper = PAPER_C_BYEYE.get(m, {}).get(p, "")
            row = [m, f"{p:.1f}", n, fm(mst, sst, 1), ncap, fm(mt, st), fm(ml, sl), paper,
                   fm(msd, ssd, 2) + (">=" if ncens else ""), ncens]
            md_rows.append(row)
            rows.append([m, p, n, mst, sst, ncap, mt, st, ml, sl, msd, ssd, ncens])
    write_csv("summary_goal2_seeds.csv", ["model", "p", "n", "stop_epoch_mean", "stop_epoch_std", "n_hit_cap",
                                         "test_err_mean", "test_err_std", "linear_mean", "linear_std",
                                         "slowdown_mean", "slowdown_std", "n_censored"], rows)
    # seed-0 re-run vs round 1 (GPU nondeterminism check)
    h2 = ["model", "p", "round1 stop epoch", "r2 seed0 stop epoch", "round1 test_err@stop", "r2 seed0 test_err@stop",
          "round1 slowdown", "r2 seed0 slowdown"]
    r2rows = []
    for m in MODEL_ORDER:
        b1 = load_round1(m, 0.0)
        for p in PS:
            r1, r0 = load_round1(m, p), L.get(("seeds", m, p, 0))
            if r1 is None or r0 is None: continue
            s1, _ = slowdown(b1, r1); s0, _ = slowdown(L.get(("seeds", m, 0.0, 0)), r0)
            r2rows.append([m, f"{p:.1f}", stop_epoch_of(r1), r0["stop_epoch"], fm(te_at_stop(r1), float("nan")),
                           fm(te_at_stop(r0), float("nan")), fm(s1, float("nan"), 2), fm(s0, float("nan"), 2)])
    return head, md_rows, h2, r2rows, stats


def fig_goal2(L):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for m in MODEL_ORDER:
        lab, col, mk = STY[m]
        ps = [p for p in PS if seed_list(L, m, p)]
        if not ps: continue
        seeds_all = sorted({s for (c, mm, p, s) in L if c == "seeds" and mm == m})
        for s in seeds_all:
            pp = [p for p in ps if ("seeds", m, p, s) in L]
            ax[0].plot(pp, [slowdown(L.get(("seeds", m, 0.0, s)), L[("seeds", m, p, s)])[0] for p in pp], color=col, alpha=0.2, lw=1)
            ax[1].plot(pp, [te_at_stop(L[("seeds", m, p, s)]) for p in pp], color=col, alpha=0.2, lw=1)
        sl = [ms([slowdown(L.get(("seeds", m, 0.0, s)), L[("seeds", m, p, s)])[0] for s in seed_list(L, m, p)]) for p in ps]
        te = [ms([te_at_stop(L[("seeds", m, p, s)]) for s in seed_list(L, m, p)]) for p in ps]
        ax[0].errorbar(ps, [a for a, _, _ in sl], yerr=[0 if math.isnan(b) else b for _, b, _ in sl], color=col, marker=mk,
                       capsize=3, label=f"{lab} ({SETTING[m]})")
        ax[1].errorbar(ps, [a for a, _, _ in te], yerr=[0 if math.isnan(b) else b for _, b, _ in te], color=col, marker=mk,
                       capsize=3, label=lab)
        r1p = [(p, load_round1(m, p)) for p in ps]
        r1b = load_round1(m, 0.0)
        pts = [(p, slowdown(r1b, r), te_at_stop(r)) for p, r in r1p if r is not None]
        ax[0].plot([p for p, _, _ in pts], [s[0] for _, s, _ in pts], "o", mfc="none", mec=col, ms=9)
        ax[1].plot([p for p, _, _ in pts], [t for _, _, t in pts], "o", mfc="none", mec=col, ms=9)
    ax[0].set(xlabel="label corruption", ylabel="time to overfit (acc>=99%, rel. to own p=0)",
              title="(b) mean +/- std over seeds; thin = seeds; o = round 1")
    ax[1].set(xlabel="label corruption", ylabel="test_error at stop", ylim=(0, 1),
              title="(c) mean +/- std over seeds; thin = seeds; o = round 1")
    ax[1].axhline(0.9, color="r", ls="--")
    for m in MODEL_ORDER:
        for p, v in PAPER_C_BYEYE[m].items():
            ax[1].plot([p], [v], "x", color=STY[m][1], ms=8)
    ax[1].plot([], [], "kx", label="paper (by eye)")
    for a in ax: a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(ARGS.out, "fig_goal2_seeds.png"), dpi=200); plt.close(fig)


# ---------------------------------------------------------------- goal 4: past interpolation
KS = [0, 10, 25, 50]


def rel_arrays(r, kmin=-15):
    """test/train error aligned to epochs-since-stop, for k in [kmin, EXT_EPOCHS]; NaN-padded."""
    ks = list(range(kmin, EXT_EPOCHS + 1))
    te, tr = np.full(len(ks), np.nan), np.full(len(ks), np.nan)
    se = r["stop_epoch"]
    for i, k in enumerate(ks):
        j = se - 1 + k
        if 0 <= j < r["epochs"]: te[i], tr[i] = r["test_err"][j], r["train_err"][j]
    return ks, te, tr


def goal4(L):
    head = ["arm", "model", "p", "n", "stop_epoch"] + [f"test_err@+{k}" for k in KS] + ["delta test_err (+50 - 0)", "train_err@+50"]
    rows = []
    for arm in ("seeds", "lrhold"):
        for m in MODEL_ORDER:
            for p in EXT_PS:
                runs = [L[(arm, m, p, s)] for s in seed_list(L, m, p, arm) if L[(arm, m, p, s)].get("stop_epoch")
                        and L[(arm, m, p, s)]["epochs"] >= L[(arm, m, p, s)]["stop_epoch"] + EXT_EPOCHS]
                if not runs: continue
                arrs = [rel_arrays(r) for r in runs]
                ki = {k: arrs[0][0].index(k) for k in KS}
                cols = [fm(*ms([a[1][ki[k]] for a in arrs])[:2]) for k in KS]
                d = ms([a[1][ki[50]] - a[1][ki[0]] for a in arrs])
                trm = ms([a[2][ki[50]] for a in arrs])
                rows.append([arm, m, f"{p:.1f}", len(runs), fm(*ms([r["stop_epoch"] for r in runs])[:2], 1)] + cols +
                            [fm(d[0], d[1], 4), fm(trm[0], trm[1], 4)])
    return head, rows


def fig_goal4(L):
    fig, ax = plt.subplots(len(EXT_PS), len(MODEL_ORDER), figsize=(5 * len(MODEL_ORDER), 3.8 * len(EXT_PS)), squeeze=False)
    for r_i, p in enumerate(EXT_PS):
        for c_i, m in enumerate(MODEL_ORDER):
            a, col = ax[r_i][c_i], STY[m][1]
            runs = [L[("seeds", m, p, s)] for s in seed_list(L, m, p) if L[("seeds", m, p, s)].get("stop_epoch")]
            arrs = [rel_arrays(r) for r in runs]
            if arrs:
                ks = arrs[0][0]
                te = np.array([x[1] for x in arrs])
                for row in te: a.plot(ks, row, color=col, alpha=0.2, lw=1)
                mean, std = np.nanmean(te, 0), (np.nanstd(te, 0, ddof=1) if te.shape[0] > 1 else np.zeros(te.shape[1]))
                a.plot(ks, mean, color=col, lw=2, label=f"{STY[m][0]} {SETTING[m]} (n={len(arrs)})")
                a.fill_between(ks, mean - std, mean + std, color=col, alpha=0.15)
            if m == "inception" and ("lrhold", m, p, 0) in L:
                ks, te, _ = rel_arrays(L[("lrhold", m, p, 0)])
                a.plot(ks, te, "k--", lw=1.5, label="LR held at stop value (seed 0)")
            a.axvline(0, color="gray", ls=":")
            a.axhline(PAPER_C_BYEYE[m][p], color="r", ls="--", lw=1, label="paper (by eye)")
            a.set(xlabel="epochs since stop rule triggered", ylabel="test_error", title=f"{STY[m][0]}, p={p}")
            a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(ARGS.out, "fig_goal4_extension.png"), dpi=200); plt.close(fig)


# ---------------------------------------------------------------- goal 3: random pixels
def bin_windows(r, k=5):
    s, l = np.array(r["win_steps"]), np.array(r["win_loss"])
    n = (len(l) // k) * k
    return s[:n].reshape(-1, k)[:, -1], l[:n].reshape(-1, k).mean(1)


def steps_to_loss(r, thr=0.1):
    s, l = bin_windows(r)
    idx = np.where(l < thr)[0]
    return int(s[idx[0]]) if len(idx) else None


def goal3(P, probe):
    head = ["variant", "seed", "first-epoch mean loss", "loss in first 50 steps", "steps to loss<0.1 (250-step bins)",
            "final train_err", "final test_err"]
    rows = []
    for v in PIXEL_VARIANTS:
        for s in PIX_SEEDS:
            r = P.get((v, s))
            if r is None: continue
            stl = steps_to_loss(r)
            rows.append([v, s, fm(r["first_epoch_loss"], float("nan"), 4), fm(r["win_loss"][0], float("nan"), 4),
                         stl if stl is not None else "not reached", fm(r["final_train_err"], float("nan"), 4),
                         fm(r["final_test_err"], float("nan"), 4)])
    ph = ["input", "probe train_err", "probe test_err (0.9 = chance)"]
    prow = [[k, fm(v["train_err"], float("nan"), 4), fm(v["test_err"], float("nan"), 4)] for k, v in (probe or {}).items()]
    return head, rows, ph, prow


def fig_goal3(P, probe):
    cols = {"random_pixels_shared": "m", "random_pixels_flat": "tab:orange", "gaussian": "k"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ref = None
    if os.path.exists(ARGS.baseline_results):
        ref = json.load(open(ARGS.baseline_results))
        for key, c, lab in (("A|true_labels", "b", "true labels (round 1)"), ("A|random_labels", "r", "random labels (round 1)")):
            if key in ref:
                ax[0].plot(np.array(ref[key]["log_steps"]) / 1000, ref[key]["log_loss"], color=c, lw=1, alpha=0.5, label=lab)
    for v, c in cols.items():
        for s in PIX_SEEDS:
            r = P.get((v, s))
            if r is None: continue
            x, y = bin_windows(r)
            ax[0].plot(x / 1000, y, color=c, ls="-" if s == PIX_SEEDS[0] else "--", label=f"{v} s{s}")
            ax[1].plot(np.array(r["win_steps"]) / 1000, r["win_loss"], color=c, ls="-" if s == PIX_SEEDS[0] else "--", lw=1, label=f"{v} s{s}")
    ax[0].set(xlabel="thousand steps", ylabel="average_loss", title="(a) loss, 250-step bins")
    ax[1].set(xlabel="thousand steps", ylabel="average_loss", xlim=(0, 3), title="first 3k steps, 50-step windows")
    ax[1].axhline(math.log(10), color="gray", ls=":", label="ln 10")
    for a in ax[:2]: a.legend(fontsize=6)
    xs = list(cols)
    for i, v in enumerate(xs):
        for s in PIX_SEEDS:
            r = P.get((v, s))
            if r is not None: ax[2].plot([i + 0.1 * s], [r["final_test_err"]], "o", color=cols[v])
        if probe and v in probe: ax[2].plot([i - 0.2], [probe[v]["test_err"]], "D", mfc="none", mec=cols[v])
    if probe and "true_images" in probe: ax[2].axhline(probe["true_images"]["test_err"], color="gray", ls=":", label="probe on true images")
    ax[2].axhline(0.9, color="r", ls="--", label="0.9 = structure destroyed")
    ax[2].set_xticks(range(len(xs))); ax[2].set_xticklabels([x.replace("random_pixels_", "rp_") for x in xs], fontsize=8)
    ax[2].set(ylim=(0, 1), ylabel="test_error", title="o = trained Inception, diamond = linear probe")
    ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(ARGS.out, "fig_goal3_pixels.png"), dpi=200); plt.close(fig)


# ---------------------------------------------------------------- outputs
def write_epoch_csvs(L, P):
    rows = []
    for (c, m, p, s), r in sorted(L.items()):
        se = r.get("stop_epoch")
        for i in range(r["epochs"]):
            rows.append([c, m, p, s, i + 1, (i + 1 - se) if se else "", r["step"][i], r["lr"][i], r["online_loss"][i],
                         r["train_err"][i], r["test_err"][i]])
    write_csv("epoch_log_labels.csv", ["cond", "model", "p", "seed", "epoch", "epochs_since_stop", "step", "lr",
                                       "online_loss", "train_err", "test_err"], rows)
    rows = [[v, s, i + 1, r["epoch_loss"][i], r["test_err"][i]] for (v, s), r in sorted(P.items()) for i in range(r["epochs"])]
    write_csv("epoch_log_pixels.csv", ["variant", "seed", "epoch", "epoch_loss", "test_err"], rows)
    rows = [[v, s, a, b] for (v, s), r in sorted(P.items()) for a, b in zip(r["win_steps"], r["win_loss"])]
    write_csv("loss_windows_pixels.csv", ["variant", "seed", "step", "mean_loss_over_window"], rows)


def verdict_pixels(err):
    return "within 0.02 of 0.9 (no usable class signal)" if abs(err - 0.9) <= 0.02 else \
        f"{0.9 - err:+.3f} below 0.9 (class signal retained)" if err < 0.9 else "above 0.9"


README = """# Phase 3, round 2: seed noise, random pixels, past interpolation

Round 1 (`train_v2.py`, `results_phase3_r1_sweep/`, `results.json`) was not modified. This folder was produced by `train_r2.py`,
which imports round-1 models, data pipeline, label corruption, evaluation and constants unchanged. Numbers below are generated from the
per-run JSONs in `runs/`; none are hardcoded. Paper reference values are my readings by eye from Fig. 1 (about +-0.02) and are used only as
dashed lines and a table column.

## What changed vs round 1
- Goal 1 (done before these runs): `param_check.py` showed all four parameter counts match Table 1 (Inception only when BN gamma/beta are
  excluded; PyTorch's total is 3,424 higher). AlexNet: conv 5x5x64 x2, padding 2, max-pool 3x3 stride 2 padding 1, LRN after each pool, flatten 3136.
- Goal 2: seeds {seeds} at the round-1 winning settings ({setting}), p in {ps}. Init, data order and label corruption all follow the seed.
  Seed 0 is a fresh re-run, compared to the round-1 JSON to separate GPU nondeterminism from seed noise. Std is the sample std (ddof=1); with
  n = {n} it is a rough noise indication, not a confidence interval.
- Goal 4: for p in {ext_ps}, runs continue {ext} epochs after the stop rule first triggers (same LR schedule). Goal-2 statistics use the first-trigger
  epoch, so they match the round-1 definition. A second arm (Inception, seed 0) holds the LR at its stop-epoch value for the continuation.
- Goal 3: Inception, true labels, {pix_epochs} epochs, three input variants, seeds {pix_seeds}; loss logged every {log_every} steps. Plus a non-training
  linear probe on per-image channel mean/std. For seed 0, `random_pixels_shared` and `gaussian` use the same input seeds as round-1 panel (a).

## Results at a glance (auto-generated from the run files; interpretation is yours to add)
{glance}

## Still unexplained (static list; edit after reading the numbers)
- The ~1.2x step-scale gap in panel (a) between round 1 and the paper is untouched by this round.
- AlexNet's panel (b) slowdown (round 1: 2.2-3.7x vs ~1.55x in the paper) is only addressed here through the seed-noise estimate.
- The paper's concave panel (c) (round 1 tracked the line e0(1-p) + 0.9p): goal 4 tests only whether later stopping under the same or a held LR changes it.
- Fig. 1c's AlexNet p=0 test error (~0.30 by eye) differs from Table 1 (23.9%); not investigated.

## Files
`runs/` per-run JSON | `epoch_log_labels.csv`, `epoch_log_pixels.csv`, `loss_windows_pixels.csv` | `summary.md` (all tables), `summary_goal2_seeds.csv` |
`fig_goal2_seeds.png`, `fig_goal3_pixels.png`, `fig_goal4_extension.png` | goal 1 is a table (param_check.py output), not a figure
"""


def aggregate():
    L, P, probe = load_all()
    if not (L or P or probe):
        print("no runs found in", runs_dir()); return
    os.makedirs(ARGS.out, exist_ok=True)
    write_epoch_csvs(L, P)
    glance, md = [], []

    if L:
        h2, r2, h2b, r2b, stats = goal2(L)
        md += ["## Goal 2: seed noise (test error read at the first stop-rule trigger; slowdown at acc>=99% rel. to the same seed's p=0)", ""]
        md += md_table(h2, r2) + ["", "### Seed-0 re-run vs round 1 (same code/seed; differences = GPU nondeterminism)", ""] + md_table(h2b, r2b) + [""]
        fig_goal2(L)
        for m in MODEL_ORDER:
            if (m, 1.0) in stats:
                s = stats[(m, 1.0)]
                glance.append(f"- Goal 2, {m}: p=1 slowdown {fm(*s['slow'], 2)}{'>=' if s['ncens'] else ''} (n={s['n']}); "
                              f"paper approx. {v2.PAPER_SLOWDOWN_P1[m]}." +
                              (f" p=0.4 test error {fm(*stats[(m, 0.4)]['te'])} vs paper ~{PAPER_C_BYEYE[m][0.4]}." if (m, 0.4) in stats else ""))
        h4, r4 = goal4(L)
        if r4:
            md += ["## Goal 4: test error vs epochs since the stop rule first triggered (mean +/- std over seeds)", ""] + md_table(h4, r4) + [""]
            fig_goal4(L)
            for row in r4:
                glance.append(f"- Goal 4, {row[0]} {row[1]} p={row[2]} (n={row[3]}): test error change over the next 50 epochs {row[-2]} "
                              f"(train err at +50: {row[-1]}).")
    if P or probe:
        h3, r3, hp, rp = goal3(P, probe)
        md += ["## Goal 3: random pixels (Inception, true labels, 64 epochs)", ""] + md_table(h3, r3) + [""]
        if rp: md += ["### Linear probe on per-image channel mean/std", ""] + md_table(hp, rp) + [""]
        if P: fig_goal3(P, probe)
        for v in PIXEL_VARIANTS:
            errs = [P[(v, s)]["final_test_err"] for s in PIX_SEEDS if (v, s) in P]
            fl = [P[(v, s)]["first_epoch_loss"] for s in PIX_SEEDS if (v, s) in P]
            if errs:
                glance.append(f"- Goal 3, {v}: final test error {np.mean(errs):.3f} ({verdict_pixels(float(np.mean(errs)))}); "
                              f"first-epoch mean loss {np.mean(fl):.3f} (ln 10 = {math.log(10):.3f}).")
        for k, v in (probe or {}).items():
            glance.append(f"- Goal 3 probe, {k}: test error {v['test_err']:.3f} ({verdict_pixels(v['test_err'])}).")
    open(os.path.join(ARGS.out, "summary.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(ARGS.out, "README.md"), "w").write(README.format(
        seeds=SEEDS, setting=SETTING, ps=PS, n=len(SEEDS), ext_ps=EXT_PS, ext=EXT_EPOCHS, pix_epochs=PIX_EPOCHS,
        pix_seeds=PIX_SEEDS, log_every=LOG_EVERY, glance="\n".join(glance) if glance else "(no results yet)"))
    print("\n".join(md)); print("\n".join(glance))
    print("saved tables, figures, README to", ARGS.out)


def main(argv=None):
    global ARGS, SEEDS, EXT_EPOCHS, PIX_EPOCHS, LOG_EVERY, PIX_SEEDS, STOP_ERR, LOSS_THRESH
    ARGS = parse(argv)
    if ARGS.smoke:                          # pipeline test only: tiny, and the stop rule always triggers at epoch 1
        SEEDS, EXT_EPOCHS, PIX_EPOCHS, LOG_EVERY, PIX_SEEDS = [0, 1], 2, 2, 2, [0]
        STOP_ERR, LOSS_THRESH = 1.0, 1e9
        for c in v2.CONDITIONS.values(): c["cap"] = 2
    os.makedirs(runs_dir(), exist_ok=True)
    if not ARGS.plot_only:
        train_all()
    aggregate()


if __name__ == "__main__":
    main()
