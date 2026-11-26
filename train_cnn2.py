"""
Train a small CNN on log-mel spectrograms to tell real from fake voices.

Training data: the TRAIN split of every source in --train (default: for-2sec AND rerec, if both exist).
Evaluation at the end (last epoch only) on: val + test of every training source, plus any
UNSEEN source you list with --eval (e.g. DEEP-VOICE). Never train on the --eval sets.

Examples (run from the folder that contains 'processed'):
    python train_cnn.py --epochs 10 --eval processed_deepvoice:test       # main run
    python train_cnn.py --train processed --name only2sec                  # for-2sec only (previous experiment)
    python train_cnn.py --no_aug --name noaug                              # ablation: no augmentation
    python train_cnn.py --no_norm --name nonorm                            # ablation: no level normalisation
    python train_cnn.py --aug_level 1 --name weakaug                       # ablation: the old, weaker augmentation
    python train_cnn.py --seed 1                                           # other seeds for the paper

Install first:  pip install torch numpy pandas scikit-learn matplotlib

Each run saves to runs/<name>_s<seed>/ :
    model.pt (weights + settings + calibrated cutoff, used by the app), history.csv, metrics.csv,
    scores_<set>.npz (for ROC curves), file_scores_<set>.csv (per-file verdicts, long recordings), curves.png
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {"processed": "for-2sec", "processed_rerec": "rerec", "processed_deepvoice": "deepvoice"}


def label_of(folder):
    return LABELS.get(folder, folder)


# ----------------------------------------------------------------------------- data
def load_npz(folder, split):
    d = np.load(f"{folder}/{split}.npz")
    return torch.from_numpy(d["X"]), torch.from_numpy(d["y"]).float(), d["files"]


def level_norm(x):
    """Subtract each clip's own average level (dB) -> volume can't be used as a cue."""
    return x - x.mean(dim=(1, 2), keepdim=True)


# -------------------------------------------------------------------- augmentation
def augment(x, level=2, p_eq=0.8, p_noise=0.6):
    """x: (B, 128, T) raw log-mel in dB. Simulates different microphones / rooms / noise.
    level 1 = random EQ (+-6 dB), noise, time shift.
    level 2 = level 1 but stronger, PLUS random gain, high-frequency roll-off and a reverb-like time smear."""
    B, M, T = x.shape
    d = x.device
    x = x.clone()
    strong = level >= 2
    eq_db, n_lo, n_rng, n_knots = (10.0, 25.0, 55.0, 8) if strong else (6.0, 35.0, 45.0, 6)

    if strong:                                                    # random overall gain
        x = x + (torch.rand(B, 1, 1, device=d) * 2 - 1) * 6.0
    # 1) random EQ: smooth random boost/cut across the mel axis
    eq_mask = (torch.rand(B, 1, 1, device=d) < p_eq).float()
    knots = (torch.rand(B, 1, n_knots, device=d) * 2 - 1) * eq_db
    curve = Fn.interpolate(knots, size=M, mode="linear", align_corners=True).transpose(1, 2)
    x = x + curve * eq_mask
    # 2) reverb-like smear along time (power domain): each frame leaks into the next few frames
    if strong:
        P = 10 ** (x / 10)
        tau = torch.where(torch.rand(B, device=d) < 0.5, torch.zeros(B, device=d), 1 + 7 * torch.rand(B, device=d))
        k = torch.arange(16, device=d).float().view(1, 16)
        w = torch.exp(-k / tau.clamp(min=1e-3).view(B, 1))
        w = w / w.sum(1, keepdim=True)
        out = torch.zeros_like(P)
        for j in range(16):
            sh = P if j == 0 else Fn.pad(P, (j, 0))[:, :, :T]
            out = out + w[:, j].view(B, 1, 1) * sh
        x = 10 * torch.log10(out + 1e-12)
        # 3) high-frequency roll-off (cheap microphones / speakers)
        cut = 40 + 60 * torch.rand(B, 1, 1, device=d)
        att = 25 * torch.rand(B, 1, 1, device=d) * (torch.rand(B, 1, 1, device=d) < 0.5).float()
        m = torch.arange(M, device=d).float().view(1, M, 1)
        x = x - att * ((m - cut) / (M - cut)).clamp(0, 1)
    # 4) random background noise with a random tilt, added in the power domain
    nz_mask = (torch.rand(B, 1, 1, device=d) < p_noise).float()
    peak = x.amax(dim=(1, 2), keepdim=True)
    n_db = peak - (n_lo + n_rng * torch.rand(B, 1, 1, device=d))
    tilt = (torch.rand(B, 1, 1, device=d) * 2 - 1) * 20 * torch.linspace(0, 1, M, device=d).view(1, M, 1)
    x_noisy = 10 * torch.log10(10 ** (x / 10) + 10 ** ((n_db + tilt) / 10))
    x = torch.where(nz_mask > 0, x_noisy, x)
    # 5) random time shift (circular)
    shifts = torch.randint(-20, 21, (B,))
    return torch.stack([torch.roll(x[i], int(shifts[i]), dims=1) for i in range(B)])


def spec_mask(x, n_time=2, max_t=20, n_freq=2, max_f=16):
    """SpecAugment on the NORMALISED input: blank random time and frequency bands."""
    B, M, T = x.shape
    x = x.clone()
    for i in range(B):
        for _ in range(n_time):
            w = int(torch.randint(0, max_t + 1, (1,)))
            s = int(torch.randint(0, max(T - w, 1), (1,)))
            x[i, :, s:s + w] = 0
        for _ in range(n_freq):
            w = int(torch.randint(0, max_f + 1, (1,)))
            s = int(torch.randint(0, max(M - w, 1), (1,)))
            x[i, s:s + w, :] = 0
    return x


# ----------------------------------------------------------------------------- model
class SmallCNN(nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        ch = [1, 16, 32, 64, 128]
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

    def forward(self, x):                        # x: (B, 128, T), level-normalised if that option is on
        x = ((x - self.mean) / self.std).unsqueeze(1)
        x = self.features(x).mean(dim=(2, 3))    # global average pooling
        return self.fc(self.drop(x)).squeeze(1)  # one logit: bigger = more "fake"


# -------------------------------------------------------------------------- metrics
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def pick_threshold(y, s):
    """Cutoff at the equal-error point of the (combined) validation scores."""
    fpr, tpr, thr = roc_curve(y, s)
    d = np.abs((1 - tpr) - fpr)
    ok = (d <= d.min() + 0.002) & np.isfinite(thr)
    return float(np.median(thr[ok]))


@torch.no_grad()
def predict(model, X, norm, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].float().to(dev)
        out.append(model(level_norm(xb) if norm else xb).cpu())
    return torch.cat(out).numpy()


def metrics(y, s, thr):
    out = {"auc": float(roc_auc_score(y, s)), "eer": eer(y, s)}
    for tag, t in (("", 0.0), ("_cal", thr)):
        pred = s > t
        out[f"acc{tag}"] = float((pred == (y > 0.5)).mean())
        out[f"fake_rate_on_real{tag}"] = float(pred[y < 0.5].mean())
        out[f"fake_rate_on_fake{tag}"] = float(pred[y > 0.5].mean())
    return out


# ------------------------------------------------------------------------------ main
def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    name = f"{a.name}_s{a.seed}"
    out_dir = Path("runs") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dirs = a.train or ([f for f in ["processed", "processed_rerec"] if Path(f"{f}/train.npz").exists()])
    parts = [load_npz(f, "train") for f in train_dirs]
    Xtr = torch.cat([p[0] for p in parts])
    ytr = torch.cat([p[1] for p in parts])
    print("training sources: " + ", ".join(f"{label_of(f)}={len(p[1])}" for f, p in zip(train_dirs, parts)))

    # warn if clips of one source's test split look like the same recordings as another source's train split
    if len(train_dirs) > 1:
        stem = lambda n: Path(str(n)).name.split(".")[0]
        for fa in train_dirs:
            tr_names = {stem(n) for n in load_npz(fa, "train")[2]}
            for fb in train_dirs:
                if fa != fb and Path(f"{fb}/test.npz").exists():
                    te = [stem(n) for n in load_npz(fb, "test")[2]]
                    hit = sum(n in tr_names for n in te)
                    print(f"overlap check: {hit}/{len(te)} of {label_of(fb)} TEST file names also appear in {label_of(fa)} TRAIN"
                          + ("   <-- possible leakage, tell me" if hit > 0.05 * len(te) else ""))

    sets, val_keys = {}, []
    for f in train_dirs:
        for sp in ("val", "test"):
            if Path(f"{f}/{sp}.npz").exists():
                k = f"{label_of(f)} {sp}"
                sets[k] = load_npz(f, sp)
                if sp == "val":
                    val_keys.append(k)
    for e in a.eval:
        folder, split = e.split(":")
        sets[f"{label_of(folder)} {split} (UNSEEN)"] = load_npz(folder, split)
    if a.limit:
        idx = torch.randperm(len(Xtr))[:a.limit]
        Xtr, ytr = Xtr[idx], ytr[idx]

    ref = Xtr[:2000].float()
    ref = level_norm(ref) if a.norm else ref
    model = SmallCNN(ref.mean().item(), ref.std().item()).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={dev} | params={n_params:,} | train clips={len(Xtr)} | aug={'off' if not a.aug else a.aug_level} "
          f"| level-norm={a.norm} | seed={a.seed}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * ((len(Xtr) + a.bs - 1) // a.bs))

    hist, t0 = [], time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot, correct, n = 0.0, 0, 0
        for i in range(0, len(Xtr), a.bs):
            idx = perm[i:i + a.bs]
            xb, yb = Xtr[idx].float().to(dev), ytr[idx].to(dev)
            if a.aug:
                xb = augment(xb, a.aug_level)
            xb = level_norm(xb) if a.norm else xb
            if a.aug:
                xb = spec_mask(xb)
            logit = model(xb)
            loss = Fn.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            correct += ((logit > 0).float() == yb).sum().item()
            n += len(idx)
        row = {"epoch": ep, "train_loss": tot / n, "train_acc": correct / n}
        if ep % a.eval_every == 0 or ep == a.epochs:
            # DIAGNOSTIC ONLY: we never pick an epoch from these. The final epoch is always the one reported.
            for sname, (X, y, _) in sets.items():
                row[f"auc[{sname}]"] = roc_auc_score(y.numpy(), predict(model, X, a.norm))
        hist.append(row)
        print(f"ep {ep:02d} loss {row['train_loss']:.3f} acc {row['train_acc']:.3f}"
              + "".join(f" | {k[4:-1]} {v:.3f}" for k, v in row.items() if k.startswith("auc")) + f" | {time.time() - t0:.0f}s")
    H = pd.DataFrame(hist)
    H.to_csv(out_dir / "history.csv", index=False)

    # ---- calibrated cutoff from the validation sets of the training sources
    scores = {k: predict(model, X, a.norm) for k, (X, y, _) in sets.items()}
    if val_keys:
        yv = np.concatenate([sets[k][1].numpy() for k in val_keys])
        sv = np.concatenate([scores[k] for k in val_keys])
        thr = pick_threshold(yv, sv)
    else:
        thr = 0.0
    print(f"\ncalibrated cutoff (from validation): {thr:.3f}   (the plain cutoff is 0)")

    rows = []
    for sname, (X, y, files) in sets.items():
        s, yy = scores[sname], y.numpy()
        rows.append({"run": name, "evaluated_on": sname, "n_clips": len(yy), **metrics(yy, s, thr)})
        tag = sname.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
        np.savez(out_dir / f"scores_{tag}.npz", scores=s, y=yy)
        if len(set(files)) < len(yy) / 3:                         # long recordings cut into windows -> per-file verdicts
            df = pd.DataFrame({"file": files, "y": yy, "score": s, "flag_cal": s > thr})
            g = df.groupby("file").agg(label=("y", "first"), windows=("y", "size"), mean_score=("score", "mean"),
                                       frac_windows_fake=("flag_cal", "mean")).reset_index()
            g["label"] = g["label"].map({0.0: "real", 1.0: "fake"})
            g.to_csv(out_dir / f"file_scores_{tag}.csv", index=False)
    M = pd.DataFrame(rows)
    M.to_csv(out_dir / "metrics.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\nFINAL RESULTS (last epoch)   *_cal = using the calibrated cutoff:")
    print(M.drop(columns=["run"]).round(3).to_string(index=False))

    torch.save({"state_dict": model.state_dict(), "norm": a.norm, "n_params": n_params, "threshold": thr,
                "config": vars(a) | {"train": train_dirs, "eval": list(a.eval)}}, out_dir / "model.pt")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(H["epoch"], H["train_loss"])
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("train loss")
    for c in [c for c in H.columns if c.startswith("auc")]:
        h = H.dropna(subset=[c])
        ax[1].plot(h["epoch"], h[c], marker="o", ms=3, label=c[4:-1])
    ax[1].axhline(0.5, color="k", lw=0.5)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("AUC")
    ax[1].legend(fontsize=7)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=140)
    plt.close(fig)
    print(f"\nSaved everything to {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cnn_mix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train", nargs="*", default=None, help="training source folders (default: processed processed_rerec)")
    ap.add_argument("--eval", nargs="*", default=[], help="UNSEEN test sets as folder:split, e.g. processed_deepvoice:test")
    ap.add_argument("--no_aug", dest="aug", action="store_false")
    ap.add_argument("--aug_level", type=int, default=2, choices=[1, 2])
    ap.add_argument("--no_norm", dest="norm", action="store_false")
    ap.add_argument("--eval_every", type=int, default=2, help="print diagnostic AUCs every N epochs (saves time)")
    ap.add_argument("--limit", type=int, default=0, help="use only N training clips (smoke test)")
    main(ap.parse_args())