"""
Train a small CNN on log-mel spectrograms to tell real from fake voices.

Trained ONLY on processed/train.npz (for-2sec). Evaluated at the end on:
    for-2sec val, for-2sec test, and any extra source you list with --eval
    (processed_rerec test is added automatically if the folder exists).

Examples (run from the folder that contains 'processed'):
    python train_cnn.py                         # augmentation ON, level-normalisation ON, seed 0
    python train_cnn.py --no_aug --name noaug   # ablation: no augmentation
    python train_cnn.py --no_norm --name nonorm # ablation: no per-clip level normalisation
    python train_cnn.py --seed 1                # another seed (run seeds 0-2 for the paper)
    python train_cnn.py --eval processed_deepvoice:test   # add another test source later

Install first:  pip install torch numpy pandas scikit-learn matplotlib

Each run saves to runs/<name>_s<seed>/ :
    model.pt            weights + settings (used later by the app)
    history.csv         loss / accuracy / AUC per epoch
    metrics.csv         final results on every evaluation set
    scores_<set>.npz    raw scores + labels (for ROC curves and score-distribution figures)
    curves.png          training curves
"""
import argparse
import json
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


# ----------------------------------------------------------------------------- data
def load_npz(folder, split):
    d = np.load(f"{folder}/{split}.npz")
    return torch.from_numpy(d["X"]), torch.from_numpy(d["y"]).float()      # X stays float16 on CPU


def level_norm(x):
    """Subtract each clip's own average level (dB) -> volume can't be used as a cue."""
    return x - x.mean(dim=(1, 2), keepdim=True)


# -------------------------------------------------------------------- augmentation
def augment(x, p_eq=0.8, p_noise=0.5):
    """x: (B, 128, T) log-mel in dB (raw, before normalisation). Simulates different channels."""
    B, M, T = x.shape
    x = x.clone()
    # 1) random EQ: a smooth random boost/cut (up to about +-6 dB) across the mel axis
    eq_mask = (torch.rand(B, 1, 1, device=x.device) < p_eq).float()
    knots = (torch.rand(B, 1, 6, device=x.device) * 2 - 1) * 6.0
    curve = Fn.interpolate(knots, size=M, mode="linear", align_corners=True).transpose(1, 2)   # (B, M, 1)
    x = x + curve * eq_mask
    # 2) random background noise, flat-ish with a random tilt, added in the POWER domain
    nz_mask = (torch.rand(B, 1, 1, device=x.device) < p_noise).float()
    peak = x.amax(dim=(1, 2), keepdim=True)
    n_db = peak - (35 + 45 * torch.rand(B, 1, 1, device=x.device))                  # 35-80 dB below the peak
    tilt = (torch.rand(B, 1, 1, device=x.device) * 2 - 1) * 20 * torch.linspace(0, 1, M, device=x.device).view(1, M, 1)
    noise_p = 10 ** ((n_db + tilt) / 10)
    x_noisy = 10 * torch.log10(10 ** (x / 10) + noise_p)
    x = torch.where(nz_mask > 0, x_noisy, x)
    # 3) random time shift (circular)
    shifts = torch.randint(-20, 21, (B,))
    x = torch.stack([torch.roll(x[i], int(shifts[i]), dims=1) for i in range(B)])
    return x


def spec_mask(x, n_time=2, max_t=20, n_freq=2, max_f=16):
    """SpecAugment on the NORMALISED input: blank random time and frequency bands (0 = clip average)."""
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

    def forward(self, x):                       # x: (B, 128, T), already level-normalised if that option is on
        x = ((x - self.mean) / self.std).unsqueeze(1)
        x = self.features(x).mean(dim=(2, 3))   # global average pooling
        return self.fc(self.drop(x)).squeeze(1)  # one logit: > 0 means "fake"


# -------------------------------------------------------------------------- metrics
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


@torch.no_grad()
def predict(model, X, norm, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].float().to(dev)
        xb = level_norm(xb) if norm else xb
        out.append(model(xb).cpu())
    return torch.cat(out).numpy()


def metrics(y, s):
    pred = s > 0
    return {
        "auc": float(roc_auc_score(y, s)), "eer": eer(y, s), "acc": float((pred == (y > 0.5)).mean()),
        "fake_rate_on_real": float(pred[y < 0.5].mean()), "fake_rate_on_fake": float(pred[y > 0.5].mean()),
    }


# ------------------------------------------------------------------------------ main
def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    name = f"{a.name}_s{a.seed}"
    out_dir = Path("runs") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr = load_npz("processed", "train")
    sets = {"for-2sec val": load_npz("processed", "val"), "for-2sec test": load_npz("processed", "test")}
    extra = list(a.eval)
    if not extra and Path("processed_rerec/test.npz").exists():
        extra = ["processed_rerec:test"]
    for e in extra:
        folder, split = e.split(":")
        sets[f"{folder} {split}"] = load_npz(folder, split)
    if a.limit:                                   # quick smoke test
        idx = torch.randperm(len(Xtr))[:a.limit]
        Xtr, ytr = Xtr[idx], ytr[idx]

    # fixed normalisation constants, from the (normalised or raw) TRAINING data only
    ref = Xtr[:2000].float()
    ref = level_norm(ref) if a.norm else ref
    model = SmallCNN(ref.mean().item(), ref.std().item()).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={dev} | params={n_params:,} | train clips={len(Xtr)} | aug={a.aug} | level-norm={a.norm} | seed={a.seed}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(Xtr) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)

    hist, t0 = [], time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot, correct, n = 0.0, 0, 0
        for i in range(0, len(Xtr), a.bs):
            idx = perm[i:i + a.bs]
            xb, yb = Xtr[idx].float().to(dev), ytr[idx].to(dev)
            if a.aug:
                xb = augment(xb)
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
        # per-epoch AUC on every set. val is the usual monitor; the others are DIAGNOSTIC ONLY:
        # we do not pick an epoch from them, the final epoch is always the one reported.
        for sname, (X, y) in sets.items():
            row[f"auc[{sname}]"] = roc_auc_score(y.numpy(), predict(model, X, a.norm))
        hist.append(row)
        print(f"ep {ep:02d} loss {row['train_loss']:.3f} acc {row['train_acc']:.3f} | " +
              " | ".join(f"{k[4:-1]} {v:.3f}" for k, v in row.items() if k.startswith("auc")) +
              f" | {time.time() - t0:.0f}s")
    H = pd.DataFrame(hist)
    H.to_csv(out_dir / "history.csv", index=False)

    # ---- final evaluation (last epoch)
    rows = []
    for sname, (X, y) in sets.items():
        s = predict(model, X, a.norm)
        m = metrics(y.numpy(), s)
        rows.append({"run": name, "evaluated_on": sname, "n_clips": len(y), **m})
        np.savez(out_dir / f"scores_{sname.replace(' ', '_').replace(':', '')}.npz", scores=s, y=y.numpy())
    M = pd.DataFrame(rows)
    M.to_csv(out_dir / "metrics.csv", index=False)
    print("\nFINAL RESULTS (last epoch):")
    print(M.round(3).to_string(index=False))

    torch.save({"state_dict": model.state_dict(), "norm": a.norm, "n_params": n_params,
                "config": vars(a) | {"eval": extra}}, out_dir / "model.pt")

    # ---- curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(H["epoch"], H["train_loss"], label="train loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    for c in [c for c in H.columns if c.startswith("auc")]:
        ax[1].plot(H["epoch"], H[c], label=c[4:-1])
    ax[1].axhline(0.5, color="k", lw=0.5)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("AUC")
    ax[1].legend(fontsize=8)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=140)
    plt.close(fig)
    print(f"\nSaved everything to {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cnn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--no_aug", dest="aug", action="store_false")
    ap.add_argument("--no_norm", dest="norm", action="store_false")
    ap.add_argument("--eval", nargs="*", default=[], help="extra sets as folder:split, e.g. processed_deepvoice:test")
    ap.add_argument("--limit", type=int, default=0, help="use only N training clips (smoke test)")
    main(ap.parse_args())