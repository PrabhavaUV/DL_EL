"""
Diagnose why test accuracy collapses: compare simple statistics of
train / val / test, per class, and check how easily a model can tell
the splits apart (a big shift = the splits come from different sources).

Run from the folder that contains 'processed':
    python diagnose.py
"""
import numpy as np
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)


def load(split):
    d = np.load(f"processed/{split}.npz")
    return d["X"].astype(np.float32), d["y"]


def clip_stats(X):
    """Simple per-clip numbers computed from the log-mel spectrograms (dB)."""
    mean_db = X.mean(axis=(1, 2))                    # overall level
    loud = X.max(axis=(1, 2))                        # loudest point in the clip
    low = X[:, :32, :].mean(axis=(1, 2))             # low-frequency energy
    high = X[:, 96:, :].mean(axis=(1, 2))            # high-frequency energy
    quiet = (X.mean(axis=1) < (X.max() - 70)).mean(axis=1)  # fraction of near-silent frames
    return np.stack([mean_db, loud, high - low, quiet], axis=1)


def mfcc_feats(X):
    out = []
    for s in X:
        m = librosa.feature.mfcc(S=s, n_mfcc=20)
        out.append(np.concatenate([m.mean(axis=1), m.std(axis=1)]))
    return np.array(out)


data = {s: load(s) for s in ["train", "val", "test"]}

print("Per-split, per-class statistics (mean over clips)")
print(f"{'split':<7}{'class':<7}{'mean dB':>9}{'peak dB':>9}{'high-low':>10}{'silent %':>10}")
for s, (X, y) in data.items():
    st = clip_stats(X)
    for label, name in [(0, "real"), (1, "fake")]:
        m = st[y == label].mean(axis=0)
        print(f"{s:<7}{name:<7}{m[0]:>9.1f}{m[1]:>9.1f}{m[2]:>10.1f}{m[3] * 100:>10.1f}")

print("\nCan a simple model tell the splits apart? (5-fold accuracy; 0.50 = cannot, 1.00 = trivially)")
feats = {s: mfcc_feats(X[rng.choice(len(X), min(1000, len(X)), replace=False)]) for s, (X, _) in data.items()}
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    F = np.vstack([feats[a], feats[b]])
    lab = np.array([0] * len(feats[a]) + [1] * len(feats[b]))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    acc = cross_val_score(clf, F, lab, cv=5).mean()
    print(f"  {a} vs {b}: {acc:.3f}")

print("\nSame check, but only on REAL clips (are the real voices different between splits?)")
for a, b in [("train", "test"), ("train", "val")]:
    Fa = mfcc_feats(data[a][0][data[a][1] == 0][:800])
    Fb = mfcc_feats(data[b][0][data[b][1] == 0][:800])
    F = np.vstack([Fa, Fb])
    lab = np.array([0] * len(Fa) + [1] * len(Fb))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    print(f"  real {a} vs real {b}: {cross_val_score(clf, F, lab, cv=5).mean():.3f}")

print("\nSame check, but only on FAKE clips (are the fake generators different between splits?)")
for a, b in [("train", "test"), ("train", "val")]:
    Fa = mfcc_feats(data[a][0][data[a][1] == 1][:800])
    Fb = mfcc_feats(data[b][0][data[b][1] == 1][:800])
    F = np.vstack([Fa, Fb])
    lab = np.array([0] * len(Fa) + [1] * len(Fb))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    print(f"  fake {a} vs fake {b}: {cross_val_score(clf, F, lab, cv=5).mean():.3f}")