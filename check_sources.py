"""
How are for-2sec and for-rerec related, and is rerec's own val/test split structured like for-2sec's?

Run from the folder that contains 'processed' and 'processed_rerec':
    python check_sources.py

Part 1: file-name overlap between the two sources (exact name, and name before the first dot).
Part 2: audio-content check. Re-recordings of the same utterance should have almost the same
        loudness-over-time pattern. For each rerec clip we find its best match among for-2sec clips
        and compare with how well for-2sec clips match each other.
Part 3: can a simple model tell rerec train from rerec val / rerec test? (same test as Table III)

Output is printed and saved to check_sources_report.txt
"""
from pathlib import Path

import numpy as np
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

A, B = "processed", "processed_rerec"
LINES = []
rng = np.random.default_rng(0)


def out(t=""):
    print(t)
    LINES.append(str(t))


def load(folder, split):
    d = np.load(f"{folder}/{split}.npz")
    return d["X"].astype(np.float32), d["y"], [str(n) for n in d["files"]]


stem = lambda n: Path(n).name.split(".")[0]

# ------------------------------------------------------------------ part 1
out("PART 1: file-name overlap (for-2sec vs rerec)")
names = {(f, s): load(f, s)[2] for f in (A, B) for s in ("train", "val", "test")}
out("example names  for-2sec: " + ", ".join(names[(A, 'train')][:3]))
out("example names  rerec   : " + ", ".join(names[(B, 'train')][:3]))
out(f"{'rerec split':<14}{'for-2sec split':<16}{'exact name':>12}{'same stem':>12}   (of rerec clips)")
for sb in ("train", "val", "test"):
    for sa in ("train", "val", "test"):
        a_exact, a_stem = set(names[(A, sa)]), {stem(n) for n in names[(A, sa)]}
        nb = names[(B, sb)]
        out(f"{sb:<14}{sa:<16}{sum(n in a_exact for n in nb):>12}{sum(stem(n) in a_stem for n in nb):>12}   / {len(nb)}")
out("If every count is 0 AND the naming styles above differ, names cannot answer the question: see Part 2.")


# ------------------------------------------------------------------ part 2
def envelope(X):
    e = X.mean(axis=1)                               # (N, T) loudness over time
    e = e - e.mean(axis=1, keepdims=True)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


def best_match(Q, R, lags=range(-20, 21, 2)):
    """For every row of Q: the highest correlation with any row of R, allowing small time shifts."""
    best = np.full(len(Q), -1.0)
    for lag in lags:
        Ql = np.roll(Q, lag, axis=1)
        best = np.maximum(best, (Ql @ R.T).max(axis=1))
    return best


out("\nPART 2: audio-content similarity (loudness-over-time pattern, best match, 1.0 = identical)")
env = {(f, s): envelope(load(f, s)[0]) for f in (A, B) for s in ("train", "val", "test")}
sub = lambda e, n=800: e[rng.choice(len(e), min(n, len(e)), replace=False)]
res = {}
res["for-2sec test vs for-2sec train (different utterances: reference)"] = best_match(sub(env[(A, 'test')]), env[(A, 'train')])
res["for-2sec val  vs for-2sec train (reference)"] = best_match(sub(env[(A, 'val')]), env[(A, 'train')])
for sb in ("test", "val"):
    for sa in ("train", "val", "test"):
        res[f"rerec {sb} vs for-2sec {sa}"] = best_match(sub(env[(B, sb)]), env[(A, sa)])
out(f"{'comparison':<68}{'median':>8}{'90th pct':>10}{'share > 0.9':>13}")
for k, v in res.items():
    out(f"{k:<68}{np.median(v):>8.3f}{np.percentile(v, 90):>10.3f}{(v > 0.9).mean():>13.2f}")
out("Reading it: if rerec clips match some for-2sec clip far better than for-2sec clips match each other")
out("(share > 0.9 much larger than the reference rows), rerec is a re-recording of the same utterances.")


# ------------------------------------------------------------------ part 3
def mfcc_feats(X):
    f = []
    for s in X:
        m = librosa.feature.mfcc(S=s, n_mfcc=20)
        f.append(np.concatenate([m.mean(axis=1), m.std(axis=1)]))
    return np.array(f)


out("\nPART 3: can a simple model tell the rerec splits apart? (5-fold accuracy, 0.5 = cannot)")
F = {}
for s in ("train", "val", "test"):
    X = load(B, s)[0]
    F[s] = mfcc_feats(X[rng.choice(len(X), min(1000, len(X)), replace=False)])
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    Z = np.vstack([F[a], F[b]])
    lab = np.array([0] * len(F[a]) + [1] * len(F[b]))
    acc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Z, lab, cv=5).mean()
    out(f"  rerec {a} vs rerec {b}: {acc:.3f}")
out("For for-2sec these were 0.511 (train vs val) and 0.950 (train vs test).")

with open("check_sources_report.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(LINES))
out("\nSaved: check_sources_report.txt")