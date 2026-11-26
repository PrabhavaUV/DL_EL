"""
Gain-normalise the spectrograms and compare before vs after.

What it does
  1. Reads  processed/{train,val,test}.npz           (your original spectrograms, left untouched)
  2. Writes processed_norm/{train,val,test}.npz      (each clip minus its own average level)
  3. Re-runs the feature-group experiments on BOTH versions and prints them side by side.

Run from the folder that contains 'processed' AND feature_analysis.py:
    python normalize_and_compare.py

Outputs: processed_norm/*.npz, normalize_report.txt, normalize_comparison.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from feature_analysis import extract, group_of, fit_eval

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

LINES = []


def out(text=""):
    print(text)
    LINES.append(str(text))


SRC, DST = Path("processed"), Path("processed_norm")
DST.mkdir(exist_ok=True)

# These two become constant (always ~0) once each clip's mean is subtracted,
# so they are removed from BOTH versions to keep the comparison fair.
INVARIANT_DROP = ["loud_mean_db", "mfcc00_mean"]

# ---------------------------------------------------------------- 1. normalise
out("Step 1/3: creating normalised copies in 'processed_norm' ...")
X, Y, F_orig, F_norm = {}, {}, {}, {}
for sp in ["train", "val", "test"]:
    d = np.load(SRC / f"{sp}.npz")
    Xo = d["X"].astype(np.float32)
    Xn = Xo - Xo.mean(axis=(1, 2), keepdims=True)       # remove each clip's overall level (dB)
    np.savez(DST / f"{sp}.npz", X=Xn.astype(np.float16), y=d["y"], files=d["files"])
    X[sp], Y[sp] = (Xo, Xn), d["y"]
    out(f"  {sp}: {Xo.shape[0]} clips, mean level before = {Xo.mean():.1f} dB, after = {Xn.mean():.4f} dB")

# ------------------------------------------------------------------ 2. features
out("\nStep 2/3: extracting features from both versions (a few minutes) ...")
for sp in ["train", "val", "test"]:
    F_orig[sp] = extract(X[sp][0])
    F_norm[sp] = extract(X[sp][1])

feats = [f for f in F_orig["train"].columns if f not in INVARIANT_DROP]
for sp in F_orig:
    F_orig[sp], F_norm[sp] = F_orig[sp][feats], F_norm[sp][feats]

# sanity check: which features did normalisation actually change?
diff = {f: np.abs(F_orig["train"][f].values - F_norm["train"][f].values).max() for f in feats}
changed = sorted([f for f, v in diff.items() if v > 1e-2])
out(f"\nSanity check: {len(changed)} of {len(feats)} features changed by normalisation:")
out("  " + ", ".join(changed))
out("  All other features (all MFCC 1-19 mean/std, spectral shape, temporal, silence/noise)")
out("  were already independent of loudness, so they are identical before and after.")

# ------------------------------------------------------------------ 3. compare
out("\nStep 3/3: group experiments, before vs after (logistic regression, trained on train only)")
groups = {g: [f for f in feats if group_of(f) == g] for g in
          ["loudness", "silence_noise", "bands", "spectral_shape", "temporal", "mfcc"]}
no_mfcc = [f for f in feats if group_of(f) != "mfcc"]
no_mfcc_no_level = [f for f in no_mfcc if group_of(f) not in ("loudness", "bands")]

settings = [("ALL features", feats)]
settings += [(f"only {g}", c) for g, c in groups.items()]
settings += [("all EXCEPT mfcc", no_mfcc),
             ("all EXCEPT mfcc, loudness, bands", no_mfcc_no_level)]

rows = []
for name, cols in settings:
    a = fit_eval(cols, F_orig, Y, name)
    b = fit_eval(cols, F_norm, Y, name)
    rows.append({
        "setting": name, "n": len(cols),
        "val_auc_before": a["val_auc"], "val_auc_after": b["val_auc"],
        "test_auc_before": a["test_auc"], "test_auc_after": b["test_auc"],
        "test_acc_before": a["test_acc"], "test_acc_after": b["test_acc"],
        "test_eer_before": a["test_eer"], "test_eer_after": b["test_eer"],
    })
C = pd.DataFrame(rows)
C.to_csv("normalize_comparison.csv", index=False)
out(C.round(3).to_string(index=False))

# ---- per-feature view of the features that changed
from sklearn.metrics import roc_auc_score
out("\nSingle-feature AUC (fake = positive) for the features that changed:")
out(f"{'feature':<18}{'train_before':>13}{'train_after':>13}{'test_before':>13}{'test_after':>12}")
for f in changed:
    vals = []
    for F in (F_orig, F_norm):
        for sp in ("train", "test"):
            v = F[sp][f].values
            vals.append(0.5 if np.std(v) < 1e-9 else roc_auc_score(Y[sp], v))
    out(f"{f:<18}{vals[0]:>13.3f}{vals[2]:>13.3f}{vals[1]:>13.3f}{vals[3]:>12.3f}")

# ---- automatic reading of the result
out("\n" + "=" * 90)
out("HOW TO READ THIS")
out("=" * 90)
r = C.set_index("setting")
b, a = r.loc["all EXCEPT mfcc", "test_auc_before"], r.loc["all EXCEPT mfcc", "test_auc_after"]
out(f"'all EXCEPT mfcc' test AUC: {b:.3f} before  ->  {a:.3f} after normalisation.")
if a < 0.55:
    out("  -> Still at or below chance (0.5 = guessing). Removing the loudness cue did not rescue generalisation.")
elif a >= b - 0.05:
    out("  -> It held up. The non-MFCC features generalise even with the loudness cue removed.")
elif a > 0.65:
    out("  -> It dropped but is still clearly above chance. Part of the earlier score was the loudness cue.")
else:
    out("  -> It fell to a weak level. Most of the earlier score came from loudness/level cues, not real artefacts.")
c = r.loc["all EXCEPT mfcc, loudness, bands", "test_auc_after"]
out(f"Using only level-independent features (shape, temporal, silence/noise): test AUC {c:.3f}.")
out("Remember: test results have been looked at many times; treat DEEP-VOICE / for-rerec as the final check.")

with open("normalize_report.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(LINES))
out("\nSaved: processed_norm/, normalize_report.txt, normalize_comparison.csv")