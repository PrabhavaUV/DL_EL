"""
Transfer evaluation: train on for-2sec, then test on another source WITHOUT retraining.

Run from the folder that contains 'processed', 'processed_rerec' and feature_analysis.py:
    python eval_transfer.py                                   # tests on processed_rerec/test.npz
    python eval_transfer.py --target processed_deepvoice      # later, for DEEP-VOICE

Models (all logistic regression, trained on processed/train.npz only):
    ALL features | only MFCC | all EXCEPT MFCC | 12 level-independent features

Outputs: eval_report.txt, eval_results.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_analysis import extract, group_of, eer

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

LINES = []


def out(text=""):
    print(text)
    LINES.append(str(text))


DROP = ["loud_mean_db", "mfcc00_mean"]     # constant after level normalisation; dropped everywhere for consistency
rng = np.random.default_rng(0)


def load_feats(folder, split):
    d = np.load(f"{folder}/{split}.npz")
    return extract(d["X"].astype(np.float32)), d["y"]


def auc_ci(y, s, n=300):
    """AUC with a rough 95% bootstrap interval (half-width)."""
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), 1.96 * np.std(vals)


def main(target, target_splits):
    out("Extracting features (a few minutes) ...")
    Ftr, ytr = load_feats("processed", "train")
    evalsets = {
        "for-2sec val": load_feats("processed", "val"),
        "for-2sec test": load_feats("processed", "test"),
    }
    for sp in target_splits:
        evalsets[f"{target} {sp}"] = load_feats(target, sp)

    feats = [c for c in Ftr.columns if c not in DROP]
    level_indep = [f for f in feats if group_of(f) not in ("mfcc", "loudness", "bands")]
    settings = [
        ("ALL features", feats),
        ("only MFCC", [f for f in feats if group_of(f) == "mfcc"]),
        ("all EXCEPT MFCC", [f for f in feats if group_of(f) != "mfcc"]),
        (f"{len(level_indep)} level-independent", level_indep),
    ]

    rows = []
    for name, cols in settings:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Ftr[cols], ytr)
        for ename, (F, y) in evalsets.items():
            s = clf.decision_function(F[cols])
            auc, hw = auc_ci(y, s)
            pred = s > 0
            rows.append({
                "model": name, "evaluated_on": ename, "n_clips": len(y),
                "auc": auc, "auc_pm95": hw, "eer": eer(y, s), "acc": (pred == y).mean(),
                "fake_rate_on_real": pred[y == 0].mean(),
                "fake_rate_on_fake": pred[y == 1].mean(),
            })
    R = pd.DataFrame(rows)
    R.to_csv("eval_results.csv", index=False)

    out("\n" + "=" * 100)
    out("RESULTS (models trained on for-2sec train only; nothing retrained for the other sets)")
    out("=" * 100)
    out("auc = ranking quality (0.5 = guessing, 1.0 = perfect); auc_pm95 = rough 95% margin of error")
    out("fake_rate_on_real = share of REAL clips called fake (want ~0);  fake_rate_on_fake = share of FAKE clips called fake (want ~1)\n")
    out(R.round(3).to_string(index=False))

    out("\n" + "=" * 100)
    out("CHANNEL CHECK: how do the sources differ? (mean feature values, test sets)")
    out("=" * 100)
    chk = ["loud_peak_db", "sil_frac_30db", "noise_floor_rel", "shape_centroid_mean", "shape_flatness", "time_spectral_flux"]
    t = []
    for ename in [k for k in evalsets if "test" in k]:
        F, y = evalsets[ename]
        for lab, nm in [(0, "real"), (1, "fake")]:
            t.append({"source": ename, "class": nm, **{c: F.loc[y == lab, c].mean() for c in chk}})
    out(pd.DataFrame(t).round(3).to_string(index=False))

    out("\n" + "=" * 100)
    out("HOW TO READ THIS")
    out("=" * 100)
    base = R[R["evaluated_on"] == "for-2sec test"].set_index("model")["auc"]
    for ename in [k for k in evalsets if k not in ("for-2sec val", "for-2sec test")]:
        sub = R[R["evaluated_on"] == ename].set_index("model")
        out(f"\n{ename}: change in AUC compared with for-2sec test")
        for m in sub.index:
            d = sub.loc[m, "auc"] - base[m]
            if sub.loc[m, "auc"] < 0.45:
                tag = "BELOW CHANCE: confidently wrong"
            elif abs(d) < 0.05:
                tag = "about the same"
            else:
                tag = "better" if d > 0 else "worse"
            out(f"  {m:<28} {base[m]:.3f} -> {sub.loc[m, 'auc']:.3f}  ({d:+.3f}, {tag}; margin +/-{sub.loc[m, 'auc_pm95']:.3f})")
    out("\nCaution: the feature set and the 'drop MFCC' choice were picked by looking at for-2sec TEST results,")
    out("so that column is optimistic. The other sources are the fairer check. Run each only once and report it.")

    with open("eval_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES))
    out("\nSaved: eval_report.txt, eval_results.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="processed_rerec", help="folder with the other source's .npz files")
    ap.add_argument("--target_splits", nargs="+", default=["test"])
    a = ap.parse_args()
    main(a.target, a.target_splits)