"""
Feature analysis: which features carry the shortcut?

Computes ~65 named, interpretable features per clip from the processed
log-mel spectrograms, then answers:

  1. For EACH feature alone: how well does it separate real/fake on train,
     val and test (AUC)?  Does that survive the move to the test set?
  2. Which features SHIFT between train and test (same class, different source)?
  3. For GROUPS of features: how does a model trained only on that group
     (or on everything except that group) do on val and test?
  4. For the final model on all features: how important is each feature,
     measured on val vs on test?

Run from the folder that contains 'processed':
    python feature_analysis.py

Install first:  pip install scikit-learn librosa numpy pandas matplotlib

Outputs: feature_report.txt (everything printed), feature_report.csv,
         group_report.csv, feature_auc_scatter.png, feature_importance.png
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 50)

LINES = []


def out(text=""):
    print(text)
    LINES.append(str(text))


# ----------------------------------------------------------------------------
# 1. Feature extraction
# ----------------------------------------------------------------------------
def extract(X, chunk=500):
    """X: (N, 128, 126) log-mel spectrograms in dB -> DataFrame of features."""
    rows = []
    bins = np.arange(X.shape[1], dtype=np.float64)
    b = bins - bins.mean()
    for i in range(0, len(X), chunk):
        Xc = X[i:i + chunk].astype(np.float64)
        n = len(Xc)
        f = {}

        frame_e = Xc.mean(axis=1)                    # (n, T) loudness per frame (dB)
        spec_mean = Xc.mean(axis=2)                  # (n, 128) average spectrum (dB)
        peak_frame = frame_e.max(axis=1)

        # --- loudness group
        f["loud_mean_db"] = Xc.mean(axis=(1, 2))
        f["loud_peak_db"] = Xc.max(axis=(1, 2))
        f["loud_std_db"] = Xc.std(axis=(1, 2))
        p10, p90 = np.percentile(frame_e, 10, axis=1), np.percentile(frame_e, 90, axis=1)
        f["loud_p10_frame"] = p10
        f["loud_p90_frame"] = p90
        f["loud_dyn_range"] = p90 - p10

        # --- silence / noise group
        f["sil_frac_20db"] = (frame_e < (peak_frame[:, None] - 20)).mean(axis=1)
        f["sil_frac_30db"] = (frame_e < (peak_frame[:, None] - 30)).mean(axis=1)
        f["noise_floor_rel"] = p10 - peak_frame      # how far the quietest frames sit below the loudest

        # --- band energy group (8 bands of 16 mel bins, low -> high pitch)
        for k in range(8):
            f[f"band_{k}"] = Xc[:, k * 16:(k + 1) * 16, :].mean(axis=(1, 2))

        # --- spectral shape group
        P = 10.0 ** (Xc / 10.0)
        tot = P.sum(axis=1) + 1e-12                              # (n, T)
        cent = (P * bins[None, :, None]).sum(axis=1) / tot       # per-frame centroid
        w = tot / tot.sum(axis=1, keepdims=True)                 # weight frames by energy
        cent_mean = (cent * w).sum(axis=1)
        f["shape_centroid_mean"] = cent_mean
        f["shape_centroid_std"] = np.sqrt((w * (cent - cent_mean[:, None]) ** 2).sum(axis=1))
        bw = np.sqrt((P * (bins[None, :, None] - cent[:, None, :]) ** 2).sum(axis=1) / tot)
        f["shape_bandwidth"] = (bw * w).sum(axis=1)
        flat = np.exp(Xc.mean(axis=1) * np.log(10) / 10) / (P.mean(axis=1) + 1e-12)
        f["shape_flatness"] = (flat * w).sum(axis=1)
        f["shape_tilt"] = (spec_mean * b[None, :]).sum(axis=1) / (b ** 2).sum()
        f["shape_high_minus_low"] = spec_mean[:, 96:].mean(axis=1) - spec_mean[:, :32].mean(axis=1)

        # --- temporal group
        f["time_energy_std"] = frame_e.std(axis=1)
        f["time_energy_delta"] = np.abs(np.diff(frame_e, axis=1)).mean(axis=1)
        f["time_spectral_flux"] = np.abs(np.diff(Xc, axis=2)).mean(axis=(1, 2))

        df = pd.DataFrame(f)

        # --- mfcc group
        mf = []
        for s in Xc.astype(np.float32):
            m = librosa.feature.mfcc(S=s, n_mfcc=20)
            mf.append(np.concatenate([m.mean(axis=1), m.std(axis=1)]))
        mf = np.array(mf)
        for k in range(20):
            df[f"mfcc{k:02d}_mean"] = mf[:, k]
        for k in range(20):
            df[f"mfcc{k:02d}_std"] = mf[:, 20 + k]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def group_of(name):
    if name.startswith("loud_"):
        return "loudness"
    if name.startswith("sil_") or name.startswith("noise_"):
        return "silence_noise"
    if name.startswith("band_"):
        return "bands"
    if name.startswith("shape_"):
        return "spectral_shape"
    if name.startswith("time_"):
        return "temporal"
    return "mfcc"


# ----------------------------------------------------------------------------
# 2. Helpers
# ----------------------------------------------------------------------------
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def safe_auc(y, v):
    if np.std(v) < 1e-12:
        return 0.5
    return roc_auc_score(y, v)


def fit_eval(cols, F, Y, name):
    """Logistic regression on a subset of columns; train on train, report val/test."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    clf.fit(F["train"][cols], Y["train"])
    res = {"setting": name, "n_features": len(cols)}
    for sp in ["val", "test"]:
        s = clf.decision_function(F[sp][cols])
        res[f"{sp}_acc"] = accuracy_score(Y[sp], s > 0)
        res[f"{sp}_auc"] = roc_auc_score(Y[sp], s)
        res[f"{sp}_eer"] = eer(Y[sp], s)
    return res


# ----------------------------------------------------------------------------
# 3. Main
# ----------------------------------------------------------------------------
def main(proc):
    out("Loading spectrograms and extracting features (about a minute)...")
    F, Y = {}, {}
    for sp in ["train", "val", "test"]:
        d = np.load(f"{proc}/{sp}.npz")
        F[sp] = extract(d["X"].astype(np.float32))
        Y[sp] = d["y"]
    feats = list(F["train"].columns)
    groups = {g: [f for f in feats if group_of(f) == g] for g in
              ["loudness", "silence_noise", "bands", "spectral_shape", "temporal", "mfcc"]}
    out(f"{len(feats)} features in {len(groups)} groups: " +
        ", ".join(f"{g}={len(v)}" for g, v in groups.items()))

    # ---- per-feature analysis
    rec = []
    for f in feats:
        r = {"feature": f, "group": group_of(f)}
        for sp in ["train", "val", "test"]:
            v, y = F[sp][f].values, Y[sp]
            r[f"real_{sp}"] = v[y == 0].mean()
            r[f"fake_{sp}"] = v[y == 1].mean()
            r[f"auc_{sp}"] = safe_auc(y, v)
        # how far the same class moves between train and test (in train std units)
        sh = []
        for c in (0, 1):
            tr = F["train"][f].values[Y["train"] == c]
            te = F["test"][f].values[Y["test"] == c]
            sh.append(abs(te.mean() - tr.mean()) / (tr.std() + 1e-9))
        r["shift_tr_te"] = max(sh)
        d_tr, d_te = r["auc_train"] - 0.5, r["auc_test"] - 0.5
        if abs(d_tr) < 0.10:
            r["verdict"] = "weak"
        elif np.sign(d_tr) == np.sign(d_te) and abs(d_te) >= 0.10:
            r["verdict"] = "transfers"
        elif np.sign(d_tr) != np.sign(d_te) and abs(d_te) >= 0.10:
            r["verdict"] = "FLIPS"
        else:
            r["verdict"] = "fades"
        rec.append(r)
    R = pd.DataFrame(rec)

    # ---- models on all features
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(F["train"], Y["train"])
    coefs = pd.Series(lr[-1].coef_[0], index=feats)
    R["lr_coef"] = R["feature"].map(coefs)

    hgb = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(F["train"], Y["train"])
    out("Computing permutation importance (about a minute)...")
    imp = {}
    for sp in ["val", "test"]:
        pi = permutation_importance(hgb, F[sp], Y[sp], scoring="roc_auc", n_repeats=5, random_state=0, n_jobs=1)
        imp[sp] = pd.Series(pi.importances_mean, index=feats)
        R[f"perm_imp_{sp}"] = R["feature"].map(imp[sp])

    R.to_csv("feature_report.csv", index=False)

    # ---- print per-feature table
    out("\n" + "=" * 100)
    out("PART 1: EVERY FEATURE, ALONE")
    out("=" * 100)
    out("AUC = how well the feature alone separates real/fake, treating 'fake' as positive.")
    out("  0.50 = useless.  >0.50 = higher value means more fake.  <0.50 = higher value means more real.")
    out("verdict (based on train vs test AUC):")
    out("  transfers = same direction and still informative on test")
    out("  FLIPS     = informative on train but points the OPPOSITE way on test  (shortcut)")
    out("  fades     = informative on train, near-useless on test")
    out("  weak      = not informative even on train")
    out("shift = how far the same class moves between train and test (0 = none, 1 = one train std)\n")
    T = R.sort_values("auc_train", key=lambda s: -(s - 0.5).abs())
    cols = ["feature", "group", "auc_train", "auc_val", "auc_test", "shift_tr_te", "verdict",
            "perm_imp_val", "perm_imp_test", "lr_coef"]
    out(T[cols].round(3).to_string(index=False))

    # ---- group analysis
    out("\n" + "=" * 100)
    out("PART 2: GROUPS OF FEATURES (logistic regression, trained on train only)")
    out("=" * 100)
    G = [fit_eval(feats, F, Y, "ALL features")]
    for g, c in groups.items():
        G.append(fit_eval(c, F, Y, f"only {g}"))
    for g, c in groups.items():
        rest = [f for f in feats if f not in c]
        G.append(fit_eval(rest, F, Y, f"all EXCEPT {g}"))
    G = pd.DataFrame(G)
    G.to_csv("group_report.csv", index=False)
    out(G.round(3).to_string(index=False))

    out("\nGradient-boosting model on ALL features:")
    for sp in ["val", "test"]:
        s = hgb.predict_proba(F[sp])[:, 1]
        out(f"  {sp}: acc={accuracy_score(Y[sp], s > 0.5):.3f}  AUC={roc_auc_score(Y[sp], s):.3f}  EER={eer(Y[sp], s):.3f}")

    # ---- automatic insights
    out("\n" + "=" * 100)
    out("PART 3: AUTOMATIC INSIGHTS")
    out("=" * 100)
    vc = R["verdict"].value_counts()
    out("Verdicts: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
    out("By group (count of each verdict):")
    out(pd.crosstab(R["group"], R["verdict"]).to_string())

    flips = R[R["verdict"] == "FLIPS"].copy()
    flips["sev"] = (flips["auc_train"] - 0.5).abs() + (flips["auc_test"] - 0.5).abs()
    flips = flips.sort_values("sev", ascending=False).head(8)
    if len(flips):
        out("\nStrongest FLIPPING features (strong cue on train, opposite cue on test):")
        out(flips[["feature", "auc_train", "auc_test", "shift_tr_te"]].round(3).to_string(index=False))

    tr = R[R["verdict"] == "transfers"].copy()
    tr["strength"] = (tr["auc_test"] - 0.5).abs()
    tr = tr.sort_values("strength", ascending=False).head(8)
    if len(tr):
        out("\nMost TRANSFERABLE features (same cue on train and test):")
        out(tr[["feature", "auc_train", "auc_test", "shift_tr_te"]].round(3).to_string(index=False))
    else:
        out("\nNo feature transfers: none keeps a clear real/fake signal from train to test.")

    out("\nMost SHIFTED features between train and test (same class, different source):")
    out(R.sort_values("shift_tr_te", ascending=False).head(8)[["feature", "group", "shift_tr_te"]].round(3).to_string(index=False))

    cand = G[G["setting"].str.startswith("only")].sort_values("test_auc", ascending=False)
    out(f"\nBest single group on test: {cand.iloc[0]['setting']} (test AUC {cand.iloc[0]['test_auc']:.3f}, "
        f"test acc {cand.iloc[0]['test_acc']:.3f})")
    allrow = G[G["setting"] == "ALL features"].iloc[0]
    dropped = G[G["setting"].str.startswith("all EXCEPT")].sort_values("test_auc", ascending=False)
    gain = dropped.iloc[0]["test_auc"] - allrow["test_auc"]
    gname = dropped.iloc[0]["setting"].replace("all EXCEPT ", "")
    if gain > 0.02:
        out(f"Removing '{gname}' helps test AUC the most: {allrow['test_auc']:.3f} -> {dropped.iloc[0]['test_auc']:.3f}")
    else:
        out(f"Removing any single group does NOT rescue test performance (best case: dropping '{gname}', "
            f"{allrow['test_auc']:.3f} -> {dropped.iloc[0]['test_auc']:.3f}). The shortcut is spread across groups.")

    risky = R[(R["perm_imp_val"] > 0.005) & (R["perm_imp_test"] < 0)].sort_values("perm_imp_val", ascending=False).head(8)
    if len(risky):
        out("\nFeatures the boosted model leans on (val importance) that HURT it on test (negative test importance):")
        out(risky[["feature", "perm_imp_val", "perm_imp_test"]].round(4).to_string(index=False))

    # ---- figures
    colors = {"transfers": "tab:green", "FLIPS": "tab:red", "fades": "tab:orange", "weak": "tab:gray"}
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for v, c in colors.items():
        s = R[R["verdict"] == v]
        ax.scatter(s["auc_train"], s["auc_test"], c=c, label=f"{v} ({len(s)})", s=28, alpha=0.8)
    ax.axhline(0.5, color="k", lw=0.6)
    ax.axvline(0.5, color="k", lw=0.6)
    ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("AUC on train (single feature)")
    ax.set_ylabel("AUC on test (single feature)")
    ax.set_title("Does each feature's real/fake cue survive to the test set?\n"
                 "red points point the opposite way on test")
    top = R.assign(sev=(R["auc_train"] - 0.5).abs()).sort_values("sev", ascending=False).head(8)
    for _, r in top.iterrows():
        ax.annotate(r["feature"], (r["auc_train"], r["auc_test"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.legend(loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig("feature_auc_scatter.png", dpi=140)
    plt.close(fig)

    topf = R.sort_values("perm_imp_val", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(topf))
    ax.barh(y + 0.2, topf["perm_imp_val"], height=0.4, label="val importance")
    ax.barh(y - 0.2, topf["perm_imp_test"], height=0.4, label="test importance")
    ax.set_yticks(y)
    ax.set_yticklabels(topf["feature"], fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("drop in AUC when the feature is shuffled")
    ax.set_title("What the boosted model relies on (val) vs what helps on test")
    ax.legend()
    fig.tight_layout()
    fig.savefig("feature_importance.png", dpi=140)
    plt.close(fig)

    out("\nSaved: feature_report.txt, feature_report.csv, group_report.csv, "
        "feature_auc_scatter.png, feature_importance.png")
    with open("feature_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="processed")
    main(ap.parse_args().processed)