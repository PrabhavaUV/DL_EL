"""
Baseline: MFCC features + classical ML (Logistic Regression and SVM).

Run from the folder that contains the 'processed' folder:
    python baseline.py

Install first:  pip install scikit-learn librosa numpy
"""
import time
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_curve, confusion_matrix

N_MFCC = 20


def load(split):
    d = np.load(f"processed/{split}.npz")
    return d["X"].astype(np.float32), d["y"]


def mfcc_features(X):
    """X: (N, 128, 126) log-mel spectrograms -> (N, 40) features.
    MFCC = DCT of the log-mel spectrogram. We summarise each clip with the
    mean and standard deviation of every coefficient over time."""
    feats = []
    for spec in X:
        m = librosa.feature.mfcc(S=spec, n_mfcc=N_MFCC)      # (20, 126)
        feats.append(np.concatenate([m.mean(axis=1), m.std(axis=1)]))
    return np.array(feats)


def eer(y_true, scores):
    """Equal Error Rate: the point where false-accept rate equals false-reject rate."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def report(name, model, Xv, yv, Xt, yt):
    for split, X, y in [("val", Xv, yv), ("test", Xt, yt)]:
        pred = model.predict(X)
        scores = model.decision_function(X)
        print(f"[{name}] {split}: acc={accuracy_score(y, pred):.4f}  "
              f"F1={f1_score(y, pred):.4f}  EER={eer(y, scores):.4f}")
        if split == "test":
            print("confusion matrix (rows=true real/fake, cols=pred real/fake):")
            print(confusion_matrix(y, pred))
    print()


if __name__ == "__main__":
    Xtr, ytr = load("train")
    Xva, yva = load("val")
    Xte, yte = load("test")

    print("Extracting MFCC features...")
    t = time.time()
    Ftr, Fva, Fte = mfcc_features(Xtr), mfcc_features(Xva), mfcc_features(Xte)
    print(f"done in {time.time() - t:.0f}s, feature shape: {Ftr.shape}\n")

    # scale features using the TRAINING set only (never fit on val/test)
    scaler = StandardScaler().fit(Ftr)
    Ftr, Fva, Fte = scaler.transform(Ftr), scaler.transform(Fva), scaler.transform(Fte)

    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=2000).fit(Ftr, ytr)
    report("LogReg", lr, Fva, yva, Fte, yte)

    print("Training SVM (RBF)... this may take a minute or two")
    svm = SVC(kernel="rbf", C=1.0).fit(Ftr, ytr)
    report("SVM", svm, Fva, yva, Fte, yte)