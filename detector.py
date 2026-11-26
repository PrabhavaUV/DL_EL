"""
Scoring engine for the real/fake voice demo. No user-interface code lives here.

Pipeline (same as training):
    audio file -> 16 kHz mono -> 2-second windows -> log-mel spectrogram (128 x 126)
    -> (optional) subtract the window's own average level -> CNN -> one score per window
Higher score = more likely fake.
"""
import io

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

SR = 16000
WIN = SR * 2
N_FFT, HOP, N_MELS = 1024, 256, 128          # identical to preprocess.py
AUDIO_TYPES = ["wav", "mp3", "flac", "ogg"]


# ---------------------------------------------------------------------------- model
class SmallCNN(nn.Module):
    """Same architecture as train_cnn2.py (97,761 parameters)."""

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

    def forward(self, x):
        x = ((x - self.mean) / self.std).unsqueeze(1)
        x = self.features(x).mean(dim=(2, 3))
        return self.fc(self.drop(x)).squeeze(1)


def load_model(path):
    """Load a model.pt written by train_cnn2.py. Returns a dict with the model and its settings."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:                       # older / richer checkpoints: our own file, so this is fine
        ck = torch.load(path, map_location="cpu", weights_only=False)
    model = SmallCNN()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return {
        "model": model,
        "norm": bool(ck.get("norm", True)),
        "threshold": float(ck.get("threshold", 0.0)),
        "n_params": int(ck.get("n_params", sum(p.numel() for p in model.parameters()))),
        "config": ck.get("config", {}),
    }


# ---------------------------------------------------------------------------- audio
def load_audio(data, max_seconds=60):
    """data: bytes or a path. Returns (waveform at 16 kHz mono, seconds analysed, total seconds in the file)."""
    raw = bytes(data) if isinstance(data, (bytes, bytearray)) else None
    y, _ = librosa.load(io.BytesIO(raw) if raw is not None else data, sr=SR, mono=True, duration=max_seconds)
    total = len(y) / SR
    try:
        total = float(sf.info(io.BytesIO(raw) if raw is not None else data).duration)
    except Exception:
        pass
    return y.astype(np.float32), len(y) / SR, total


def make_windows(y, overlap=0.0, min_rms=1e-4):
    """Cut into 2-second windows. Returns (list of windows, start times in seconds, number of silent windows skipped).
    A clip shorter than 2 s is zero-padded to one window (as in training)."""
    if len(y) < WIN:
        w = np.pad(y, (0, WIN - len(y)))
        return [w], [0.0], 0
    step = max(int(WIN * (1.0 - overlap)), 1)
    wins, starts, skipped = [], [], 0
    for s in range(0, len(y) - WIN + 1, step):
        w = y[s:s + WIN]
        if np.sqrt(np.mean(w ** 2)) < min_rms:
            skipped += 1
            continue
        wins.append(w)
        starts.append(s / SR)
    return wins, starts, skipped


def logmel(w):
    """128 x 126 log-mel in dB, exactly as in training (including the half-precision rounding used when stored)."""
    S = librosa.feature.melspectrogram(y=w, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=1.0, top_db=80.0).astype(np.float16).astype(np.float32)


@torch.no_grad()
def score_windows(bundle, windows):
    X = torch.from_numpy(np.stack([logmel(w) for w in windows]))
    if bundle["norm"]:
        X = X - X.mean(dim=(1, 2), keepdim=True)
    return bundle["model"](X).numpy()


def analyze(bundle, data, overlap=0.0, max_seconds=60, min_rms=1e-4):
    y, seconds, total = load_audio(data, max_seconds)
    wins, starts, skipped = make_windows(y, overlap, min_rms)
    if not wins:
        return {"ok": False, "reason": "The clip is silent (or too quiet) everywhere, so there is nothing to analyse."}
    scores = score_windows(bundle, wins)
    return {"ok": True, "y": y, "seconds": seconds, "total_seconds": total, "starts": np.array(starts),
            "scores": scores, "mean_score": float(scores.mean()), "n_windows": len(wins), "skipped": skipped,
            "short_clip": len(y) < WIN}


# -------------------------------------------------------------------------- verdicts
def verdict(mean_score, cutoff, band):
    """Three-way verdict from the distance between the clip's score and the cutoff."""
    m = mean_score - cutoff
    if m > band:
        return "likely fake"
    if m < -band:
        return "likely real"
    return "uncertain"


def suggest_cutoff(real_scores, fake_scores):
    """Cutoff that maximises balanced accuracy on labelled example clips, plus the AUC.
    With very few clips this is only a rough guide."""
    r, f = np.asarray(real_scores, float), np.asarray(fake_scores, float)
    allv = np.unique(np.concatenate([r, f]))
    if len(allv) < 2:
        return None
    cands = np.concatenate([[allv[0] - 1], (allv[:-1] + allv[1:]) / 2, [allv[-1] + 1]])
    best, best_ba = cands[0], -1.0
    for c in cands:
        ba = 0.5 * ((r <= c).mean() + (f > c).mean())
        if ba > best_ba:
            best, best_ba = c, ba
    auc = float(np.mean([(fv > rv) + 0.5 * (fv == rv) for fv in f for rv in r]))
    return {"cutoff": float(best), "balanced_accuracy": float(best_ba), "auc": auc}