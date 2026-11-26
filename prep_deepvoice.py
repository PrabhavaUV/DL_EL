"""
Prepare DEEP-VOICE for testing: cut long recordings into 2-second windows and
convert them to the same log-mel spectrograms used for FoR.

DEEP-VOICE is used as a TEST set only. Never train on it.

Step 1 (look at the data first):
    python prep_deepvoice.py inspect --data_dir "C:/Users/Admin/Desktop/KAGGLE/AUDIO"

Step 2 (convert):
    python prep_deepvoice.py run --data_dir "C:/Users/Admin/Desktop/KAGGLE/AUDIO" --out_dir processed_deepvoice

The script looks (recursively) for folders named REAL and FAKE under --data_dir.

Install first:  pip install librosa soundfile numpy tqdm
Output: processed_deepvoice/test.npz  (X, y, files)   -> one row per 2-second window
        processed_deepvoice/windows_per_file.csv
"""
import argparse
import random
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

SR = 16000
WIN = SR * 2
N_FFT, HOP, N_MELS = 1024, 256, 128          # same settings as preprocess.py
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}


def find_files(data_dir: Path):
    """Return {0: [real files], 1: [fake files]} from any folders named real / fake under data_dir."""
    out = {0: [], 1: []}
    for d in [data_dir] + [p for p in data_dir.rglob("*") if p.is_dir()]:
        n = d.name.lower()
        if n in ("real", "fake"):
            label = 0 if n == "real" else 1
            out[label] += [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXT]
    for k in out:
        out[k] = sorted(set(out[k]))
    return out


def inspect(data_dir: Path):
    files = find_files(data_dir)
    if not files[0] and not files[1]:
        raise SystemExit(f"No folders named REAL / FAKE with audio found under {data_dir}.\n"
                         "Send me a screenshot or the output of:  dir /s /b \"<that folder>\" | more")
    for label, name in [(0, "REAL"), (1, "FAKE")]:
        fl = files[label]
        infos = []
        for f in fl:
            try:
                i = sf.info(str(f))
                infos.append((f.name, i.samplerate, i.duration))
            except Exception as e:
                print(f"[warn] could not read {f.name}: {e}")
        if not infos:
            continue
        durs = np.array([d for _, _, d in infos])
        print(f"\n{name}: {len(fl)} files | total {durs.sum() / 60:.1f} min | "
              f"duration min/mean/max {durs.min():.1f}/{durs.mean():.1f}/{durs.max():.1f} s | "
              f"sampling rates {sorted(set(s for _, s, _ in infos))}")
        print(f"  about {int((durs // 2).sum())} non-overlapping 2-second windows")
        for n, s, d in infos[:6]:
            print(f"    {n}  ({d:.0f} s, {s} Hz)")
        if len(infos) > 6:
            print(f"    ... and {len(infos) - 6} more")


def logmel(y):
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=1.0, top_db=80.0).astype(np.float16)


def run(data_dir: Path, out_dir: Path, max_per_class: int, seed: int, min_rms: float):
    random.seed(seed)
    files = find_files(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wins = {0: [], 1: []}            # (file name, window array)
    for label in (0, 1):
        for f in tqdm(files[label], desc="REAL" if label == 0 else "FAKE"):
            try:
                y, _ = librosa.load(str(f), sr=SR, mono=True)
            except Exception as e:
                print(f"[skip] {f.name}: {e}")
                continue
            for s in range(0, len(y) - WIN + 1, WIN):          # non-overlapping 2 s windows, remainder dropped
                w = y[s:s + WIN]
                if np.sqrt(np.mean(w ** 2)) >= min_rms:        # skip digital silence
                    wins[label].append((f.name, w))
    n = min(len(wins[0]), len(wins[1]))
    if max_per_class > 0:
        n = min(n, max_per_class)
    print(f"\nwindows available: real={len(wins[0])}, fake={len(wins[1])} -> using {n} per class (balanced)")
    X, y, names = [], [], []
    for label in (0, 1):
        pick = random.sample(wins[label], n)
        for fname, w in tqdm(pick, desc="spectrograms " + ("real" if label == 0 else "fake")):
            X.append(logmel(w))
            y.append(label)
            names.append(fname)
    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    np.savez(out_dir / "test.npz", X=X, y=y, files=np.array(names))
    import collections
    cnt = collections.Counter(zip(names, y.tolist()))
    with open(out_dir / "windows_per_file.csv", "w", encoding="utf-8") as fh:
        fh.write("file,label,windows\n")
        for (fn, lab), c in sorted(cnt.items()):
            fh.write(f"{fn},{'fake' if lab else 'real'},{c}\n")
    print(f"saved {out_dir / 'test.npz'}: X={X.shape}, real={int((y == 0).sum())}, fake={int((y == 1).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["inspect", "run"])
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--out_dir", default=Path("processed_deepvoice"), type=Path)
    ap.add_argument("--max_per_class", type=int, default=3000, help="windows per class (0 = as many as balance allows)")
    ap.add_argument("--min_rms", type=float, default=1e-4, help="skip windows quieter than this")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    inspect(a.data_dir) if a.mode == "inspect" else run(a.data_dir, a.out_dir, a.max_per_class, a.seed, a.min_rms)