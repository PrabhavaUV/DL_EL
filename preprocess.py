import argparse
import random
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

SR = 16000            # sampling rate we resample everything to
DURATION = 2          # seconds
N_SAMPLES = SR * DURATION
N_FFT = 1024
HOP = 256
N_MELS = 128
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}

# accepted folder names for each split / class (case-insensitive)
SPLIT_NAMES = {
    "train": ["training", "train"],
    "val": ["validation", "val", "valid"],
    "test": ["testing", "test"],
}
CLASS_NAMES = {0: ["real"], 1: ["fake"]}


def find_dir(parent: Path, names):
    """Find a direct subfolder of `parent` whose name matches one of `names`."""
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() in names:
            return child
    return None


def find_root(data_dir: Path):
    """Find the folder that actually contains the train/val/test folders
    (looks inside data_dir and up to 2 levels below it)."""
    candidates = [data_dir] + [p for p in data_dir.rglob("*") if p.is_dir() and len(p.relative_to(data_dir).parts) <= 2]
    for c in candidates:
        if find_dir(c, SPLIT_NAMES["train"]) is not None:
            return c
    return None


def list_audio(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXT)


def get_layout(data_dir: Path):
    root = find_root(data_dir)
    if root is None:
        raise SystemExit(
            f"Could not find a 'training' folder inside {data_dir}.\n"
            "Point --data_dir at the extracted folder, and tell me what the folder structure looks like."
        )
    layout = {}
    for split, names in SPLIT_NAMES.items():
        split_dir = find_dir(root, names)
        if split_dir is None:
            print(f"[warn] no folder found for split '{split}'")
            continue
        layout[split] = {}
        for label, cnames in CLASS_NAMES.items():
            class_dir = find_dir(split_dir, cnames)
            if class_dir is None:
                print(f"[warn] no '{cnames[0]}' folder inside {split_dir}")
                continue
            layout[split][label] = list_audio(class_dir)
    return root, layout


def inspect(data_dir: Path):
    root, layout = get_layout(data_dir)
    print(f"\nDataset root: {root}\n")
    print(f"{'split':<8}{'real':>10}{'fake':>10}")
    for split, classes in layout.items():
        print(f"{split:<8}{len(classes.get(0, [])):>10}{len(classes.get(1, [])):>10}")

    # check a random sample of files for sampling rate and duration
    all_files = [f for classes in layout.values() for files in classes.values() for f in files]
    sample = random.sample(all_files, min(30, len(all_files)))
    srs, durs, exts = set(), [], set()
    for f in sample:
        try:
            info = sf.info(str(f))
            srs.add(info.samplerate)
            durs.append(info.duration)
            exts.add(f.suffix.lower())
        except Exception as e:
            print(f"[warn] could not read {f.name}: {e}")
    if durs:
        print(f"\nFile types: {exts}")
        print(f"Sampling rates seen (in 30 random files): {srs}")
        print(f"Duration min/mean/max: {min(durs):.2f} / {np.mean(durs):.2f} / {max(durs):.2f} s")
    print("\nNow listen to ~10 real and ~10 fake clips by hand before moving on.")


def to_logmel(path: Path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    # force exactly 2 seconds: crop or zero-pad
    if len(y) > N_SAMPLES:
        y = y[:N_SAMPLES]
    elif len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=1.0, top_db=80.0).astype(np.float16)


def run(data_dir: Path, out_dir: Path, max_per_class: int, seed: int):
    random.seed(seed)
    root, layout = get_layout(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, classes in layout.items():
        X, y, names = [], [], []
        for label, files in classes.items():
            files = list(files)
            random.shuffle(files)
            if max_per_class > 0:
                # use a smaller subset for val/test so they don't dwarf training
                limit = max_per_class if split == "train" else max(max_per_class // 5, 500)
                files = files[:limit]
            for f in tqdm(files, desc=f"{split}/{'real' if label == 0 else 'fake'}"):
                try:
                    X.append(to_logmel(f))
                    y.append(label)
                    names.append(str(f.name))
                except Exception as e:
                    print(f"[skip] {f.name}: {e}")
        if not X:
            continue
        X = np.stack(X)
        y = np.array(y, dtype=np.int64)
        np.savez(out_dir / f"{split}.npz", X=X, y=y, files=np.array(names))
        print(f"saved {split}: X={X.shape}, real={int((y == 0).sum())}, fake={int((y == 1).sum())}")

    print("\nDone. Next: plot a few spectrograms from the .npz files and check they look sensible.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["inspect", "run"])
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--out_dir", default=Path("processed"), type=Path)
    ap.add_argument("--max_per_class", type=int, default=5000,
                    help="max training clips per class (0 = use everything)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mode == "inspect":
        inspect(args.data_dir)
    else:
        run(args.data_dir, args.out_dir, args.max_per_class, args.seed)