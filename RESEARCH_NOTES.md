# Research & Experiment Log: Cross-Source Audio Deepfake Detection (2-Second Clips)

> Living lab notebook for journal paper preparation. All metrics and observations below are recorded directly from executed workspace scripts, reports, and saved artifacts.

---

## 1. Research Question & Core Problem
Can a lightweight deepfake audio detector trained on 2-second speech clips generalize across **recording/channel shifts** and **unseen synthesis/voice-conversion generators** without leaking dataset shortcuts?

### Datasets Used
1. **Fake-or-Real (`for-2sec`, folder `processed/`):**
   - `train`: 13,956 clips | `val`: 2,826 clips | `test`: 1,088 clips
   - Log-mel spectrograms: `16 kHz` mono, `2 s` (`32,000` samples, crop/zero-pad first 2s), `N_FFT=1024`, `HOP=256`, `N_MELS=128`, power-to-dB (`top_db=80.0`, `float16`) via `preprocess.py`.
2. **Fake-or-Real Re-recorded (`for-rerec`, folder `processed_rerec/`):**
   - `train`: 10,208 clips | `val`: 2,244 clips | `test`: 816 clips
3. **DEEP-VOICE (`processed_deepvoice/`, strictly UNSEEN test set):**
   - `test`: 3,740 balanced 2-second non-overlapping windows (`1,870` real, `1,870` fake) across 64 long recordings (8 real speakers + 56 voice-converted fake files), skipping digital silence (`min_rms=1e-4`) via `prep_deepvoice.py`.

---

## 2. Diagnostic Findings: Dataset Shift & Shortcuts

### Finding 1 — `for-2sec` has a severe train/val vs. test domain shift (`diagnose.py`)
A 5-fold Logistic Regression on 40 MFCC summary features (mean + std of 20 MFCCs) measuring how easily splits can be told apart (`0.50` = identical distribution, `1.00` = trivially separable):
- `for-2sec train vs val`: **`0.511`** (same source/channel)
- `for-2sec train vs test`: **`0.950`** (completely different channel/source)

### Finding 2 — `for-rerec` is a re-recording of the same utterances in the same splits (`check_sources.py`, `check_sources_report.txt`)
- **Filename overlap:** `0` exact and `0` stem matches (`file*.wav...` vs `recording*.wav`), so filenames alone cannot verify overlap.
- **Audio loudness-envelope correlation (lags `-20..20`):**
  - `rerec val vs for-2sec val`: median `0.886`, 90th pct `0.962`, share `>0.9` = **`0.43`**
  - `rerec test vs for-2sec test`: median `0.872`, 90th pct `0.945`, share `>0.9` = **`0.34`**
  - Cross-split comparisons (`rerec val/test vs for-2sec train`, etc.): share `>0.9` = `0.00–0.01` (matching reference `0.02–0.04`).
  - **Implication:** `for-rerec` re-records the same utterances in matching splits, so combining `for-2sec/train` and `rerec/train` does **not** leak `val`/`test` utterances.
- **Internal split shift in `for-rerec`:**
  - `rerec train vs val`: **`0.544`**
  - `rerec train vs test`: **`0.905`**
  - `rerec val vs test`: **`0.915`**

### Finding 3 — MFCCs act as inverted channel shortcuts (`feature_analysis.py`, `feature_report.txt`)
Across 66 interpretable features (6 groups: `loudness=6`, `silence_noise=3`, `bands=8`, `spectral_shape=6`, `temporal=3`, `mfcc=40`), evaluated with Logistic Regression trained only on `for-2sec train`:

| Feature Subset | # Features | `for-2sec val` AUC | `for-2sec test` AUC | `for-2sec test` EER |
| :--- | :---: | :---: | :---: | :---: |
| `ALL features` | 66 | 0.969 | 0.418 | 0.559 |
| `only mfcc` | 40 | 0.950 | **0.184** (inverted) | 0.729 |
| `all EXCEPT mfcc` | 26 | 0.917 | **0.872** | **0.211** |
| `only loudness` | 6 | 0.750 | 0.759 | 0.306 |
| `only temporal` | 3 | 0.737 | 0.636 | 0.399 |
| `only spectral_shape` | 6 | 0.735 | 0.616 | 0.463 |
| Gradient Boosting (`ALL`) | 66 | 1.000 | 0.647 | 0.384 |

- 13 of 40 MFCC features **flipped** direction (`auc_train > 0.60`, `auc_test < 0.35`), e.g., `mfcc16_std` (`0.658` → `0.079`), `mfcc13_std` (`0.643` → `0.087`), `mfcc08_std` (`0.717` → `0.164`).
- Most transferable individual features: `band_0` (`0.616` → `0.738`), `mfcc01_mean` (`0.639` → `0.711`), `shape_tilt` (`0.351` → `0.319`), `shape_bandwidth` (`0.397` → `0.320`), `time_spectral_flux` (`0.734` → `0.677`).

### Finding 4 — Per-clip loudness normalization check (`normalize_and_compare.py`, `normalize_report.txt`)
- Subtracting each clip's mean dB (`processed_norm/`) changed only 11 of 64 features (`band_0..band_7`, `loud_p10_frame`, `loud_p90_frame`, `loud_peak_db`; `loud_mean_db` and `mfcc00_mean` dropped as constants).
- `all EXCEPT mfcc` (25 features) held at **`0.872` test AUC** before and after normalization (`val_auc = 0.917`).
- Purely level-independent features (12 features: `spectral_shape`, `temporal`, `silence_noise`) achieved **`0.798` test AUC** (`val_auc = 0.847`).

---

## 3. Cross-Source Transfer Results (Linear Models & CNNs)

### A. Logistic Regression trained on `for-2sec train` only (`eval_transfer.py`)
From `eval_results_rerec.csv` (`eval_report_rerec.txt`) and `eval_results.csv` (`eval_report.txt`):

| Model | `for-2sec val` AUC | `for-2sec test` AUC | `rerec test` AUC (Acc) | `deepvoice test` AUC (Acc) |
| :--- | :---: | :---: | :---: | :---: |
| `ALL features` | 0.969 | 0.416 | 0.498 (0.439) | 0.605 (0.508) |
| `only MFCC` | 0.948 | 0.234 | 0.303 (0.412) | 0.632 (0.510) |
| `all EXCEPT MFCC` | 0.917 | 0.872 | **0.853** (0.566) | 0.632 (0.503) |
| `12 level-independent` | 0.847 | 0.798 | **0.864** (0.516) | **0.679** (0.510) |

- **Key failure mode:** Even when AUC stays high (`0.853`–`0.864` on `rerec test`), threshold `0` breaks completely due to channel shift (`fake_rate_on_real` = `0.863–0.968` on `rerec test`, and `fake_rate_on_fake` = `0.008–0.032` on `deepvoice test`).

### B. `SmallCNN` on `for-2sec` only (`train_cnn1.py`, run `runs/cnn_s0`)
- Architecture: 4 Conv2d+BatchNorm+ReLU+MaxPool2d blocks (`1->16->32->64->128`), global average pooling, Dropout(`0.3`), Linear(`128->1`), level-norm ON, level-1 augmentation + SpecAugment, 10 epochs.
- Results (`runs/cnn_s0/metrics.csv`):
  - `for-2sec val`: AUC `1.000`, EER `0.004`, Acc `0.997`
  - `for-2sec test`: AUC `0.892`, EER `0.203`, Acc `0.575` (`fake_rate_on_real=0.000`, `fake_rate_on_fake=0.151`)
  - `processed_rerec test`: AUC **`0.546`**, EER `0.458`, Acc `0.500` (`0.000` / `0.000`) — fails to transfer across re-recording channel.

### C. `SmallCNN` on `for-2sec + rerec` with Level-2 Augmentation (`train_cnn2.py`, run `runs/cnn_mix_s0`)
- Added `rerec/train`, random overall gain (`±6 dB`), stronger EQ (`±10 dB`, 8 knots), exponential time-smear (reverb proxy), high-frequency roll-off (`up to 25 dB`), tilted noise, time roll, SpecAugment, and validation-EER threshold calibration (`thr`).
- Results (`runs/cnn_mix_s0/metrics.csv`) + saved logit analysis:

| Evaluation Set | N Clips | AUC | EER | Acc (raw `@0`) | Acc (`_cal`) | Real Mean Logit | Fake Mean Logit | Oracle Best Acc (`@thr`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `for-2sec val` | 2,826 | 1.000 | 0.004 | 0.991 | 0.993 | -9.10 | +14.01 | 0.995 (`@ 3.30`) |
| `for-2sec test` | 1,088 | 0.830 | 0.261 | 0.631 | 0.597 | -5.54 | -1.72 | **0.750** (`@ -3.31`) |
| `rerec val` | 2,244 | 0.997 | 0.024 | 0.976 | 0.974 | -7.51 | +6.67 | 0.978 (`@ -0.52`) |
| `rerec test` | 816 | 0.815 | 0.267 | 0.701 | 0.657 | -3.33 | -0.19 | **0.740** (`@ -1.52`) |
| `deepvoice test (UNSEEN)` | 3,740 | **0.709** | 0.347 | 0.517 | 0.514 | -10.13 | -7.44 | 0.660 (`@ -8.26`) |

- **Per-file unseen `DEEP-VOICE` behavior (`file_scores_deepvoice_test_UNSEEN.csv`):** All 64 files had negative mean logits (`-12.55` to `-5.01`), causing `fake_rate_on_fake_cal = 0.028` (97.2% of fake windows missed at the validation-calibrated cutoff).

---

## 4. Root Causes of False Negatives on Generated Audio
1. **Validation threshold mismatch (Calibration Collapse):** `val` is in-distribution (`AUC ~0.998–1.000`), producing extreme positive/negative logits. On shifted or unseen audio, logits shift strongly negative, so a validation-calibrated cutoff classifies almost everything as "real".
2. **Closed-set generator overfitting:** `for-2sec` and `for-rerec` share the same legacy TTS artifacts. A tiny CNN trained from scratch learns *"FoR TTS artifact = fake; anything else = real"*, failing on unseen voice conversion / modern TTS.
3. **Lossy preprocessing & weak inductive bias:** `preprocess.py` crops only the first 2 seconds and discards phase into 128-bin mel spectrograms; `SmallCNN` uses global average pooling, which dilutes localized vocoder artifacts.

---

## 5. Hardware Constraints & Action Plan (Same-Day Target: ≥75% Cross-Source Accuracy)

### Observed Environment
- **CPU:** AMD Ryzen 5 3600 (6 cores / 12 threads)
- **GPU:** NVIDIA GeForce GTX 1050 Ti (4 GB VRAM, Pascal `sm_61`, driver `582.78`) — *currently unused because Python `3.14.5` has `torch 2.15.0+cpu` (`torch.cuda.is_available() == False`). Needs `cu126` build (since `cu128` drops `sm_61`).*

### Threshold Sweep on Saved `cnn_mix_s0` Logits (`model.pt` saved val `thr = 0.573`)
| Cutoff (`thr`) | `for-2sec test` Acc (R→F / F→F) | `rerec test` Acc (R→F / F→F) | `deepvoice` Acc (R→F / F→F) |
| :---: | :---: | :---: | :---: |
| `+0.57` (val) | 0.597 (0.01 / 0.20) | 0.657 (0.05 / 0.36) | 0.514 (0.00 / 0.03) |
| `0.00` | 0.631 (0.02 / 0.28) | 0.701 (0.08 / 0.49) | 0.517 (0.00 / 0.03) |
| `-1.50` | 0.696 (0.08 / 0.47) | **0.739** (0.25 / 0.72) | 0.528 (0.00 / 0.06) |
| `-2.00` | 0.724 (0.10 / 0.55) | 0.729 (0.32 / 0.78) | 0.535 (0.00 / 0.07) |
| `-2.50` | 0.734 (0.14 / 0.61) | 0.722 (0.40 / 0.84) | 0.543 (0.01 / 0.09) |
| `-3.00` | **0.744** (0.19 / 0.68) | 0.712 (0.46 / 0.89) | 0.554 (0.01 / 0.12) |

### Project Constraint & Planned Improvements (100% Trained From Scratch — No Pretrained Weights)
1. **Enable GPU acceleration:** Switch from `2.15.0+cpu` to CUDA-enabled PyTorch (`cu126` wheel compatible with `sm_61` and Python 3.14) for GTX 1050 Ti.
2. **Upgrade the from-scratch architecture (replacing plain 4-layer `SmallCNN`):**
   - **Better input:** LFCC or Mel + Delta/Delta-Delta with per-clip/per-band CMVN (mean + std normalization, not just global mean subtraction) to stop logit drift.
   - **Ideal lightweight from-scratch model:** **SE-ResNet (6–8 residual blocks) or LCNN (Light CNN with Max-Feature-Map)** + **Attentive Statistics Pooling** (captures localized vocoder glitches instead of blurring them via global average pooling).
3. **Out-of-domain threshold calibration & multi-window inference:**
   - Calibrate threshold on **channel-augmented / shifted validation** (never clean `val`), and aggregate overlapping 2s windows via top-k / high-percentile pooling.
4. **Generator diversity (if targeting modern unseen fakes):** Add at least 1 modern fake dataset (e.g., WaveFake) to training while keeping `DEEP-VOICE` strictly unseen for evaluation.
5. **Journal Paper Framing Notes:**
   - **Contribution 1:** Systematic proof that standard MFCC/CNN pipelines on FoR exploit channel/dataset shortcuts (`0.95` split-separability, inverted MFCC test AUC `0.184`).
   - **Contribution 2:** Controlled ablation showing non-MFCC / level-independent features + channel augmentation + re-recorded training stabilize same-generator cross-channel AUC (`0.815–0.872`), while exposing **logit-shift calibration collapse** (`0.51–0.66` accuracy despite `0.71–0.83` AUC).
   - **Contribution 3:** From-scratch **SE-ResNet / LCNN + Attentive Pooling + CMVN + OOD calibration** for fast, lightweight cross-domain detection without pretrained foundation models.

---

## 6. Repository State
- Remote: `https://github.com/PrabhavaUV/DL_EL.git` (committed locally on `main` at `d91ef1f`, excluding raw audio and `.npz` caches via `.gitignore`).
