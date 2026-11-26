"""
Real or fake voice? A research demo.

Run:   streamlit run app.py
"""
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import detector as det

CANDIDATE_MODELS = ["runs/cnn_mix_s0/model.pt", "../runs/cnn_mix_s0/model.pt", "model.pt"]
VERDICT_TEXT = {
    "likely fake": "LIKELY FAKE (synthetic or converted voice)",
    "likely real": "LIKELY REAL (human voice)",
    "uncertain": "UNCERTAIN: the score is too close to the cutoff to call",
}


# ----------------------------------------------------------------------------- cached work
@st.cache_resource(show_spinner=False)
def get_model(path):
    return det.load_model(path)


@st.cache_data(show_spinner=False, max_entries=64)
def run_analysis(model_path, data, overlap, max_seconds):
    return det.analyze(get_model(model_path), data, overlap=overlap, max_seconds=max_seconds)


def set_cutoff(value):
    st.session_state["cutoff_input"] = float(round(value, 3))


# ----------------------------------------------------------------------------- drawing
def draw_ruler(score, cutoff, band):
    fig, ax = plt.subplots(figsize=(8, 1.5))
    lo, hi = min(score, cutoff) - 4 * band - 1, max(score, cutoff) + 4 * band + 1
    ax.axvspan(lo, cutoff - band, color="tab:green", alpha=0.25)
    ax.axvspan(cutoff - band, cutoff + band, color="tab:orange", alpha=0.25)
    ax.axvspan(cutoff + band, hi, color="tab:red", alpha=0.25)
    ax.axvline(cutoff, color="k", lw=1, ls="--")
    ax.plot([score], [0.5], marker="v", color="k", ms=14)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("average score (left = more real-like, right = more fake-like)")
    ax.text(lo, 0.08, " real zone", color="darkgreen", fontsize=9)
    ax.text(hi, 0.08, "fake zone ", color="darkred", fontsize=9, ha="right")
    fig.tight_layout()
    return fig


def draw_spectrogram(y):
    S = librosa.feature.melspectrogram(y=y, sr=det.SR, n_fft=det.N_FFT, hop_length=det.HOP, n_mels=det.N_MELS)
    D = librosa.power_to_db(S, ref=1.0, top_db=80.0)
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(D, origin="lower", aspect="auto", extent=[0, len(y) / det.SR, 0, det.N_MELS])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel band (low to high pitch)")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    return fig


def draw_timeline(starts, scores, cutoff, band):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axhspan(cutoff - band, cutoff + band, color="tab:orange", alpha=0.2, label="uncertain zone")
    ax.axhline(cutoff, color="k", ls="--", lw=1, label="cutoff")
    ax.plot(starts + 1, scores, marker="o", color="tab:blue", label="2-second windows")
    ax.set_xlabel("time (s), centre of each window")
    ax.set_ylabel("score")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    return fig


def render_result(res, cutoff, band):
    """Show the verdict and the evidence for one analysed clip."""
    if not res.get("ok"):
        st.warning(res.get("reason", "Could not analyse this clip."))
        return
    v = det.verdict(res["mean_score"], cutoff, band)
    box = {"likely fake": st.error, "likely real": st.success, "uncertain": st.warning}[v]
    box(f"### {VERDICT_TEXT[v]}")
    frac = float((res["scores"] > cutoff).mean())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average score", f"{res['mean_score']:.2f}")
    c2.metric("Cutoff", f"{cutoff:.2f}")
    c3.metric("Windows above cutoff", f"{frac * 100:.0f}%")
    c4.metric("Windows analysed", f"{res['n_windows']}")
    st.pyplot(draw_ruler(res["mean_score"], cutoff, band))
    if res["short_clip"]:
        st.info("This clip is shorter than 2 seconds, so it was padded with silence. Short clips are less reliable.")
    if res["total_seconds"] > res["seconds"] + 0.5:
        st.info(f"Only the first {res['seconds']:.0f} s of this {res['total_seconds']:.0f} s file were analysed "
                "(change this in the sidebar).")
    if res["skipped"]:
        st.caption(f"{res['skipped']} silent window(s) were skipped.")
    with st.expander("Spectrogram and window-by-window scores", expanded=True):
        st.pyplot(draw_spectrogram(res["y"]))
        st.pyplot(draw_timeline(res["starts"], res["scores"], cutoff, band))
        st.dataframe(pd.DataFrame({"start (s)": res["starts"], "score": res["scores"].round(3),
                                   "above cutoff": res["scores"] > cutoff}), hide_index=True)


# ----------------------------------------------------------------------------- tabs
def tab_check(model_path, overlap, max_seconds, cutoff, band):
    st.write("Upload a speech clip. It is cut into 2-second pieces, each piece is scored by the model, "
             "and the scores are averaged.")
    up = st.file_uploader("Audio file", type=det.AUDIO_TYPES, key="single")
    if up is None:
        st.caption("Accepted: " + ", ".join(det.AUDIO_TYPES) + ". English speech works best.")
        return
    st.audio(up)
    try:
        with st.spinner("Analysing..."):
            res = run_analysis(model_path, up.getvalue(), overlap, max_seconds)
    except Exception as e:
        st.error("Could not read this file. Try converting it to WAV. "
                 f"(mp3 may need extra codecs on some computers.)\n\nDetails: {e}")
        return
    render_result(res, cutoff, band)


def tab_batch(model_path, overlap, max_seconds, cutoff, band):
    st.write("The model's cutoff was learned on one set of recordings and can be wrong for other recordings. "
             "Upload some clips whose answer you KNOW (real voices and fake voices). The app scores them, "
             "measures how well they separate, and suggests a cutoff that fits them.")
    c1, c2 = st.columns(2)
    real_files = c1.file_uploader("Known REAL clips", type=det.AUDIO_TYPES, accept_multiple_files=True, key="real")
    fake_files = c2.file_uploader("Known FAKE clips", type=det.AUDIO_TYPES, accept_multiple_files=True, key="fake")
    if not real_files and not fake_files:
        return
    rows = []
    for label, files in (("real", real_files), ("fake", fake_files)):
        for f in files or []:
            try:
                r = run_analysis(model_path, f.getvalue(), overlap, max_seconds)
            except Exception as e:
                rows.append({"file": f.name, "truth": label, "windows": 0, "score": np.nan, "verdict": f"unreadable ({e})"})
                continue
            if not r.get("ok"):
                rows.append({"file": f.name, "truth": label, "windows": 0, "score": np.nan, "verdict": "silent"})
                continue
            rows.append({"file": f.name, "truth": label, "windows": r["n_windows"], "score": round(r["mean_score"], 3),
                         "verdict": det.verdict(r["mean_score"], cutoff, band)})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)
    ok = df.dropna(subset=["score"])
    real_s, fake_s = ok[ok.truth == "real"]["score"].values, ok[ok.truth == "fake"]["score"].values
    if len(real_s) and len(fake_s):
        sug = det.suggest_cutoff(real_s, fake_s)
        correct = (((ok.truth == "real") & (ok.verdict == "likely real")) |
                   ((ok.truth == "fake") & (ok.verdict == "likely fake"))).mean()
        m1, m2, m3 = st.columns(3)
        m1.metric("Separation (AUC)", f"{sug['auc']:.2f}", help="1.0 = every fake scores above every real; 0.5 = no better than guessing")
        m2.metric("Clips called correctly at the current cutoff", f"{correct * 100:.0f}%", help="'uncertain' counts as not correct")
        m3.metric("Suggested cutoff", f"{sug['cutoff']:.2f}")
        st.button("Use the suggested cutoff", on_click=set_cutoff, args=(sug["cutoff"],))
        if len(real_s) + len(fake_s) < 10:
            st.caption("With so few clips the suggestion is only a rough guide.")
    else:
        st.info("Add at least one real and one fake clip to measure separation.")
    st.download_button("Download results (CSV)", df.to_csv(index=False), "batch_results.csv", "text/csv")


def tab_about(bundle):
    st.markdown(f"""
**How it works.** The clip is resampled to 16 kHz and cut into 2-second windows. Each window becomes a
log-mel spectrogram (a picture of which pitches are present over time). A small convolutional neural network
({bundle['n_params']:,} parameters, trained on a CPU-class machine) gives each window a score; the scores are averaged.

**What it was trained and tested on.** English speech from the Fake-or-Real (FoR) dataset, in two versions
(clean and re-recorded through a speaker and microphone). It was then tested once on DEEP-VOICE, a different
dataset of real recordings and voice-converted fakes of eight public figures.

**Measured results (AUC: 0.5 = guessing, 1.0 = perfect).**
FoR test splits: 0.83 and 0.82. DEEP-VOICE (never seen in training): 0.71.
On DEEP-VOICE the model's fixed cutoff labelled almost everything as real, so accuracy there was about 50%.

**Limits you should know about.**
- Ranking is better than the fixed cutoff. A cutoff learned on one set of recordings does not transfer reliably,
  which is why the app has a calibration tab and an 'uncertain' zone.
- Cues that separate real from fake in one dataset can weaken or reverse in another.
- Untested: other languages, music, phone-call audio, noisy rooms, and modern voice-cloning systems.
- A verdict is a statistical guess about a recording, not proof about a person.
""")


# ----------------------------------------------------------------------------- main
def main():
    st.set_page_config(page_title="Real or fake voice?", page_icon="🎙️", layout="wide")
    st.title("🎙️ Real or fake voice?")
    st.caption("A lightweight deep-learning research demo (CNN on log-mel spectrograms). Not a forensic tool.")

    with st.sidebar:
        st.header("Settings")
        default_path = next((p for p in CANDIDATE_MODELS if Path(p).exists()), CANDIDATE_MODELS[0])
        model_path = st.text_input("Model file", default_path)
        if not Path(model_path).exists():
            st.error("Model file not found. Put model.pt (from runs/cnn_mix_s0/) here, or type its path.")
            st.stop()
        bundle = get_model(model_path)
        if st.session_state.get("cutoff_model") != model_path:
            st.session_state["cutoff_input"] = float(round(bundle["threshold"], 3))
            st.session_state["cutoff_model"] = model_path
        cutoff = st.number_input("Cutoff (score above this = fake)", step=0.1, key="cutoff_input",
                                 help="Starts at the value learned from training validation data. "
                                      "Use the Calibrate tab to fit it to your own example clips.")
        band = st.slider("Uncertain zone (+/-)", 0.1, 5.0, 1.0, 0.1,
                         help="Scores within this distance of the cutoff are reported as 'uncertain'.")
        overlap = st.selectbox("Window overlap", [0.0, 0.5], format_func=lambda v: f"{int(v * 100)}%")
        max_seconds = st.slider("Max seconds analysed", 10, 120, 60, 10)

    t1, t2, t3 = st.tabs(["Check a clip", "Calibrate / batch test", "About and limits"])
    with t1:
        tab_check(model_path, overlap, max_seconds, cutoff, band)
    with t2:
        tab_batch(model_path, overlap, max_seconds, cutoff, band)
    with t3:
        tab_about(bundle)


if __name__ == "__main__":
    main()