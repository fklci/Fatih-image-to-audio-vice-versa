"""
Super Simple Image↔Audio Converter (Streamlit)

- Image → 10s Audio (RGB → melody)
- 10s Audio → 600×600 Image (spectrum → RGB bands)

Run locally:
  pip install -r requirements.txt
  streamlit run img-audio-converter-app.py

Notes:
- Audio I/O uses the standard library `wave` module (16‑bit PCM WAV only).
- If your input audio isn't WAV, convert it to WAV first (e.g., with ffmpeg).
- Duration is fixed at 10 seconds. Inputs longer than 10s are truncated; shorter are zero‑padded.
"""

import io
import math
import struct
import wave
from typing import Tuple

import numpy as np
from PIL import Image
import streamlit as st

SR = 44100           # sample rate
DURATION = 10.0      # seconds
N_COLS = 600         # image width & time slices
IMG_SIZE = (600, 600)
SAMPLES_PER_COL = int(SR * DURATION / N_COLS)

# ----- Helpers -----

def _read_wav_bytes(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Return mono float32 in [-1,1], sample_rate from a 16-bit PCM WAV file."""
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        if sampwidth != 2:
            raise ValueError("Only 16-bit PCM WAV supported.")
        pcm = wf.readframes(n_frames)
    dtype = np.int16
    audio = np.frombuffer(pcm, dtype=dtype).astype(np.float32) / 32768.0
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    return audio, framerate


def _write_wav_bytes(audio: np.ndarray, sample_rate: int = SR) -> bytes:
    """Encode mono float32 [-1,1] to 16-bit PCM WAV and return bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    int16 = (audio * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())
    return buf.getvalue()


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    minv, maxv = float(v.min()), float(v.max())
    if maxv - minv < eps:
        return np.zeros_like(v)
    return (v - minv) / (maxv - minv)


# ----- Image → Audio -----

def image_to_audio(img: Image.Image, duration: float = DURATION, sr: int = SR) -> np.ndarray:
    """Convert an image to a 10s mono audio track.

    Method (intentionally simple):
    - Resize image to 600×600.
    - For each of 600 columns (time slices), compute mean R,G,B (0..255).
    - Map R,G,B to three sine frequencies in distinct bands and amplitudes from 0..1.
    - Synthesize SAMPLES_PER_COL samples per slice keeping oscillator phase continuity.
    """
    img = img.convert('RGB').resize(IMG_SIZE, Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32)  # (H,W,3)
    # Mean per column: shape (W, 3)
    mean_rgb = arr.mean(axis=0) / 255.0  # 0..1

    # Frequency maps (choose pleasant distinct ranges)
    def map_freq(x, f_lo, f_hi):
        return f_lo + x * (f_hi - f_lo)

    r_freqs = map_freq(mean_rgb[:, 0], 220.0, 880.0)   # A3..A5
    g_freqs = map_freq(mean_rgb[:, 1], 146.8, 587.3)   # D3..D5
    b_freqs = map_freq(mean_rgb[:, 2], 261.6, 1046.5)  # C4..C6

    # Amplitude based on color intensity with soft companding
    r_amp = np.sqrt(mean_rgb[:, 0]) * 0.6
    g_amp = np.sqrt(mean_rgb[:, 1]) * 0.6
    b_amp = np.sqrt(mean_rgb[:, 2]) * 0.6

    # Simple per-slice synthesis with phase continuity across slices
    total_samples = int(sr * duration)
    out = np.zeros(total_samples, dtype=np.float32)

    # phase accumulators
    ph_r = 0.0
    ph_g = 0.0
    ph_b = 0.0

    t = np.arange(SAMPLES_PER_COL, dtype=np.float32) / sr

    idx = 0
    for i in range(N_COLS):
        fr, fg, fb = r_freqs[i], g_freqs[i], b_freqs[i]
        ar, ag, ab = r_amp[i], g_amp[i], b_amp[i]

        # generate slice
        sr_r = 2 * math.pi * fr
        sr_g = 2 * math.pi * fg
        sr_b = 2 * math.pi * fb

        # sine with current phase
        slice_r = np.sin(ph_r + sr_r * t)
        slice_g = np.sin(ph_g + sr_g * t)
        slice_b = np.sin(ph_b + sr_b * t)

        slice_sum = ar * slice_r + ag * slice_g + ab * slice_b

        # gentle 5ms fade in/out to reduce clicks between slices
        fade = int(0.005 * sr)
        if fade > 0:
            w = np.ones_like(slice_sum)
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            w[:fade] *= ramp
            w[-fade:] *= ramp[::-1]
            slice_sum *= w

        out[idx:idx + SAMPLES_PER_COL] = slice_sum[: max(0, min(SAMPLES_PER_COL, total_samples - idx))]

        # update phases for continuity
        ph_r = (ph_r + sr_r * (SAMPLES_PER_COL / sr)) % (2 * math.pi)
        ph_g = (ph_g + sr_g * (SAMPLES_PER_COL / sr)) % (2 * math.pi)
        ph_b = (ph_b + sr_b * (SAMPLES_PER_COL / sr)) % (2 * math.pi)

        idx += SAMPLES_PER_COL
        if idx >= total_samples:
            break

    # final normalization to -1..1
    out = out / (np.max(np.abs(out)) + 1e-9)
    return out.astype(np.float32)


# ----- Audio → Image -----

def audio_to_image(audio: np.ndarray, sr: int) -> Image.Image:
    """Convert a 10s mono audio signal to a 600×600 image.

    Method (intentionally simple):
    - Split audio into 600 equal windows (≈16.67 ms each).
    - Compute magnitude spectrum per window via FFT.
    - Aggregate energy into 3 bands: Low(20–400 Hz)→R, Mid(400–2000)→G, High(2000–8000)→B.
    - Normalize across time and paint vertical stripes for each time slice.
    """
    target_len = int(SR * DURATION)
    # resample if needed (naive — speed/quality tradeoff); otherwise pad/truncate
    if sr != SR:
        # simple linear resample to SR
        x_old = np.linspace(0, 1, num=audio.shape[0], endpoint=False)
        x_new = np.linspace(0, 1, num=target_len, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    else:
        if audio.shape[0] < target_len:
            audio = np.pad(audio, (0, target_len - audio.shape[0]))
        else:
            audio = audio[:target_len]

    # Windowing into 600 slices
    slices = audio.reshape(N_COLS, SAMPLES_PER_COL)

    # FFT params
    nfft = 2048
    freqs = np.fft.rfftfreq(nfft, d=1.0 / SR)

    # Band indices
    def band_mask(f_lo, f_hi):
        return (freqs >= f_lo) & (freqs < f_hi)

    m_low = band_mask(20, 400)
    m_mid = band_mask(400, 2000)
    m_high = band_mask(2000, 8000)

    R, G, B = [], [], []
    window = np.hanning(SAMPLES_PER_COL)
    for sl in slices:
        x = sl * window
        X = np.fft.rfft(x, n=nfft)
        mag = np.abs(X)
        R.append(float(mag[m_low].mean() if m_low.any() else 0.0))
        G.append(float(mag[m_mid].mean() if m_mid.any() else 0.0))
        B.append(float(mag[m_high].mean() if m_high.any() else 0.0))

    R = _normalize(np.array(R))
    G = _normalize(np.array(G))
    B = _normalize(np.array(B))

    # scale to 0..255 and build 600×600 image as vertical stripes
    col_stack = np.stack([
        (R * 255).astype(np.uint8),
        (G * 255).astype(np.uint8),
        (B * 255).astype(np.uint8)
    ], axis=1)  # (600, 3)

    # tile across rows
    img_arr = np.repeat(col_stack[np.newaxis, :, :], IMG_SIZE[1], axis=0)  # (600, 600, 3)
    img_arr = np.transpose(img_arr, (0, 1, 2))  # no-op for clarity
    return Image.fromarray(img_arr, mode='RGB')


# ----- Streamlit UI -----

st.set_page_config(page_title="Image ↔ Audio (10s)", page_icon="🎵", layout="centered")
st.title("🎵 Super Simple Image ↔ Audio Converter")
st.caption("Images → 10s WAV via RGB→melody • 10s WAV → 600×600 image via spectrum bands")

mode = st.tabs(["Image → Audio (10s WAV)", "Audio → Image (600×600)"])

with mode[0]:
    st.subheader("Image → 10s Audio")
    img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    if img_file is not None:
        img = Image.open(img_file)
        st.image(img, caption="Input image (resized to 600×600 internally)", use_column_width=True)
        if st.button("Convert to 10s WAV"):
            audio = image_to_audio(img, duration=DURATION, sr=SR)
            wav_bytes = _write_wav_bytes(audio, sample_rate=SR)
            st.audio(wav_bytes, format='audio/wav')
            st.download_button("Download WAV (10s)", data=wav_bytes, file_name="image_to_audio.wav", mime="audio/wav")
        st.info("Tip: Larger/redder areas emphasize A‑range tones; greener/blue areas emphasize D/C ranges.")

with mode[1]:
    st.subheader("Audio → 600×600 Image")
    wav_file = st.file_uploader("Upload a 10s WAV (16‑bit PCM preferred)", type=["wav"])
    if wav_file is not None:
        try:
            data = wav_file.read()
            audio, sr = _read_wav_bytes(data)
            img = audio_to_image(audio, sr)
            st.image(img, caption="Reconstructed image (vertical time stripes)", use_column_width=True)
            # save to PNG for download
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            st.download_button("Download PNG (600×600)", data=buf.getvalue(), file_name="audio_to_image.png", mime="image/png")
            st.audio(data, format='audio/wav')
        except Exception as e:
            st.error(f"Could not read WAV: {e}")
    st.caption("If your source audio is not WAV, convert to WAV first (e.g., `ffmpeg -i in.mp3 -ar 44100 -ac 1 out.wav`).")
