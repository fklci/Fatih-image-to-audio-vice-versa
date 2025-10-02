# Image ↔ Audio (10s) — Streamlit

Convert images to 10‑second WAV audio (RGB→melody) and 10‑second WAV audio back to 600×600 images (spectrum bands→RGB).

## Run locally
```bash
pip install -r requirements.txt
streamlit run img-audio-converter-app.py
```

## Deploy options

### A) Streamlit Community Cloud (fastest)
1. Push this folder to a **public GitHub repo**.
2. Go to **streamlit.io → Community Cloud → New app** and connect your GitHub.
3. Select your repo/branch and set the main file to `img-audio-converter-app.py`.
4. Click **Deploy**. That URL is your live app.

> No secrets needed. Supports WAV uploads (16‑bit PCM). Enforces 10s duration.

### B) Hugging Face Spaces
1. Create a Space (`New Space`) and choose **Streamlit**.
2. Upload `img-audio-converter-app.py` (or rename to `app.py`), `requirements.txt`, and this README.
3. The Space builds automatically and gives you a public URL.

### C) Docker (any VPS / Render / Railway / Fly.io)
Build and run locally:
```bash
docker build -t img-audio-converter .
docker run -p 8501:8501 img-audio-converter
```
Then visit http://localhost:8501

**Render.com** example (Web Service):
- Build Command: `pip install -r requirements.txt`
- Start Command: `streamlit run img-audio-converter-app.py --server.port $PORT --server.address 0.0.0.0`

## Notes
- Audio must be WAV (16‑bit PCM). Convert other formats first (e.g., with `ffmpeg`).
- The app trims or pads to 10 seconds; image side is 600×600 fixed.
- RGB→Audio uses three sine oscillators (A/D/C bands). Audio→RGB uses low/mid/high spectral energy.
