# Stem Splitter: a local AI vocal / instrumental separator

Everything runs on your own computer. A FastAPI backend (`app.py`) runs Meta's open-source
**Demucs** model, and the browser UI (`index.html`) plays, mixes and exports the stems.
Keep `app.py`, `index.html` and `requirements.txt` in the same folder.

## Setup

You need **Python 3.10+** and **FFmpeg**.

### 1. Install FFmpeg
| OS | Command |
|---|---|
| Windows | `winget install Gyan.FFmpeg` (then open a **new** terminal) |
| macOS | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt install ffmpeg` |

To check that it worked, run `ffmpeg -version`.

### 2. Create a virtual environment
```bash
cd stem-splitter
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
```

### 3. (Optional) Install the GPU build of PyTorch
* **NVIDIA on Windows:** `pip install torch --index-url https://download.pytorch.org/whl/cu124`
  (choose the CUDA version listed at https://pytorch.org/get-started/locally/)
* **Linux:** the default `pip install torch` already includes CUDA.
* **Apple Silicon:** the default install uses the Metal (MPS) GPU automatically.
* **CPU only:** skip this step.

### 4. Install the rest
```bash
pip install -r requirements.txt
```

### 5. Run
```bash
python app.py
```
Open **http://127.0.0.1:8000** in your browser. The first separation downloads the model
weights (about 80 MB). This happens only once.

## Options
```bash
python app.py --model htdemucs_ft     # best quality, about 4x slower
python app.py --shifts 2              # averages 2 passes: slightly cleaner, 2x slower
python app.py --device cpu            # force CPU (for example, if MPS errors on a Mac)
python app.py --port 9000
python app.py --host 0.0.0.0          # reach it from other devices on your network
```

## How it works
1. **Upload.** `POST /api/separate` saves the file and queues a job. Jobs run one at a time.
2. **Separate.** A worker thread decodes the file with ffmpeg and runs Demucs. Drums, bass
   and other are summed into **Instrumental**. The worker writes `vocals.mp3` and
   `instrumental.mp3` at 320 kbps, plus lossless float copies that are used for exporting.
3. **Play.** The page polls `GET /api/jobs/{id}`, then loads both MP3s into the Web Audio
   API. Both stems start on the same AudioContext clock tick, each through its own gain
   node, so they stay sample-locked while you move the faders.
4. **Export.** **Download Mixed MP3** calls
   `GET /api/jobs/{id}/mix.mp3?vocals=…&instrumental=…`. The server mixes the lossless stems
   at exactly those gains, respecting mute and solo. It turns the whole mix down only if it
   would otherwise clip.

Temporary files are stored in your system temp folder under `stem-splitter-jobs/`. Only the
5 most recent jobs are kept.
