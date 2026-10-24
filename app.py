from __future__ import annotations

import argparse
import logging
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response


BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "index.html"
WORK_DIR = Path(tempfile.gettempdir()) / "stem-splitter-jobs" 
ALLOWED_EXT = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif", ".wma"}
MAX_UPLOAD_MB = 500
KEEP_JOBS = 5                    
MP3_BITRATE = "320k"

log = logging.getLogger("stem-splitter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

SETTINGS = {"model": "htdemucs", "device": "auto", "shifts": 1, "overlap": 0.25}



def require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise SystemExit(
            "\n[!] ffmpeg was not found on your PATH.\n"
            "    Windows: winget install Gyan.FFmpeg   (then open a NEW terminal)\n"
            "    macOS:   brew install ffmpeg\n"
            "    Linux:   sudo apt install ffmpeg\n"
        )
    return exe


def decode_audio(path: Path, samplerate: int, channels: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:a:0",              
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", str(channels), "-ar", str(samplerate), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg could not decode this file: "
                           + proc.stderr.decode(errors="replace").strip()[-400:])
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("The file contains no audio.")
    return audio.reshape(-1, channels).T.copy()


def encode_mp3(audio: np.ndarray, samplerate: int, out_path: Path) -> None:
    channels = audio.shape[0]
    pcm = np.ascontiguousarray(audio.T, dtype=np.float32).tobytes()
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-f", "f32le", "-ar", str(samplerate), "-ac", str(channels), "-i", "-",
        "-codec:a", "libmp3lame", "-b:a", MP3_BITRATE, str(out_path),
    ]
    proc = subprocess.run(cmd, input=pcm, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg MP3 encoding failed: "
                           + proc.stderr.decode(errors="replace").strip()[-400:])


def prevent_clipping(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio / peak * 0.99 if peak > 0.99 else audio



_model = None
_model_lock = threading.Lock()


def pick_device() -> str:
    if SETTINGS["device"] != "auto":
        return SETTINGS["device"]
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from demucs.pretrained import get_model as demucs_get_model
            log.info("Loading Demucs model '%s' (first run downloads the weights)…", SETTINGS["model"])
            _model = demucs_get_model(SETTINGS["model"])
            _model.eval()
            log.info("Model ready. Stems: %s  |  device: %s", _model.sources, pick_device())
        return _model


def separate(input_path: Path, out_dir: Path, report) -> float:

    from demucs.apply import apply_model

    report("loading", 0.02, "Loading AI model…")
    model = get_model()
    sr, ch = model.samplerate, model.audio_channels
    device = pick_device()

    report("decoding", 0.05, "Decoding audio…")
    wav = torch.from_numpy(decode_audio(input_path, sr, ch))
    n_samples = wav.shape[-1]

    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    mix = (wav - mean) / std

    best = [0.0]

    def on_chunk(d: dict):
        if d.get("state") != "end":
            return
        models = max(1, d.get("models", 1))
        frac = (d.get("model_idx_in_bag", 0) + min(1.0, d.get("segment_offset", 0) / n_samples)) / models
        best[0] = max(best[0], min(frac, 1.0))
        report("separating", 0.08 + 0.80 * best[0], f"Separating stems on {device.upper()}… {best[0]:.0%}")

    report("separating", 0.08, f"Separating stems on {device.upper()}…")
    kwargs = dict(device=device, shifts=SETTINGS["shifts"], split=True,
                  overlap=SETTINGS["overlap"], progress=False)
    with torch.no_grad():
        try:
            sources = apply_model(model, mix[None], callback=on_chunk, **kwargs)[0]
        except TypeError:  
            sources = apply_model(model, mix[None], **kwargs)[0]
    sources = sources * std + mean                         
    sources = sources.cpu().numpy().astype(np.float32)      

    names = list(model.sources)
    if "vocals" not in names:
        raise RuntimeError(f"Model '{SETTINGS['model']}' has no vocals stem: {names}")
    vocals = sources[names.index("vocals")]
    instrumental = np.sum([sources[i] for i, n in enumerate(names) if n != "vocals"], axis=0)

    report("encoding", 0.90, "Encoding MP3 stems…")
    for name, stem in (("vocals", vocals), ("instrumental", instrumental)):
        np.save(out_dir / f"{name}.npy", stem)
        encode_mp3(prevent_clipping(stem), sr, out_dir / f"{name}.mp3")
    return n_samples / sr

@dataclass
class Job:
    id: str
    filename: str
    dir: Path
    input_path: Path
    status: str = "queued"    
    progress: float = 0.0
    message: str = "Waiting in queue…"
    duration: float = 0.0
    samplerate: int = 44100
    created: float = field(default_factory=time.time)


JOBS: dict[str, Job] = {}
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()


def worker_loop():
    while True:
        job = JOBS.get(JOB_QUEUE.get())
        if job is None:
            continue

        def report(status, progress, message, job=job):
            job.status, job.progress, job.message = status, round(progress, 4), message

        started = time.time()
        try:
            job.duration = separate(job.input_path, job.dir, report)
            job.samplerate = get_model().samplerate
            report("done", 1.0, f"Done in {time.time() - started:.0f}s")
            log.info("Job %s finished in %.1fs", job.id, time.time() - started)
        except Exception as exc:   # surface any failure to the UI instead of hanging
            log.exception("Job %s failed", job.id)
            report("error", job.progress, str(exc))
        finally:
            job.input_path.unlink(missing_ok=True)
            prune_old_jobs()


def prune_old_jobs():
    finished = sorted((j for j in JOBS.values() if j.status in ("done", "error")),
                      key=lambda j: j.created, reverse=True)
    for old in finished[KEEP_JOBS:]:
        shutil.rmtree(old.dir, ignore_errors=True)
        JOBS.pop(old.id, None)


def get_job(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job (it may have been cleaned up — upload again).")
    return job


def safe_stem_name(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"[^\w\-. ]+", "_", stem).strip() or "track"

app = FastAPI(title="Stem Splitter")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/api/info")
def info():
    return {"model": SETTINGS["model"], "device": pick_device(), "torch": torch.__version__}


@app.post("/api/separate")
async def create_job(file: UploadFile = File(...)):
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use MP3, WAV, FLAC or M4A.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / f"input{ext}"

    size = 0
    with input_path.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                fh.close()
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
            fh.write(chunk)

    JOBS[job_id] = Job(id=job_id, filename=filename, dir=job_dir, input_path=input_path)
    ahead = sum(1 for j in JOBS.values() if j.status not in ("done", "error")) - 1
    if ahead > 0:
        JOBS[job_id].message = f"Waiting in queue ({ahead} ahead)…"
    JOB_QUEUE.put(job_id)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    j = get_job(job_id)
    return {"id": j.id, "filename": j.filename, "status": j.status, "progress": j.progress,
            "message": j.message, "duration": j.duration}

@app.get("/api/jobs/{job_id}/mix.mp3")
def download_mix(job_id: str,
                 vocals: float = Query(1.0, ge=0.0, le=4.0),
                 instrumental: float = Query(1.0, ge=0.0, le=4.0)):
    j = get_job(job_id)
    if j.status != "done":
        raise HTTPException(409, "Separation is not finished yet.")
    mix = vocals * np.load(j.dir / "vocals.npy") + instrumental * np.load(j.dir / "instrumental.npy")
    mix = prevent_clipping(mix)
    out = j.dir / f"mix_{vocals:.3f}_{instrumental:.3f}.mp3"
    if not out.exists():
        encode_mp3(mix, j.samplerate, out)
    name = (f"{safe_stem_name(j.filename)} - Mix "
            f"(vox {round(vocals * 100)}%, inst {round(instrumental * 100)}%).mp3")
    return FileResponse(out, media_type="audio/mpeg", filename=name)


@app.get("/api/jobs/{job_id}/{stem}.mp3")
def download_stem(job_id: str, stem: str, download: bool = False):
    j = get_job(job_id)
    if stem not in ("vocals", "instrumental"):
        raise HTTPException(404, "Stem must be 'vocals' or 'instrumental'.")
    path = j.dir / f"{stem}.mp3"
    if j.status != "done" or not path.exists():
        raise HTTPException(409, "Separation is not finished yet.")
    name = f"{safe_stem_name(j.filename)} - {stem.capitalize()}.mp3"
    return FileResponse(path, media_type="audio/mpeg", filename=name if download else None)

def main():
    parser = argparse.ArgumentParser(description="Local AI vocal/instrumental separator")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other devices on your LAN")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="htdemucs",
                        help="htdemucs (fast, default) | htdemucs_ft (best quality, ~4x slower) | "
                             "htdemucs_6s | mdx_extra")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--shifts", type=int, default=1,
                        help="random-shift passes averaged together; 2–5 = slightly cleaner, proportionally slower")
    parser.add_argument("--overlap", type=float, default=0.25, help="chunk overlap (0.25 default)")
    args = parser.parse_args()

    SETTINGS.update(model=args.model, device=args.device, shifts=args.shifts, overlap=args.overlap)
    require_ffmpeg()
    if not INDEX_HTML.exists():
        raise SystemExit(f"[!] index.html must sit next to app.py ({INDEX_HTML})")
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=lambda: _safe_warmup(), daemon=True).start()

    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print(f"\n  🎧  Stem Splitter running →  {url}\n      model: {args.model} | device: {pick_device()}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def _safe_warmup():
    try:
        get_model()
    except Exception as exc:
        log.warning("Model warm-up failed (will retry on first job): %s", exc)


if __name__ == "__main__":
    main()
