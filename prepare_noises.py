"""Build the five-noise bank as 16 kHz mono WAVs of NOISE_SECONDS each.

    python prepare_noises.py            # rain, engine, traffic, babble (+ screaming if raw clips exist)
    python prepare_noises.py --only rain engine

Sources
  rain, engine : ESC-50 (CC BY-NC 3.0). 5 s clips, several concatenated with a crossfade.
  traffic      : DEMAND STRAFFIC_16k (CC BY-SA 3.0), channel 1.   ~119 MB download
  babble       : DEMAND PRESTO_16k  (CC BY-SA 3.0), channel 1.    ~111 MB download
  screaming    : ESC-50 has no screaming class. Download CC0 / CC BY clips from
                 freesound.org by hand, put them in noises/raw/screaming/, and list each
                 clip (title, author, URL, licence) in noises.json -> screaming.clips.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "noises"
RAW = OUT / "raw"
SR = 16_000
NOISE_SECONDS = 30
XFADE_S = 0.05

ESC50_META = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
ESC50_AUDIO = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{}"
DEMAND = "https://zenodo.org/records/1227121/files/{}_16k.zip?download=1"


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total > 5_000_000:
                    print(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        if total > 5_000_000:
            print()
        tmp.replace(dest)
    return dest


def load_16k_mono(path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y.astype(np.float32)


def concat_crossfade(clips: list[np.ndarray]) -> np.ndarray:
    n = int(XFADE_S * SR)
    out = clips[0]
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    for c in clips[1:]:
        if len(out) < n or len(c) < n:
            out = np.concatenate([out, c])
            continue
        mid = out[-n:] * (1 - ramp) + c[:n] * ramp
        out = np.concatenate([out[:-n], mid, c[n:]])
    return out


def finish(y: np.ndarray, name: str, meta: dict, clips: list[dict]):
    """Loop/trim to NOISE_SECONDS, peak-normalise to -1 dBFS, write WAV, record clips."""
    target = NOISE_SECONDS * SR
    while len(y) < target:
        y = concat_crossfade([y, y])
    y = y[:target]
    y = y - y.mean()
    y = 0.89 * y / (np.max(np.abs(y)) + 1e-9)
    OUT.mkdir(exist_ok=True)
    sf.write(OUT / f"{name}.wav", y, SR, subtype="PCM_16")
    meta[name]["clips"] = clips
    print(f"  wrote noises/{name}.wav ({len(y) / SR:.0f} s, {len(clips)} source clip(s))")


def esc50(name: str, category: str, meta: dict, n_clips: int = 6):
    print(f"[{name}] ESC-50 '{category}'")
    rows = list(csv.DictReader(io.StringIO(requests.get(ESC50_META, timeout=60).text)))
    rows = [r for r in rows if r["category"] == category][:n_clips]
    clips, info = [], []
    for r in rows:
        path = download(ESC50_AUDIO.format(r["filename"]), RAW / "esc50" / r["filename"])
        y = load_16k_mono(path)
        y, _ = librosa.effects.trim(y, top_db=60)   # ESC-50 pads short clips with silence
        if len(y) > SR:
            clips.append(y)
            info.append({"file": r["filename"], "freesound_id": r["src_file"],
                         "url": f"https://freesound.org/s/{r['src_file']}/"})
    finish(concat_crossfade(clips), name, meta, info)


def demand(name: str, env: str, meta: dict, start_s: float = 60.0):
    print(f"[{name}] DEMAND {env}")
    wav = RAW / "demand" / f"{env}_ch01.wav"
    if not wav.exists():
        zpath = download(DEMAND.format(env), RAW / "demand" / f"{env}_16k.zip")
        with zipfile.ZipFile(zpath) as z:
            member = next(m for m in z.namelist() if m.lower().endswith("ch01.wav"))
            wav.write_bytes(z.read(member))
        zpath.unlink()  # keep only channel 1
    y = load_16k_mono(wav)
    s = int(start_s * SR)
    finish(y[s: s + NOISE_SECONDS * SR], name, meta,
           [{"file": f"{env}/ch01.wav", "url": "https://zenodo.org/records/1227121"}])


def screaming(meta: dict):
    print("[screaming] local Freesound clips")
    folder = RAW / "screaming"
    folder.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3"})
    if not files:
        print(f"  SKIPPED: put CC0 / CC BY screaming clips in {folder} and run again.")
        return
    clips = [librosa.effects.trim(load_16k_mono(p), top_db=50)[0] for p in files]
    known = {c.get("file"): c for c in meta["screaming"].get("clips", [])}
    info = [known.get(p.name, {"file": p.name, "author": "TODO", "url": "TODO", "licence": "TODO"})
            for p in files]
    finish(concat_crossfade(clips), "screaming", meta, info)
    if any("TODO" in c.values() for c in info):
        print("  NOTE: fill in author / url / licence for each clip in noises.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=["rain", "engine", "traffic", "babble", "screaming"])
    args = ap.parse_args()
    todo = args.only or ["rain", "engine", "traffic", "babble", "screaming"]

    meta_path = ROOT / "noises.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    jobs = {
        "rain": lambda: esc50("rain", "rain", meta),
        "engine": lambda: esc50("engine", "engine", meta),
        "traffic": lambda: demand("traffic", "STRAFFIC", meta),
        "babble": lambda: demand("babble", "PRESTO", meta),
        "screaming": lambda: screaming(meta),
    }
    for name in todo:
        try:
            jobs[name]()
        except Exception as e:
            print(f"  FAILED {name}: {e}")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
