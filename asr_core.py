"""Shared pipeline for the noise-robust ASR comparison.

Everything that affects the comparison lives here so the live app (app.py) and the
fixed benchmark (benchmark.py) apply exactly the same processing:

    load_audio -> prepare_speech -> mix_at_snr -> [reduce_noise] -> ASRModel.transcribe
    -> normalise -> score
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from transformers.utils import logging as _hf_logging  # noqa: E402

_hf_logging.set_verbosity_error()  # hide deprecation chatter from generate()
_hf_logging.disable_progress_bar()

ROOT = Path(__file__).resolve().parent
SR = 16_000  # both models expect 16 kHz mono

# Same device and thread count for every model, so timing is comparable.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_THREADS = min(4, os.cpu_count() or 4)
torch.set_num_threads(NUM_THREADS)


# --------------------------------------------------------------------------- data files
def load_json(name: str) -> dict:
    data = json.loads((ROOT / name).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_comment")}


def load_registry() -> tuple[dict, list[str]]:
    reg = load_json("models.json")
    defaults = reg.pop("_defaults", None) or list(reg)[:2]
    return reg, defaults


def load_noise_bank() -> tuple[dict, dict[str, np.ndarray]]:
    """Return (metadata, {noise_id: waveform}); noises whose file is missing are skipped."""
    meta = load_json("noises.json")
    audio = {}
    for nid, info in meta.items():
        path = ROOT / info["file"]
        if path.exists():
            audio[nid] = load_audio(path)
    return meta, audio


# --------------------------------------------------------------------------- audio
def load_audio(src) -> np.ndarray:
    """Load a file path or a (sr, array) tuple as 16 kHz mono float32."""
    if isinstance(src, tuple):
        sr, data = src
        data = np.asarray(data)
    else:
        try:
            data, sr = sf.read(str(src), dtype="float32", always_2d=True)
        except Exception:  # formats libsndfile can't read (e.g. webm) need ffmpeg via librosa
            data, sr = librosa.load(str(src), sr=None, mono=False)
            data = np.atleast_2d(data).T
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    return np.ascontiguousarray(data, dtype=np.float32)


def prepare_speech(speech: np.ndarray, pad_s: float = 0.2) -> np.ndarray:
    """Remove DC offset and trim leading/trailing silence (keeping a short pad).

    Trimming matters for the SNR: P_speech is a mean over the whole clip, so two
    seconds of silence before the user starts talking would make the real SNR lower
    than the one requested.
    """
    speech = speech - speech.mean()
    _, (start, end) = librosa.effects.trim(speech, top_db=40)
    pad = int(pad_s * SR)
    return speech[max(0, start - pad): min(len(speech), end + pad)].astype(np.float32)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, seed: int):
    """Add noise at an exact speech-to-noise power ratio.

    The noise is looped or trimmed to the speech length from a seeded random offset and
    scaled by g = sqrt(P_speech / (P_noise * 10^(SNR/10))). Returns (mixture, offset).
    If the mixture would clip, the whole mixture is scaled down; that keeps the SNR.
    """
    n = len(speech)
    if len(noise) < n:
        noise = np.tile(noise, math.ceil(n / len(noise)) + 1)
    rng = np.random.default_rng(seed)
    offset = int(rng.integers(0, len(noise) - n + 1))
    seg = noise[offset: offset + n].astype(np.float64)
    p_speech = float(np.mean(speech.astype(np.float64) ** 2))
    p_noise = float(np.mean(seg ** 2))
    if p_noise == 0 or p_speech == 0:
        return speech.copy(), offset
    g = math.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    mix = speech + g * seg
    peak = float(np.max(np.abs(mix)))
    if peak > 0.99:
        mix *= 0.99 / peak
    return mix.astype(np.float32), offset


_A_W = None


def a_weighted_power(x: np.ndarray) -> float:
    """Mean power after A-weighting: how loud a sound *seems*, not its raw energy.

    Low rumble (engine, traffic) carries lots of energy the ear barely hears, so at the same
    raw SNR it sounds far quieter than rain or babble. Matching A-weighted power fixes that.
    """
    global _A_W
    if _A_W is None:
        f = librosa.fft_frequencies(sr=SR, n_fft=1024)
        _A_W = (10 ** (librosa.A_weighting(f) / 10))[:, None]
    S = np.abs(librosa.stft(np.asarray(x, dtype=np.float32), n_fft=1024, hop_length=256)) ** 2
    return float(np.mean(S * _A_W))


def mix_noises(speech: np.ndarray, noises: list[np.ndarray], snr_db: float, seed: int):
    """Add several noises, each as loud *to the ear* as speech / 10^(snr/10).

    Every noise is scaled on its own (A-weighted), so two noises are louder than one.
    Returns (mixture, overall_snr_db), where the overall SNR is plain (unweighted) power.
    """
    if not noises:
        return speech.copy(), None
    rng = np.random.default_rng(seed)
    n = len(speech)
    target = a_weighted_power(speech) / 10 ** (snr_db / 10)
    total = np.zeros(n, dtype=np.float64)
    for noise in noises:
        if len(noise) < n:
            noise = np.tile(noise, math.ceil(n / len(noise)) + 1)
        off = int(rng.integers(0, len(noise) - n + 1))
        seg = noise[off: off + n].astype(np.float64)
        p = a_weighted_power(seg)
        if p > 0:
            total += math.sqrt(target / p) * seg
    overall = 10 * math.log10(np.mean(speech.astype(np.float64) ** 2) / max(np.mean(total ** 2), 1e-20))
    mix = speech + total
    peak = float(np.max(np.abs(mix)))
    if peak > 0.99:
        mix *= 0.99 / peak
    return mix.astype(np.float32), overall


def measured_snr(speech: np.ndarray, mix: np.ndarray) -> float:
    """SNR recovered from a mixture (for checks). Undoes any anti-clipping rescale."""
    scale = float(np.dot(mix, speech) / np.dot(speech, speech))  # mixture = scale*(speech+gn)
    noise = mix / scale - speech
    return 10 * math.log10(np.mean(speech ** 2) / np.mean(noise ** 2))


def reduce_noise(audio: np.ndarray) -> np.ndarray:
    """Spectral-gating noise reduction. Changes the input audio only, never the model."""
    import noisereduce as nr
    out = nr.reduce_noise(y=audio, sr=SR, stationary=False)
    return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- models
class ASRModel:
    """A pretrained recogniser with greedy decoding and honest timing."""

    def __init__(self, key: str, spec: dict):
        from transformers import (AutoModelForCTC, AutoModelForSpeechSeq2Seq,
                                  AutoProcessor)
        self.key, self.spec = key, spec
        self.label = spec.get("label", key)
        self.kind = spec["kind"]
        self.processor = AutoProcessor.from_pretrained(spec["hf_id"])
        cls = AutoModelForCTC if self.kind == "ctc" else AutoModelForSpeechSeq2Seq
        self.model = cls.from_pretrained(spec["hf_id"]).to(DEVICE).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        # Warm-up run so the first real trial doesn't pay one-off start-up costs.
        rng = np.random.default_rng(0)
        self.transcribe((0.01 * rng.standard_normal(SR)).astype(np.float32))

    def _sync(self):
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Return (text, seconds). Time covers the forward pass and decoding only."""
        inputs = self.processor(audio, sampling_rate=SR, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items() if torch.is_tensor(v)}
        self._sync()
        t0 = time.perf_counter()
        if self.kind == "ctc":
            logits = self.model(inputs["input_values"]).logits
            ids = torch.argmax(logits, dim=-1)                      # greedy CTC, no LM
            text = self.processor.batch_decode(ids)[0]
        else:
            dur = len(audio) / SR
            ids = self.model.generate(
                **inputs,
                num_beams=1, do_sample=False,                       # greedy, temperature 0
                max_new_tokens=int(dur * 8) + 16,                   # stops endless loops
            )
            text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        self._sync()
        return text.strip(), time.perf_counter() - t0


_MODEL_CACHE: dict[str, ASRModel] = {}


def get_model(key: str, registry: dict) -> ASRModel:
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = ASRModel(key, registry[key])
    return _MODEL_CACHE[key]


# --------------------------------------------------------------------------- scoring
def normalise(text: str) -> str:
    """One rule set for references and both models: lowercase, no punctuation, single spaces."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)          # 1,000 -> 1000
    text = re.sub(r"\d+", lambda m: " " + number_words(int(m.group())) + " ", text)
    text = re.sub(r"[-–—/]", " ", text)           # well-known -> well known
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)     # keep only in-word apostrophes
    words = [ABBREV.get(w, w) for w in text.split()]        # Whisper writes "Mr.", CTC "MISTER"
    return " ".join(words)


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> str:
    """4 -> 'four', 25 -> 'twenty five'. Whisper writes digits, Wav2Vec2 spells words out."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    if n < 1_000_000:
        return number_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + number_words(n % 1000))
    return " ".join(_ONES[int(d)] for d in str(n))


ABBREV = {"mr": "mister", "mrs": "missus", "dr": "doctor", "ok": "okay"}


@dataclass
class Score:
    wer: float
    S: int
    D: int
    I: int
    N: int
    ops: list = field(default_factory=list)  # (type, ref_words, hyp_words) for highlighting


def score(reference: str, hypothesis: str) -> Score:
    ref, hyp = normalise(reference), normalise(hypothesis)
    ref_w, hyp_w = ref.split(), hyp.split()
    N = len(ref_w)
    if N == 0:
        raise ValueError("Reference is empty after normalisation")
    if not hyp_w:
        return Score(1.0, 0, N, 0, N, [("delete", ref_w, [])])
    out = jiwer.process_words(ref, hyp)
    ops = [(c.type, ref_w[c.ref_start_idx:c.ref_end_idx], hyp_w[c.hyp_start_idx:c.hyp_end_idx])
           for c in out.alignments[0]]
    S, D, I = out.substitutions, out.deletions, out.insertions
    return Score((S + D + I) / N, S, D, I, N, ops)


def agreement(a: str, b: str) -> float:
    """Word-level agreement between two transcripts (1 - WER of b against a). NOT a WER."""
    a, b = normalise(a), normalise(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return max(0.0, 1.0 - jiwer.wer(a, b))


def winner(wer_a: float, wer_b: float) -> str:
    if abs(wer_a - wer_b) < 1e-9:
        return "tie"
    return "a" if wer_a < wer_b else "b"
