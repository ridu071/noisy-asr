# Noise-robust speech recognition: Whisper base.en vs Wav2Vec2 Base

Record a sentence, mix in one of five real noises at a chosen SNR, and compare two
pretrained recognisers on the identical audio: transcripts, WER (with S/D/I), speed and
a winner for each trial. A fixed benchmark gives repeatable numbers for the report.

| File | Purpose |
|---|---|
| `app.py` | Gradio web app (live demo) |
| `benchmark.py` | 10 sentences × 16 conditions × noise reduction off/on, writes `results/` |
| `asr_core.py` | Shared pipeline: audio prep, SNR mixing, noise reduction, models, normaliser, WER |
| `plots.py` | Charts (same colour per model everywhere) |
| `prepare_noises.py` | Downloads and builds the noise bank in `noises/` |
| `models.json` | The two models and their details |
| `noises.json` | Noise names, sources, licences, per-clip credits |
| `recordings/` | Your saved recordings: `rec_NN.wav` + `rec_NN.txt` (what you said) |

## Setup (Windows, laptop CPU)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python prepare_noises.py      # ~230 MB of downloads, once
```

**Screaming noise:** ESC-50 has no screaming class. Download 3–6 CC0 or CC BY clips from
freesound.org, put them in `noises/raw/screaming/`, run `python prepare_noises.py --only screaming`,
then fill in author, URL and licence for each clip in `noises.json`.

## Run

```powershell
.venv\Scripts\python benchmark.py --sanity   # step 0: clean LibriSpeech check, both WERs should be low
.venv\Scripts\python app.py                  # open http://127.0.0.1:7860
```

In the app: record yourself saying anything, type exactly what you said, pick noise / SNR /
noise reduction, press **Run**. Listen to the original and the noisy version under the verdict.
**Run all noises** sweeps every noise at the chosen SNR. **Save to benchmark set** stores the
recording and your transcript as `recordings/rec_NN.wav` + `rec_NN.txt`.

## Benchmark for the report

1. In the app, record 10 different sentences, type each transcript, and press **Save to benchmark set**.
2. `python benchmark.py` → 160 mixtures × 2 models × NR off/on = 640 transcriptions.
3. Read `results/summary.md`; charts are `wer_by_noise.png`, `wer_vs_snr.png`, `nr_effect.png`.

No recordings yet? `python benchmark.py --librispeech 10` runs the same grid on 10 LibriSpeech clips.

## Colab

```python
!git clone <your repo> asr && cd asr && pip install -r requirements.txt
!python prepare_noises.py
!python app.py --share     # https link: browsers only allow the microphone over https
```

## Method notes (what the code actually does)

- **Audio:** 16 kHz mono float32. Leading/trailing silence is trimmed (40 dB, 0.2 s pad) so that
  P_speech is speech power, not diluted by silence.
- **Mixing:** noise looped/trimmed from a seeded offset, gain g = √(P_s / (P_n·10^(SNR/10))).
  If the mixture would clip, the whole mixture is scaled down, which keeps the SNR.
  In the benchmark one seed per (sentence, noise) means the same noise segment at 10, 5 and 0 dB.
- **Noise reduction:** `noisereduce` spectral gating (non-stationary mode) on the mixture.
- **Decoding:** greedy for both. Whisper: 1 beam, no sampling, no timestamps, `max_new_tokens`
  capped at 8 per second of audio + 16 to stop runaway loops. Wav2Vec2: argmax CTC, no LM.
- **Timing:** forward pass + decoding only, after a warm-up, same device and 4 threads.
- **Normaliser:** lowercase, hyphens → spaces, punctuation removed (in-word apostrophes kept),
  numbers turned into words ("4" = "four"), Mr/Mrs/Dr expanded, spaces collapsed.
  Applied to your transcript and both outputs.
- **Transcript required:** WER is always computed against what you typed; the app refuses to run without it.
