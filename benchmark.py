"""Fixed benchmark: every sentence x 16 conditions x noise reduction off/on, both models.

    python benchmark.py --sanity            # clean LibriSpeech check: both WERs should be low
    python benchmark.py                     # your recordings/s01..s10.wav (saved from the app)
    python benchmark.py --librispeech 10    # use 10 LibriSpeech clips instead (no recording needed)
    python benchmark.py --nr off            # skip the noise-reduction runs (half the time)

Outputs in results/: results.csv (one row per transcription), summary.md, and charts.
"""
from __future__ import annotations

import argparse
import io
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf

import asr_core as core
import plots

SNRS = [10, 5, 0]
RES = core.ROOT / "results"


def librispeech_clips(n: int):
    """First n clips of the LibriSpeech dummy set (clean, read speech): [(id, text, audio)]."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    repo = "hf-internal-testing/librispeech_asr_dummy"
    files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
    path = hf_hub_download(repo, sorted(files, key=lambda f: "clean" not in f)[0], repo_type="dataset")
    table = pq.read_table(path).to_pylist()
    out = []
    for r in table[:n]:
        data, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        out.append((str(r.get("id", len(out))), r["text"], core.load_audio((sr, data))))
    return out


def own_recordings():
    """recordings/rec_NN.wav + rec_NN.txt pairs saved from the app."""
    out = []
    for wav in sorted((core.ROOT / "recordings").glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            print(f"  skipping {wav.name}: no {txt.name} transcript")
            continue
        out.append((wav.stem, txt.read_text(encoding="utf-8").strip(), core.load_audio(wav)))
    print(f"Found {len(out)} recordings with transcripts")
    return out


def sanity(models):
    print("Clean-audio check on LibriSpeech (no noise added):")
    ok = True
    for cid, ref, audio in librispeech_clips(3):
        for m in models:
            text, secs = m.transcribe(audio)
            w = core.score(ref, text).wer
            ok &= w <= 0.2
            print(f"  {cid:>18} {m.label:<24} WER {w * 100:5.1f}%  {secs:.2f}s  {text[:60]!r}")
    print("PASS: both models transcribe clean speech well." if ok else
          "FAIL: a model has high WER on clean speech, so check the pipeline before adding noise.")
    return ok


def run(clips, models, noise_audio, nr_flags, base_seed):
    noise_ids = list(noise_audio)
    conditions = [("clean", None)] + [(n, s) for n in noise_ids for s in SNRS]
    total = len(clips) * len(conditions) * len(nr_flags) * len(models)
    print(f"{len(clips)} sentences x {len(conditions)} conditions x NR {nr_flags} x "
          f"{len(models)} models = {total} transcriptions")
    rows, done, t0 = [], 0, time.time()
    for i, (sid, ref, raw) in enumerate(clips):
        speech = core.prepare_speech(raw)
        for j, (nid, snr) in enumerate(conditions):
            # One seed per (sentence, noise): the same noise segment at every SNR,
            # so only the loudness changes along an SNR curve.
            seed = base_seed + 1000 * i + (noise_ids.index(nid) + 1 if nid != "clean" else 0)
            if nid == "clean":
                mix, offset = speech, None
            else:
                mix, offset = core.mix_at_snr(speech, noise_audio[nid], snr, seed)
            for nr in nr_flags:
                x = core.reduce_noise(mix) if nr else mix
                dur = len(x) / core.SR
                for m in models:
                    text, secs = m.transcribe(x)
                    sc = core.score(ref, text)
                    rows.append({"sentence_id": sid, "reference": ref, "noise": nid, "snr_db": snr,
                                 "noise_reduction": nr, "seed": seed, "noise_offset": offset,
                                 "duration_s": round(dur, 3), "model": m.key, "transcript": text,
                                 "wer": sc.wer, "S": sc.S, "D": sc.D, "I": sc.I, "N": sc.N,
                                 "time_s": secs, "rtf": secs / dur})
                    done += 1
            eta = (time.time() - t0) / done * (total - done)
            print(f"\r  {done}/{total}  eta {eta / 60:4.1f} min", end="", flush=True)
    print()
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, registry, noise_names):
    RES.mkdir(exist_ok=True)
    label = lambda k: registry[k]["label"]  # noqa: E731
    names = noise_names | {"clean": "Clean"}
    g = ["noise", "snr_db", "noise_reduction", "model"]
    summ = (df.groupby(g, dropna=False)
              .agg(wer_mean=("wer", "mean"), wer_sd=("wer", "std"), n=("wer", "size"),
                   errors=("S", "sum"), S=("S", "sum"), D=("D", "sum"), I=("I", "sum"),
                   words=("N", "sum"), rtf=("rtf", "mean"))
              .reset_index())
    summ["errors"] = summ["S"] + summ["D"] + summ["I"]
    summ["pooled_wer"] = summ["errors"] / summ["words"]
    summ.to_csv(RES / "summary_by_condition.csv", index=False)

    out = ["# Benchmark summary", "",
           f"{df['sentence_id'].nunique()} sentences, {len(df)} transcriptions. "
           "WER in %. Mean ± SD across sentences; pooled = total errors / total reference words.",
           "With this few sentences, differences of a few points are indicative only.", ""]

    for nr in sorted(df["noise_reduction"].unique()):
        s = summ[summ["noise_reduction"] == nr]
        piv = s.assign(cell=lambda d: (d["wer_mean"] * 100).round(1).astype(str) + " ± "
                       + (d["wer_sd"].fillna(0) * 100).round(1).astype(str),
                       cond=lambda d: d["noise"].map(names) + d["snr_db"].map(
                           lambda v: "" if pd.isna(v) else f" {v:g} dB"),
                       model=lambda d: d["model"].map(label))
        tab = piv.pivot(index="cond", columns="model", values="cell")
        order = ["Clean"] + [f"{names[n]} {s:g} dB" for n in dict.fromkeys(df["noise"]) if n != "clean"
                             for s in SNRS]
        tab = tab.reindex([c for c in order if c in tab.index])
        tab.index.name = "condition"
        out += [f"## Mean WER by condition, noise reduction {'ON' if nr else 'OFF'}", "",
                tab.to_markdown(), ""]

    by_snr = (df[df["noise"] != "clean"].groupby(["noise_reduction", "snr_db", "model"])["wer"].mean()
              .mul(100).round(1).unstack("model").rename(columns=label)
              .sort_index(level=[0, 1], ascending=[True, False]))
    by_snr.index = [f"NR {'on' if nr else 'off'}, {s:g} dB" for nr, s in by_snr.index]
    out += ["## Mean WER by SNR (all noises pooled)", "", by_snr.to_markdown(), ""]

    # wins: compare models on each identical mixture
    key = ["sentence_id", "noise", "snr_db", "noise_reduction"]
    w = (df.assign(snr_db=df["snr_db"].fillna(-1))            # clean has no SNR
           .pivot_table(index=key, columns="model", values="wer").dropna())
    win = w.apply(lambda r: "tie" if r.max() - r.min() < 1e-9 else r.idxmin(), axis=1)
    counts = win.map(lambda k: label(k) if k in registry else k).value_counts()
    out += ["## Wins per mixture", "", counts.to_frame("wins").to_markdown(), ""]

    err = (df.groupby("model")[["S", "D", "I", "N"]].sum()
           .assign(**{"WER pooled %": lambda d: ((d.S + d.D + d.I) / d.N * 100).round(1)})
           .rename(index=label))
    out += ["## Error split (all conditions)", "",
            "High insertion counts in heavy noise point to hallucinated words.", "", err.to_markdown(), ""]

    rtf = (df.groupby("model").agg(time_mean_s=("time_s", "mean"), time_median_s=("time_s", "median"),
                                   rtf_mean=("rtf", "mean"), rtf_median=("rtf", "median"))
             .round(3).rename(index=label))
    out += [f"## Speed ({core.DEVICE}, {core.NUM_THREADS} threads)", "",
            "Median is the fairer figure: background load on a laptop causes occasional slow runs.", "",
            rtf.to_markdown(), ""]

    noisy = df[df["noise"] != "clean"]
    if noisy["noise_reduction"].nunique() == 2:
        d = (noisy.groupby(["noise", "model", "noise_reduction"])["wer"].mean().unstack()
             .pipe(lambda t: t[True] - t[False]).rename("delta").reset_index())
        out += ["## Effect of noise reduction (WER with NR minus without, points; negative = helped)", "",
                d.assign(delta=(d["delta"] * 100).round(1), noise=d["noise"].map(names),
                         model=d["model"].map(label))
                 .pivot(index="noise", columns="model", values="delta").to_markdown(), ""]
        plots.nr_effect(d, registry, names, "Effect of noise reduction, averaged over SNRs") \
             .savefig(RES / "nr_effect.png", dpi=150)

    (RES / "summary.md").write_text("\n".join(out), encoding="utf-8")

    off = summ[~summ["noise_reduction"].astype(bool)]
    if len(off):
        plots.wer_by_noise_per_snr(off, registry, names, "Mean WER by noise (noise reduction off)") \
             .savefig(RES / "wer_by_noise.png", dpi=150)
        plots.wer_vs_snr(off, registry, names, "WER against SNR (noise reduction off)") \
             .savefig(RES / "wer_vs_snr.png", dpi=150)
    print("\n".join(out))
    print(f"\nWrote results/results.csv, summary.md and charts to {RES}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sanity", action="store_true", help="only run the clean LibriSpeech check")
    ap.add_argument("--librispeech", type=int, metavar="N", help="use N LibriSpeech clips, not recordings")
    ap.add_argument("--nr", choices=["both", "off", "on"], default="off",
                    help="spectral-gating noise reduction runs (default off)")
    ap.add_argument("--models", nargs="+", help="model keys from models.json (default: the two defaults)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--summarise-only", action="store_true", help="rebuild summary from results/results.csv")
    args = ap.parse_args()

    registry, defaults = core.load_registry()
    meta, noise_audio = core.load_noise_bank()
    names = {k: v["name"] for k, v in meta.items()}
    if args.summarise_only:
        summarise(pd.read_csv(RES / "results.csv"), registry, names)
        return

    keys = args.models or defaults
    print("Loading models:", ", ".join(keys))
    models = [core.get_model(k, registry) for k in keys]
    if args.sanity:
        sys.exit(0 if sanity(models) else 1)

    missing = [names[k] for k in meta if k not in noise_audio]
    if missing:
        print("WARNING: missing noises (run prepare_noises.py):", ", ".join(missing))
    clips = librispeech_clips(args.librispeech) if args.librispeech else own_recordings()
    if not clips:
        sys.exit("No recordings found in recordings/. Record them in the app, or use --librispeech 10.")
    nr_flags = {"both": [False, True], "off": [False], "on": [True]}[args.nr]
    df = run(clips, models, noise_audio, nr_flags, args.seed)
    RES.mkdir(exist_ok=True)
    df.to_csv(RES / "results.csv", index=False)
    summarise(df, registry, names)


if __name__ == "__main__":
    main()
