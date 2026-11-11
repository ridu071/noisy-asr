"""Noisy speech recognition: compare any 2 of 3 pretrained models, live.

Record, type what you said, add a noise (and listen to it), pick 2 models, press Run.

    python app.py            # http://127.0.0.1:7860  (microphone works on localhost)
    python app.py --share    # public https link, needed for the microphone in Colab
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import random

import gradio as gr

import asr_core as core
import plots

REGISTRY, DEFAULTS = core.load_registry()
NOISE_META, NOISES = core.load_noise_bank()
SNR_DB = 0          # each noise sounds as loud as the speech (A-weighted, 0 dB)

NOISE_ICONS = {"rain": "🌧️", "traffic": "🚗", "babble": "☕", "engine": "⚙️", "screaming": "😱"}
MODEL_DOTS = ["🔵", "🟠", "🟢", "🟡"]  # same order as the colours in plots.SERIES
CLEAN, NOISY = "🔇 Clean (no noise)", "🔊 Add noise"


def noise_name(nid: str) -> str:
    return "Clean" if nid == "clean" else NOISE_META.get(nid, {}).get("name", nid)


def noise_label(ids) -> str:
    return " + ".join(noise_name(i) for i in ids) if ids else "Clean (no noise)"


def label(key: str) -> str:
    return REGISTRY[key]["label"]


def noise_choices():
    return [(f"{NOISE_ICONS.get(k, '🔊')} {m['name']}", k) for k, m in NOISE_META.items() if k in NOISES]


def model_choices():
    return [(f"{MODEL_DOTS[i % len(MODEL_DOTS)]} {v['label']} · {v['params_m']:g}M", k)
            for i, (k, v) in enumerate(REGISTRY.items())]


# --------------------------------------------------------------------------- audio
def load_speech(audio_path):
    if not audio_path:
        raise gr.Error("Record something first (or upload a file).")
    speech = core.prepare_speech(core.load_audio(audio_path))
    if len(speech) < 0.3 * core.SR:
        raise gr.Error("The recording is too short or silent.")
    return speech


def make_noisy(speech, noise_ids, seed):
    """The exact audio the models hear: speech + every selected noise. Returns (audio, overall SNR)."""
    return core.mix_noises(speech, [NOISES[i] for i in noise_ids or []], SNR_DB, seed)


# --------------------------------------------------------------------------- one trial
def run_trial(speech, reference, noise_ids, seed, keys):
    """Mix once, send the identical array to every chosen model, score each one."""
    noisy, overall = make_noisy(speech, noise_ids, seed)
    dur = len(noisy) / core.SR
    row = {"time": dt.datetime.now().isoformat(timespec="seconds"), "reference": reference,
           "noise": "+".join(noise_ids) if noise_ids else "clean",
           "snr_db": None if overall is None else round(overall, 1),
           "seed": int(seed), "duration_s": round(dur, 2)}
    results = []
    for key in keys:
        text, secs = core.get_model(key, REGISTRY).transcribe(noisy)
        results.append({"key": key, "text": text, "score": core.score(reference, text),
                        "time": secs, "rtf": secs / dur})
    # Rank by words right (more is better); WER only breaks ties between equal word counts.
    best = min(rank_key(r) for r in results)
    winners = [r["key"] for r in results if rank_key(r) == best]
    best_correct = -best[0]
    # No contest with one model, and no "best" when nobody got a single word right.
    row["winners"] = [] if len(results) == 1 or best_correct == 0 else winners
    row["nobody"] = len(results) > 1 and best_correct == 0
    return row, results


def rank_key(r):
    sc = r["score"]
    return (-(sc.N - sc.S - sc.D), round(sc.wer, 9))


# --------------------------------------------------------------------------- rendering
KIND = {"correct": ("✅", "ok"), "misheard": ("✏️", "sub"), "missed": ("❌", "del"), "extra": ("➕", "ins")}


def word_pairs(sc: core.Score):
    """Line up what you said with what the model heard: (you_said, model_heard, kind)."""
    for typ, ref_w, hyp_w in sc.ops:
        if typ in ("equal", "substitute"):
            for r, h in zip(ref_w, hyp_w):
                yield r, h, "correct" if typ == "equal" else "misheard"
        elif typ == "delete":
            for r in ref_w:
                yield r, "", "missed"
        elif typ == "insert":
            for h in hyp_w:
                yield "", h, "extra"


def word_chips(sc: core.Score) -> str:
    cols = []
    for r, h, kind in word_pairs(sc):
        icon, cls = KIND[kind]
        cols.append(f'<div class="wcol {cls}"><span class="wsaid">{html.escape(r) or "—"}</span>'
                    f'<span class="wheard">{html.escape(h) or "—"}</span><span class="wicon">{icon}</span></div>')
    legend = ('<div class="wlegend"><span>top: you said</span><span>bottom: model heard</span></div>')
    return f'{legend}<div class="wrow">{"".join(cols)}</div>'


def correct_words(sc: core.Score) -> int:
    return sc.N - sc.S - sc.D


def model_card(row, res):
    key, sc = res["key"], res["score"]
    spec = REGISTRY[key]
    color = plots.model_color(key, REGISTRY)
    correct = correct_words(sc)
    acc = round(100 * correct / sc.N)
    errors = sc.S + sc.D + sc.I
    winners = row["winners"]
    if key in winners and len(winners) > 1:
        badge = '<span class="badge tie">🤝 Tied best</span>'
    elif key in winners:
        badge = '<span class="badge win">🏆 Did best</span>'
    else:
        badge = ""
    heard = html.escape(res["text"])
    heard_html = f"“{heard}”" if heard else '<i class="muted">nothing: no words recognised</i>'
    counts = (f'<span class="cnt ok">✅ {correct} correct</span><span class="cnt sub">✏️ {sc.S} misheard</span>'
              f'<span class="cnt del">❌ {sc.D} missed</span><span class="cnt ins">➕ {sc.I} extra</span>')
    plural = "s" if errors != 1 else ""
    best = " best" if winners == [key] else ""   # only a clear winner is highlighted, never a tie
    return f"""
<div class="rcard{best}" style="--c:{color}">
  <div class="rtop"></div>
  <div class="rbody">
    <div class="rhead">
      <div><div class="rname">{html.escape(spec['label'])}</div>
           <div class="muted small">{spec['params_m']:g}M params · {html.escape(spec['architecture'])}</div>
           <div style="margin-top:8px">{badge}</div></div>
      <div class="ring" style="--p:{max(acc, 0)}"><div><b>{acc}%</b><span>words right</span></div></div>
    </div>
    <div class="lbl">You said</div>
    <div class="said">“{html.escape(row['reference'].strip())}”</div>
    <div class="lbl">The model heard</div>
    <div class="heard">{heard_html}</div>
    <div class="lbl">Word by word</div>
    {word_chips(sc)}
    <div class="cnts">{counts}</div>
    <div class="result">
      <div>Got <b>{correct} of {sc.N}</b> words right</div>
      <div class="wer"><b>{sc.wer * 100:.0f}%</b> WER <span class="muted small">= {errors} error{plural} ÷ {sc.N} words · lower is better</span></div>
    </div>
    <div class="muted small" style="margin-top:8px">⏱ {res['time']:.2f} s to transcribe {row['duration_s']:.1f} s of audio</div>
  </div>
</div>"""


def cards_html(row, results):
    """One card per selected model, side by side (1, 2 or 3 columns)."""
    return (f'<div class="cards" style="--n:{len(results)}">'
            + "".join(model_card(row, r) for r in results) + "</div>")


def verdict_html(row, results):
    cond = noise_label([] if row["noise"] == "clean" else row["noise"].split("+"))
    if row["snr_db"] is not None:
        cond += f" · overall SNR {row['snr_db']:.1f} dB"
    winners = row["winners"]
    if row.get("nobody"):
        head = "😶 None of the models understood you"
    elif not winners:
        head = f"📝 {html.escape(label(results[0]['key']))} result"
    elif len(winners) == 1:
        head = f"🏆 {html.escape(label(winners[0]))} did best"
    elif len(winners) == len(results):
        head = "🤝 It's a tie"
    else:
        head = "🤝 Tie: " + " and ".join(html.escape(label(k)) for k in winners)
    bars = []
    for r in sorted(results, key=rank_key):      # best first: most words right, then lowest WER
        sc = r["score"]
        correct = correct_words(sc)
        note = " · heard nothing" if not r["text"].strip() else ""
        medal = " 🏆" if winners == [r["key"]] else ""
        color = plots.model_color(r["key"], REGISTRY)
        bars.append(
            f'<div class="hrow"><div class="hname">{html.escape(label(r["key"]))}{medal}</div>'
            f'<div class="htrack"><div class="hfill" style="width:{100 * correct / sc.N:.0f}%;'
            f'background:{color}"></div></div>'
            f'<div class="hval"><b>{correct}/{sc.N}</b> words · {sc.wer * 100:.0f}% WER{note}</div></div>')
    note = ""
    if transcript_looks_wrong(results):
        note = ('<div class="mismatch">⚠️ <b>Your typed text may not match the recording.</b> The models '
                'mostly agree with <i>each other</i> but not with what you typed, so their high error '
                'rates may be wrong. Check the text box and run again.</div>')
    return (f'<div class="verdict"><div class="vhead">{head}</div>'
            f'<div class="muted small">Noise: {html.escape(cond)}</div>{"".join(bars)}{note}</div>')


def transcript_looks_wrong(results) -> bool:
    """Spot a typed transcript that doesn't match the audio.

    If every model scores badly against your text, yet the models largely agree with each
    other, the likeliest explanation is the text, not the models. Needs at least 2 models
    that actually produced words (two empty outputs would 'agree' trivially).
    """
    texts = [r["text"] for r in results]
    if len(results) < 2 or not all(core.normalise(t) for t in texts):
        return False
    if min(r["score"].wer for r in results) < 0.5:
        return False
    pairs = [(a, b) for i, a in enumerate(texts) for b in texts[i + 1:]]
    return sum(core.agreement(a, b) for a, b in pairs) / len(pairs) >= 0.6


def loading_html(keys):
    names = [html.escape(label(k)) for k in keys]
    joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    sub = ("The models are listening to the same noisy audio" if len(keys) > 1
           else "The model is listening to the noisy audio")
    return (f'<div class="verdict loading"><div class="spinner"></div>'
            f'<div><div class="vhead">Running {joined}…</div><div class="muted small">{sub}</div></div></div>')


def order(keys):
    """Selected models in registry order, so each one always sits in the same place."""
    return [k for k in REGISTRY if k in (keys or [])]


# --------------------------------------------------------------------------- handlers
def active_noises(mode, noise_ids):
    """The noises actually used: none in Clean mode, the ticked ones in Add-noise mode."""
    return [] if mode == CLEAN else list(noise_ids or [])


def on_mode(mode):
    return gr.Column(visible=(mode == NOISY))


def on_preview(audio_path, mode, noise_ids):
    """Runs as soon as a recording exists or the noise choice changes: build and play it."""
    if not audio_path:
        return None, gr.Audio(value=None, label="🎧 Preview (record first)")
    noise_ids = active_noises(mode, noise_ids)
    if mode == NOISY and not noise_ids:
        return None, gr.Audio(value=None, label="🎧 Pick at least one noise to hear it")
    seed = random.randrange(1_000_000)
    noisy, overall = make_noisy(load_speech(audio_path), noise_ids, seed)
    if not noise_ids:
        title = "🎧 Your recording, no noise added"
    else:
        title = f"🎧 What the models will hear: {noise_label(noise_ids).lower()} · SNR {overall:.1f} dB"
    return seed, gr.Audio(value=(core.SR, noisy), label=title)


def _validate(audio_path, reference, mode, noise_ids, keys):
    if not keys:
        raise gr.Error("Select at least one model.")
    if mode == NOISY and not active_noises(mode, noise_ids):
        raise gr.Error("Pick at least one noise, or choose 'Clean (no noise)'.")
    if not core.normalise(reference or ""):
        raise gr.Error("Type exactly what you said. The models are scored against it.")
    return load_speech(audio_path)


def what_is_missing(audio_path, reference, mode, noise_ids, keys):
    """First thing the user still has to do, as (friendly message, field id), or None if ready."""
    if not audio_path:
        return "Please record yourself first 🎙️", "rec"
    try:
        load_speech(audio_path)
    except gr.Error:
        return "That recording is too short or silent. Please record again 🎙️", "rec"
    if not core.normalise(reference or ""):
        return "Please type what you said ✍️ The models are scored against it.", "said"
    if not keys:
        return "Please pick at least one model.", "models"
    if mode == NOISY and not active_noises(mode, noise_ids):
        return "Please pick a noise, or choose Clean (no noise).", "noise"
    return None


def on_start(audio_path, reference, mode, noise_ids, keys):
    """Check everything is filled in. If not: a small pop-up, and nothing else happens."""
    missing = what_is_missing(audio_path, reference, mode, noise_ids, keys)
    if missing:
        gr.Warning(missing[0], duration=4, title="Almost there")
        return gr.skip(), gr.skip(), missing[1]
    return loading_html(order(keys)), "", ""


def on_run(audio_path, reference, mode, noise_ids, keys, seed, missing):
    if missing:                       # on_start already showed the pop-up
        return gr.skip(), gr.skip()
    speech = _validate(audio_path, reference, mode, noise_ids, keys)
    seed = random.randrange(1_000_000) if seed is None else seed
    row, results = run_trial(speech, reference, active_noises(mode, noise_ids), seed, order(keys))
    return verdict_html(row, results), cards_html(row, results)


# --------------------------------------------------------------------------- layout
def credits_md():
    parts = ["**Credits.** "]
    for k, m in NOISE_META.items():
        if k in NOISES:
            parts.append(f"{m['name']}: {m['source']}, [{m['licence']}]({m['url']}). ")
    parts.append("Models: " + ", ".join(f"{v['label']} ({v['hf_id']}, {v['licence']})"
                                        for v in REGISTRY.values()) + ".")
    return "".join(parts)


def models_table():
    rows = [f"| {v['label']} | `{v['hf_id']}` | {v['params_m']}M | {v['architecture']} | {v['train_hours']:,} h |"
            for v in REGISTRY.values()]
    return ("| Model | Hugging Face ID | Params | Architecture | Training audio |\n|---|---|---|---|---|\n"
            + "\n".join(rows))


def noises_table():
    rows = []
    for k, m in NOISE_META.items():
        if k in NOISES:
            secs = len(NOISES[k]) / core.SR
            rows.append(f"| {NOISE_ICONS.get(k, '')} {m['name']} | {m['character']} | {m['source']} | "
                        f"[{m['licence']}]({m['url']}) | {secs:.0f} s |")
    return ("| Noise | What it sounds like | Source | Licence | Clip length |\n|---|---|---|---|---|\n"
            + "\n".join(rows) + "\n\nEach clip is 16 kHz mono. A random part of it is used for every "
            "recording, looped if your recording is longer than the clip.")


def hero_html():
    return f"""
<div id="hero">
  <div class="hero-icon">🎙️</div>
  <h1>Speech recognition in noise</h1>
  <div class="hero-line">
    <p>Say something, add real-world noise, and watch AI models try to understand you.</p>
    <div class="tags"><span class="tag">🤖 {len(REGISTRY)} models</span><span class="tag">🔊 {len(NOISES)} noises</span><span class="tag">💻 runs on your laptop</span></div>
  </div>
</div>"""


def step_head(n, title, sub):
    return (f'<div class="step-head"><span class="step-num">{n}</span><span>{title}</span></div>'
            f'<div class="step-sub">{sub}</div>')


CSS = """
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* hero */
#hero-wrap, #hero-wrap .html-container { padding: 0 !important; border: none !important; background: none !important; }
#hero .hero-line { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: nowrap;
  position: relative; z-index: 1; }
#hero .hero-line p { margin: 0 !important; flex: 1 1 auto; min-width: 0; }
#hero .tags { display: flex; gap: 8px; flex: none; }
#hero .tags .tag { margin: 0; }
#hero { position: relative; overflow: hidden; border-radius: 22px; padding: 26px 34px; color: #fff;
  background: linear-gradient(135deg, #4338ca 0%, #7c3aed 55%, #c026d3 100%);
  box-shadow: 0 14px 34px rgba(79, 70, 229, .28); }
#hero h1 { color: #fff; margin: 0 0 6px; font-size: 32px; font-weight: 800; letter-spacing: -.02em; }
#hero p { color: rgba(255,255,255,.9); margin: 0 0 16px; font-size: 16px; }
#hero .hero-icon { position: absolute; right: 26px; top: 10px; font-size: 96px; opacity: .18; transform: rotate(-12deg); }
#hero .tag { display: inline-block; margin: 0 8px 4px 0; padding: 5px 13px; border-radius: 999px; font-size: 13px;
  background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.32); color: #fff; }

/* step cards */
.step-card { border: 1.5px solid var(--border-color-primary) !important; border-radius: 18px !important;
  padding: 16px !important; background: var(--block-background-fill) !important; gap: 10px !important;
  transition: border-color .2s, box-shadow .2s; }
.step-card:hover, .step-card:focus-within { border-color: #8b5cf6 !important; box-shadow: 0 0 0 4px rgba(139, 92, 246, .14); }
.step-head { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 17px; }
.step-num { width: 30px; height: 30px; border-radius: 50%; color: #fff; font-size: 15px; display: grid; place-items: center;
  background: linear-gradient(135deg, #6366f1, #a855f7); box-shadow: 0 3px 10px rgba(124, 58, 237, .35); flex: none; }
.step-sub { opacity: .65; font-size: 13px; margin: 2px 0 0 40px; }

/* recorder: microphone picker pinned to the bottom-right corner */
#rec-audio { border-radius: 14px !important; min-height: 0 !important; }
#rec-audio .mic-select { max-width: 230px; font-size: 13px !important; }
#rec-audio .mic-wrap { margin: 10px 12px !important; }
#rec-audio button[aria-label="Clear"], #rec-audio button[aria-label="Clear audio"] { display: none !important; }
#rerecord-btn { align-self: flex-start; border-radius: 999px !important; font-weight: 600 !important;
  border: 1.5px solid var(--border-color-primary) !important; padding: 6px 16px !important; }
#rerecord-btn:hover { border-color: #8b5cf6 !important; }
#said-box textarea { font-size: 15px !important; padding: 12px 14px !important; border-radius: 12px !important; }

/* layout: cards 1 and 2 equal height, models strip full width */
.top-row { align-items: stretch !important; }
.top-row > .step-card { align-self: stretch !important; justify-content: flex-start !important; }
.top-row > .step-card > * { flex: 0 0 auto !important; }
.models-strip { align-items: center !important; flex-wrap: wrap !important; gap: 18px !important; }
.models-strip > .html-container, .models-strip > div:first-child { flex: 0 0 auto !important; min-width: 260px !important; }
.models-strip .pills { flex: 1 1 auto !important; min-width: 0 !important; }
#results { scroll-margin-top: 16px; }
.nudge { animation: nudge .5s ease; border-radius: 12px; box-shadow: 0 0 0 3px rgba(250, 178, 25, .7) !important; }
@keyframes nudge { 0%, 100% { transform: translateX(0); } 20%, 60% { transform: translateX(-6px); }
  40%, 80% { transform: translateX(6px); } }
.mismatch { margin-top: 12px; padding: 10px 14px; border-radius: 12px; font-size: 14px;
  background: rgba(250, 178, 25, .14); border: 1px solid rgba(250, 178, 25, .5); }

/* Clean / Add noise: one two-way switch */
.segmented .wrap { display: grid !important; grid-template-columns: 1fr 1fr; gap: 4px !important; padding: 4px;
  border-radius: 14px; background: var(--background-fill-secondary); border: 1.5px solid var(--border-color-primary); }
.segmented label { justify-content: center; border: none !important; background: transparent !important;
  border-radius: 10px !important; padding: 10px !important; font-weight: 700 !important; box-shadow: none !important;
  transition: all .15s ease !important; }
.segmented label input { display: none !important; }
.segmented label:has(input:checked), .segmented label.selected {
  background: linear-gradient(135deg, #6366f1, #a855f7) !important; color: #fff !important;
  box-shadow: 0 4px 12px rgba(124, 58, 237, .35) !important; }
.segmented label:has(input:checked) *, .segmented label.selected * { color: #fff !important; }

/* noise picker: its own panel with outlined chips */
.noise-panel { border: 1.5px dashed rgba(139, 92, 246, .45) !important; border-radius: 14px !important;
  padding: 12px 14px !important; background: rgba(139, 92, 246, .06) !important; gap: 6px !important; }
.noise-panel .html-container { padding: 0 !important; }
.noise-panel .form, .noise-panel .block, .noise-panel fieldset { background: transparent !important;
  border: none !important; box-shadow: none !important; padding: 0 !important; }
.panel-title { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; opacity: .7; }
.chips, .chips .wrap { background: transparent !important; border: none !important; }
.chips .wrap { gap: 8px !important; }
.chips label { border-radius: 10px !important; padding: 7px 14px !important; font-weight: 600 !important;
  border: 1.5px solid var(--border-color-primary) !important; background: var(--block-background-fill) !important;
  box-shadow: none !important; transition: all .15s ease !important; }
.chips label:hover { border-color: #a855f7 !important; }
.chips label input { display: none !important; }
.chips label:has(input:checked), .chips label.selected {
  border-color: #a855f7 !important; background: rgba(168, 85, 247, .20) !important; }
.chips label:has(input:checked)::after, .chips label.selected::after {
  content: "✓"; margin-left: 8px; font-weight: 800; color: #a855f7; }

/* pill toggles (noise mode, noises, models) */
.pills .wrap { gap: 10px !important; }
.pills label { border-radius: 999px !important; padding: 9px 18px !important; font-weight: 600 !important;
  border: 1.5px solid var(--border-color-primary) !important; background: var(--background-fill-secondary) !important;
  box-shadow: none !important; transition: all .15s ease !important; }
.pills label:hover { border-color: #8b5cf6 !important; transform: translateY(-1px); }
.pills label input { display: none !important; }
.pills label:has(input:checked), .pills label.selected {
  background: linear-gradient(135deg, #6366f1, #a855f7) !important; color: #fff !important;
  border-color: transparent !important; box-shadow: 0 5px 14px rgba(124, 58, 237, .35) !important; }
.pills label:has(input:checked) *, .pills label.selected * { color: #fff !important; }

/* run button */
#run-btn { background: linear-gradient(135deg, #4338ca, #9333ea) !important; color: #fff !important; border: none !important;
  font-size: 19px !important; font-weight: 800 !important; padding: 16px !important; border-radius: 16px !important;
  box-shadow: 0 10px 24px rgba(79, 70, 229, .35) !important; transition: transform .15s, filter .15s !important; }
#run-btn:hover { transform: translateY(-2px); filter: brightness(1.1); }

/* verdict + head-to-head bars */
.verdict { border-radius: 18px; padding: 18px 22px; border: 1.5px solid var(--border-color-primary);
  background: linear-gradient(135deg, rgba(99,102,241,.10), rgba(168,85,247,.06)); }
.vhead { font-size: 22px; font-weight: 800; margin-bottom: 2px; }
.hrow { display: grid; grid-template-columns: 190px 1fr 150px; align-items: center; gap: 14px; margin-top: 12px; }
.hname { font-weight: 700; }
.htrack { height: 16px; border-radius: 999px; background: var(--border-color-primary); overflow: hidden; }
.hfill { height: 100%; border-radius: 999px; min-width: 4px; transition: width .6s ease; }
.hval { font-size: 14px; }
.loading { display: flex; align-items: center; gap: 16px; }
.spinner { width: 34px; height: 34px; border-radius: 50%; border: 4px solid rgba(139,92,246,.25); border-top-color: #8b5cf6;
  animation: spin .8s linear infinite; flex: none; }
@keyframes spin { to { transform: rotate(360deg); } }

/* result cards: one column per selected model */
.cards { display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); gap: 16px; align-items: start; }
@media (max-width: 1000px) { .cards { grid-template-columns: 1fr; } }
/* result card */
.rcard { border-radius: 20px; overflow: hidden; border: 1.5px solid var(--border-color-primary);
  background: var(--block-background-fill); box-shadow: 0 6px 20px rgba(0,0,0,.06); }
.rtop { height: 7px; background: var(--c); }
.rcard.best { border: 2.5px solid var(--c); transform: translateY(-4px);
  box-shadow: 0 0 0 5px color-mix(in srgb, var(--c) 22%, transparent),
              0 16px 36px color-mix(in srgb, var(--c) 35%, transparent); }
.rcard.best .rtop { height: 10px; }
.rbody { padding: 18px 22px 20px; }
.rhead { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
.rname { font-size: 20px; font-weight: 800; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 13px; font-weight: 700; }
.badge.win { background: #0ca30c; color: #fff; }
.badge.tie { border: 1.5px solid currentColor; opacity: .8; }
.ring { width: 96px; height: 96px; border-radius: 50%; flex: none; display: grid; place-items: center;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), var(--border-color-primary) 0); }
.ring > div { width: 76px; height: 76px; border-radius: 50%; background: var(--block-background-fill);
  display: flex; flex-direction: column; align-items: center; justify-content: center; line-height: 1.1; }
.ring b { font-size: 22px; } .ring span { font-size: 10px; opacity: .7; text-align: center; }
.lbl { font-size: 11px; text-transform: uppercase; letter-spacing: .07em; opacity: .6; font-weight: 700; margin-top: 16px; }
.said { font-size: 16px; opacity: .9; }
.heard { font-size: 21px; font-weight: 700; }
.wlegend { display: flex; gap: 14px; font-size: 11px; opacity: .55; margin: 4px 0; }
.wrow { display: flex; flex-wrap: wrap; gap: 8px; }
.wcol { display: flex; flex-direction: column; align-items: stretch; gap: 3px; text-align: center; }
.wcol span { padding: 3px 10px; border-radius: 9px; font-size: 14px; white-space: nowrap; }
.wsaid { opacity: .75; }
.wheard { font-weight: 700; }
.wicon { font-size: 12px; padding: 0 !important; }
.wcol.ok .wsaid, .wcol.ok .wheard { background: rgba(12,163,12,.16); }
.wcol.sub .wsaid, .wcol.sub .wheard { background: rgba(237,161,0,.24); }
.wcol.del .wsaid, .wcol.del .wheard { background: rgba(227,73,72,.22); }
.wcol.ins .wsaid, .wcol.ins .wheard { background: rgba(98,80,214,.24); }
.cnts { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.cnt { padding: 4px 11px; border-radius: 999px; font-size: 13px; font-weight: 600; }
.cnt.ok { background: rgba(12,163,12,.14); } .cnt.sub { background: rgba(237,161,0,.2); }
.cnt.del { background: rgba(227,73,72,.18); } .cnt.ins { background: rgba(98,80,214,.2); }
.result { margin-top: 16px; padding: 12px 14px; border-radius: 14px; background: var(--background-fill-secondary); font-size: 15px; }
.result .wer b { font-size: 24px; }
.muted { opacity: .65; } .small { font-size: 12.5px; }

/* footer */
.credits { opacity: .5; font-size: 12px !important; }
.credits p { font-size: 12px !important; }
.credits a { color: inherit !important; text-decoration: underline; }

@media (max-width: 720px) {
  .hrow { grid-template-columns: 1fr; gap: 4px; }
  #hero h1 { font-size: 25px; } #hero .hero-icon { font-size: 64px; }
  #hero .hero-line { flex-wrap: wrap; } #hero .tags { flex-wrap: wrap; }
}
"""

# Ready: scroll to the results. Something missing: jump to that field and give it a short shake.
SCROLL_JS = """(missing) => { setTimeout(() => {
  if (!missing) {
    const r = document.getElementById('results');
    if (r) r.scrollIntoView({behavior: 'smooth', block: 'start'});
    return;
  }
  const where = {said: '#said-box', rec: '#rec-audio', models: '.models-strip', noise: '.noise-panel'}[missing];
  const el = where && document.querySelector(where);
  if (!el) return;
  el.scrollIntoView({behavior: 'smooth', block: 'center'});
  el.classList.remove('nudge'); void el.offsetWidth; el.classList.add('nudge');
  setTimeout(() => el.classList.remove('nudge'), 1600);
  const t = el.querySelector('textarea'); if (t) t.focus();
}, 120); }"""

THEME = gr.themes.Soft(
    primary_hue="violet", secondary_hue="indigo", neutral_hue="slate", radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)


def build_ui():
    with gr.Blocks(title="Speech recognition in noise") as demo:
        gr.HTML(hero_html(), elem_id="hero-wrap")
        seed = gr.State(None)  # set by the preview, so Run uses exactly the audio you heard

        with gr.Row(equal_height=True, elem_classes="top-row"):
            with gr.Column(elem_classes="step-card"):
                gr.HTML(step_head(1, "Record yourself, then type what you said",
                                  "Both models are scored against your text"))
                audio_in = gr.Audio(sources=["microphone"], type="filepath", show_label=False,
                                    elem_id="rec-audio")
                rerecord_btn = gr.Button("🔄 Re-record", size="sm", visible=False, elem_id="rerecord-btn")
                reference = gr.Textbox(show_label=False, lines=1, max_lines=3, elem_id="said-box",
                                       placeholder="✍️  Type exactly what you said, e.g. I am studying in college")
            with gr.Column(elem_classes="step-card"):
                gr.HTML(step_head(2, "Add noise", "Each noise sounds as loud as your voice. Listen before you run"))
                noise_mode = gr.Radio([CLEAN, NOISY], value=NOISY, show_label=False, elem_classes="segmented")
                with gr.Column(elem_classes="noise-panel") as noise_panel:
                    gr.HTML('<div class="panel-title">Choose noises · one or more</div>')
                    noise = gr.CheckboxGroup(noise_choices(), value=["babble"] if "babble" in NOISES else [],
                                             show_label=False, elem_classes="chips")
                noisy_play = gr.Audio(label="🎧 Preview (record first)", interactive=False)

        with gr.Row(elem_classes="step-card models-strip"):
            gr.HTML(step_head(3, "Pick models", "Select one, two or all three. Each one gets its own result card"))
            models = gr.CheckboxGroup(model_choices(), value=DEFAULTS[:2], show_label=False,
                                      elem_classes="pills")
        run_btn = gr.Button("▶  Run comparison", elem_id="run-btn")

        verdict = gr.HTML(elem_id="results")
        cards = gr.HTML()

        with gr.Accordion("ℹ️ About the models and noises", open=False):
            gr.Markdown("### Models\n" + models_table() + "\n\n### Noises\n" + noises_table()
                        + "\n\n### How it works\nEach selected noise is scaled to *sound* as loud as your "
                        f"voice ({SNR_DB} dB, A-weighted, so low rumbles like engine and traffic are not "
                        "quieter to the ear than rain). Several noises add up, so the overall SNR "
                        "shown is lower. Greedy decoding for all models (no beam search, no language model). "
                        f"Device: {core.DEVICE}, {core.NUM_THREADS} threads. Timing covers the forward "
                        "pass and decoding only, after a warm-up run. Before scoring, your transcript "
                        "and the outputs are lowercased, punctuation is removed and numbers become "
                        "words (\"4\" = \"four\"). WER (word error rate) = (misheard + missed + extra "
                        "words) ÷ words you said.")
        gr.Markdown(credits_md(), elem_classes="credits")

        preview_in = [audio_in, noise_mode, noise]
        audio_in.change(on_preview, preview_in, [seed, noisy_play]).then(
            lambda a: gr.Button(visible=bool(a)), audio_in, rerecord_btn)
        rerecord_btn.click(lambda: (None, gr.Button(visible=False)), None, [audio_in, rerecord_btn])
        noise.change(on_preview, preview_in, [seed, noisy_play])
        noise_mode.change(on_mode, noise_mode, noise_panel).then(on_preview, preview_in, [seed, noisy_play])
        missing = gr.State("")
        run_btn.click(on_start, [audio_in, reference, noise_mode, noise, models],
                      [verdict, cards, missing], show_progress="hidden").then(
            None, missing, None, js=SCROLL_JS).then(
            on_run, [audio_in, reference, noise_mode, noise, models, seed, missing],
            [verdict, cards], show_progress="hidden").then(
            None, missing, None, js=SCROLL_JS)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public https link (needed in Colab)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    for key in REGISTRY:
        print(f"Loading {label(key)} ...", flush=True)
        core.get_model(key, REGISTRY)
    build_ui().queue(default_concurrency_limit=1).launch(
        share=args.share, server_port=args.port, theme=THEME, css=CSS)
