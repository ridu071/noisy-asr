"""Matplotlib charts shared by the app and the benchmark.

Each model keeps one colour everywhere (its position in models.json), so Whisper is
always blue and Wav2Vec2 always orange, whatever subset is plotted.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9890"


def model_color(key: str, registry: dict) -> str:
    keys = [k for k in registry if not k.startswith("_")]
    return SERIES[keys.index(key) % len(SERIES)] if key in keys else NEUTRAL


def _noise_order(present, noise_names: dict) -> list[str]:
    """Noises in noises.json order (not alphabetical), clean excluded."""
    present = set(present)
    return [n for n in noise_names if n in present and n != "clean"]


def _style(ax, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def _fig(ncols=1, w=7.0, h=3.6, sharey=True):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), sharey=sharey, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes[0]


def _legend(fig, handles, labels):
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9,
               labelcolor=INK_2, ncols=len(labels), handlelength=1.0)


def grouped_bars(ax, df, x_col, x_order, value_col, registry, err_col=None):
    """Bars of value_col per x, one bar per model. Returns legend handles."""
    models = [m for m in registry if m in set(df["model"])]
    n = len(models)
    width = 0.8 / max(n, 1)
    xs = np.arange(len(x_order))
    handles = []
    for i, m in enumerate(models):
        sub = df[df["model"] == m].set_index(x_col).reindex(x_order)
        vals = sub[value_col].to_numpy(dtype=float)
        err = None
        if err_col:  # mean +/- SD, but never below 0 % WER
            sd = np.nan_to_num(sub[err_col].to_numpy(dtype=float))
            err = np.vstack([np.minimum(sd, np.nan_to_num(vals)), sd])
        h = ax.bar(xs + (i - (n - 1) / 2) * width, vals, width, color=model_color(m, registry),
                   edgecolor=SURFACE, linewidth=1.5, yerr=err,
                   error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 0})
        handles.append(h)
    ax.set_xticks(xs, x_order)
    return handles, [registry[m].get("label", m) for m in models]


def wer_by_noise(df: pd.DataFrame, registry: dict, title: str, noise_names: dict | None = None):
    """df columns: noise, model, wer (0-1). One bar per model per noise."""
    noise_names = noise_names or {}
    d = df.copy()
    d["noise_label"] = d["noise"].map(lambda n: noise_names.get(n, n))
    order = list(dict.fromkeys(d["noise_label"]))
    d["wer"] = d["wer"] * 100
    fig, (ax,) = _fig(w=7.5, h=3.6)
    handles, labels = grouped_bars(ax, d, "noise_label", order, "wer", registry)
    _style(ax, "WER (%)")
    ax.set_ylim(0, max(100, float(np.nanmax(d["wer"])) * 1.1 if len(d) else 100))
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    _legend(fig, handles, labels)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def wer_by_noise_per_snr(summary: pd.DataFrame, registry: dict, noise_names: dict, title: str):
    """summary columns: noise, snr_db, model, wer_mean, wer_sd. One panel per SNR."""
    snrs = sorted(summary.loc[summary["noise"] != "clean", "snr_db"].dropna().unique(), reverse=True)
    noises = _noise_order(summary["noise"], noise_names)
    order = [noise_names.get(n, n) for n in noises]
    fig, axes = _fig(ncols=len(snrs), w=4.2 * len(snrs), h=3.8)
    handles = labels = None
    for ax, snr in zip(axes, snrs):
        d = summary[(summary["snr_db"] == snr) & (summary["noise"] != "clean")].copy()
        d["noise_label"] = d["noise"].map(lambda n: noise_names.get(n, n))
        d["wer_mean"] *= 100
        d["wer_sd"] *= 100
        handles, labels = grouped_bars(ax, d, "noise_label", order, "wer_mean", registry, "wer_sd")
        _style(ax, "Mean WER (%)" if ax is axes[0] else None)
        ax.set_title(f"SNR {snr:g} dB", loc="left", color=INK_2, fontsize=10)
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].set_ylim(bottom=0)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    _legend(fig, handles, labels)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def wer_vs_snr(summary: pd.DataFrame, registry: dict, noise_names: dict, title: str):
    """Small multiples, one panel per noise: WER against SNR, one line per model.

    The clean baseline is drawn at the left as 'clean' so each line starts from it.
    """
    noises = _noise_order(summary["noise"], noise_names)
    snrs = sorted(summary.loc[summary["noise"] != "clean", "snr_db"].dropna().unique(), reverse=True)
    xlabels = ["clean"] + [f"{s:g}" for s in snrs]
    fig, axes = _fig(ncols=len(noises), w=2.6 * len(noises), h=3.4)
    handles, labels = [], []
    for ax, noise in zip(axes, noises):
        for m in [k for k in registry if k in set(summary["model"])]:
            clean = summary[(summary["noise"] == "clean") & (summary["model"] == m)]["wer_mean"]
            vals = [float(clean.iloc[0]) if len(clean) else np.nan]
            for s in snrs:
                v = summary[(summary["noise"] == noise) & (summary["snr_db"] == s) & (summary["model"] == m)]
                vals.append(float(v["wer_mean"].iloc[0]) if len(v) else np.nan)
            (h,) = ax.plot(range(len(xlabels)), np.array(vals) * 100, color=model_color(m, registry),
                           linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                           markeredgewidth=1.5)
            if ax is axes[0]:
                handles.append(h)
                labels.append(registry[m].get("label", m))
        _style(ax, "Mean WER (%)" if ax is axes[0] else None)
        ax.set_xticks(range(len(xlabels)), xlabels)
        ax.set_xlabel("SNR (dB)", color=INK_2, fontsize=9)
        ax.set_title(noise_names.get(noise, noise), loc="left", color=INK_2, fontsize=10)
    axes[0].set_ylim(bottom=0)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    _legend(fig, handles, labels)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def nr_effect(delta: pd.DataFrame, registry: dict, noise_names: dict, title: str):
    """delta columns: noise, model, delta (WER with NR minus without, 0-1). Below 0 = NR helped."""
    d = delta.copy()
    d["noise_label"] = d["noise"].map(lambda n: noise_names.get(n, n))
    d["delta"] *= 100
    order = [noise_names.get(n, n) for n in _noise_order(d["noise"], noise_names)]
    fig, (ax,) = _fig(w=7.5, h=3.6)
    handles, labels = grouped_bars(ax, d, "noise_label", order, "delta", registry)
    _style(ax, "WER change with noise reduction (points)")
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_title(title + "  (below 0 = noise reduction helped)", loc="left", color=INK, fontsize=11)
    _legend(fig, handles, labels)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig
