"""
EFD fault localisation analysis.

  Step 1 : what the healthy circuit looks like + which metrics are dead
  Step 2 : spatial plots, healthy vs hard short (the standing wave)
  Step 3 : metric usefulness table (evidence for the gap analysis)
  Step 4 : cross-run fingerprint check (do location codes mean the same on every day?)
  Step 5 : sliding-window jump localisation (localize.py): ranked candidates,
           confidence, false-alarm test, calibration, plots and text report

Run:      python analyse_efd.py
Outputs:  plots/ directory (PNG / CSV / TXT)

NOTE: the filename-encoded fault location/resistance is used ONLY to group and
validate results, never as a model input.
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
from load_efd_data import load_all
import localize as L

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

RUNS = ("T0001", "T0002", "T0003")
LOCS = ["1", "2", "3", "4"]
RESS = ["0", "1", "2", "3", "4", "5", "8"]
RES_LABEL = {"0": "0 Ω", "1": "5k", "2": "14.3k", "3": "19.3k", "4": "30k", "5": "35k", "8": "50k"}
LOW_RES = {"0", "1", "2"}          # 0 Ω – 14.3 kΩ
X = np.arange(1001, 1091)

# ── chart style (validated reference palette; first 3 slots are CVD-safe all-pairs) ──
RUN_COLORS = {"T0001": "#2a78d6", "T0002": "#eb6834", "T0003": "#1baf7a"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#f0efec", "#b3302f"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#f0efec"] + SEQ_BLUE)
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})


def _fixture_xaxis(ax):
    ax.set_xlim(1000.5, 1090.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.set_xlabel("Fixture ID (assumed physical order along the loop)")


def _save(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def _label(frames, key):
    return str(frames["COMM"][key].iloc[0])


def _features(data):
    """feats[run][fname] = (loc, res, per-fixture feature DataFrame). Computed once."""
    out = {}
    for run in RUNS:
        out[run] = {}
        for fname, fr in sorted(data[run].items()):
            out[run][fname] = (_label(fr, "faultLocation"), _label(fr, "faultResistance"),
                               L.per_fixture_features(fr))
    return out


def _healthy_mean(feats, run, exclude=None):
    rows = [f.to_numpy(float) for fn, (loc, _, f) in feats[run].items() if loc == "0" and fn != exclude]
    return np.nanmean(rows, axis=0)


def consistency_flags(df: pd.DataFrame, col: str = "b") -> pd.Series:
    """
    An answer is 'consistent' if it lies within ±2 fixtures of at least half of the
    answers for the SAME run and location code at OTHER resistances.
    (The true positions are confidential, so agreement is the only check available.)
    """
    ok = []
    for _, r in df.iterrows():
        others = df[(df.run == r.run) & (df["loc"] == r["loc"]) & (df.res != r.res)][col]
        ok.append(bool(len(others)) and (np.abs(others - r[col]) <= 2).mean() >= 0.5)
    return pd.Series(ok, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — healthy circuit, dead metrics, cumulative counters
# ══════════════════════════════════════════════════════════════════════════════

def step1(data):
    print("\n" + "=" * 60 + "\nSTEP 1 — HEALTHY CIRCUIT & RAW METRIC CHECK\n" + "=" * 60)
    from load_efd_data import SCHEMA
    rows = []
    for mt in ("COMM", "RATES", "EXCOMM", "CORR"):
        df = pd.concat([fr[mt].assign(file=fn) for run in RUNS for fn, fr in data[run].items()],
                       ignore_index=True)
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        for c in SCHEMA[mt][3:]:
            v = pd.to_numeric(df[c], errors="coerce")
            sample = df.assign(v=v, t=ts).sort_values("t")
            # counters reset at the start of each recording, so test per (file, fixture);
            # "mostly increasing" allows for the occasional wrap-around of 12/16-bit counters
            mono = sample.groupby(["file", "fixture_id"]).v.apply(
                lambda s: np.nan if s.nunique() <= 2 else float((np.diff(s.to_numpy(float)) >= 0).mean() > 0.9)).mean()
            rows.append({"message": mt, "field": c, "min": v.min(), "max": v.max(),
                         "n_distinct": v.nunique(),
                         "cumulative_counter": bool(mono > 0.5) if mono == mono else False,
                         "constant": v.nunique() <= 1})
    t = pd.DataFrame(rows)
    t.to_csv(PLOTS_DIR / "step1_raw_field_check.csv", index=False)
    print(t.to_string(index=False))
    print("\n  -> constant fields carry no information; cumulative counters must be turned into rates.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — the standing wave: healthy vs hard short, per run
# ══════════════════════════════════════════════════════════════════════════════

def step2(feats):
    print("\n" + "=" * 60 + "\nSTEP 2 — SPATIAL PROFILES (standing wave)\n" + "=" * 60)
    loc_style = {"1": "-", "2": "--", "3": "-.", "4": ":"}
    for metric in ("peakInputLevel", "peakUncorrLevel_dB"):
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for ax, run in zip(axes, RUNS):
            j = list(L.FEATURES).index(metric)
            ax.plot(X, _healthy_mean(feats, run)[:, j], color=MUTED, lw=2.5, label="healthy (mean of 4)")
            for fn, (loc, res, f) in feats[run].items():
                if loc != "0" and res == "0":
                    ax.plot(X, f[metric], color=RUN_COLORS[run], ls=loc_style[loc], lw=1.6,
                            label=f"hard short, loc code {loc}")
            ax.set_title(f"{run} — {metric}: healthy vs 0 Ω faults")
            ax.set_ylabel(metric)
            ax.legend(fontsize=7, ncol=5, loc="upper left")
        _fixture_xaxis(axes[-1])
        fig.tight_layout()
        _save(fig, f"step2_profile_{metric}.png")

    # delta profiles, one panel per (run, location), one line per resistance
    fig, axes = plt.subplots(3, 4, figsize=(18, 9), sharex=True)
    for r_i, run in enumerate(RUNS):
        bm = _healthy_mean(feats, run)[:, 0]
        for l_i, loc in enumerate(LOCS):
            ax = axes[r_i, l_i]
            items = sorted([(res, f) for fn, (lc, res, f) in feats[run].items() if lc == loc])
            shades = SEQUENTIAL(np.linspace(1.0, 0.35, max(len(items), 2)))
            for k, (res, f) in enumerate(items):
                shade = shades[k]
                ax.plot(X, f["peakInputLevel"].to_numpy(float) - bm, color=shade, lw=1.3,
                        label=RES_LABEL[res])
            ax.axhline(0, color=AXIS, lw=1)
            ax.set_title(f"{run}  loc code {loc}", fontsize=9)
            if l_i == 0:
                ax.set_ylabel("Δ peakInputLevel\n(fault − healthy)")
            if r_i == 0 and l_i == 0:
                ax.legend(fontsize=6, title="resistance", title_fontsize=6)
    for ax in axes[-1]:
        _fixture_xaxis(ax)
        ax.set_xlabel("Fixture ID")
    fig.suptitle("Change caused by the fault (darker = lower resistance). Same shape at every "
                 "resistance within a run; amplitude shrinks as resistance rises.", fontsize=10)
    fig.tight_layout()
    _save(fig, "step2_delta_peakInputLevel.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — which metrics carry signal?
# ══════════════════════════════════════════════════════════════════════════════

def _single_metric_consistency(data, metric):
    """Run the localizer using ONE metric only (no bootstrap) -> consistency of top answers."""
    saved = L.LEVEL_METRICS
    L.LEVEL_METRICS = [metric]
    try:
        rows = []
        for run in RUNS:
            healthy = [fr for fr in data[run].values() if _label(fr, "faultLocation") == "0"]
            bl = L.build_baseline(healthy)
            for fn, fr in data[run].items():
                if _label(fr, "faultLocation") == "0":
                    continue
                r = L.localize(fr, bl, run=run, n_bootstrap=0)
                top = max(r.candidates, key=lambda c: c.ensemble_score) if r.candidates else None
                rows.append({"run": run, "loc": _label(fr, "faultLocation"),
                             "res": _label(fr, "faultResistance"),
                             "b": top.fixture_boundary if top else np.nan})
        df = pd.DataFrame(rows)
        ok = consistency_flags(df)
        lo = df.res.isin(LOW_RES)
        # attractor share: fraction of a run's answers within ±2 of that run's most common answer.
        # Four different locations should NOT all land on one spot (ideal ≈ 0.25–0.3).
        attr = np.mean([(np.abs(g.b - g.b.mode().iloc[0]) <= 2).mean() for _, g in df.groupby("run")])
        return ok[lo].mean(), ok[~lo].mean(), attr
    finally:
        L.LEVEL_METRICS = saved


def step3(data, feats):
    print("\n" + "=" * 60 + "\nSTEP 3 — METRIC USEFULNESS\n" + "=" * 60)
    names = list(L.FEATURES)
    rows = []
    for j, m in enumerate(names):
        det_hard, det_weak, same, diff = [], [], [], []
        for run in RUNS:
            healthy = [fn for fn, (loc, _, _) in feats[run].items() if loc == "0"]
            null_rms = np.nanmean([np.sqrt(np.nanmean(
                (feats[run][fn][2].to_numpy(float)[:, j] - _healthy_mean(feats, run, fn)[:, j]) ** 2))
                for fn in healthy])
            bm = _healthy_mean(feats, run)[:, j]
            prof = {}
            for fn, (loc, res, f) in feats[run].items():
                if loc == "0":
                    continue
                d = f.to_numpy(float)[:, j] - bm
                ratio = np.sqrt(np.nanmean(d ** 2)) / max(null_rms, 1e-12)
                (det_hard if res == "0" else det_weak if res == "8" else []).append(ratio)
                prof[(loc, res)] = np.nan_to_num(d - np.nanmean(d))
            keys = list(prof)
            for a in range(len(keys)):
                for b in range(a + 1, len(keys)):
                    if keys[a][1] == keys[b][1]:
                        continue
                    c = np.corrcoef(prof[keys[a]], prof[keys[b]])[0, 1]
                    if np.isfinite(c):
                        (same if keys[a][0] == keys[b][0] else diff).append(c)
        lo, hi, attr = _single_metric_consistency(data, m)
        rows.append({
            "metric": m, "used_for_localization": m in L.LEVEL_METRICS,
            "detect_hard_short_x_noise": round(float(np.nanmedian(det_hard)), 1),
            "detect_50k_x_noise": round(float(np.nanmedian(det_weak)), 1),
            "fingerprint_corr_same_loc": round(float(np.nanmean(same)), 2),
            "fingerprint_corr_diff_loc": round(float(np.nanmean(diff)), 2),
            "jump_consistency_lowR": round(float(lo), 2),
            "jump_consistency_highR": round(float(hi), 2),
            "attractor_share": round(float(attr), 2),
        })
    t = pd.DataFrame(rows)
    t.to_csv(PLOTS_DIR / "step3_metric_usefulness.csv", index=False)
    print(t.to_string(index=False))
    print("\n  detect_*      : RMS change across fixtures, in units of healthy-vs-healthy noise")
    print("  fingerprint_* : within a run, how alike the change patterns are (same vs different location)")
    print("  jump_*        : consistency of the localizer if it used this metric alone")
    print("  attractor_share: share of ALL answers in a run landing on one spot. High = the metric keeps")
    print("                  pointing at the same glitchy fixture whatever the fault, so its 'consistency' is fake")
    return t


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — do location codes mean the same physical point on every day?
# ══════════════════════════════════════════════════════════════════════════════

def step4(feats, sw_top: pd.DataFrame):
    print("\n" + "=" * 60 + "\nSTEP 4 — CROSS-RUN FINGERPRINT CHECK\n" + "=" * 60)
    hard = {}
    for run in RUNS:
        bm = _healthy_mean(feats, run)[:, 0]
        for fn, (loc, res, f) in feats[run].items():
            if loc != "0" and res == "0":
                d = f["peakInputLevel"].to_numpy(float) - bm
                hard[(run, loc)] = np.nan_to_num(d - np.nanmean(d))
    top = {(r.run, r["loc"]): int(r.b) for _, r in sw_top[sw_top.res == "0"].iterrows()}

    pairs = [("T0001", "T0002"), ("T0001", "T0003"), ("T0002", "T0003")]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    fig.subplots_adjust(wspace=0.55)
    lines = []
    for ax, (a, b) in zip(axes, pairs):
        M = np.array([[np.corrcoef(hard[(a, i)], hard[(b, k)])[0, 1] for k in LOCS] for i in LOCS])
        im = ax.imshow(M, cmap=DIVERGING, vmin=-1, vmax=1)
        for i in range(4):
            for k in range(4):
                ax.text(k, i, f"{M[i, k]:.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if abs(M[i, k]) > 0.6 else INK)
        ax.set_xticks(range(4), [f"loc {l}\n→{top.get((b, l), '?')}" for l in LOCS])
        ax.set_yticks(range(4), [f"loc {l} →{top.get((a, l), '?')}" for l in LOCS])
        ax.set_xlabel(b)
        ax.set_ylabel(a)
        ax.grid(False)
        same = float(np.mean(np.diag(M)))
        lines.append(f"  {a} vs {b}: mean corr same code = {same:+.2f}; best match per {a} code: " +
                     ", ".join(f"{l}->{LOCS[int(np.argmax(M[i]))]} ({M[i].max():.2f})"
                               for i, l in enumerate(LOCS)))
    fig.colorbar(im, ax=axes, fraction=0.015, label="correlation of hard-short change patterns")
    fig.suptitle("Do the same location codes give the same fingerprint on different days?  "
                 "(→ = fixture boundary the localizer chose)", fontsize=10)
    _save(fig, "step4_cross_run_fingerprints.png")
    print("\n".join(lines))
    print("  -> if the diagonal is not the strongest cell, the code does NOT name the same physical point"
          " on both days (or the rig behaved differently).")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — sliding-window localisation (localize.py)
# ══════════════════════════════════════════════════════════════════════════════

BAND_ORDER = ["HIGH", "MEDIUM", "LOW", "NOISE"]


def _top_table(results) -> pd.DataFrame:
    rows = []
    for r in results:
        c = r.candidates[0] if r.candidates else None
        rows.append({
            "run": r.run, "file": r.file, "loc": r.fault_location_label, "res": r.fault_resistance_label,
            "b": c.fixture_boundary if c else np.nan,
            "zone_start": c.fixture_range[0] if c else np.nan,
            "zone_end": c.fixture_range[1] if c else np.nan,
            "confidence": round(c.confidence, 3) if c else 0.0,
            "band": c.confidence_band if c else "NOISE",
            "detection": round(c.detection, 3) if c else 0.0,
            "vote_share": round(c.vote_share, 3) if c else 0.0,
            "peak_to_null": round(r.peak_to_null, 2), "signal_strength": r.signal_strength,
            "rank2_b": r.candidates[1].fixture_boundary if len(r.candidates) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def _plot_grid(f: pd.DataFrame):
    """Rows = (run, location code), cols = resistance; text = predicted boundary, colour = confidence."""
    keys = [(run, loc) for run in RUNS for loc in LOCS]
    M = np.full((len(keys), len(RESS)), np.nan)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for i, (run, loc) in enumerate(keys):
        for k, res in enumerate(RESS):
            s = f[(f.run == run) & (f["loc"] == loc) & (f.res == res)]
            if s.empty:
                continue
            M[i, k] = s.confidence.iloc[0]
    ax.imshow(np.ma.masked_invalid(M), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for i, (run, loc) in enumerate(keys):
        for k, res in enumerate(RESS):
            s = f[(f.run == run) & (f["loc"] == loc) & (f.res == res)]
            if s.empty:
                ax.text(k, i, "—", ha="center", va="center", color=MUTED, fontsize=8)
                continue
            r = s.iloc[0]
            txt = f"{int(r.b)}\n{r.band} {r.confidence:.0%}" + ("" if r.ok else "  ✗")
            ax.text(k, i, txt, ha="center", va="center", fontsize=7,
                    color="#ffffff" if r.confidence > 0.55 else INK)
    ax.set_xticks(range(len(RESS)), [RES_LABEL[r] for r in RESS])
    ax.set_yticks(range(len(keys)), [f"{run}  loc {loc}" for run, loc in keys])
    for y in (3.5, 7.5):
        ax.axhline(y, color="#fcfcfb", lw=3)
    ax.grid(False)
    ax.set_xlabel("Fault resistance")
    ax.set_title("Rank-1 fault boundary per scenario (colour = confidence; ✗ = disagrees with the "
                 "same location at other resistances)")
    _save(fig, "sw_prediction_grid.png")


def _plot_profiles(results, bl_valleys):
    lookup = {(r.run, r.fault_location_label, r.fault_resistance_label): r for r in results}
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    for ax, loc in zip(axes.ravel(), LOCS):
        for run in RUNS:
            for v in bl_valleys[run]:
                ax.axvspan(1001 + v - L.VALLEY_GUARD - 0.5, 1001 + v + L.VALLEY_GUARD + 0.5,
                           color=GRID, alpha=0.5, lw=0)
            r = lookup.get((run, loc, "0"))
            if r is None:
                continue
            ax.plot(X, r.ensemble, color=RUN_COLORS[run], lw=1.6, label=run)
            if r.candidates:
                ax.axvline(r.candidates[0].fixture_boundary, color=RUN_COLORS[run], ls="--", lw=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
        ax.set_title(f"Location code {loc}, hard short (0 Ω)")
        ax.set_ylabel("Jump score (× healthy noise)")
        ax.legend(fontsize=7)
    for ax in axes[-1]:
        _fixture_xaxis(ax)
    fig.suptitle("Sliding-window jump score along the loop. Dashed = chosen boundary. "
                 "Grey bands = standing-wave valley zones (down-weighted).", fontsize=10)
    fig.tight_layout()
    _save(fig, "sw_profiles_hard_short.png")


def _plot_confidence(f: pd.DataFrame, null_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    for run in RUNS:
        pts = [(k, f[(f.run == run) & (f.res == res)].confidence.mean()) for k, res in enumerate(RESS)
               if ((f.run == run) & (f.res == res)).any()]
        ax.plot([k for k, _ in pts], [y for _, y in pts], marker="o", ms=8, color=RUN_COLORS[run],
                label=run)
    ax.plot([len(RESS)], [null_df.confidence.mean()], marker="s", ms=8, color=MUTED, ls="none",
            label="healthy (false-alarm test)")
    for thr, name in ((0.70, "HIGH"), (0.40, "MEDIUM"), (0.15, "LOW")):
        ax.axhline(thr, color=AXIS, ls=":", lw=1)
        ax.text(-0.4, thr + 0.01, name, fontsize=7, color=MUTED)
    ax.set_xticks(range(len(RESS) + 1), [RES_LABEL[r] for r in RESS] + ["no fault"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Fault resistance")
    ax.set_ylabel("Mean rank-1 confidence")
    ax.set_title("Confidence falls as the fault gets weaker")
    ax.legend(fontsize=7)

    ax = axes[1]
    stats = f.groupby("band").ok.agg(["mean", "size"]).reindex(BAND_ORDER)
    bars = ax.bar(range(4), stats["mean"].fillna(0), color=SEQ_BLUE[3], width=0.55)
    for i, (m, n) in enumerate(zip(stats["mean"], stats["size"])):
        ax.text(i, (0 if np.isnan(m) else m) + 0.02, f"{0 if np.isnan(m) else m:.0%}\n(n={int(n) if n == n else 0})",
                ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(4), BAND_ORDER)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Share of answers that are consistent")
    ax.set_title("Is the confidence honest? (higher band → more often consistent)")
    fig.tight_layout()
    _save(fig, "sw_confidence.png")


def step5(data, feats):
    print("\n" + "=" * 60 + "\nSTEP 5 — SLIDING-WINDOW JUMP LOCALISATION\n" + "=" * 60)
    results = L.localize_all(data, include_null_checks=True)
    faulted = [r for r in results if r.fault_location_label != "0"]
    nulls = [r for r in results if r.fault_location_label == "0"]

    top = _top_table(results)
    f = top[top["loc"] != "0"].copy()
    f["ok"] = consistency_flags(f).values
    null_df = top[top["loc"] == "0"].copy()

    f.to_csv(PLOTS_DIR / "sliding_window_top1.csv", index=False)
    L.results_table(faulted).to_csv(PLOTS_DIR / "sliding_window_candidates.csv", index=False)
    null_df.to_csv(PLOTS_DIR / "sliding_window_false_alarm_test.csv", index=False)
    report = "\n\n".join(r.summary() for r in faulted)
    report += "\n\n" + "#" * 70 + "\n# FALSE-ALARM TEST: each healthy file vs the other healthy files\n" + \
              "#" * 70 + "\n\n" + "\n\n".join(r.summary() for r in nulls)
    (PLOTS_DIR / "sliding_window_report.txt").write_text(report, encoding="utf-8")
    print("  Saved sliding_window_top1.csv, sliding_window_candidates.csv, "
          "sliding_window_false_alarm_test.csv, sliding_window_report.txt")

    lo = f.res.isin(LOW_RES)
    print(f"\n  Consistent answers: overall {f.ok.mean():.0%} | 0–14.3 kΩ {f[lo].ok.mean():.0%} "
          f"| 19.3–50 kΩ {f[~lo].ok.mean():.0%}")
    print("\n  Calibration (share consistent per confidence band):")
    print(f.groupby("band").ok.agg(["mean", "size"]).reindex(BAND_ORDER).to_string())
    print(f"\n  False-alarm test: {null_df.band.isin(['HIGH', 'MEDIUM']).sum()} of {len(null_df)} healthy "
          f"files reached MEDIUM/HIGH (bands: {dict(null_df.band.value_counts())})")
    print("\n  Rank-1 boundary per (run, location) x resistance:")
    print(f.pivot_table(index=["run", "loc"], columns="res", values="b").to_string())

    valleys = {}
    for run in RUNS:
        healthy = [fr for fr in data[run].values() if _label(fr, "faultLocation") == "0"]
        valleys[run] = L.build_baseline(healthy).valleys
        print(f"  {run} standing-wave valleys at fixtures {[1001 + v for v in valleys[run]]}")

    _plot_grid(f)
    _plot_profiles(results, valleys)
    _plot_confidence(f, null_df)
    return f, null_df


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data = load_all()
    feats = _features(data)
    step1(data)
    step2(feats)
    sw_top, null_df = step5(data, feats)
    step4(feats, sw_top)
    step3(data, feats)
    print("\nDone. All outputs saved to:", PLOTS_DIR)
