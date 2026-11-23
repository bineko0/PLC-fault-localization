"""
localize.py — Sliding-window jump localization for EFD data
(trend-robust, null-calibrated, bootstrap confidence).

Given one CSV's frames dict (from load_efd_data._load_file, or indexed from
load_all()) and a Baseline built from healthy recordings, produce a ranked
list of probable fault zones with honest confidence scores.

Usage (standalone):   python localize.py
Usage (imported):     from localize import build_baseline, localize, localize_all

Physical reasoning
------------------
1. On a healthy circuit the received PLC level is NOT flat along the loop.
   peakInputLevel follows a smooth wave over the 90 fixtures: the carrier forms
   a standing-wave pattern on the long series cable.
2. An earth fault does two things:
   a. It changes the electrical termination of the line, so the whole standing
      wave shifts. This is a SMOOTH change seen at every fixture. It tells you
      THAT there is a fault, not WHERE.
   b. It leaks part of the loop current to ground at ONE point. Fixtures sense
      the series loop current, so the received level shows an ABRUPT JUMP
      between the two fixtures on either side of the leak. This is the
      localization signal.
3. Therefore the detector must react to sudden jumps and ignore smooth slopes.
   The previous version compared plain window averages; on a wave-shaped
   profile every slope looked like a "step", which made it unstable.
4. Exception: at a VALLEY of the standing wave the profile is V-shaped. If the
   wave slides sideways or the valley gets shallower, the change there looks
   like a jump even without a leak. In validation, every answer within a few
   fixtures of a valley was inconsistent, so valley zones are down-weighted.

Pipeline per call to localize()
-------------------------------
1. Per-fixture features with robust statistics (median per fixture, so the
   occasional dropout reading of 0 cannot drag a fixture's value down).
   Cumulative counters (berCount, rxCrcFailCount) become rates per minute.
2. delta = faulted profile - healthy baseline profile (per fixture).
3. For every boundary b (between fixture b-1 and b): fit one local model to the
   W fixtures on each side: a line on the left, a line on the right, plus a
   "healthy wave slid sideways" term (B'(x)). The model's jump at the boundary,
   divided by its standard error, is the score. Smooth slopes and pure wave
   shifts are absorbed; only genuine discontinuities score.
4. Each metric's score is divided by that metric's 99th-percentile score on
   healthy-vs-healthy comparisons (the "null"), so all metrics are on the same
   scale and naturally noisy metrics are down-weighted. The ensemble is the
   mean of the squared normalized scores over the level metrics, multiplied by
   VALLEY_PENALTY within VALLEY_GUARD fixtures of a standing-wave valley.
5. Confidence for each candidate = detection x stability
     detection  : how far the peak rises above the largest false peak ever seen
                  in healthy-vs-healthy comparisons (0 at <= 0.5x, 1 at >= 2x).
     stability  : bootstrap vote share. Resample the recording's snapshots,
                  re-run the whole localizer N_BOOTSTRAP times, and count how
                  often the top answer lands within +/-VOTE_TOL fixtures of
                  this candidate.

Known limits (by design)
------------------------
* Boundaries need MIN_WINDOW fixtures on each side, so a fault before fixture
  1004 or after fixture 1088 cannot be scored.
* Fixture IDs are assumed to be in physical cable order. The smoothness of the
  healthy standing-wave profile supports this, but it is not confirmed metadata.
* The null is built from the run's other healthy recordings (leave-one-out).
  With only four healthy files per run it is coarse. Production use would need
  more healthy reference recordings.
* A real fault sitting inside a valley zone (about 1009–1020 and 1070–1083 on
  this rig) is partly masked by the valley penalty.
* VALLEY_GUARD/VALLEY_PENALTY were chosen after looking at validation results.
  They are physically motivated but have not been tested on fresh data.
* The filename labels (faultLocation / faultResistance) are copied into the
  result for VALIDATION ONLY. They are never used to compute a prediction.
  The only label-derived input is the choice of which recordings are healthy
  (faultLocation == 0), the equivalent of a commissioning reference recording.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── constants ─────────────────────────────────────────────────────────────────

ALL_FIXTURE_IDS: list[str] = [str(f) for f in range(1001, 1091)]  # 90 fixtures
N_FIX = len(ALL_FIXTURE_IDS)

# Feature definitions: name -> (message type, column, aggregation)
#   median : robust per-fixture level
#   mean   : per-fixture average (used for 0..1 rates)
#   rate   : cumulative counter -> increments per minute, log1p-scaled
#            (berCount and rxCrcFailCount are running totals that wrap around)
#   last   : last value of a cumulative counter
FEATURES: dict[str, tuple[str, str, str]] = {
    # signal-LEVEL metrics: used for localization
    "peakInputLevel":     ("EXCOMM", "peakInputLevel",     "median"),
    "peakCorrLevel_dB":   ("CORR",   "peakCorrLevel_dB",   "median"),
    "peakUncorrLevel_dB": ("CORR",   "peakUncorrLevel_dB", "median"),
    # diagnostic metrics: computed and reported, NOT used for localization
    "phaseNoise_dB":      ("EXCOMM", "phaseNoise_dB",      "median"),
    "berRate":            ("EXCOMM", "berCount",           "rate"),
    "crcFailRate":        ("EXCOMM", "rxCrcFailCount",     "rate"),
    "dsRawRate0":         ("RATES",  "dsRawRate0",         "mean"),
    "usRawRate0":         ("RATES",  "usRawRate0",         "mean"),
    "commFailures":       ("COMM",   "failures",           "last"),
}

# Counter widths from the manual (16-bit BER counter, 12-bit CRC counter).
COUNTER_MODULUS = {"berCount": 65536, "rxCrcFailCount": 4096}

LEVEL_METRICS: list[str] = ["peakInputLevel", "peakCorrLevel_dB", "peakUncorrLevel_dB"]
DIAGNOSTIC_METRICS: list[str] = [m for m in FEATURES if m not in LEVEL_METRICS]

WINDOW_SIZE: int = 5        # fixtures per side used to fit each local trend line
MIN_WINDOW: int = 3         # minimum fixtures per side (allows scoring near the ends)
TOP_K: int = 5              # candidates returned
PEAK_MIN_GAP: int = 5       # minimum distance between returned candidates
N_BOOTSTRAP: int = 30       # bootstrap repetitions for the stability vote
VOTE_TOL: int = 2           # a bootstrap answer within +/-2 fixtures "agrees"
NULL_QUANTILE: float = 99.0 # per-metric normalization quantile on the null
DETECT_LO: float = 0.5      # peak / null-max ratio at which detection = 0
DETECT_HI: float = 2.0      # peak / null-max ratio at which detection = 1
RANDOM_SEED: int = 0
# Changes the healthy standing wave can make by itself; each window is allowed to
# explain these before a jump is credited to a leak. Options: "slope", "curvature".
NUISANCE_SHAPES: tuple[str, ...] = ("slope",)
# Standing-wave valleys (local minima of the healthy peakInputLevel profile) are
# where wave re-tuning alone creates jump-like changes. Jumps within VALLEY_GUARD
# fixtures of a valley have their score multiplied by VALLEY_PENALTY.
VALLEY_GUARD: int = 5
VALLEY_PENALTY: float = 0.1


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class Baseline:
    """Healthy reference for one circuit (built only from no-fault recordings)."""
    mean: pd.DataFrame              # (90 fixtures x features) healthy profile
    roughness_floor: dict[str, float]
    null_q: dict[str, float]        # per-metric normalization (99th pct of null |t|)
    null_max: float                 # largest ensemble peak seen on healthy-vs-healthy
    null_maxima: np.ndarray         # all leave-one-out ensemble peaks
    n_files: int
    wave_shapes: dict[str, np.ndarray]  # (N, k) healthy-wave change shapes per level metric
    valleys: list[int]                  # indices of standing-wave valleys
    site_weight: np.ndarray             # (N,) 1.0, or VALLEY_PENALTY near a valley


@dataclass
class FaultZoneCandidate:
    """One entry in the ranked localization output."""
    rank: int
    fixture_boundary: int           # jump sits between fixture_boundary-1 and fixture_boundary
    fixture_range: tuple[int, int]  # zone to inspect (boundary +/- VOTE_TOL)
    confidence: float               # 0.0 – 1.0  (= detection x vote_share)
    ensemble_score: float           # null-normalized jump energy at this boundary
    detection: float                # 0..1, peak relative to largest false peak on healthy data
    vote_share: float               # 0..1, bootstrap agreement
    contributing_metrics: list[str] # level metrics ordered by contribution
    metric_scores: dict[str, float] # signed null-normalized jump per metric
    confidence_band: str            # HIGH / MEDIUM / LOW / NOISE
    notes: str = ""


@dataclass
class LocalizationResult:
    """Full output from one localize() call."""
    run: str
    fault_location_label: str       # from filename (validation only)
    fault_resistance_label: str
    candidates: list[FaultZoneCandidate]
    overall_confidence: float
    signal_strength: str            # STRONG / MODERATE / WEAK / NONE
    n_metrics_active: int           # level metrics whose peak exceeds their own null
    peak_to_null: float             # top ensemble score / null_max
    warnings: list[str] = field(default_factory=list)
    ensemble: Optional[np.ndarray] = None
    file: str = ""

    def summary(self) -> str:
        lines = [
            "=== EFD Localization Result ===",
            f"  Run: {self.run}  |  File: {self.file}",
            f"  (validation labels) loc={self.fault_location_label}  "
            f"res={self.fault_resistance_label}",
            f"  Signal strength : {self.signal_strength}   "
            f"(peak = {self.peak_to_null:.2f}x largest healthy false peak)",
            f"  Active metrics  : {self.n_metrics_active}/{len(LEVEL_METRICS)}",
            f"  Top confidence  : {self.overall_confidence:.0%}",
            "",
            f"  {'Rank':<5} {'Boundary':>14} {'Zone':>11} {'Conf':>6} {'Band':<7}"
            f" {'Detect':>6} {'Votes':>6} {'Score':>8}  Metrics",
            "  " + "-" * 88,
        ]
        for c in self.candidates:
            rng = f"{c.fixture_range[0]}–{c.fixture_range[1]}"
            lines.append(
                f"  #{c.rank:<4} {c.fixture_boundary-1}|{c.fixture_boundary:<9} {rng:>11} "
                f"{c.confidence:>5.0%}  {c.confidence_band:<7}"
                f" {c.detection:>5.0%} {c.vote_share:>6.0%} {c.ensemble_score:>8.2f}  "
                f"{', '.join(c.contributing_metrics)}"
            )
        if not self.candidates:
            lines.append("  (no candidates)")
        for c in self.candidates:
            if c.rank == 1 or c.confidence >= 0.15 or c.detection >= 0.7:
                lines.append(f"  Note #{c.rank}: {c.notes}")
        for w in self.warnings:
            lines.append(f"  [WARN] {w}")
        return "\n".join(lines)


# ── features ──────────────────────────────────────────────────────────────────

def _counter_rate(sub: pd.DataFrame, col: str) -> float:
    """Increments per minute of a wrapping cumulative counter, log1p-scaled."""
    sub = sub.sort_values("timestamp")
    v = sub[col].to_numpy(float)
    t = sub["timestamp"].to_numpy(float)
    ok = ~np.isnan(v)
    v, t = v[ok], t[ok]
    if len(v) < 2 or t[-1] <= t[0]:
        return np.nan
    d = np.diff(v)
    d = np.where(d < 0, d + COUNTER_MODULUS[col], d)  # counter wrapped around
    return float(np.log1p(d.sum() / (t[-1] - t[0]) * 60.0))


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _resample_rows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Bootstrap: resample each fixture's snapshots with replacement."""
    idx = []
    for _, pos in df.groupby("fixture_id").indices.items():
        idx.append(rng.choice(pos, size=len(pos), replace=True))
    if not idx:
        return df
    return df.iloc[np.concatenate(idx)]


def per_fixture_features(
    frames: dict[str, pd.DataFrame],
    metrics: Optional[list[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    One value per fixture per metric -> DataFrame (90 fixtures x metrics).
    If rng is given, each message stream is bootstrap-resampled first.
    """
    metrics = metrics or list(FEATURES)
    by_type: dict[str, list[str]] = defaultdict(list)
    for m in metrics:
        by_type[FEATURES[m][0]].append(m)

    out: dict[str, pd.Series] = {}
    for msg_type, ms in by_type.items():
        cols = sorted({FEATURES[m][1] for m in ms})
        df = _numeric(frames[msg_type], cols + ["timestamp"])
        if rng is not None:
            df = _resample_rows(df, rng)
        g = df.groupby("fixture_id")
        for m in ms:
            _, col, agg = FEATURES[m]
            if agg == "median":
                out[m] = g[col].median()
            elif agg == "mean":
                out[m] = g[col].mean()
            elif agg == "last":
                out[m] = df.sort_values("timestamp").groupby("fixture_id")[col].last()
            elif agg == "rate":
                out[m] = g[["timestamp", col]].apply(lambda s, c=col: _counter_rate(s, c))
    return pd.DataFrame(out).reindex(ALL_FIXTURE_IDS)[metrics]


# ── sliding-window jump detector ──────────────────────────────────────────────

_WEIGHT_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _jump_weights(mask: np.ndarray, window: int, min_window: int,
                  shape: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear weights so that  jump[b] = weights[b] @ profile.

    For each boundary b (between fixture b-1 and b) we fit ONE model to the
    `window` fixtures on each side:
        y(x) = left line            for x <  b
               left line + J + slope change   for x >= b
               (+ c * shape(x) on both sides, if `shape` is given)
    J is the jump at the boundary. Without `shape` this is exactly "fit a line
    on each side and compare them at the boundary".
    `shape` = slope of the healthy wave (B'(x)). Its term absorbs the fake step
    created when a V-shaped wave valley slides sideways.
    Also returns the variance factor of J for unit-variance noise.
    """
    key = (mask.tobytes(), window, min_window,
           None if shape is None else np.round(shape, 9).tobytes())
    if key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[key]
    x = np.arange(N_FIX, dtype=float)
    weights = np.zeros((N_FIX, N_FIX))
    varf = np.full(N_FIX, np.nan)
    for b in range(min_window, N_FIX - min_window + 1):
        left = np.arange(max(0, b - window), b)
        right = np.arange(b, min(N_FIX, b + window))
        left, right = left[mask[left]], right[mask[right]]
        if len(left) < min_window or len(right) < min_window:
            continue
        idx = np.concatenate([left, right])
        u = x[idx] - (b - 0.5)
        h = (idx >= b).astype(float)
        cols = [np.ones_like(u), u, h, u * h]
        if shape is not None:
            for k in range(shape.shape[1]):
                s = shape[idx, k]
                if np.std(s) > 1e-12:
                    cols.append((s - s.mean()) / np.std(s))
        X = np.column_stack(cols)
        XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
        inv = np.linalg.inv(XtX)
        w = (inv @ X.T)[2]               # row that returns J
        weights[b, idx] = w
        varf[b] = inv[2, 2]
    _WEIGHT_CACHE[key] = (weights, varf)
    return weights, varf


def _roughness(z: np.ndarray) -> float:
    """Robust point-to-point noise level of a profile (from second differences)."""
    d2 = np.diff(z[~np.isnan(z)], 2)
    if len(d2) < 5:
        return np.nan
    r = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6)
    return float(r if r > 0 else np.std(d2) / np.sqrt(6))


def _jump_t(z: np.ndarray, floor: float, window: int, min_window: int,
            shape: Optional[np.ndarray] = None) -> np.ndarray:
    """Signed jump score per boundary (jump / its standard error)."""
    mask = ~np.isnan(z)
    weights, varf = _jump_weights(mask, window, min_window, shape)
    sigma = max(np.nan_to_num(_roughness(z), nan=0.0), floor, 1e-9)
    jump = weights @ np.where(mask, z, 0.0)
    return jump / (sigma * np.sqrt(varf))


def _wave_shapes(b: np.ndarray) -> np.ndarray:
    """
    Shapes of change that the healthy standing wave can produce by itself,
    as columns of an (N, k) array:
      slope     B'(x)  : the wave slides sideways (phase shift)
      curvature B''(x) : valleys become shallower / rounder (wave ratio changes)
    """
    x = np.arange(N_FIX, dtype=float)
    ok = ~np.isnan(b)
    bf = np.interp(x, x[ok], b[ok])
    bs = np.convolve(np.pad(bf, 1, mode="edge"), np.ones(3) / 3, mode="valid")  # light smoothing
    cols = []
    if "slope" in NUISANCE_SHAPES:
        cols.append(np.gradient(bf))
    if "curvature" in NUISANCE_SHAPES:
        cols.append(np.gradient(np.gradient(bs)))
    return np.column_stack(cols) if cols else np.zeros((N_FIX, 0))


def _scores(delta: pd.DataFrame, bl: Baseline, window: int, min_window: int
            ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-metric null-normalized signed jump scores and their ensemble."""
    per: dict[str, np.ndarray] = {}
    for m in LEVEL_METRICS:
        shape = bl.wave_shapes[m] if NUISANCE_SHAPES else None
        t = _jump_t(delta[m].to_numpy(float), bl.roughness_floor[m], window, min_window, shape)
        per[m] = t / bl.null_q[m]
    ens = np.nanmean(np.vstack([np.nan_to_num(per[m]) ** 2 for m in LEVEL_METRICS]), axis=0)
    ens[np.isnan(per[LEVEL_METRICS[0]]) & np.isnan(per[LEVEL_METRICS[-1]])] = 0.0
    return ens * bl.site_weight, per


def _find_valleys(b: np.ndarray, half_width: int = 8) -> list[int]:
    """Local minima of the (lightly smoothed) healthy profile."""
    x = np.arange(N_FIX, dtype=float)
    ok = ~np.isnan(b)
    bs = np.convolve(np.pad(np.interp(x, x[ok], b[ok]), 2, mode="edge"), np.ones(5) / 5, "valid")
    return [i for i in range(2, N_FIX - 2)
            if bs[i] == bs[max(0, i - half_width): i + half_width + 1].min()]


def _site_weight(valleys: list[int]) -> np.ndarray:
    w = np.ones(N_FIX)
    for v in valleys:
        # boundary b sits between fixtures b-1 and b; guard both sides of the valley
        w[max(0, v - VALLEY_GUARD): min(N_FIX, v + VALLEY_GUARD + 2)] = VALLEY_PENALTY
    return w


# ── baseline ──────────────────────────────────────────────────────────────────

def build_baseline(
    baseline_frames_list: list[dict[str, pd.DataFrame]],
    window: int = WINDOW_SIZE,
    min_window: int = MIN_WINDOW,
) -> Baseline:
    """
    Build the healthy reference from >= 2 no-fault recordings of the same circuit.
    The null (false-alarm level) is measured leave-one-out: each healthy file is
    compared against the mean of the others, exactly as a faulted file would be.
    """
    if len(baseline_frames_list) < 2:
        raise ValueError("Need at least 2 healthy recordings to calibrate the null.")
    feats = [per_fixture_features(fr) for fr in baseline_frames_list]
    stack = np.array([f.to_numpy(float) for f in feats])          # (n, 90, M)
    mean = pd.DataFrame(np.nanmean(stack, axis=0), index=ALL_FIXTURE_IDS,
                        columns=list(FEATURES))

    # leave-one-out null deltas
    loo = []
    for k in range(len(feats)):
        others = np.delete(stack, k, axis=0)
        loo.append(pd.DataFrame(stack[k] - np.nanmean(others, axis=0),
                                index=ALL_FIXTURE_IDS, columns=list(FEATURES)))

    floor: dict[str, float] = {}
    for j, m in enumerate(FEATURES):
        r = np.nanmedian([_roughness(d[m].to_numpy(float)) for d in loo])
        floor[m] = float(max(np.nan_to_num(r), 0.25 * np.nanstd(stack[:, :, j] - np.nanmean(stack[:, :, j], 0)), 1e-6))

    wave_shapes = {m: _wave_shapes(mean[m].to_numpy(float)) for m in LEVEL_METRICS}
    null_q: dict[str, float] = {}
    for m in LEVEL_METRICS:
        shape = wave_shapes[m] if NUISANCE_SHAPES else None
        allt = np.concatenate([np.abs(_jump_t(d[m].to_numpy(float), floor[m], window,
                                              min_window, shape)) for d in loo])
        null_q[m] = float(max(np.nanpercentile(allt, NULL_QUANTILE), 1e-6))

    valleys = _find_valleys(mean["peakInputLevel"].to_numpy(float))
    tmp = Baseline(mean, floor, null_q, 1.0, np.array([]), len(feats), wave_shapes,
                   valleys, _site_weight(valleys))
    maxima = np.array([np.nanmax(_scores(d, tmp, window, min_window)[0]) for d in loo])
    tmp.null_maxima = maxima
    tmp.null_max = float(max(maxima.max(), 1e-6))
    return tmp


# ── peaks, confidence ─────────────────────────────────────────────────────────

def _find_peaks(scores: np.ndarray, top_k: int = TOP_K, min_gap: int = PEAK_MIN_GAP) -> list[int]:
    """Highest local maxima, at least min_gap apart."""
    s = np.nan_to_num(scores)
    is_peak = np.zeros_like(s, dtype=bool)
    is_peak[1:-1] = (s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:])
    order = [i for i in np.argsort(s)[::-1] if is_peak[i] and s[i] > 0]
    selected: list[int] = []
    for i in order:
        if all(abs(i - j) >= min_gap for j in selected):
            selected.append(int(i))
        if len(selected) >= top_k:
            break
    return selected


def _detection(score: float, null_max: float) -> float:
    ratio = score / null_max
    return float(np.clip((ratio - DETECT_LO) / (DETECT_HI - DETECT_LO), 0.0, 1.0))


def _confidence_band(conf: float) -> str:
    if conf >= 0.70:
        return "HIGH"
    if conf >= 0.40:
        return "MEDIUM"
    if conf >= 0.15:
        return "LOW"
    return "NOISE"


def _signal_strength(peak_to_null: float) -> str:
    if peak_to_null >= DETECT_HI:
        return "STRONG"
    if peak_to_null >= 1.0:
        return "MODERATE"
    if peak_to_null >= DETECT_LO:
        return "WEAK"
    return "NONE"


# ── main localize function ────────────────────────────────────────────────────

def localize(
    frames: dict[str, pd.DataFrame],
    baseline: Baseline,
    run: str = "?",
    top_k: int = TOP_K,
    window: int = WINDOW_SIZE,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    file: str = "",
) -> LocalizationResult:
    """Localize the fault in one recording. Call .summary() on the result to print it."""
    warn_msgs: list[str] = []
    try:   # labels are carried along for validation only
        fault_loc = str(frames["COMM"]["faultLocation"].iloc[0])
        fault_res = str(frames["COMM"]["faultResistance"].iloc[0])
    except (KeyError, IndexError):
        fault_loc = fault_res = "?"

    feats = per_fixture_features(frames, LEVEL_METRICS)
    missing = feats.isna().all(axis=1).sum()
    if missing:
        warn_msgs.append(f"{missing} fixture(s) have no level data.")
    delta = feats - baseline.mean[LEVEL_METRICS]
    ensemble, per = _scores(delta, baseline, window, MIN_WINDOW)

    peaks = _find_peaks(ensemble, top_k=top_k)
    top_score = float(ensemble[peaks[0]]) if peaks else 0.0
    peak_to_null = top_score / baseline.null_max

    # bootstrap stability vote
    rng = np.random.default_rng(seed)
    boot_top: list[int] = []
    for _ in range(n_bootstrap):
        fb = per_fixture_features(frames, LEVEL_METRICS, rng=rng)
        eb, _ = _scores(fb - baseline.mean[LEVEL_METRICS], baseline, window, MIN_WINDOW)
        pb = _find_peaks(eb, top_k=1)
        if pb:
            boot_top.append(pb[0])
    boot_top_arr = np.array(boot_top)

    fixture_ids = [int(f) for f in ALL_FIXTURE_IDS]
    candidates: list[FaultZoneCandidate] = []
    for idx in peaks:
        score = float(ensemble[idx])
        det = _detection(score, baseline.null_max)
        votes = float((np.abs(boot_top_arr - idx) <= VOTE_TOL).mean()) if len(boot_top_arr) else 0.0
        conf = det * votes
        contrib = {m: float(per[m][idx]) for m in LEVEL_METRICS if not np.isnan(per[m][idx])}
        active = sorted([m for m in contrib if abs(contrib[m]) >= 1.0],
                        key=lambda m: -abs(contrib[m]))
        if conf >= 0.70:
            note = "Clear, repeatable jump well above the healthy noise floor."
        elif conf >= 0.40:
            note = "Jump present but not fully stable; inspect this zone and the next candidate."
        elif conf >= 0.15:
            note = "Weak or unstable; treat as a hint only."
        elif det >= 0.7:
            note = ("Real jump (well above healthy noise) but always weaker than a higher-ranked "
                    "candidate; second place to inspect.")
        else:
            note = "Indistinguishable from noise."
        if baseline.site_weight[idx] < 1.0:
            note += " Sits at a standing-wave valley: could be wave re-tuning, not the leak."
        lo = max(0, idx - 1 - VOTE_TOL)
        hi = min(N_FIX - 1, idx + VOTE_TOL)
        candidates.append(FaultZoneCandidate(
            rank=0, fixture_boundary=fixture_ids[idx],
            fixture_range=(fixture_ids[lo], fixture_ids[hi]),
            confidence=conf, ensemble_score=score, detection=det, vote_share=votes,
            contributing_metrics=active,
            metric_scores={m: round(v, 2) for m, v in contrib.items()},
            confidence_band=_confidence_band(conf), notes=note,
        ))

    # rank by confidence, then by raw score
    candidates.sort(key=lambda c: (-c.confidence, -c.ensemble_score))
    for r, c in enumerate(candidates, start=1):
        c.rank = r

    n_active = sum(1 for m in LEVEL_METRICS if np.nanmax(np.abs(per[m])) >= 1.0)
    if peak_to_null < 1.0:
        warn_msgs.append("Top peak is not above the largest false peak seen on healthy data: "
                         "fault may be absent or too weak to localize.")
    if candidates and candidates[0].vote_share < 0.5:
        warn_msgs.append("Bootstrap answers are scattered: location is not well constrained.")

    return LocalizationResult(
        run=run, fault_location_label=fault_loc, fault_resistance_label=fault_res,
        candidates=candidates,
        overall_confidence=candidates[0].confidence if candidates else 0.0,
        signal_strength=_signal_strength(peak_to_null),
        n_metrics_active=n_active, peak_to_null=peak_to_null,
        warnings=warn_msgs, ensemble=ensemble, file=file,
    )


# ── batch helper ──────────────────────────────────────────────────────────────

def _is_healthy(frames: dict[str, pd.DataFrame]) -> bool:
    """Which recordings are healthy references (in production: a commissioning recording)."""
    return str(frames["COMM"]["faultLocation"].iloc[0]) == "0"


def localize_all(
    data: dict,
    include_null_checks: bool = False,
    n_bootstrap: int = N_BOOTSTRAP,
    verbose: bool = True,
) -> list[LocalizationResult]:
    """
    Run localize() on every faulted recording in load_all() output, one baseline per run.
    include_null_checks=True also localizes each healthy file against the OTHER healthy
    files: those results should come out with low confidence (false-alarm test).
    """
    results: list[LocalizationResult] = []
    for run in sorted(data):
        healthy = {fn: fr for fn, fr in data[run].items() if _is_healthy(fr)}
        if len(healthy) < 2:
            print(f"[WARN] {run}: fewer than 2 healthy recordings, skipping run.")
            continue
        bl = build_baseline(list(healthy.values()))
        if verbose:
            print(f"  {run}: baseline from {bl.n_files} healthy files, "
                  f"largest healthy false peak = {bl.null_max:.2f}")
        for fname, frames in sorted(data[run].items()):
            if fname in healthy:
                continue
            results.append(localize(frames, bl, run=run, n_bootstrap=n_bootstrap, file=fname))
        if include_null_checks:
            for fname in sorted(healthy):
                others = [fr for fn, fr in healthy.items() if fn != fname]
                bl_loo = build_baseline(others)
                results.append(localize(healthy[fname], bl_loo, run=run,
                                        n_bootstrap=n_bootstrap, file=fname))
    return results


def results_table(results: list[LocalizationResult]) -> pd.DataFrame:
    """Long format: one row per (recording, ranked candidate)."""
    rows = []
    for r in results:
        for c in r.candidates:
            rows.append({
                "run": r.run, "file": r.file,
                "true_fault_location": r.fault_location_label,
                "fault_resistance": r.fault_resistance_label,
                "signal_strength": r.signal_strength,
                "peak_to_null": round(r.peak_to_null, 3),
                "rank": c.rank, "fixture_boundary": c.fixture_boundary,
                "zone_start": c.fixture_range[0], "zone_end": c.fixture_range[1],
                "confidence": round(c.confidence, 4), "confidence_band": c.confidence_band,
                "detection": round(c.detection, 3), "vote_share": round(c.vote_share, 3),
                "ensemble_score": round(c.ensemble_score, 3),
                "top_metric": c.contributing_metrics[0] if c.contributing_metrics else "",
                "notes": c.notes,
            })
    return pd.DataFrame(rows)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, str(Path(__file__).parent))
    from load_efd_data import load_all

    print("Loading data...")
    data = load_all()
    print("\nRunning localization on all scenarios...\n")
    results = localize_all(data)
    for r in results:
        print(r.summary())
        print()
    out_df = results_table(results)
    out_path = Path(__file__).parent / "localization_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Results exported to: {out_path}")
