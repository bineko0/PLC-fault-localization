# EFD Localization: Sliding-Window Jump Detection

**Question:** does the PLC telemetry already collected from every fixture carry information about *where* an earth fault is?

**Short answer:** yes, for low-impedance faults. Using only existing telemetry, a sliding-window jump detector produces the same fault boundary at 0 Ω, 5 kΩ and 14.3 kΩ in **81 %** of scenarios, versus **11 %** for the previous approach. It gets 33–50 % at 30–50 kΩ. The confidence score separates trustworthy answers from untrustworthy ones: answers marked **HIGH are consistent 86 %** of the time, answers marked LOW or NOISE only about 10 %. In the false-alarm test, healthy recordings never reached MEDIUM or HIGH (0 of 12).

> Ground truth for fault locations 1–4 is confidential. "Consistent" therefore means *agrees with the same location code at the other resistances of the same test day*. It is a repeatability check, not an accuracy check against a known position.

---

## 1. What the data showed

| Finding | Consequence |
|---|---|
| `response_rate` = 100 % and every repeater depth = 0 in **every** file, including hard shorts. All `RATES` depths 1–7 are zero. | COMM and RATES carry almost no signal on this rig. The channel never degraded enough to need repeaters. |
| `berCount`, `rxCrcFailCount`, `attempts`, `failures` and `userCorrTrigCount` are **running totals** that reset at each recording and wrap at 12/16 bits. | Averaging them, as the previous code did, is meaningless. They are converted to per-minute rates. |
| Single snapshots occasionally read 0 (`peakInputLevel`, `peakCorrLevel_dB`). | Per-fixture **medians**, not means. |
| The healthy `peakInputLevel` profile is a smooth **standing wave** along the fixtures: valleys near 1014 and ~1076, peak near 1040–1050. The same shape appears on all three days. | Fixture IDs are almost certainly in physical cable order. "Normal" is not flat. |
| A fault (a) **re-shapes the whole wave** and (b) adds a **sharp jump between two neighbouring fixtures**. Within one day the jump stays at the same place for every resistance. | (a) says *that* there is a fault. (b) says *where*. |

**Physical interpretation.** In a series loop, each fixture's modem couples to the loop **current**. A leak to ground removes part of the current at one point, so the received level is discontinuous across the leak. That jump is the localization signal. The wave re-shaping is a global change of the line's termination: it is visible at every fixture, so it detects a fault but cannot place it.

## 2. Method (`localize.py`)

1. **Features per fixture:** median `peakInputLevel`, `peakCorrLevel_dB` and `peakUncorrLevel_dB`. These three signal-level metrics are used for localization; the others are diagnostics only (see §5).
2. **Δ profile:** faulted minus healthy baseline, per fixture. The baseline is the mean of that day's healthy recordings.
3. **Sliding-window jump test** at every boundary between fixture *b−1* and *b*. Fit one model to the 5 fixtures on each side: a line on the left, a line on the right, plus a "healthy wave slid sideways" term (the slope of the healthy profile). The fitted jump divided by its standard error is the score. Smooth slopes and pure wave shifts are absorbed; only real discontinuities score.
4. **Null calibration:** each healthy file is compared with the day's other healthy files (leave-one-out). Each metric's score is divided by its 99th-percentile score on that null, and the three metrics are averaged as squared scores.
5. **Valley guard:** near a standing-wave valley (±5 fixtures), a sideways shift of the V-shaped valley produces a step even without a leak. In validation, **all 15** answers that landed near a valley were inconsistent. These boundaries have their score multiplied by 0.1.
6. **Confidence = detection × stability.**
   - *Detection:* the peak relative to the largest false peak ever seen on healthy data (0 at ≤ 0.5×, 1 at ≥ 2×).
   - *Stability:* bootstrap vote share. Resample the recording's snapshots 30 times and count how often the top answer lands within ±2 fixtures.
   - *Bands:* HIGH ≥ 0.70, MEDIUM ≥ 0.40, LOW ≥ 0.15, NOISE below that.

**Output per recording:** up to 5 ranked fault zones. Each is a boundary "fixture *a* | fixture *b*" with a ±2-fixture inspection zone, a confidence, its band and a plain-language note.

## 3. Results (80 recordings: 68 faulted, 12 healthy)

| Resistance | n | Consistent | Mean confidence | HIGH answers |
|---|---|---|---|---|
| 0 Ω | 12 | 83 % | 1.00 | 12 |
| 5 kΩ | 12 | 83 % | 0.86 | 10 |
| 14.3 kΩ | 12 | 75 % | 0.69 | 8 |
| 19.3 kΩ | 4 | 50 % | 0.53 | 2 |
| 30 kΩ | 12 | 33 % | 0.43 | 4 |
| 35 kΩ | 4 | 0 % | 0.20 | 0 |
| 50 kΩ | 12 | 50 % | 0.69 | 7 |

| Confidence band | Answers | Share consistent |
|---|---|---|
| HIGH | 43 | **86 %** |
| MEDIUM | 5 | 40 % |
| LOW | 7 | 14 % |
| NOISE | 13 | 8 % |

**Before and after**, measured with the same yardstick:

| | Old method | New method |
|---|---|---|
| Consistent, 0–14.3 kΩ | 11 % | **81 %** |
| Consistent, 19.3–50 kΩ | 12 % | 38 % |
| Answers labelled HIGH | 100 % | 63 % |
| Healthy files reaching MEDIUM/HIGH | not tested | **0 / 12** |

The old method failed for three reasons. It averaged cumulative counters. Near-zero healthy spreads made its z-scores explode (up to 12 000), and the confidence formula saturated at 12. And its plain left/right window averages flagged every slope of the standing wave as a "step".

**Remaining failure modes**, all visible in `plots/sw_prediction_grid.png`:
- **T0002 at 30–50 kΩ → 1079**, with HIGH confidence. This is the valley artifact: strong enough to survive the guard. It accounts for 3 of the 6 "HIGH but inconsistent" answers.
- **T0003 location 4:** only the hard short gives a clear answer (1048). Every other resistance is NOISE, which is correct behaviour for a weak signal.
- **T0003 location 1:** splits between 1069 (0–14.3 kΩ) and 1033 (30–50 kΩ).

## 4. The location codes do not mean the same thing on every day

Within a day, the fault's change pattern for one location code is nearly identical across resistances (correlation 0.85; 0.0 between different codes). **Across days, the same code does not match** (mean correlation −0.07 to +0.47). Instead, the patterns match a *different* code on the other day, and **the jump positions agree whenever the patterns agree**:

| Pattern match (correlation) | Jump found |
|---|---|
| T0001 code 1 ↔ T0003 code 2 (0.96) | 1065 / 1069 |
| T0001 code 2 ↔ T0003 code 3 (0.97) | 1025 / 1022 |
| T0001 code 3 ↔ T0003 code 4 (0.93) | 1053 / 1048 |
| T0002 code 1 ↔ T0003 code 3 (0.98) | 1023 / 1022 |

So the detector finds the same physical spot from two independent angles (whole-loop pattern and local jump), but the code-to-position mapping seems to change between days. **Please confirm with the rig operators.** If the mapping really is fixed, then day-to-day conditions change the circuit's response, and a per-day baseline (as used here) is mandatory.

Secondary jumps also appear at *other* codes' positions. For example, the T0001 code-2 hard short also shows a strong jump at 1065, which is code 1's spot. One hypothesis is that the fault-injection wiring at each test point stays connected and adds a small discontinuity of its own.

## 5. Gap analysis

**Useful** (from `plots/step3_metric_usefulness.csv`):
- `peakInputLevel` is the backbone. It responds about 100× the healthy noise for a hard short and about 30× at 50 kΩ, and it alone gives 81 % low-resistance consistency.
- `peakCorrLevel_dB` and `peakUncorrLevel_dB` add confirmation and detection strength. They are weaker for location.

**Unreliable for location:**
- `berCount` and `rxCrcFailCount` jump circuit-wide during hard shorts, so they are good *detectors*. But used alone they send about half of a day's answers to the same glitchy fixtures (~1033–1038, ~1051–1055) whatever the fault position.
- `phaseNoise_dB`, `dsRawRate0`/`usRawRate0` and COMM `failures` sit at the noise floor.
- `response_rate`, repeater depths and RATES depths 1–7 never change.

**Data that would lift this toward production:**
1. **Ground-truth fault positions** (cable metres or the fixture pair). This is the biggest gap: today only repeatability can be measured, not accuracy.
2. **Fixture position metadata:** cable distance per fixture, and confirmation that ID order is physical order. That would turn fixture boundaries into metres and allow uneven spacing.
3. **More healthy recordings per circuit and per day.** With only 4 baselines, the false-alarm calibration is coarse.
4. **CCR-side measurements:** loop current, output voltage and insulation resistance (from the EFD itself) for each recording. These would give fault resistance directly, and let the localizer know in advance when to expect weak evidence.
5. **A second carrier frequency or channel.** The standing-wave valleys would sit elsewhere, so a fault hidden in one channel's valley would be visible on the other. This is the most direct fix for the valley blind spot.
6. **Raw received amplitude per fixture** instead of a peak level. A clean amplitude measurement would sharpen the current-step signature.
7. **Recordings with faults inside the valley zones (~1009–1020, ~1070–1083)** and at the ends of the loop (< 1004, > 1088), to measure the blind spots.

## 6. Limitations (stated plainly)

- Validation is **consistency, not accuracy**.
- The **valley guard settings (±5, ×0.1) were chosen after seeing validation results**. They are physically motivated and computed from healthy data only, but they are not independently tested. Fresh data is needed to confirm them.
- Faults **inside a valley zone** are partly masked, and faults **before fixture 1004 or after 1088** cannot be scored (the window needs 3 fixtures on each side).
- **The only label used as an input** is which recordings are healthy (code 0). This stands in for a commissioning reference recording. Fault location and resistance labels are used only for validation.

## 7. Reproduce

```
python analyse_efd.py    # full analysis + all plots/tables in plots/   (a few minutes)
python localize.py       # localizer only -> localization_results.csv  (~1 min)
```

Key outputs: `plots/sw_prediction_grid.png`, `plots/sw_confidence.png`, `plots/step2_delta_peakInputLevel.png`, `plots/step4_cross_run_fingerprints.png`, `plots/sliding_window_report.txt`. The previous method's outputs are archived in `plots/old_method_outputs/`.
