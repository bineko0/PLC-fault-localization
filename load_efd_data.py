"""
Loader for EFD T0001–T0003 CSV datasets.

Filename format: <epoch>-<commCh>-<powerStep>-<faultLocation>-<faultResistance>.csv
Each CSV has no header; rows are interleaved COMM / RATES / EXCOMM / CORR messages
identified by column index 1 (0-based).
"""

import os
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd

# Data root: use current script directory (allows portability across systems)
DATA_ROOT = Path(__file__).parent

SCHEMA = {
    "COMM": [
        "timestamp", "type", "fixture_id",
        "response_rate", "attempts", "failures",
        "ds_avg_depth", "us_avg_depth",
        "ds_current_depth", "us_current_depth",
        "avg_signal_strength", "incorrect_count",
    ],
    "RATES": (
        ["timestamp", "type", "fixture_id"]
        + [f"dsRawRate{i}" for i in range(8)]
        + [f"usRawRate{i}" for i in range(8)]
    ),
    "EXCOMM": [
        "timestamp", "type", "fixture_id",
        "peakInputLevel", "overflowStatus", "berCount",
        "phaseNoise_dB", "rxCrcFailCount",
    ],
    "CORR": [
        "timestamp", "type", "fixture_id",
        "peakCorrLevel_dB", "peakUncorrLevel_dB",
        "ftryCorrTrigCount", "userCorrTrigCount",
        "corrFired", "ftryCorrFired",
    ],
}

EXPECTED_COLS = {k: len(v) for k, v in SCHEMA.items()}

_FNAME_RE = re.compile(
    r"^(?P<epoch>\d+)-(?P<commCh>[^-]+)-(?P<powerStep>[^-]+)"
    r"-(?P<faultLocation>[^-]+)-(?P<faultResistance>[^-]+)\.csv$"
)


def _parse_filename(fname: str) -> dict:
    m = _FNAME_RE.match(fname)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {fname!r}")
    return {
        "faultLocation": m.group("faultLocation"),
        "faultResistance": m.group("faultResistance"),
    }


def _load_file(csv_path: Path, run: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Return (frames_dict, warnings).  frames_dict keys are COMM/RATES/EXCOMM/CORR."""
    meta = _parse_filename(csv_path.name)
    fault_loc = meta["faultLocation"]
    fault_res = meta["faultResistance"]

    buckets: dict[str, list[list]] = {k: [] for k in SCHEMA}
    warnings: list[str] = []

    with csv_path.open(newline="", encoding="utf-8") as fh:
        import csv as _csv
        reader = _csv.reader(fh, quoting=_csv.QUOTE_NONE)
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            msg_type = row[1].strip() if len(row) > 1 else ""
            if msg_type not in SCHEMA:
                warnings.append(
                    f"{csv_path.name} row {row_idx}: unknown message type {msg_type!r}"
                )
                continue

            values = [v.strip() for v in row]
            expected = EXPECTED_COLS[msg_type]
            if len(values) != expected:
                warnings.append(
                    f"{csv_path.name} row {row_idx}: {msg_type} expected {expected} cols,"
                    f" got {len(values)}"
                )
                # Pad or trim so we can still build a DataFrame
                if len(values) < expected:
                    values += [None] * (expected - len(values))
                else:
                    values = values[:expected]

            buckets[msg_type].append(values)

    frames: dict[str, pd.DataFrame] = {}
    for msg_type, rows in buckets.items():
        cols = SCHEMA[msg_type]
        df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
        df["run"] = run
        df["faultLocation"] = fault_loc
        df["faultResistance"] = fault_res
        frames[msg_type] = df

    return frames, warnings


def load_all(data_root: Path = DATA_ROOT) -> dict:
    """
    Returns data[run][filename] = {COMM: df, RATES: df, EXCOMM: df, CORR: df}
    and prints a loading summary.
    """
    data: dict[str, dict[str, dict[str, pd.DataFrame]]] = defaultdict(dict)
    all_warnings: list[str] = []

    run_dirs = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.startswith("EFD_T")
    )

    for run_dir in run_dirs:
        run = run_dir.name.replace("EFD_", "")  # e.g. "T0001"
        csv_files = sorted(run_dir.glob("*.csv"))
        for csv_path in csv_files:
            frames, warnings = _load_file(csv_path, run)
            data[run][csv_path.name] = frames
            all_warnings.extend(warnings)

    _print_summary(data, all_warnings)
    return dict(data)


def _print_summary(
    data: dict[str, dict[str, dict[str, pd.DataFrame]]],
    warnings: list[str],
) -> None:
    print("=" * 60)
    print("EFD DATA LOAD SUMMARY")
    print("=" * 60)

    totals: dict[str, int] = defaultdict(int)

    for run, files in sorted(data.items()):
        run_counts: dict[str, int] = defaultdict(int)
        for fname, frames in files.items():
            for msg_type, df in frames.items():
                run_counts[msg_type] += len(df)
                totals[msg_type] += len(df)

        print(f"\nRun {run}: {len(files)} files")
        for msg_type in ("COMM", "RATES", "EXCOMM", "CORR"):
            print(f"  {msg_type:8s}: {run_counts[msg_type]:>7,} rows")

    print("\nTotals across all runs:")
    for msg_type in ("COMM", "RATES", "EXCOMM", "CORR"):
        print(f"  {msg_type:8s}: {totals[msg_type]:>7,} rows")

    print(f"\nWarnings (unexpected column counts / unknown types): {len(warnings)}")
    for w in warnings[:20]:
        print(f"  [WARN] {w}")
    if len(warnings) > 20:
        print(f"  ... and {len(warnings) - 20} more")

    print("=" * 60)


if __name__ == "__main__":
    data = load_all()
