"""
Convert a raw Muse 2 CSV export into an MNE .fif file with a standard-1020
montage attached, ready for zuna.reconstruct_fif.

Two export layouts are accepted:
  - the LSL CSV written by ingestion/src/record_eeg.py
    (lsl_timestamp, TP9, AF7, AF8, TP10[, AUX])
  - a Mind Monitor CSV export
    (TimeStamp, RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10, ... other sensors)

Recording gaps (Bluetooth dropouts) are detected from the timestamp column and
marked with BAD_gap annotations, so downstream fixed-length epoching drops the
windows that straddle a discontinuity instead of treating them as continuous
signal.

Without an explicit output path, the CSV filename is checked for "open"/"closed"
and the .fif is routed to data/input_eo/ or data/input_ec/ accordingly, so the
eo_ec_pipeline branch-comparison scripts can pick it straight up.

Usage:
    python muse_to_fif.py data/gary_open_2026-08-09.csv     # -> data/input_eo/gary_open_2026-08-09_raw.fif
    python muse_to_fif.py data/gary_closed_2026-08-09.csv   # -> data/input_ec/gary_closed_2026-08-09_raw.fif
    python muse_to_fif.py <input.csv> <output.fif>          # explicit output path, no routing
"""

import sys
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from eeg_io import MUSE_EEG_CHANNELS, numpy_to_fif

# Timestamp columns, most specific first: LSL float seconds, Mind Monitor datetime.
LSL_TIME_COLUMN = "lsl_timestamp"
MIND_MONITOR_TIME_COLUMN = "TimeStamp"

# A sample interval this many times the median is treated as a dropout, not jitter.
GAP_FACTOR = 5.0


def route_output_dir(csv_path: str) -> Path:
    name = Path(csv_path).stem.lower()
    if "open" in name:
        return Path("data/input_eo")
    if "closed" in name:
        return Path("data/input_ec")
    raise ValueError(
        f"Can't tell EO/EC condition from filename {csv_path!r}: "
        "expected 'open' or 'closed' somewhere in the name, or pass an explicit output path."
    )


def _channel_columns(df: pd.DataFrame) -> dict:
    """Map each Muse channel to its column name in this export layout."""
    for prefix in ("", "RAW_"):
        columns = {ch: f"{prefix}{ch}" for ch in MUSE_EEG_CHANNELS}
        if all(c in df.columns for c in columns.values()):
            return columns
    raise ValueError(
        f"CSV is missing expected Muse channels {MUSE_EEG_CHANNELS} "
        "(as written by record_eeg.py) and their RAW_* form (as written by a Mind Monitor export)"
    )


def _timestamps_seconds(df: pd.DataFrame) -> np.ndarray:
    """Timestamp column as float seconds, whichever export layout wrote it."""
    if LSL_TIME_COLUMN in df.columns:
        return df[LSL_TIME_COLUMN].to_numpy(dtype=float)
    if MIND_MONITOR_TIME_COLUMN in df.columns:
        stamps = pd.to_datetime(df[MIND_MONITOR_TIME_COLUMN])
        return stamps.astype("int64").to_numpy() / 1e9
    raise ValueError(
        f"CSV has no timestamp column: expected {LSL_TIME_COLUMN!r} (record_eeg.py) "
        f"or {MIND_MONITOR_TIME_COLUMN!r} (Mind Monitor export)"
    )


def _sfreq_and_gaps(timestamps: np.ndarray, csv_path: str) -> tuple[float, mne.Annotations, float]:
    """Sampling rate, plus a BAD_gap annotation per dropout.

    RawArray assumes uniform sampling, so a dropout silently compresses the time
    axis. Annotating the seam keeps the compression visible and makes MNE's
    epoching reject the window that spans it.

    The rate comes from the mean interval between consecutive samples with the
    dropouts excluded: the median alone is robust to dropouts but biased by
    timestamp quantization (a Mind Monitor export is rounded to milliseconds, and
    at 256 Hz that median lands on 4 ms == 250 Hz).
    """
    diffs = np.diff(timestamps)
    median = float(np.median(diffs))
    if not median > 0:
        raise ValueError(
            f"Median sample interval in {csv_path} is {median}: timestamps are duplicated "
            "or non-monotonic, so the sampling rate cannot be inferred."
        )

    is_gap = diffs > GAP_FACTOR * median
    sfreq = 1.0 / float(diffs[~is_gap].mean())

    gap_idx = np.flatnonzero(is_gap)
    onsets = gap_idx / sfreq          # seam sits between sample i and i+1
    durations = np.full(len(gap_idx), 1.0 / sfreq)
    lost_s = float(diffs[gap_idx].sum() - len(gap_idx) / sfreq) if len(gap_idx) else 0.0
    annotations = mne.Annotations(onset=onsets, duration=durations,
                                  description=["BAD_gap"] * len(gap_idx))
    return sfreq, annotations, lost_s


def muse_csv_to_fif(csv_path: str, out_fif: str) -> str:
    df = pd.read_csv(csv_path)

    columns = _channel_columns(df)
    timestamps = _timestamps_seconds(df)

    # Mind Monitor interleaves other sensors, leaving the EEG columns empty on those rows.
    keep = df[list(columns.values())].notna().all(axis=1).to_numpy()
    df, timestamps = df[keep], timestamps[keep]
    if len(df) < 2:
        raise ValueError(f"{csv_path} has fewer than 2 EEG samples")

    sfreq, annotations, lost_s = _sfreq_and_gaps(timestamps, csv_path)
    if len(annotations):
        warnings.warn(
            f"{csv_path}: {len(annotations)} recording gap(s) totalling {lost_s:.1f} s "
            f"({100 * lost_s / (timestamps[-1] - timestamps[0]):.1f}% of wall clock) — "
            "marked BAD_gap; the .fif time axis is that much shorter than the session."
        )

    # Muse CSV values are in microvolts; MNE expects volts.
    data_v = df[[columns[ch] for ch in MUSE_EEG_CHANNELS]].to_numpy(dtype=float).T * 1e-6

    Path(out_fif).parent.mkdir(parents=True, exist_ok=True)
    return numpy_to_fif(data_v, MUSE_EEG_CHANNELS, sfreq, out_fif, annotations=annotations)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python muse_to_fif.py <input.csv> [output.fif]")
    csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        out_fif = sys.argv[2]
    else:
        out_fif = str(route_output_dir(csv_path) / f"{Path(csv_path).stem}_raw.fif")
    path = muse_csv_to_fif(csv_path, out_fif)
    print(f"wrote {path}")
