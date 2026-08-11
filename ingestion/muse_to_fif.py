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

The export is also checked against the 60-second block the study design specifies: a
short block, or one that lost too many samples to dropouts, is refused rather than
written as an ordinary .fif that nothing downstream can tell apart from a clean one.

Usage:
    python muse_to_fif.py data/gary_open_2026-08-09.csv     # -> data/input_eo/gary_open_2026-08-09_raw.fif
    python muse_to_fif.py data/gary_closed_2026-08-09.csv   # -> data/input_ec/gary_closed_2026-08-09_raw.fif
    python muse_to_fif.py <input.csv> <output.fif>          # explicit output path, no routing
    python muse_to_fif.py <input.csv> --allow-incomplete    # convert a short/gappy block anyway
"""

import argparse
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

# The design is one 60-second block per condition per subject, and a short or gappy block
# has to fail here rather than become an ordinary .fif: nothing downstream re-checks a
# recording's length, classify.classify pools epochs unweighted (a 20 s block contributes
# a third of the epochs and a third-sized cross-validation group), and spectral_check
# weights every subject equally however little signal it kept. record_eeg.py enforces the
# same two thresholds on the LSL path; a Mind Monitor export enters through here instead,
# so the guard has to exist on both routes.
EXPECTED_BLOCK_SECONDS = 60.0
MIN_BLOCK_FRACTION = 0.9      # of EXPECTED_BLOCK_SECONDS
MIN_SAMPLE_FRACTION = 0.9     # of the samples the wall-clock span should have yielded


def route_output_dir(csv_path: str) -> Path:
    name = Path(csv_path).stem.lower()
    # Both markers is as undecidable as neither, and worse: the directory is the only
    # record of the condition downstream, so guessing would silently invert a subject's
    # EC/EO contrast instead of failing.
    markers = [marker for marker in ("open", "closed") if marker in name]
    if len(markers) == 1:
        return Path("data/input_eo" if markers[0] == "open" else "data/input_ec")
    found = f"found both {markers}" if markers else "found neither"
    raise ValueError(
        f"Can't tell EO/EC condition from filename {csv_path!r} ({found}): "
        "expected exactly one of 'open' or 'closed' somewhere in the name, "
        "or pass an explicit output path."
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


def sfreq_and_gaps(timestamps: np.ndarray, csv_path: str) -> tuple[float, mne.Annotations, float]:
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


def check_block_complete(n_samples: int, sfreq: float, timestamps: np.ndarray, csv_path: str) -> None:
    """Raise unless this export is a usable block, i.e. long enough and not full of holes.

    Two independent ways an export falls short, both invisible in the resulting .fif:
    the session was stopped early (short wall clock), or it ran the full minute but the
    headset dropped a large share of it (samples missing from a full-length span). The
    second is what BAD_gap annotations mark; annotating them keeps epoching honest, but
    epoching alone cannot tell that only 35 s of a 60 s block survived.
    """
    signal_s = n_samples / sfreq
    wall_s = float(timestamps[-1] - timestamps[0])
    kept = signal_s / wall_s if wall_s > 0 else 1.0

    problems = []
    if signal_s < MIN_BLOCK_FRACTION * EXPECTED_BLOCK_SECONDS:
        problems.append(
            f"only {signal_s:.1f} s of signal, against the {EXPECTED_BLOCK_SECONDS:.0f} s "
            f"block the design specifies ({signal_s / EXPECTED_BLOCK_SECONDS:.0%})"
        )
    if kept < MIN_SAMPLE_FRACTION:
        problems.append(
            f"kept {n_samples} samples, {kept:.0%} of the ~{wall_s * sfreq:.0f} that "
            f"{wall_s:.1f} s at {sfreq:.1f} Hz should have yielded (dropouts)"
        )
    if problems:
        raise ValueError(
            f"INCOMPLETE RECORDING {csv_path}: {'; '.join(problems)}. Re-record this block, "
            "or pass --allow-incomplete to convert it anyway and flag it in the writeup."
        )


def muse_csv_to_fif(csv_path: str, out_fif: str, allow_incomplete: bool = False) -> str:
    df = pd.read_csv(csv_path)

    columns = _channel_columns(df)
    timestamps = _timestamps_seconds(df)

    # Mind Monitor interleaves other sensors, leaving the EEG columns empty on those rows.
    keep = df[list(columns.values())].notna().all(axis=1).to_numpy()
    df, timestamps = df[keep], timestamps[keep]
    if len(df) < 2:
        raise ValueError(f"{csv_path} has fewer than 2 EEG samples")

    sfreq, annotations, lost_s = sfreq_and_gaps(timestamps, csv_path)
    if len(annotations):
        warnings.warn(
            f"{csv_path}: {len(annotations)} recording gap(s) totalling {lost_s:.1f} s "
            f"({100 * lost_s / (timestamps[-1] - timestamps[0]):.1f}% of wall clock) — "
            "marked BAD_gap; the .fif time axis is that much shorter than the session."
        )
    if allow_incomplete:
        warnings.warn(f"{csv_path}: completeness check skipped (--allow-incomplete); "
                      "flag this block as short/gappy wherever its results are reported.")
    else:
        check_block_complete(len(df), sfreq, timestamps, csv_path)

    # Muse CSV values are in microvolts; MNE expects volts.
    data_v = df[[columns[ch] for ch in MUSE_EEG_CHANNELS]].to_numpy(dtype=float).T * 1e-6

    Path(out_fif).parent.mkdir(parents=True, exist_ok=True)
    return numpy_to_fif(data_v, MUSE_EEG_CHANNELS, sfreq, out_fif, annotations=annotations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Muse CSV export (record_eeg.py or Mind Monitor)")
    parser.add_argument("out_fif", nargs="?", default=None,
                        help="explicit output path; omit to route on 'open'/'closed' in the name")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help=f"convert even if the block is shorter than "
                             f"{MIN_BLOCK_FRACTION:.0%} of {EXPECTED_BLOCK_SECONDS:.0f} s or lost "
                             f"more than {1 - MIN_SAMPLE_FRACTION:.0%} of its samples to dropouts")
    args = parser.parse_args()

    out_fif = args.out_fif or str(route_output_dir(args.csv) / f"{Path(args.csv).stem}_raw.fif")
    path = muse_csv_to_fif(args.csv, out_fif, allow_incomplete=args.allow_incomplete)
    print(f"wrote {path}")
