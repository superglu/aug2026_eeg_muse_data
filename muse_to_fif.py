"""
Convert a raw Muse 2 LSL CSV export (lsl_timestamp, TP9, AF7, AF8, TP10, AUX)
into an MNE .fif file with a standard-1020 montage attached, ready for
zuna.reconstruct_fif.

Usage:
    python muse_to_fif.py data/gary_2026-08-09_13-06-56.csv
    python muse_to_fif.py <input.csv> <output.fif>
"""

import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

MUSE_EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]


def muse_csv_to_fif(csv_path: str, out_fif: str) -> str:
    df = pd.read_csv(csv_path)

    missing = [c for c in MUSE_EEG_CHANNELS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected Muse channels: {missing}")

    sfreq = 1.0 / df["lsl_timestamp"].diff().median()

    # Muse CSV values are in microvolts; MNE expects volts.
    data_uv = df[MUSE_EEG_CHANNELS].to_numpy().T  # (n_channels, n_samples)
    data_v = data_uv * 1e-6

    info = mne.create_info(ch_names=MUSE_EEG_CHANNELS, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data_v, info, verbose="ERROR")
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), verbose="ERROR")

    Path(out_fif).parent.mkdir(parents=True, exist_ok=True)
    raw.save(out_fif, overwrite=True, verbose="ERROR")
    return out_fif


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python muse_to_fif.py <input.csv> [output.fif]")
    csv_path = sys.argv[1]
    default_out = Path("fif_in") / f"{Path(csv_path).stem}_raw.fif"
    out_fif = sys.argv[2] if len(sys.argv) > 2 else str(default_out)
    path = muse_csv_to_fif(csv_path, out_fif)
    print(f"wrote {path}")
