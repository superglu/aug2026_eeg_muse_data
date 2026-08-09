"""
Convert a raw Muse 2 LSL CSV export (lsl_timestamp, TP9, AF7, AF8, TP10, AUX)
into an MNE .fif file with a standard-1020 montage attached, ready for
zuna.reconstruct_fif.

Without an explicit output path, the CSV filename is checked for "open"/"closed"
and the .fif is routed to data/input_eo/ or data/input_ec/ accordingly, so the
eo_ec_pipeline branch-comparison scripts can pick it straight up.

Usage:
    python muse_to_fif.py data/gary_open_2026-08-09.csv     # -> data/input_eo/gary_open_2026-08-09_raw.fif
    python muse_to_fif.py data/gary_closed_2026-08-09.csv   # -> data/input_ec/gary_closed_2026-08-09_raw.fif
    python muse_to_fif.py <input.csv> <output.fif>          # explicit output path, no routing
"""

import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

MUSE_EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]


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
    if len(sys.argv) > 2:
        out_fif = sys.argv[2]
    else:
        out_fif = str(route_output_dir(csv_path) / f"{Path(csv_path).stem}_raw.fif")
    path = muse_csv_to_fif(csv_path, out_fif)
    print(f"wrote {path}")
