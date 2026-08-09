"""
Convert a raw Muse 2 LSL CSV export into a labeled .fif: attaches a standard-1020
montage and annotates fixed eyes-open (EO) / eyes-closed (EC) blocks, so every
downstream branch (raw / zuna-denoise / zuna-upsample) epochs consistently.

Edit BLOCKS below to match your recording log (block boundaries in seconds,
relative to the start of the recording).
"""

import sys
from pathlib import Path

import mne
import pandas as pd

MUSE_EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

# Example: 6 alternating 60s blocks starting with eyes-closed. Replace with your
# actual protocol timing before running.
BLOCKS = [
    (0, 60, "EC"), (60, 120, "EO"),
    (120, 180, "EC"), (180, 240, "EO"),
    (240, 300, "EC"), (300, 360, "EO"),
]


def convert_and_label(csv_path: str, out_fif: str, blocks=BLOCKS) -> str:
    df = pd.read_csv(csv_path)
    sfreq = 1.0 / df["lsl_timestamp"].diff().median()
    data_v = df[MUSE_EEG_CHANNELS].to_numpy().T * 1e-6  # muV -> V

    info = mne.create_info(ch_names=MUSE_EEG_CHANNELS, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data_v, info, verbose="ERROR")
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), verbose="ERROR")

    duration_s = raw.n_times / raw.info["sfreq"]
    blocks = [b for b in blocks if b[0] < duration_s]
    if not blocks:
        raise ValueError(f"No blocks fit inside a {duration_s:.1f}s recording; edit BLOCKS.")

    onsets = [start for start, _, _ in blocks]
    durations = [min(stop, duration_s) - start for start, stop, _ in blocks]
    labels = [label for _, _, label in blocks]
    raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=labels))

    Path(out_fif).parent.mkdir(parents=True, exist_ok=True)
    raw.save(out_fif, overwrite=True, verbose="ERROR")
    return out_fif


if __name__ == "__main__":
    csv_path = sys.argv[1]
    out_fif = sys.argv[2]
    convert_and_label(csv_path, out_fif)
    print(f"wrote {out_fif}")
