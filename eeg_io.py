"""Helpers for moving EEG between numpy arrays and the .fif files ZUNA1.1 consumes."""

import mne


def numpy_to_fif(data, ch_names, sfreq, out_path, montage="standard_1020"):
    """Save a (n_channels, n_samples) float array (volts) as a .fif for fif_in/.

    Electrode positions matter — ZUNA1.1 predicts from scalp coordinates, so
    ch_names must exist in the montage (default: standard 10-20).
    """
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    raw.set_montage(montage)
    raw.save(out_path, overwrite=True)
    return out_path


def load_cleaned(fif_path):
    """Load a cleaned .fif written by clean_eeg.py (fif_out/...)."""
    return mne.io.read_raw_fif(fif_path, preload=True)
