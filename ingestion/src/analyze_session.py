"""Load a recorded EEG session into MNE and show a first-pass analysis.

Takes a CSV written by src/record_eeg.py, builds an MNE Raw object with
proper channel locations, bandpass filters it, and opens two windows:
the power spectrum and the filtered raw traces.

Usage:
    python src/analyze_session.py data/eeg_2026-08-09_13-00-00.csv
    python src/analyze_session.py data/rest.csv --no-plot   # just print info
"""

import argparse

import mne
import numpy as np

CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]
SAMPLING_RATE = 256.0


def load_raw(csv_path: str) -> mne.io.Raw:
    """Build an MNE Raw object from a record_eeg.py CSV."""
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if data.ndim != 2 or data.shape[1] < 1 + len(CHANNEL_NAMES):
        raise SystemExit(f"{csv_path} does not look like a record_eeg.py CSV")

    # Columns: lsl_timestamp, TP9, AF7, AF8, TP10, AUX. MNE expects volts.
    eeg = data[:, 1 : 1 + len(CHANNEL_NAMES)].T * 1e-6

    info = mne.create_info(CHANNEL_NAMES, SAMPLING_RATE, ch_types="eeg")
    raw = mne.io.RawArray(eeg, info)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), match_case=False)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="First-pass MNE analysis of a recorded session")
    parser.add_argument("csv", help="CSV file written by src/record_eeg.py")
    parser.add_argument("--no-plot", action="store_true", help="Print info only, no windows")
    args = parser.parse_args()

    raw = load_raw(args.csv)
    print(raw.info)
    print(f"Duration: {raw.times[-1]:.1f} s")

    raw.filter(l_freq=1.0, h_freq=50.0)

    spectrum = raw.compute_psd(fmax=60)
    band_powers = {}
    psds, freqs = spectrum.get_data(return_freqs=True)
    for band, (lo, hi) in {
        "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
        "beta": (13, 30), "gamma": (30, 50),
    }.items():
        # Integrate over the band (not mean it): bandwidths differ by ~6x here, so a
        # ratio of per-band means is not a fraction of total power.
        mask = (freqs >= lo) & (freqs < hi)
        band_powers[band] = float(np.trapezoid(psds[:, mask].mean(axis=0), freqs[mask]))
    total = sum(band_powers.values())
    print("\nRelative band power (all channels):")
    for band, power in band_powers.items():
        print(f"  {band:6s} {power / total:6.1%}")

    if not args.no_plot:
        spectrum.plot()
        raw.plot(scalings=dict(eeg=100e-6), block=True)


if __name__ == "__main__":
    main()
