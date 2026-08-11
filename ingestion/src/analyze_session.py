"""Load a recorded EEG session into MNE and show a first-pass analysis.

Takes a CSV written by src/record_eeg.py, builds an MNE Raw object with
proper channel locations, reports relative band power from the unfiltered
spectrum, and opens two windows: the power spectrum and the bandpass-filtered
raw traces.

The sampling rate and the Bluetooth dropouts come from the CSV's timestamp
column via ingestion/muse_to_fif.py, so a gappy session is not silently
analysed as a continuous one.

Usage:
    python src/analyze_session.py data/eeg_2026-08-09_13-00-00.csv
    python src/analyze_session.py data/rest.csv --no-plot   # just print info
"""

import argparse
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muse_to_fif import sfreq_and_gaps  # noqa: E402 - ingestion/ is not an installed package

CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]


def load_raw(csv_path: str) -> tuple[mne.io.Raw, float]:
    """Build an MNE Raw object from a record_eeg.py CSV, plus the seconds lost to dropouts."""
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if data.ndim != 2 or data.shape[1] < 1 + len(CHANNEL_NAMES):
        raise SystemExit(f"{csv_path} does not look like a record_eeg.py CSV")

    # Columns: lsl_timestamp, TP9, AF7, AF8, TP10, AUX. MNE expects volts.
    eeg = data[:, 1 : 1 + len(CHANNEL_NAMES)].T * 1e-6

    # The rate is derived from the timestamps, never assumed to be 256 Hz: RawArray
    # assumes uniform sampling, so a Bluetooth dropout would otherwise become a step
    # discontinuity in an over-long time axis. The step is broadband, and it lands in
    # the relative band powers below (mostly inflating delta). Same treatment as
    # muse_to_fif.py, so the CSV and the .fif made from it agree.
    sfreq, annotations, lost_s = sfreq_and_gaps(data[:, 0], csv_path)

    info = mne.create_info(CHANNEL_NAMES, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg, info)
    raw.set_annotations(annotations)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), match_case=False)
    return raw, lost_s


def main() -> None:
    parser = argparse.ArgumentParser(description="First-pass MNE analysis of a recorded session")
    parser.add_argument("csv", help="CSV file written by src/record_eeg.py")
    parser.add_argument("--no-plot", action="store_true", help="Print info only, no windows")
    args = parser.parse_args()

    raw, lost_s = load_raw(args.csv)
    print(raw.info)
    print(f"Duration: {raw.times[-1]:.1f} s of signal at {raw.info['sfreq']:.1f} Hz")
    if lost_s:
        print(f"WARNING: {len(raw.annotations)} recording gap(s) totalling {lost_s:.1f} s "
              f"were dropped by the headset — the session ran "
              f"{raw.times[-1] + lost_s:.1f} s on the wall clock. Flag this block.")

    # reject_by_annotation is explicit, not defaulted: the BAD_gap seams are step
    # discontinuities, and their broadband power would skew every band below.
    #
    # The PSD is computed on the *unfiltered* data, before the bandpass below, because
    # Raw.filter has no annotation awareness: it convolves straight across a seam. Each
    # BAD_gap spans a single sample (muse_to_fif.sfreq_and_gaps), while an l_freq=1.0 FIR
    # runs ~845 taps at 256 Hz, so filtering first would smear the step's energy as
    # ringing over ~+/-1.6 s while reject_by_annotation still dropped only the ~4 ms
    # annotated -- leaving essentially all of the artifact in the bands (mostly delta),
    # which is exactly what rejecting the seams is meant to prevent. fmin=1.0 does the
    # highpass's job here anyway: the drift below 1 Hz is outside every band reported.
    # Same convention as eo_ec_pipeline/band_power_check.py, which also never filters.
    spectrum = raw.compute_psd(fmin=1.0, fmax=60, reject_by_annotation=True)
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
        # Filtering is for the trace window only, so the numbers above are never taken
        # from data whose gap seams have been smeared across their annotations.
        raw.filter(l_freq=1.0, h_freq=50.0)
        spectrum.plot()
        raw.plot(scalings=dict(eeg=100e-6), block=True)


if __name__ == "__main__":
    main()
