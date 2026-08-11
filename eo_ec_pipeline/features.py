"""
Epoch a branch's .fif into fixed windows, compute per-epoch per-channel alpha-band
(8-13 Hz) log-power via Welch PSD, and write a tidy feature table for classify.py.

The whole recording is one condition (a file living in data/input_ec == EC or
data/input_eo == EO) and the file itself (via --block-id, default the filename
stem) is the group used by StratifiedGroupKFold in classify.py.

  eeg/bin/python3 features.py <fif_path> <branch_name> <out_csv> --condition EC
"""

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd

EPOCH_SEC = 2.0
ALPHA_BAND = (8.0, 13.0)


def _epoch_rows(seg, branch: str, block_idx, condition: str) -> list[dict]:
    # reject_by_annotation is explicit, not defaulted: ingestion/muse_to_fif.py marks every
    # Bluetooth dropout with a BAD_gap annotation precisely so the window straddling the seam
    # is dropped here. A discontinuity is a step artifact with broadband power that lands in
    # the 8-13 Hz band, i.e. straight into the EC-EO numbers this table feeds.
    epochs = mne.make_fixed_length_epochs(seg, duration=EPOCH_SEC, overlap=0.0, preload=True,
                                          reject_by_annotation=True, verbose="ERROR")
    # method="welch" is explicit, not defaulted: Epochs.compute_psd defaults to multitaper,
    # whose time-bandwidth smoothing on 2s windows drags power from outside 8-13 Hz into the
    # alpha estimate. Welch is what this module documents and what band_power_check.py uses
    # (Raw.compute_psd, which does default to Welch), so the two spectral checks stay comparable.
    psd = epochs.compute_psd(method="welch", fmin=ALPHA_BAND[0], fmax=ALPHA_BAND[1], verbose="ERROR")
    alpha_power = psd.get_data().mean(axis=-1)  # (n_epochs, n_channels), mean linear power over alpha bins

    rows = []
    for ep_idx in range(alpha_power.shape[0]):
        for ch_idx, ch_name in enumerate(epochs.ch_names):
            rows.append({
                "branch": branch,
                "block_idx": block_idx,
                "condition": condition,
                "epoch_idx": ep_idx,
                "channel": ch_name,
                "log_alpha_power": float(np.log(alpha_power[ep_idx, ch_idx] + 1e-20)),
            })
    return rows


def extract_features(fif_path: str, branch: str, condition: str, block_id=None) -> pd.DataFrame:
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    block_id = block_id if block_id is not None else Path(fif_path).stem
    return pd.DataFrame(_epoch_rows(raw, branch, block_id, condition))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fif_path")
    parser.add_argument("branch")
    parser.add_argument("out_csv")
    parser.add_argument("--condition", choices=["EC", "EO"], required=True,
                         help="the single condition this whole recording was collected under")
    parser.add_argument("--block-id", default=None, help="group id for this file (default: filename stem)")
    args = parser.parse_args()

    df = extract_features(args.fif_path, args.branch, condition=args.condition, block_id=args.block_id)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {len(df)} rows -> {args.out_csv}")
