"""
Quick real-data sanity check across all five classic EEG bands (delta/theta/alpha/
beta/gamma), comparing eyes-closed vs eyes-open on the raw (unprocessed) Muse
recordings in data/input_ec and data/input_eo -- no ZUNA branches involved.

Takes up to --max-per-condition recordings per condition (default 2, so 4 total),
computes per-channel Welch PSD band power over the full recording, averages
across channels and recordings, and plots a grouped bar chart.

  eeg/bin/python3 eo_ec_pipeline/band_power_check.py
  eeg/bin/python3 eo_ec_pipeline/band_power_check.py --max-per-condition 3
"""

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd

CONDITION_DIRS = {"EC": "data/input_ec", "EO": "data/input_eo"}
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
COLORS = {"EC": "#2a78d6", "EO": "#eb6834"}  # validated categorical slots 1 & 2


def band_power_table(fif_paths_by_condition: dict) -> pd.DataFrame:
    """Relative band power (each band's mean power as a fraction of total 1-45Hz
    power), same convention as ingestion/src/analyze_session.py -- gives every
    band a meaningful, comparable 0-1 baseline instead of raw (very negative,
    arbitrarily-offset) log power."""
    rows = []
    for condition, fif_paths in fif_paths_by_condition.items():
        for fif_path in fif_paths:
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
            psd = raw.compute_psd(fmin=1.0, fmax=45.0, verbose="ERROR")
            freqs = psd.freqs
            power = psd.get_data().mean(axis=0)  # mean over channels, (n_freqs,)

            band_power = {band: power[(freqs >= fmin) & (freqs < fmax)].mean()
                          for band, (fmin, fmax) in BANDS.items()}
            total = sum(band_power.values())
            for band, p in band_power.items():
                rows.append({
                    "condition": condition,
                    "recording": Path(fif_path).stem,
                    "band": band,
                    "relative_power": float(p / total),
                })
    return pd.DataFrame(rows)


def plot_band_power(df: pd.DataFrame, out_png: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    means = df.groupby(["band", "condition"])["relative_power"].mean().unstack("condition")
    means = means.reindex(BANDS.keys())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(means.index))
    width = 0.32

    for i, condition in enumerate(["EC", "EO"]):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means[condition], width, label=f"{condition} (raw)",
                       color=COLORS[condition], edgecolor="none")
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in means[condition]],
                     padding=2, fontsize=8, color="#52514e")

    ax.set_xticks(x)
    ax.set_xticklabels(means.index)
    ax.set_ylabel("relative band power (% of 1-45Hz total)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title("EEG relative band power: eyes-closed vs eyes-open (raw recordings)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#e5e4e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main(max_per_condition: int = 2, out_png: str = "data/figures/band_power_eo_ec.png") -> pd.DataFrame:
    fif_paths_by_condition = {}
    for condition, input_dir in CONDITION_DIRS.items():
        fifs = sorted(Path(input_dir).glob("*.fif"))[:max_per_condition]
        if not fifs:
            raise FileNotFoundError(f"No .fif files found in {input_dir} ({condition})")
        fif_paths_by_condition[condition] = fifs
        print(f"{condition}: using {[f.name for f in fifs]}")

    df = band_power_table(fif_paths_by_condition)

    print()
    print("=== Mean relative band power by condition ===")
    print(df.groupby(["band", "condition"])["relative_power"].mean().unstack("condition").reindex(BANDS.keys()).to_string())

    plot_band_power(df, out_png)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-condition", type=int, default=2)
    parser.add_argument("--out-png", default="data/figures/band_power_eo_ec.png")
    args = parser.parse_args()
    main(args.max_per_condition, args.out_png)
