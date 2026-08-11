"""
Quick real-data sanity check across all five classic EEG bands (delta/theta/alpha/
beta/gamma), comparing eyes-closed vs eyes-open on the raw (unprocessed) Muse
recordings in data/input_ec and data/input_eo -- no ZUNA branches involved.

Takes up to --max-per-condition *subjects* that have a recording in both conditions
(default 2, so 4 recordings), computes per-channel Welch PSD band power over the full
recording, averages across channels and recordings, and plots a grouped bar chart.
The selection is paired on purpose: comparing EC and EO over different subjects would
report between-subject alpha differences as an eyes-closed/eyes-open effect.

  eeg/bin/python3 eo_ec_pipeline/band_power_check.py
  eeg/bin/python3 eo_ec_pipeline/band_power_check.py --max-per-condition 3
"""

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from branch_files import CONDITION_INPUT_DIRS, subject_key

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
COLORS = {"EC": "#2a78d6", "EO": "#eb6834"}  # validated categorical slots 1 & 2


def integrated_band_power(power: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    """Power in [fmin, fmax) — the PSD integrated over the band, not its mean.

    Bandwidths here differ by ~5x (delta 3 Hz vs beta 17 Hz), so a mean PSD value
    discards exactly the factor that makes bands comparable: a ratio of per-band
    means over-credits the narrow low-frequency bands and is not a fraction of
    total power at all.
    """
    mask = (freqs >= fmin) & (freqs < fmax)
    return float(np.trapezoid(power[mask], freqs[mask]))


def band_power_table(fif_paths_by_condition: dict) -> pd.DataFrame:
    """Relative band power: each band's integrated power as a fraction of the
    total 1-45Hz power, same convention as ingestion/src/analyze_session.py --
    gives every band a meaningful, comparable 0-1 baseline instead of raw (very
    negative, arbitrarily-offset) log power."""
    rows = []
    for condition, fif_paths in fif_paths_by_condition.items():
        for fif_path in fif_paths:
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
            # reject_by_annotation is explicit, not defaulted: Raw.compute_psd defaults it to
            # False, which would integrate every BAD_gap seam marked by ingestion/muse_to_fif.py
            # straight into these bands. A dropout is a broadband step artifact with most of its
            # energy in delta, so a single one inflates delta and deflates every band's share of
            # the 1-45 Hz total -- and dropouts aren't equally likely in EC vs EO, i.e. it biases
            # the exact comparison this module exists to make. Same convention as features.py.
            psd = raw.compute_psd(fmin=1.0, fmax=45.0, reject_by_annotation=True, verbose="ERROR")
            freqs = psd.freqs
            power = psd.get_data().mean(axis=0)  # mean over channels, (n_freqs,)

            band_power = {band: integrated_band_power(power, freqs, fmin, fmax)
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


def _recordings_by_subject(condition: str, input_dir: str) -> dict:
    """{subject: fif_path} for one condition, one file per subject."""
    fifs = sorted(Path(input_dir).glob("*.fif"))
    if not fifs:
        raise FileNotFoundError(f"No .fif files found in {input_dir} ({condition})")
    by_subject = {}
    for fif in fifs:
        subject = subject_key(fif.name)
        if subject in by_subject:
            raise ValueError(
                f"Two {condition} recordings in {input_dir} key to subject {subject!r}: "
                f"{by_subject[subject].name} and {fif.name} — this comparison is paired, so "
                "each subject must have exactly one file per condition."
            )
        by_subject[subject] = fif
    return by_subject


def select_subject_pairs(max_subjects: int) -> dict:
    """Pick the same subjects for both conditions, so the contrast stays within-subject.

    Choosing each condition's files independently (sorted glob, truncated) makes the
    EC/EO contrast within-subject only by coincidence of filename sorting: one missing
    or differently-named export is enough to compare EC{a,b} against EO{b,c}, and
    between-subject differences in resting alpha dominate this feature space more than
    the EO/EC effect does. So subjects are paired first, then truncated.
    """
    by_condition = {c: _recordings_by_subject(c, d) for c, d in CONDITION_INPUT_DIRS.items()}
    paired = sorted(set.intersection(*(set(s) for s in by_condition.values())))
    if not paired:
        raise FileNotFoundError(
            "No subject has a recording in every condition: "
            + "; ".join(f"{c}={sorted(s)}" for c, s in by_condition.items())
        )
    for condition, subjects in by_condition.items():
        unpaired = sorted(set(subjects) - set(paired))
        if unpaired:
            print(f"WARNING: skipping {condition} recording(s) with no counterpart in the "
                  f"other condition: {unpaired}")

    selected = paired[:max_subjects]
    print(f"subjects: using {selected} ({len(paired)} paired, {len(selected)} kept)")
    return {c: [by_condition[c][s] for s in selected] for c in by_condition}


def main(max_per_condition: int = 2, out_png: str = "data/figures/band_power_eo_ec.png") -> pd.DataFrame:
    fif_paths_by_condition = select_subject_pairs(max_per_condition)
    for condition, fifs in fif_paths_by_condition.items():
        print(f"{condition}: using {[f.name for f in fifs]}")

    df = band_power_table(fif_paths_by_condition)

    print()
    print("=== Mean relative band power by condition ===")
    print(df.groupby(["band", "condition"])["relative_power"].mean().unstack("condition").reindex(BANDS.keys()).to_string())

    plot_band_power(df, out_png)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-condition", type=int, default=2,
                         help="how many subjects (EC/EO pairs) to include, i.e. this many "
                              "recordings per condition")
    parser.add_argument("--out-png", default="data/figures/band_power_eo_ec.png")
    args = parser.parse_args()
    main(args.max_per_condition, args.out_png)
