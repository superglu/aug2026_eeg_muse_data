"""
Per-channel raw-signal figure for each recording in data/input_ec and data/input_eo,
matching the visual style (layout, demeaned uV, per-channel stacked rows, figure size)
of the denoised/upsampled overlay figures ZUNA writes via plot_reconstruction_overlay --
so raw/denoised/upsampled can be visually compared side by side for the same recording.

Unlike the denoised/upsampled figures, there's no "reconstruction" to overlay (raw is
untouched), so this draws a single trace per channel and no inferred-region shading.

  eeg/bin/python3 eo_ec_pipeline/plot_raw_figures.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

CONDITION_DIRS = {"EC": "data/input_ec", "EO": "data/input_eo"}


def plot_raw_figure(fif_path, out_path, title, demean: bool = True) -> Path:
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, exclude=[])

    ch_names = raw.ch_names
    sfreq = raw.info["sfreq"]
    data = raw.get_data() * 1e6  # uV, (n_channels, n_times)
    t = np.arange(data.shape[1]) / sfreq
    n = len(ch_names)

    fig, axes = plt.subplots(n, 1, figsize=(16, max(2, 1.4 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]
        trace = data[i].copy()
        if demean:
            trace -= trace.mean()
        ax.plot(t, trace, color="#1f77b4", lw=0.7, alpha=0.9, label="input")
        ax.set_ylabel(ch_names[i], rotation=0, ha="right", va="center", fontsize=9)
        ax.set_yticks([])
        ax.margins(x=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(f"{title}  —  raw", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(branch_root: str = "data/branches") -> None:
    branch_root = Path(branch_root)
    for condition, input_dir in CONDITION_DIRS.items():
        fifs = sorted(Path(input_dir).glob("*.fif"))
        if not fifs:
            print(f"no files under {input_dir}; skipping {condition}")
            continue
        out_dir = branch_root / condition / "raw_figures"
        for fif in fifs:
            base = fif.stem.replace("_raw", "")
            out_path = plot_raw_figure(fif, out_dir / f"{base}__raw.png", title=base)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-root", default="data/branches")
    args = parser.parse_args()
    main(args.branch_root)
