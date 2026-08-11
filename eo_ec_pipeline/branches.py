"""
Build the three comparison branches for the ZUNA1.1 EO/EC preservation test:
  raw/        the labeled input, untouched
  denoised/   ZUNA1.1 output with the same 4 real channels forced through reconstruction
  upsampled/  ZUNA1.1 output with virtual posterior/occipital channels added

Run with the venv that has zuna installed:
  eeg/bin/python3 branches.py data/labeled data/branches
"""

import shutil
import sys
from pathlib import Path

from zuna import reconstruct_fif

MUSE_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
POSTERIOR_TARGETS = ["O1", "O2", "Oz", "Pz"]  # the channels Muse doesn't have


def build_branches(input_dir: str, out_root: str, gpu_device=0) -> None:
    out_root = Path(out_root)

    # Wipe previous output first: these directories are read back by globbing, so a
    # recording dropped from input_dir since the last run would otherwise survive
    # here and keep being pooled into the results.
    for stale in ("raw", "denoised", "denoised_figures", "upsampled", "upsampled_figures"):
        shutil.rmtree(out_root / stale, ignore_errors=True)

    raw_dir = out_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fif in Path(input_dir).glob("*.fif"):
        shutil.copy(fif, raw_dir / fif.name)

    reconstruct_fif(
        input_dir=input_dir,
        output_dir=str(out_root / "denoised"),
        figures_dir=str(out_root / "denoised_figures"),
        gpu_device=gpu_device,
        repair_channels=MUSE_CHANNELS,   # mask+reconstruct all 4 real channels in full
    )

    reconstruct_fif(
        input_dir=input_dir,
        output_dir=str(out_root / "upsampled"),
        figures_dir=str(out_root / "upsampled_figures"),
        gpu_device=gpu_device,
        target_channel_count=POSTERIOR_TARGETS,   # hallucinate posterior channels only
    )

    print(f"branches written under {out_root}")
    print("  denoised -> read from denoised/full_reconstruction/*.fif (model output on all 4ch)")
    print("  upsampled -> read from upsampled/full_reconstruction/*.fif (all 8ch model output)")


if __name__ == "__main__":
    build_branches(sys.argv[1], sys.argv[2])
