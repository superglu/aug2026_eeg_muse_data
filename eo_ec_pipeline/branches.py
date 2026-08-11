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

import mne
from zuna import reconstruct_fif

MUSE_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
POSTERIOR_TARGETS = ["O1", "O2", "Oz", "Pz"]  # the channels Muse doesn't have


def _propagate_annotations(input_dir, out_dir) -> None:
    """Copy each input recording's annotations onto the branch output built from it.

    features.py epochs with reject_by_annotation=True so that the window straddling a
    Bluetooth dropout (marked BAD_gap by ingestion/muse_to_fif.py) is dropped. The raw
    branch is a file copy and keeps those annotations; the ZUNA branches are new .fif
    files that need not. Without this, gap-straddling epochs would be rejected in raw
    only, feeding step-artifact power into exactly the branches raw is compared against
    and leaving the branches with unequal epoch counts. Outputs that already carry
    annotations are left alone.
    """
    inputs = {p.stem: p for p in Path(input_dir).glob("*.fif")}
    # longest stem first: ZUNA suffixes its outputs, so match the most specific input
    stems = sorted(inputs, key=len, reverse=True)

    for out_fif in sorted(Path(out_dir).rglob("*.fif")):
        stem = next((s for s in stems if out_fif.stem.startswith(s)), None)
        if stem is None:
            continue

        src = mne.io.read_raw_fif(inputs[stem], preload=False, verbose="ERROR")
        if len(src.annotations) == 0:
            continue

        out = mne.io.read_raw_fif(out_fif, preload=True, verbose="ERROR")
        if len(out.annotations) > 0:  # ZUNA carried them through; don't duplicate
            continue

        # Re-express onsets relative to the start of the data, so they land in the right
        # place whether or not the output kept the input's meas_date and first_samp.
        offset = src.first_time if src.annotations.orig_time is not None else 0.0
        out.set_annotations(mne.Annotations(onset=src.annotations.onset - offset,
                                            duration=src.annotations.duration,
                                            description=src.annotations.description),
                            verbose="ERROR")
        out.save(out_fif, overwrite=True, verbose="ERROR")
        print(f"  re-applied {len(src.annotations)} annotation(s) from {inputs[stem].name} -> {out_fif.name}")


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
    _propagate_annotations(input_dir, out_root / "denoised")

    reconstruct_fif(
        input_dir=input_dir,
        output_dir=str(out_root / "upsampled"),
        figures_dir=str(out_root / "upsampled_figures"),
        gpu_device=gpu_device,
        target_channel_count=POSTERIOR_TARGETS,   # hallucinate posterior channels only
    )
    _propagate_annotations(input_dir, out_root / "upsampled")

    print(f"branches written under {out_root}")
    print("  denoised -> read from denoised/full_reconstruction/*.fif (model output on all 4ch)")
    print("  upsampled -> read from upsampled/full_reconstruction/*.fif (all 8ch model output)")


if __name__ == "__main__":
    build_branches(sys.argv[1], sys.argv[2])
