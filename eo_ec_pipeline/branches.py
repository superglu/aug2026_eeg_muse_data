"""
Build the three comparison branches for the ZUNA1.1 EO/EC preservation test:
  raw/        the labeled input, untouched
  denoised/   ZUNA1.1 output with the same 4 real channels forced through reconstruction
  upsampled/  ZUNA1.1 output with virtual posterior/occipital channels added

Run with the venv that has zuna installed:
  eeg/bin/python3 branches.py data/labeled data/branches
"""

import os
import shutil
import sys
from pathlib import Path

import mne
import numpy as np
from zuna import reconstruct_fif

MUSE_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
POSTERIOR_TARGETS = ["O1", "O2", "Oz", "Pz"]  # the channels Muse doesn't have

GAP_DESCRIPTION = "BAD_gap"       # written by ingestion/muse_to_fif.py
ONSET_TOLERANCE_S = 1e-3          # two gap onsets this close are the same gap


def _relative_annotations(raw) -> mne.Annotations:
    """A recording's annotations with onsets relative to the start of its data.

    Onsets are stored relative to meas_date when orig_time is set, so they have to be
    rebased before they can be compared across, or copied between, two files that need
    not share a meas_date or first_samp.
    """
    annotations = raw.annotations
    offset = raw.first_time if annotations.orig_time is not None else 0.0
    return mne.Annotations(onset=annotations.onset - offset,
                           duration=annotations.duration,
                           description=annotations.description)


def _gap_onsets(raw) -> tuple[np.ndarray, np.ndarray]:
    """Onsets (relative to the data start) and durations of a recording's BAD_gap marks."""
    annotations = _relative_annotations(raw)
    is_gap = np.asarray(annotations.description) == GAP_DESCRIPTION
    return annotations.onset[is_gap], annotations.duration[is_gap]


def _propagate_annotations(input_dir, out_dir) -> None:
    """Copy each input recording's BAD_gap annotations onto the branch output built from it.

    features.py epochs with reject_by_annotation=True so that the window straddling a
    Bluetooth dropout (marked BAD_gap by ingestion/muse_to_fif.py) is dropped. The raw
    branch is a file copy and keeps those annotations; the ZUNA branches are new .fif
    files that need not. Without this, gap-straddling epochs would be rejected in raw
    only, feeding step-artifact power into exactly the branches raw is compared against
    and leaving the branches with unequal epoch counts. Outputs that already carry every
    input BAD_gap are left alone; an output that carries annotations of its own (ZUNA may
    mark the spans it reconstructed) still gets the missing gaps added, since the presence
    of *some* annotation says nothing about whether the gaps survived.
    """
    inputs = {p.stem: p for p in Path(input_dir).glob("*.fif")}
    # longest stem first: ZUNA suffixes its outputs, so match the most specific input
    stems = sorted(inputs, key=len, reverse=True)

    for out_fif in sorted(Path(out_dir).rglob("*.fif")):
        stem = next((s for s in stems if out_fif.stem.startswith(s)), None)
        if stem is None:
            continue

        src = mne.io.read_raw_fif(inputs[stem], preload=False, verbose="ERROR")
        src_onsets, src_durations = _gap_onsets(src)
        if len(src_onsets) == 0:
            continue

        out = mne.io.read_raw_fif(out_fif, preload=True, verbose="ERROR")
        out_onsets, _ = _gap_onsets(out)
        # Only the gaps ZUNA didn't carry through: matching on the BAD_gap onsets rather
        # than on "the output has some annotation" keeps ZUNA's own annotations (e.g. of
        # the spans it reconstructed) from being mistaken for the propagated gaps.
        missing = [i for i, onset in enumerate(src_onsets)
                   if len(out_onsets) == 0 or np.min(np.abs(out_onsets - onset)) > ONSET_TOLERANCE_S]
        if not missing:
            continue

        kept = _relative_annotations(out).append(
            onset=src_onsets[missing],
            duration=src_durations[missing],
            description=[GAP_DESCRIPTION] * len(missing),
        )
        out.set_annotations(kept, verbose="ERROR")
        # Write beside the original and rename into place: MNE refuses to save a Raw onto
        # the file it was read from, and an in-place write that failed partway would leave
        # a truncated .fif that branch_fifs() would still hand to features.py by name.
        tmp_fif = out_fif.with_name(f".tmp_{out_fif.stem}_raw.fif")
        try:
            out.save(tmp_fif, overwrite=True, verbose="ERROR")
            os.replace(tmp_fif, out_fif)
        finally:
            tmp_fif.unlink(missing_ok=True)
        print(f"  re-applied {len(missing)} {GAP_DESCRIPTION} annotation(s) "
              f"from {inputs[stem].name} -> {out_fif.name}")


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
