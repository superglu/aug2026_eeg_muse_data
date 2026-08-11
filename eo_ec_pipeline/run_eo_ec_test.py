"""
End-to-end EO/EC preservation test across two per-condition folders of already-labeled
recordings (whole file = one condition, no in-file block annotations needed):

  data/input_ec/*.fif   eyes-closed (EC) recordings
  data/input_eo/*.fif   eyes-open (EO) recordings

For each condition, builds the raw/denoised/upsampled branches via branches.build_branches,
then extracts alpha-power features from the full_reconstruction .fif of each of the
three branches (hybrid is skipped for both -- identical to full_reconstruction for
denoised, and full_reconstruction is used instead of hybrid for upsampled here for a
consistent "model regenerates everything" comparison across both). One group per
recording, so StratifiedGroupKFold in classify.py never splits epochs from the same
file across folds.

  eeg/bin/python3 eo_ec_pipeline/run_eo_ec_test.py
  eeg/bin/python3 eo_ec_pipeline/run_eo_ec_test.py --gpu-device ""   # CPU
"""

import argparse
from pathlib import Path

import pandas as pd

from branch_files import BRANCH_SUBDIR, CONDITION_INPUT_DIRS, branch_fifs
from branches import build_branches
from classify import classify, per_subject_spectral_check, spectral_check
from features import extract_features


def run_eo_ec_test(branch_root: str = "data/branches", gpu_device=0, skip_build: bool = False) -> pd.DataFrame:
    branch_root = Path(branch_root)

    if not skip_build:
        for condition, input_dir in CONDITION_INPUT_DIRS.items():
            fifs = list(Path(input_dir).glob("*.fif"))
            if not fifs:
                raise FileNotFoundError(f"No .fif files found in {input_dir} ({condition})")
            build_branches(input_dir, branch_root / condition, gpu_device=gpu_device)

    dfs = []
    for condition in CONDITION_INPUT_DIRS:
        for branch, subdir in BRANCH_SUBDIR.items():
            branch_dir = branch_root / condition / subdir
            for fif in branch_fifs(branch_dir, condition):
                block_id = f"{condition}_{fif.stem}"
                dfs.append(extract_features(str(fif), branch, condition=condition, block_id=block_id))

    df = pd.concat(dfs, ignore_index=True)

    out_csv = branch_root / "features.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {len(df)} feature rows -> {out_csv}")
    print()
    print("=== Spectral sanity check: EC - EO log-alpha-power (dB) per subject per channel ===")
    print(per_subject_spectral_check(df).to_string(index=False))
    print()
    print("=== Spectral sanity check: EC - EO log-alpha-power (dB) pooled over subjects ===")
    print(spectral_check(df).to_string(index=False))
    print()
    print("=== Classification accuracy per branch (StratifiedGroupKFold by recording) ===")
    print(classify(df).to_string(index=False))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-root", default="data/branches")
    parser.add_argument("--gpu-device", default=0)
    parser.add_argument("--skip-build", action="store_true",
                         help="reuse existing data/branches/*/{raw,denoised,upsampled} output instead of re-running ZUNA")
    args = parser.parse_args()
    run_eo_ec_test(args.branch_root, gpu_device=args.gpu_device, skip_build=args.skip_build)
