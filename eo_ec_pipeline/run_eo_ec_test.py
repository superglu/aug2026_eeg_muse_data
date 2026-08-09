"""
End-to-end EO/EC preservation test across two per-condition folders of already-labeled
recordings (whole file = one condition, no in-file block annotations needed):

  data/input_ec/*.fif   eyes-closed (EC) recordings
  data/input_eo/*.fif   eyes-open (EO) recordings

For each condition, builds the three ZUNA1.1 comparison branches (raw / denoised /
upsampled) via branches.build_branches, extracts alpha-power features per branch
(one group per recording, so GroupKFold in classify.py never splits epochs from the
same file across folds), and runs the EO/EC spectral + classification comparison.

  eeg/bin/python3 eo_ec_pipeline/run_eo_ec_test.py
  eeg/bin/python3 eo_ec_pipeline/run_eo_ec_test.py --gpu-device ""   # CPU
"""

import argparse
from pathlib import Path

import pandas as pd

from branches import build_branches
from classify import classify, spectral_check
from features import extract_features

CONDITION_INPUT_DIRS = {"EC": "data/input_ec", "EO": "data/input_eo"}
BRANCH_SUBDIR = {
    "raw": "raw",
    "denoised": "denoised/full_reconstruction",
    "upsampled": "upsampled/hybrid",
}


def run_eo_ec_test(branch_root: str = "data/branches", gpu_device=0) -> pd.DataFrame:
    branch_root = Path(branch_root)

    for condition, input_dir in CONDITION_INPUT_DIRS.items():
        fifs = list(Path(input_dir).glob("*.fif"))
        if not fifs:
            raise FileNotFoundError(f"No .fif files found in {input_dir} ({condition})")
        build_branches(input_dir, branch_root / condition, gpu_device=gpu_device)

    dfs = []
    for condition in CONDITION_INPUT_DIRS:
        for branch, subdir in BRANCH_SUBDIR.items():
            branch_dir = branch_root / condition / subdir
            for fif in sorted(branch_dir.glob("*.fif")):
                block_id = f"{condition}_{fif.stem}"
                dfs.append(extract_features(str(fif), branch, condition=condition, block_id=block_id))

    df = pd.concat(dfs, ignore_index=True)

    out_csv = branch_root / "features.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {len(df)} feature rows -> {out_csv}")
    print()
    print("=== Spectral sanity check: EC - EO log-alpha-power (dB) per channel ===")
    print(spectral_check(df).to_string(index=False))
    print()
    print("=== Classification accuracy per branch (GroupKFold by recording) ===")
    print(classify(df).to_string(index=False))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-root", default="data/branches")
    parser.add_argument("--gpu-device", default=0)
    args = parser.parse_args()
    run_eo_ec_test(args.branch_root, gpu_device=args.gpu_device)
