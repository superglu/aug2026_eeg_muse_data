"""
Compare EO/EC discriminability across branches (raw / denoised / upsampled):
  1. spectral sanity check: EC vs EO alpha-power per channel per branch (dB)
  2. classification accuracy: grouped cross-validated logistic regression per branch
     (grouped by block_idx so folds never mix epochs from the same EC/EO block, and
     stratified so every test fold holds both conditions -- see n_splits_for)

  eeg/bin/python3 classify.py raw.csv denoised.csv upsampled.csv
"""

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

MAX_SPLITS = 5


def spectral_check(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["branch", "channel", "condition"])["log_alpha_power"].mean().unstack("condition")
    g["EC_minus_EO_db"] = (g["EC"] - g["EO"]) * 10 / np.log(10)
    return g.reset_index()


def n_splits_for(y: np.ndarray, groups: np.ndarray) -> int:
    """Most folds that can still hold both classes.

    Every group here is one whole recording of a single condition, so a fold's
    class balance is decided entirely by which recordings land in it. With more
    folds than recordings-per-condition, some test fold necessarily contains a
    single class, and accuracy on it is 0.0 or 1.0 regardless of what was learned
    -- the reported mean stops estimating anything. Capping the split count at the
    smaller per-class recording count lets StratifiedGroupKFold put at least one
    EC and one EO recording in every test fold.
    """
    groups_per_class = [len(set(groups[y == label])) for label in np.unique(y)]
    if len(groups_per_class) < 2 or min(groups_per_class) < 2:
        raise ValueError(
            "Need at least 2 recordings per condition for grouped cross-validation; "
            f"got {dict(zip(np.unique(y).tolist(), groups_per_class))} (0=EO, 1=EC)."
        )
    return min(MAX_SPLITS, min(groups_per_class))


def classify(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for branch, sub in df.groupby("branch"):
        wide = sub.pivot_table(index=["block_idx", "epoch_idx"], columns="channel",
                                values="log_alpha_power").reset_index()
        labels = sub.drop_duplicates(["block_idx", "epoch_idx"])[["block_idx", "epoch_idx", "condition"]]
        wide = wide.merge(labels, on=["block_idx", "epoch_idx"])

        X = wide.drop(columns=["block_idx", "epoch_idx", "condition"]).to_numpy()
        y = (wide["condition"] == "EC").astype(int).to_numpy()
        groups = wide["block_idx"].to_numpy()

        cv = StratifiedGroupKFold(n_splits=n_splits_for(y, groups), shuffle=True, random_state=0)
        scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, groups=groups, cv=cv)
        results.append({"branch": branch, "mean_acc": scores.mean(), "std_acc": scores.std(), "n_epochs": len(y)})
    return pd.DataFrame(results)


if __name__ == "__main__":
    feature_csvs = sys.argv[1:]
    df = pd.concat([pd.read_csv(f) for f in feature_csvs], ignore_index=True)

    print("=== Spectral sanity check: EC - EO log-alpha-power (dB) per channel ===")
    print(spectral_check(df).to_string(index=False))
    print()
    print("=== Classification accuracy per branch (StratifiedGroupKFold by block) ===")
    print(classify(df).to_string(index=False))
