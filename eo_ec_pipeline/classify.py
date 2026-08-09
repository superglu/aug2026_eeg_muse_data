"""
Compare EO/EC discriminability across branches (raw / denoised / upsampled):
  1. spectral sanity check: EC vs EO alpha-power per channel per branch (dB)
  2. classification accuracy: grouped cross-validated logistic regression per branch
     (grouped by block_idx so folds never mix epochs from the same EC/EO block)

  eeg/bin/python3 classify.py raw.csv denoised.csv upsampled.csv
"""

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score


def spectral_check(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["branch", "channel", "condition"])["log_alpha_power"].mean().unstack("condition")
    g["EC_minus_EO_db"] = (g["EC"] - g["EO"]) * 10 / np.log(10)
    return g.reset_index()


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

        n_groups = len(set(groups))
        cv = GroupKFold(n_splits=min(5, n_groups))
        scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, groups=groups, cv=cv)
        results.append({"branch": branch, "mean_acc": scores.mean(), "std_acc": scores.std(), "n_epochs": len(y)})
    return pd.DataFrame(results)


if __name__ == "__main__":
    feature_csvs = sys.argv[1:]
    df = pd.concat([pd.read_csv(f) for f in feature_csvs], ignore_index=True)

    print("=== Spectral sanity check: EC - EO log-alpha-power (dB) per channel ===")
    print(spectral_check(df).to_string(index=False))
    print()
    print("=== Classification accuracy per branch (GroupKFold by block) ===")
    print(classify(df).to_string(index=False))
