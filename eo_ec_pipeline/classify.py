"""
Compare EO/EC discriminability across branches (raw / denoised / upsampled):
  1. spectral sanity check: EC vs EO alpha-power per channel per branch (dB),
     reported per subject and pooled over subjects (per_subject_spectral_check /
     spectral_check) -- the README requires both, not pooled alone
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

from branch_files import subject_key

MAX_SPLITS = 5


def _db(ec: pd.Series, eo: pd.Series) -> pd.Series:
    return (ec - eo) * 10 / np.log(10)


def per_subject_spectral_check(df: pd.DataFrame) -> pd.DataFrame:
    """EC - EO log-alpha-power (dB) per subject per channel per branch.

    The README requires this alongside the pooled number ("Report per-subject alpha
    suppression index ... not just pooled"): at n=4, one contaminated block or one
    low-alpha subject can carry a pooled mean, and pooling alone can't tell "consistent
    moderate effect" from "huge effect in some, absent in others". Each subject's two
    recordings are matched by subject_key, since the feature table carries one block_idx
    per recording (one condition each), not per subject.
    """
    # astype(str): a block id read back from features.csv can arrive as a number if the
    # operator passed a numeric --block-id, and subject_key takes a name.
    tidy = df.assign(subject=df["block_idx"].astype(str).map(subject_key))
    per_subject = (tidy.groupby(["branch", "channel", "subject", "condition"])["log_alpha_power"]
                   .mean().unstack("condition"))
    for condition in ("EC", "EO"):
        if condition not in per_subject.columns:
            per_subject[condition] = np.nan
    per_subject["EC_minus_EO_db"] = _db(per_subject["EC"], per_subject["EO"])
    return per_subject.reset_index()


def spectral_check(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled EC - EO log-alpha-power (dB), each subject weighted equally.

    Averaging raw epochs would weight each subject by its number of surviving epochs,
    and those counts are not equal: features.py drops epochs straddling a BAD_gap, so a
    subject with several Bluetooth dropouts would count for less than a clean one and
    data quality would silently reweight the reported group dB. Averaging the per-subject
    values instead makes this the mean of the effect over subjects. Subjects missing one
    of the two conditions have no effect to contribute and are excluded.
    """
    per_subject = per_subject_spectral_check(df)
    paired = per_subject.dropna(subset=["EC_minus_EO_db"])
    if paired.empty:
        # Every subject missing a condition is not "nothing interesting to report": it is
        # the headline metric of the study vanishing. It happens whenever subject_key
        # mis-keys the recordings (the two conditions then land under different subjects),
        # so it has to fail like band_power_check._recordings_by_subject does rather than
        # print an empty table.
        raise ValueError(
            "No subject has both an EC and an EO recording, so there is no EC-EO effect "
            f"to report. Subjects keyed from block_idx: {sorted(per_subject['subject'].unique())} "
            "— check that each recording's filename carries its subject and exactly one of "
            "'open'/'closed' (see branch_files.subject_key)."
        )
    return paired.groupby(["branch", "channel"]).agg(
        EC=("EC", "mean"),
        EO=("EO", "mean"),
        EC_minus_EO_db=("EC_minus_EO_db", "mean"),
        n_subjects=("subject", "nunique"),
    ).reset_index()


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

    print("=== Spectral sanity check: EC - EO log-alpha-power (dB) per subject per channel ===")
    print(per_subject_spectral_check(df).to_string(index=False))
    print()
    print("=== Spectral sanity check: EC - EO log-alpha-power (dB) pooled over subjects ===")
    print(spectral_check(df).to_string(index=False))
    print()
    print("=== Classification accuracy per branch (StratifiedGroupKFold by block) ===")
    print(classify(df).to_string(index=False))
