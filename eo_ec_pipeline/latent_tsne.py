"""
t-SNE visualization of per-window alpha-power feature vectors across every branch
and reconstruction-kind produced by run_eo_ec_test.py, colored three ways:

  1. by subject (gary / kotora / vishwani)
  2. by branch (raw / denoised / upsampled)
  3. by reconstruction kind (full_reconstruction / hybrid) -- raw has neither, shown
     as a muted "n/a" category for spatial context

Feature vector per window = log alpha-band (8-13Hz) power on the 4 REAL Muse
channels only (TP9, AF7, AF8, TP10) -- kept to these 4 so raw/denoised (4ch) and
upsampled (8ch, includes hallucinated posterior channels) are directly comparable
in the same feature space. One t-SNE embedding is fit on the combined data; the
three plots just recolor the same 2D layout.

  eeg/bin/python3 eo_ec_pipeline/latent_tsne.py
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from features import extract_features

REAL_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
CONDITION_DIRS = {"EC": "data/input_ec", "EO": "data/input_eo"}

# (branch, recon_kind, subdir-under-data/branches/<condition>/)
VARIANTS = [
    ("raw", None, "raw"),
    ("denoised", "full_reconstruction", "denoised/full_reconstruction"),
    ("denoised", "hybrid", "denoised/hybrid"),
    ("upsampled", "full_reconstruction", "upsampled/full_reconstruction"),
    ("upsampled", "hybrid", "upsampled/hybrid"),
]

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated slots 1-4
MUTED_GRAY = "#9a9a95"


def build_feature_table(branch_root: str = "data/branches") -> pd.DataFrame:
    branch_root = Path(branch_root)
    rows = []
    for condition in CONDITION_DIRS:
        for branch, recon_kind, subdir in VARIANTS:
            variant_dir = branch_root / condition / subdir
            fifs = sorted(variant_dir.glob("*.fif"))
            if not fifs:
                warnings.warn(f"no files under {variant_dir}; skipping this variant/condition")
                continue
            for fif in fifs:
                subject = fif.stem.split("_")[0]
                block_id = f"{condition}_{branch}_{recon_kind}_{fif.stem}"
                df = extract_features(str(fif), branch, condition=condition, block_id=block_id)
                wide = df[df["channel"].isin(REAL_CHANNELS)].pivot_table(
                    index="epoch_idx", columns="channel", values="log_alpha_power"
                ).reindex(columns=REAL_CHANNELS)
                wide["subject"] = subject
                wide["condition"] = condition
                wide["branch"] = branch
                wide["recon_kind"] = recon_kind if recon_kind is not None else "n/a"
                wide["recording"] = fif.stem
                rows.append(wide.reset_index())
    if not rows:
        raise FileNotFoundError(f"No branch output found under {branch_root} -- run run_eo_ec_test.py first")
    return pd.concat(rows, ignore_index=True)


def fit_tsne(df: pd.DataFrame, perplexity: float = 30.0, random_state: int = 0) -> np.ndarray:
    X = StandardScaler().fit_transform(df[REAL_CHANNELS].to_numpy())
    perplexity = min(perplexity, max(5.0, (len(df) - 1) / 3))
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=random_state).fit_transform(X)


def scatter_by(df: pd.DataFrame, xy: np.ndarray, color_col: str, title: str, out_png: str,
               color_map: dict | None = None) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    categories = [c for c in df[color_col].unique() if c != "n/a"]
    categories = sorted(categories)
    if "n/a" in df[color_col].unique():
        mask = df[color_col] == "n/a"
        ax.scatter(xy[mask.to_numpy(), 0], xy[mask.to_numpy(), 1], s=10, c=MUTED_GRAY,
                   label="n/a (raw)", alpha=0.5, linewidths=0)

    palette = color_map or {cat: CATEGORICAL[i % len(CATEGORICAL)] for i, cat in enumerate(categories)}
    for cat in categories:
        mask = (df[color_col] == cat).to_numpy()
        ax.scatter(xy[mask, 0], xy[mask, 1], s=10, c=palette[cat], label=cat, alpha=0.7, linewidths=0)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title)
    ax.legend(frameon=False, markerscale=1.5)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main(branch_root: str = "data/branches", out_dir: str = "data/figures") -> pd.DataFrame:
    df = build_feature_table(branch_root)
    print(f"{len(df)} windows total across {df['recording'].nunique()} recordings x {len(VARIANTS)} variants")

    xy = fit_tsne(df)
    df["tsne_x"], df["tsne_y"] = xy[:, 0], xy[:, 1]

    scatter_by(df, xy, "subject", "t-SNE of alpha-power windows, colored by subject",
               f"{out_dir}/tsne_by_subject.png")
    scatter_by(df, xy, "branch", "t-SNE of alpha-power windows, colored by branch (raw/denoised/upsampled)",
               f"{out_dir}/tsne_by_branch.png")
    scatter_by(df, xy, "recon_kind", "t-SNE of alpha-power windows, colored by reconstruction kind",
               f"{out_dir}/tsne_by_recon_kind.png",
               color_map={"full_reconstruction": CATEGORICAL[0], "hybrid": CATEGORICAL[1]})
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-root", default="data/branches")
    parser.add_argument("--out-dir", default="data/figures")
    args = parser.parse_args()
    main(args.branch_root, args.out_dir)
