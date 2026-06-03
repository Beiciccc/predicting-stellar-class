from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET = "class"
PUBLIC_STACKER_SLUG = "gpu-logistic-regression-stacker"
PHOTOMETRY_COLUMNS = ["u", "g", "r", "i", "z", "redshift"]
COLOR_PAIRS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]


def find_data_dir():
    candidates = [
        Path("/kaggle/input/playground-series-s6e6"),
        Path("/kaggle/input/predicting-stellar-class"),
    ]
    candidates.extend(path.parent for path in Path("/kaggle/input").glob("**/train.csv"))
    for candidate in candidates:
        if (candidate / "train.csv").exists() and (candidate / "test.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find train.csv and test.csv under /kaggle/input")


def find_public_stacker_submission(sample_submission):
    input_root = Path("/kaggle/input")
    for path in input_root.glob(f"**/{PUBLIC_STACKER_SLUG}/submission.csv"):
        candidate = pd.read_csv(path)
        if list(candidate.columns) == ["id", TARGET] and candidate["id"].equals(sample_submission["id"]):
            return candidate
    for path in input_root.glob("**/submission.csv"):
        if PUBLIC_STACKER_SLUG not in str(path):
            continue
        candidate = pd.read_csv(path)
        if list(candidate.columns) == ["id", TARGET] and candidate["id"].equals(sample_submission["id"]):
            return candidate
    raise FileNotFoundError("Could not find the public stacker submission.csv source")


def add_color_features(df):
    out = df.copy()
    for left, right in COLOR_PAIRS:
        out[f"{left}-{right}"] = out[left] - out[right]
    return out


def print_basic_eda(train, test):
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("\nTarget distribution:")
    target_summary = (
        train[TARGET]
        .value_counts()
        .rename_axis(TARGET)
        .reset_index(name="count")
    )
    target_summary["share"] = target_summary["count"] / len(train)
    print(target_summary.to_string(index=False))

    print("\nMissing values in core columns:")
    core_columns = ["id", TARGET, *PHOTOMETRY_COLUMNS, "spectral_type", "galaxy_population"]
    missing = train[[col for col in core_columns if col in train.columns]].isna().sum()
    print(missing[missing > 0].to_string() if missing.sum() else "No missing values in core train columns.")

    print("\nPhotometry summary by target:")
    print(train.groupby(TARGET)[PHOTOMETRY_COLUMNS].agg(["mean", "std", "min", "max"]).round(4))

    feature_train = add_color_features(train)
    color_columns = [f"{left}-{right}" for left, right in COLOR_PAIRS]
    print("\nColor index means by target:")
    print(feature_train.groupby(TARGET)[color_columns].mean().round(4))


def save_eda_figures(train):
    work_dir = Path("/kaggle/working")
    work_dir.mkdir(parents=True, exist_ok=True)

    target_counts = train[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    target_counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("Target class distribution")
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(work_dir / "eda_target_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    redshift_clip = train["redshift"].clip(
        lower=train["redshift"].quantile(0.001),
        upper=train["redshift"].quantile(0.999),
    )
    for label, group in train.assign(redshift_clip=redshift_clip).groupby(TARGET):
        ax.hist(group["redshift_clip"], bins=80, alpha=0.45, density=True, label=label)
    ax.set_title("Redshift distribution by class")
    ax.set_xlabel("redshift, clipped at 0.1% and 99.9%")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(work_dir / "eda_redshift_by_class.png", dpi=160)
    plt.close(fig)

    feature_train = add_color_features(train)
    color_columns = [f"{left}-{right}" for left, right in COLOR_PAIRS]
    fig, ax = plt.subplots(figsize=(7, 4))
    feature_train.groupby(TARGET)[color_columns].mean().T.plot(ax=ax, marker="o")
    ax.set_title("Mean color indices by class")
    ax.set_xlabel("color index")
    ax.set_ylabel("mean value")
    fig.tight_layout()
    fig.savefig(work_dir / "eda_color_indices.png", dpi=160)
    plt.close(fig)


data_dir = find_data_dir()
train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

print_basic_eda(train, test)
save_eda_figures(train)

submission = find_public_stacker_submission(sample_submission)
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("\nSubmission preview:")
print(submission.head())
