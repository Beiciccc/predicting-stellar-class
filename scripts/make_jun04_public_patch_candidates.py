#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "jun04_public_patches"
TARGET = "class"

FILES = {
    "best": ROOT / "outputs/public_blends/qso_strict3_lr_flex_nina_lgbm_ours.csv",
    "fachri": ROOT / "references/kaggle_outputs/fachri_weighted_patch/submission.csv",
    "nina_simple": ROOT / "references/kaggle_outputs/nina_simple_vote/submission.csv",
    "deeplearn": ROOT / "references/kaggle_outputs/deeplearnerrr_blenders/submission.csv",
    "ektarr": ROOT / "references/kaggle_outputs/ektarr_ensemble_tuning/submission.csv",
    "stpete": ROOT / "references/kaggle_outputs/stpete_logistic_stacking/submission_final_stacking_logistic.csv",
    "flex": ROOT / "references/kaggle_outputs/flex_blender/submission.csv",
    "nina_old": ROOT / "references/kaggle_outputs/nina_vote/submission.csv",
    "lgbm_cal": ROOT / "references/kaggle_outputs/lgbm_single_96728/submission.csv",
}


def read_frames() -> dict[str, pd.DataFrame]:
    frames = {name: pd.read_csv(path) for name, path in FILES.items()}
    ids = frames["best"]["id"]
    for name, frame in frames.items():
        if list(frame.columns) != ["id", TARGET]:
            raise ValueError(f"{name}: unexpected columns {list(frame.columns)}")
        if not frame["id"].equals(ids):
            raise ValueError(f"{name}: id order mismatch")
    return frames


def save(name: str, ids: pd.Series, labels: pd.Series, anchor: pd.Series) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, TARGET: labels})
    path = OUT_DIR / f"{name}.csv"
    out.to_csv(path, index=False)
    print(
        f"{name}: diff_vs_best={int(labels.ne(anchor).sum())} "
        f"counts={labels.value_counts().to_dict()} path={path}"
    )


def unanimous_patch(
    frames: dict[str, pd.DataFrame],
    keys: list[str],
    name: str,
) -> None:
    best = frames["best"][TARGET]
    patched = best.copy()
    for idx in range(len(best)):
        labels = [frames[key][TARGET].iat[idx] for key in keys]
        if len(set(labels)) == 1 and labels[0] != best.iat[idx]:
            patched.iat[idx] = labels[0]
    save(name, frames["best"]["id"], patched, best)


def all_new_majority_patch(frames: dict[str, pd.DataFrame]) -> None:
    best = frames["best"][TARGET]
    patched = best.copy()
    keys = ["fachri", "nina_simple", "deeplearn", "ektarr", "stpete"]
    for idx in range(len(best)):
        counts = Counter(frames[key][TARGET].iat[idx] for key in keys)
        label, count = counts.most_common(1)[0]
        if count >= 5 and label != best.iat[idx]:
            patched.iat[idx] = label
    save("new5_unanimous_patch", frames["best"]["id"], patched, best)


def main() -> None:
    frames = read_frames()
    unanimous_patch(
        frames,
        ["fachri", "nina_simple", "deeplearn", "flex"],
        "fachri_nina_deep_flex_unanimous_patch",
    )
    unanimous_patch(
        frames,
        ["fachri", "flex", "nina_old"],
        "fachri_flex_nina_unanimous_patch",
    )
    unanimous_patch(
        frames,
        ["fachri", "nina_simple", "deeplearn", "lgbm_cal"],
        "fachri_nina_deep_lgbm_unanimous_patch",
    )
    all_new_majority_patch(frames)


if __name__ == "__main__":
    main()
