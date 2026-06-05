#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "jun05"
TARGET = "class"
CLASSES = ["GALAXY", "QSO", "STAR"]

FILES = {
    "best": ROOT / "outputs/jun04_adaptive3/s38_plus_new5.csv",
    "lr_v1": ROOT / "references/kaggle_outputs/cdeotte_gpu_lr_stacker/submission.csv",
    "lr_v3": ROOT / "references/kaggle_outputs/cdeotte_gpu_lr_stacker_latest/submission.csv",
    "xgb_v3": ROOT / "references/kaggle_outputs/cdeotte_xgb_v3/submission.csv",
    "lgbm_v3": ROOT / "references/kaggle_outputs/cdeotte_lgbm_v3/subs/lgbm-3_submission.csv",
    "nn_v1": ROOT / "references/kaggle_outputs/cdeotte_nn_v1/subs/nn-1_submission.csv",
    "tabicl_v2": ROOT / "references/kaggle_outputs/cdeotte_tabicl_v2/subs/tabicl-2_submission.csv",
    "kospintr": ROOT / "references/kaggle_outputs/kospintr_baseline/submission.csv",
    "tuan60": ROOT / "references/kaggle_outputs/tuannm_ensemble_97025/submission_alpha_0.60.csv",
    "fachri": ROOT / "references/kaggle_outputs/fachri_weighted_patch/submission.csv",
    "nina_simple": ROOT / "references/kaggle_outputs/nina_simple_vote/submission.csv",
    "deeplearn": ROOT / "references/kaggle_outputs/deeplearnerrr_blenders/submission.csv",
    "ektarr": ROOT / "references/kaggle_outputs/ektarr_ensemble_tuning/submission.csv",
    "stpete": ROOT / "references/kaggle_outputs/stpete_logistic_stacking/submission_final_stacking_logistic.csv",
    "flex": ROOT / "references/kaggle_outputs/flex_blender/submission.csv",
    "nina_old": ROOT / "references/kaggle_outputs/nina_vote/submission.csv",
    "lgbm_old": ROOT / "references/kaggle_outputs/lgbm_single_96728/submission.csv",
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


def min_vote_patch(
    frames: dict[str, pd.DataFrame],
    keys: list[str],
    min_count: int,
    name: str,
) -> None:
    best = frames["best"][TARGET]
    patched = best.copy()
    for idx in range(len(best)):
        counts = Counter(frames[key][TARGET].iat[idx] for key in keys)
        label, count = counts.most_common(1)[0]
        if count >= min_count and label != best.iat[idx]:
            patched.iat[idx] = label
    save(name, frames["best"]["id"], patched, best)


def min_vote_label_patch(
    frames: dict[str, pd.DataFrame],
    keys: list[str],
    min_count: int,
    label: str,
    name: str,
) -> None:
    best = frames["best"][TARGET]
    patched = best.copy()
    for idx in range(len(best)):
        votes = [frames[key][TARGET].iat[idx] for key in keys]
        if votes.count(label) >= min_count and label != best.iat[idx]:
            patched.iat[idx] = label
    save(name, frames["best"]["id"], patched, best)


def lr3_margin_patches(frames: dict[str, pd.DataFrame]) -> None:
    best = frames["best"]
    pred = np.load(ROOT / "references/kaggle_outputs/cdeotte_gpu_lr_stacker_latest/pred_lr_stacker_v3.npy")
    label_to_idx = {label: idx for idx, label in enumerate(CLASSES)}
    lr_idx = pred.argmax(axis=1)
    lr_label = pd.Series([CLASSES[idx] for idx in lr_idx])
    best_idx = best[TARGET].map(label_to_idx).to_numpy()
    margin = pred[np.arange(len(pred)), lr_idx] - pred[np.arange(len(pred)), best_idx]
    rows = (
        pd.DataFrame({"id": best["id"], "lr_label": lr_label, "margin": margin})
        .loc[lr_label.ne(best[TARGET].to_numpy())]
        .sort_values("margin", ascending=False)
    )
    for n in [5, 10, 20, 50, 100, 200]:
        patched = best[TARGET].copy()
        top = rows.head(n).set_index("id")["lr_label"]
        mask = best["id"].isin(top.index)
        patched.loc[mask] = best.loc[mask, "id"].map(top)
        save(f"s40_lr3_top{n}_margin_patch", best["id"], patched, best[TARGET])


def main() -> None:
    frames = read_frames()
    core13 = [
        "lr_v1",
        "lr_v3",
        "flex",
        "nina_old",
        "nina_simple",
        "deeplearn",
        "fachri",
        "ektarr",
        "stpete",
        "lgbm_old",
        "lgbm_v3",
        "xgb_v3",
        "tuan60",
    ]
    public12 = [name for name in core13 if name != "tuan60"]
    patch8 = ["lr_v3", "xgb_v3", "tuan60", "fachri", "nina_simple", "deeplearn", "ektarr", "stpete"]
    public15 = core13 + ["nn_v1", "tabicl_v2"]
    new7 = ["lr_v3", "lgbm_v3", "xgb_v3", "tuan60", "nn_v1", "tabicl_v2", "kospintr"]

    min_vote_label_patch(frames, core13, 10, "GALAXY", "s40_core13_10of13_gal")
    min_vote_label_patch(frames, public12, 9, "GALAXY", "s40_public12_9of12_gal")
    min_vote_label_patch(frames, core13, 9, "GALAXY", "s40_core13_9of13_gal")
    min_vote_patch(frames, patch8, 7, "s40_patch8_7of8")
    unanimous_patch(frames, ["tuan60", "lr_v3", "xgb_v3"], "s40_tuan60_lr3_xgb3_unanimous")
    unanimous_patch(frames, ["tuan60", "fachri", "xgb_v3"], "s40_tuan60_fachri_xgb3_unanimous")
    unanimous_patch(frames, ["tuan60", "fachri", "lr_v3"], "s40_tuan60_fachri_lr3_unanimous")
    min_vote_label_patch(frames, patch8, 6, "GALAXY", "s40_patch8_6of8_gal")
    min_vote_label_patch(frames, public15, 11, "GALAXY", "s40_public15_11of15_gal")
    min_vote_patch(frames, new7, 6, "s40_new7_6of7")
    lr3_margin_patches(frames)

    unanimous_patch(frames, ["lr_v3", "tuan60", "lgbm_old"], "s40_lr3_tuan60_lgbmold_unanimous")
    unanimous_patch(
        frames,
        ["lr_v3", "xgb_v3", "lgbm_v3", "nn_v1", "tabicl_v2"],
        "s40_cdeotte_v3_all5_unanimous",
    )
    unanimous_patch(frames, ["lr_v3", "tuan60", "fachri"], "s40_lr3_tuan60_fachri_unanimous")
    min_vote_patch(
        frames,
        ["lr_v3", "tuan60", "fachri", "nina_simple", "deeplearn", "ektarr", "stpete"],
        6,
        "s40_new_lr3_tuan60_6of7",
    )
    unanimous_patch(frames, ["lr_v3", "tuan60", "xgb_v3"], "s40_lr3_tuan60_xgb3_unanimous")
    unanimous_patch(frames, ["lr_v3", "tuan60", "flex"], "s40_lr3_tuan60_flex_unanimous")
    unanimous_patch(frames, ["lr_v3", "tuan60", "nina_old"], "s40_lr3_tuan60_nina_unanimous")
    unanimous_patch(frames, ["lr_v3", "tuan60"], "s40_lr3_tuan60_unanimous")
    min_vote_patch(
        frames,
        ["fachri", "nina_simple", "deeplearn", "ektarr", "stpete"],
        4,
        "s40_new_public_4of5",
    )
    min_vote_patch(
        frames,
        ["lr_v3", "tuan60", "fachri", "nina_simple", "deeplearn", "ektarr", "stpete"],
        5,
        "s40_new_lr3_tuan60_5of7",
    )


if __name__ == "__main__":
    main()
