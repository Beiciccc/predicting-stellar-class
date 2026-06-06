#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "jun06"
TARGET = "class"
CLASSES = ["GALAXY", "QSO", "STAR"]

FILES = {
    "s40": ROOT / "outputs/jun04_adaptive3/s38_plus_new5.csv",
    "s47": ROOT / "outputs/jun05/s40_lr3_top10_margin_patch.csv",
    "s48": ROOT / "outputs/jun05/s40_lr3_top20_margin_patch.csv",
    "lr_v3": ROOT / "references/kaggle_outputs/cdeotte_gpu_lr_stacker_latest/submission.csv",
    "lr_v7": ROOT / "references/kaggle_outputs/jun06_cdeotte_lr_latest/submission.csv",
    "flex_anchor": ROOT / "references/kaggle_outputs/jun06_flex_blender/submission_anchor.csv",
    "flex_star": ROOT / "references/kaggle_outputs/jun06_flex_blender/submission_star_lean.csv",
    "flex_conservative": ROOT / "references/kaggle_outputs/jun06_flex_blender/submission_conservative.csv",
    "flex_consensus": ROOT / "references/kaggle_outputs/jun06_flex_blender/submission_consensus.csv",
    "flex_democratic": ROOT / "references/kaggle_outputs/jun06_flex_blender/submission_democratic.csv",
    "nybbler": ROOT / "references/kaggle_outputs/jun06_nybbler_key_stack/submission.csv",
    "adolf_fw_lgb": ROOT / "references/kaggle_outputs/jun06_adolf_fw_lgb/submission.csv",
    "kirill_tabm": ROOT / "references/kaggle_outputs/jun06_kirill_tabm_ovr/submission.csv",
    "kirill_xgb": ROOT / "references/kaggle_outputs/jun06_kirill_xgb_ovr/submission.csv",
    "torres": ROOT / "references/kaggle_outputs/jun06_torres_stacking/submission_ensemble1.csv",
    "kospintr": ROOT / "references/kaggle_outputs/jun06_kospintr_baseline/submission.csv",
    "tuan60": ROOT / "references/kaggle_outputs/tuannm_ensemble_97025/submission_alpha_0.60.csv",
    "fachri": ROOT / "references/kaggle_outputs/fachri_weighted_patch/submission.csv",
    "nina_simple": ROOT / "references/kaggle_outputs/nina_simple_vote/submission.csv",
    "deeplearn": ROOT / "references/kaggle_outputs/deeplearnerrr_blenders/submission.csv",
    "ektarr": ROOT / "references/kaggle_outputs/ektarr_ensemble_tuning/submission.csv",
    "stpete": ROOT / "references/kaggle_outputs/stpete_logistic_stacking/submission_final_stacking_logistic.csv",
}

SUPPORT_KEYS = [
    "flex_anchor",
    "nybbler",
    "adolf_fw_lgb",
    "kirill_tabm",
    "kirill_xgb",
    "torres",
    "kospintr",
    "tuan60",
    "fachri",
    "nina_simple",
    "deeplearn",
    "ektarr",
    "stpete",
]


def read_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for name, path in FILES.items():
        if path.exists():
            frame = pd.read_csv(path)
            if list(frame.columns) != ["id", TARGET]:
                raise ValueError(f"{name}: unexpected columns {list(frame.columns)}")
            frames[name] = frame
    ids = frames["s47"]["id"]
    for name, frame in frames.items():
        if not frame["id"].equals(ids):
            raise ValueError(f"{name}: id order mismatch")
    return frames


def save(name: str, ids: pd.Series, labels: pd.Series, anchor: pd.Series) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, TARGET: labels})
    path = OUT_DIR / f"{name}.csv"
    out.to_csv(path, index=False)
    print(
        f"{name}: diff_vs_anchor={int(labels.ne(anchor).sum())} "
        f"counts={labels.value_counts().to_dict()} path={path}"
    )


def margin_rows(anchor: pd.DataFrame, pred_path: Path) -> pd.DataFrame:
    pred = np.load(pred_path)
    label_to_idx = {label: idx for idx, label in enumerate(CLASSES)}
    pred_idx = pred.argmax(axis=1)
    pred_label = pd.Series([CLASSES[idx] for idx in pred_idx])
    anchor_idx = anchor[TARGET].map(label_to_idx).to_numpy()
    margin = pred[np.arange(len(pred)), pred_idx] - pred[np.arange(len(pred)), anchor_idx]
    rows = (
        pd.DataFrame(
            {
                "id": anchor["id"],
                "anchor": anchor[TARGET],
                "pred": pred_label,
                "margin": margin,
            }
        )
        .loc[pred_label.ne(anchor[TARGET].to_numpy())]
        .sort_values("margin", ascending=False)
        .reset_index(drop=True)
    )
    rows["rank"] = np.arange(1, len(rows) + 1)
    rows["transition"] = rows["anchor"] + "->" + rows["pred"]
    return rows


def add_support(rows: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = rows.copy()
    for key in SUPPORT_KEYS:
        if key in frames:
            out[key] = out["id"].map(frames[key].set_index("id")[TARGET])
    support_cols = [key for key in SUPPORT_KEYS if key in out.columns]
    out["support_pred"] = out.apply(lambda row: sum(row[col] == row["pred"] for col in support_cols), axis=1)
    out["support_anchor"] = out.apply(lambda row: sum(row[col] == row["anchor"] for col in support_cols), axis=1)
    return out


def patch_by_ids(anchor: pd.DataFrame, rows: pd.DataFrame, name: str) -> None:
    patched = anchor[TARGET].copy()
    mapping = rows.set_index("id")["pred"]
    mask = anchor["id"].isin(mapping.index)
    patched.loc[mask] = anchor.loc[mask, "id"].map(mapping)
    save(name, anchor["id"], patched, anchor[TARGET])


def adolf_lr7_agreement_patches(frames: dict[str, pd.DataFrame]) -> None:
    if "adolf_fw_lgb" not in frames:
        return
    anchor = frames["flex_anchor"]
    label_to_idx = {label: idx for idx, label in enumerate(CLASSES)}
    lr7 = frames["lr_v7"][TARGET]
    adolf = frames["adolf_fw_lgb"][TARGET]
    agree = lr7.eq(adolf) & lr7.ne(anchor[TARGET])

    lr_pred = np.load(ROOT / "references/kaggle_outputs/jun06_cdeotte_lr_latest/pred_lr_stacker_v7.npy")
    adolf_pred = np.load(ROOT / "references/kaggle_outputs/jun06_adolf_fw_lgb/pred_fw_lgb_stacker_v6.npy")
    pred_idx = lr7.map(label_to_idx).to_numpy()
    anchor_idx = anchor[TARGET].map(label_to_idx).to_numpy()
    lr_margin = lr_pred[np.arange(len(anchor)), pred_idx] - lr_pred[np.arange(len(anchor)), anchor_idx]
    adolf_margin = adolf_pred[np.arange(len(anchor)), pred_idx] - adolf_pred[np.arange(len(anchor)), anchor_idx]

    rows = pd.DataFrame(
        {
            "id": anchor["id"],
            "anchor": anchor[TARGET],
            "pred": lr7,
            "lr_margin": lr_margin,
            "adolf_margin": adolf_margin,
        }
    ).loc[agree]
    rows["margin"] = rows[["lr_margin", "adolf_margin"]].min(axis=1)
    rows["transition"] = rows["anchor"] + "->" + rows["pred"]
    rows = rows.sort_values("margin", ascending=False).reset_index(drop=True)
    rows["rank"] = np.arange(1, len(rows) + 1)

    for n in [5, 10, 20, 24, 25, 26, 30, 35, 40, 50]:
        patch_by_ids(anchor, rows[rows["rank"].le(n)], f"jun06_flex_anchor_lr7_adolf_agree_top{n}")
    patch_by_ids(
        anchor,
        rows[rows["rank"].le(50) & rows["transition"].isin(["GALAXY->STAR", "QSO->STAR"])],
        "jun06_flex_anchor_lr7_adolf_agree_top50_star_target",
    )


def copy_source(frames: dict[str, pd.DataFrame], source: str, name: str, anchor: str = "s47") -> None:
    save(name, frames[source]["id"], frames[source][TARGET].copy(), frames[anchor][TARGET])


def main() -> None:
    frames = read_frames()

    copy_source(frames, "flex_anchor", "jun06_flex_anchor")
    copy_source(frames, "lr_v7", "jun06_lr_v7")
    copy_source(frames, "flex_star", "jun06_flex_star_lean")
    copy_source(frames, "flex_conservative", "jun06_flex_conservative")
    copy_source(frames, "flex_consensus", "jun06_flex_consensus")
    copy_source(frames, "flex_democratic", "jun06_flex_democratic")
    copy_source(frames, "nybbler", "jun06_nybbler_key_stack")
    if "adolf_fw_lgb" in frames:
        copy_source(frames, "adolf_fw_lgb", "jun06_adolf_fw_lgb")

    lr3_rows = add_support(
        margin_rows(frames["s40"], ROOT / "references/kaggle_outputs/cdeotte_gpu_lr_stacker_latest/pred_lr_stacker_v3.npy"),
        frames,
    )
    patch_by_ids(
        frames["s40"],
        lr3_rows[lr3_rows["rank"].between(1, 10) | lr3_rows["rank"].between(16, 20)],
        "jun06_s40_lr3_top10_plus16_20",
    )
    patch_by_ids(
        frames["s40"],
        lr3_rows[lr3_rows["rank"].le(20) & ~lr3_rows["rank"].isin([13])],
        "jun06_s48_drop_lr3_rank13",
    )
    patch_by_ids(
        frames["s40"],
        lr3_rows[lr3_rows["rank"].le(20) & ~lr3_rows["rank"].isin([14])],
        "jun06_s48_drop_lr3_rank14",
    )
    patch_by_ids(
        frames["s40"],
        lr3_rows[lr3_rows["rank"].le(20) & ~lr3_rows["rank"].isin([11, 12])],
        "jun06_s48_drop_lr3_r11_r12",
    )
    patch_by_ids(
        frames["s40"],
        lr3_rows[lr3_rows["rank"].le(20) & ~lr3_rows["transition"].eq("STAR->GALAXY")],
        "jun06_s40_lr3_top20_no_star_to_galaxy",
    )

    lr7_rows = add_support(
        margin_rows(frames["s47"], ROOT / "references/kaggle_outputs/jun06_cdeotte_lr_latest/pred_lr_stacker_v7.npy"),
        frames,
    )
    for n in [5, 10, 20, 30, 50]:
        patch_by_ids(frames["s47"], lr7_rows[lr7_rows["rank"].le(n)], f"jun06_s47_lr7_top{n}")
    patch_by_ids(
        frames["s47"],
        lr7_rows[lr7_rows["rank"].le(50) & lr7_rows["support_pred"].ge(7)],
        "jun06_s47_lr7_top50_support_ge7",
    )
    patch_by_ids(
        frames["s47"],
        lr7_rows[
            lr7_rows["rank"].le(50)
            & lr7_rows["support_pred"].ge(7)
            & lr7_rows["transition"].isin(["GALAXY->STAR", "QSO->STAR"])
        ],
        "jun06_s47_lr7_top50_star_target_support_ge7",
    )

    flex_rows = add_support(
        margin_rows(frames["flex_anchor"], ROOT / "references/kaggle_outputs/jun06_cdeotte_lr_latest/pred_lr_stacker_v7.npy"),
        frames,
    )
    for n in [10, 20, 50]:
        patch_by_ids(frames["flex_anchor"], flex_rows[flex_rows["rank"].le(n)], f"jun06_flex_anchor_lr7_top{n}")
    patch_by_ids(
        frames["flex_anchor"],
        flex_rows[flex_rows["rank"].le(50) & flex_rows["support_pred"].ge(7)],
        "jun06_flex_anchor_lr7_top50_support_ge7",
    )
    adolf_lr7_agreement_patches(frames)


if __name__ == "__main__":
    main()
