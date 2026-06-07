#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "jun07"
TARGET = "class"

FILES = {
    "best_top25": ROOT / "outputs/jun06/jun06_flex_anchor_lr7_adolf_agree_top25.csv",
    "nina_vote2": ROOT / "references/kaggle_outputs/jun07_nina_simple_vote2/submission.csv",
    "amry_one": ROOT / "references/kaggle_outputs/jun07_amry_meta_patch_97108/submission_gpu_meta_one_row_patch.csv",
    "amry_top2": ROOT / "references/kaggle_outputs/jun07_amry_meta_patch_97108/submission_gpu_meta_top2_patch.csv",
    "amry_top3": ROOT / "references/kaggle_outputs/jun07_amry_meta_patch_97108/submission_gpu_meta_top3_patch.csv",
    "amry_top5": ROOT / "references/kaggle_outputs/jun07_amry_meta_patch_97108/submission_gpu_meta_top5_patch.csv",
    "amry_top8": ROOT / "references/kaggle_outputs/jun07_amry_meta_patch_97108/submission_gpu_meta_top8_patch.csv",
    "amry_top13": ROOT / "references/kaggle_outputs/jun07_amry_meta_patch_97108/submission_gpu_meta_top13_patch.csv",
    "amry_mlp_raw": ROOT / "references/kaggle_outputs/jun07_amry_meta_patch_97108/submission_raw_mlp_b1_00.csv",
    "deb_saga": ROOT / "references/kaggle_outputs/jun07_deb_saga_97105/submission.csv",
    "cdeotte_v8": ROOT / "references/kaggle_outputs/jun07_cdeotte_lr_latest/submission.csv",
    "kaisei_oof": ROOT / "references/kaggle_outputs/jun07_kaisei_oof_stacker/submission.csv",
    "amry_oof": ROOT / "references/kaggle_outputs/jun07_amry_oof_stacker_97106/submission.csv",
}
AMRY_CANDIDATES = ROOT / "references/kaggle_outputs/jun07_amry_meta_patch_97108/gpu_patch_candidates.csv"


def read_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for name, path in FILES.items():
        if path.exists():
            frame = pd.read_csv(path)
            if list(frame.columns) != ["id", TARGET]:
                raise ValueError(f"{name}: unexpected columns {list(frame.columns)}")
            frames[name] = frame
    ids = frames["best_top25"]["id"]
    for name, frame in frames.items():
        if not frame["id"].equals(ids):
            raise ValueError(f"{name}: id order mismatch")
    return frames


def save(name: str, frame: pd.DataFrame, anchor: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = frame[["id", TARGET]].copy()
    path = OUT_DIR / f"{name}.csv"
    out.to_csv(path, index=False)
    print(
        f"{name}: diff_vs_best={int(out[TARGET].ne(anchor[TARGET]).sum())} "
        f"counts={out[TARGET].value_counts().to_dict()} path={path}"
    )


def make_amry_candidate_patches(frames: dict[str, pd.DataFrame]) -> None:
    if "nina_vote2" not in frames or not AMRY_CANDIDATES.exists():
        return
    base = frames["nina_vote2"]
    best = frames["best_top25"]
    candidates = pd.read_csv(AMRY_CANDIDATES)
    candidates = candidates[~candidates["already_tried"].astype(bool)].copy()
    candidates = candidates.sort_values("score", ascending=False).reset_index(drop=True)
    candidates["transition"] = candidates["anchor_class"] + "->" + candidates["gpu_meta_class"]

    for n in [6, 7, 20, 30, 50]:
        patched = base.copy()
        rows = candidates.head(n)
        mapping = rows.set_index("id")["gpu_meta_class"]
        mask = patched["id"].isin(mapping.index)
        patched.loc[mask, TARGET] = patched.loc[mask, "id"].map(mapping)
        save(f"jun07_amry_custom_top{n}", patched, best)

    star_rows = candidates[candidates["transition"].isin(["GALAXY->STAR", "QSO->STAR"])]
    for n in [20, 30, 50]:
        patched = base.copy()
        rows = star_rows.head(n)
        mapping = rows.set_index("id")["gpu_meta_class"]
        mask = patched["id"].isin(mapping.index)
        patched.loc[mask, TARGET] = patched.loc[mask, "id"].map(mapping)
        save(f"jun07_amry_custom_star_target_top{n}", patched, best)


def main() -> None:
    frames = read_frames()
    anchor = frames["best_top25"]
    for name in [
        "nina_vote2",
        "amry_one",
        "amry_top2",
        "amry_top3",
        "amry_top5",
        "amry_top8",
        "amry_top13",
        "amry_mlp_raw",
        "deb_saga",
        "cdeotte_v8",
        "kaisei_oof",
        "amry_oof",
    ]:
        if name in frames:
            save(f"jun07_{name}", frames[name], anchor)
    make_amry_candidate_patches(frames)


if __name__ == "__main__":
    main()
