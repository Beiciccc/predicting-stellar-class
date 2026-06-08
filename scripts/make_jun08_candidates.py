#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "jun08"
TARGET = "class"

NINA_VOTE2 = ROOT / "references/kaggle_outputs/jun07_nina_simple_vote2/submission.csv"
AMRY_CANDIDATES = ROOT / "references/kaggle_outputs/jun07_amry_meta_patch_97108/gpu_patch_candidates.csv"
AMRY_TOP8 = ROOT / "references/kaggle_outputs/jun07_amry_meta_patch_97108/submission_gpu_meta_top8_patch.csv"
AMRY_TOP13 = ROOT / "references/kaggle_outputs/jun07_amry_meta_patch_97108/submission_gpu_meta_top13_patch.csv"


def read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", TARGET]:
        raise ValueError(f"{path}: unexpected columns {list(frame.columns)}")
    return frame


def load_ranked_candidates() -> pd.DataFrame:
    candidates = pd.read_csv(AMRY_CANDIDATES)
    candidates = candidates[~candidates["already_tried"].astype(bool)].copy()
    candidates = candidates.sort_values("score", ascending=False).reset_index(drop=True)
    candidates["rank"] = candidates.index + 1
    candidates["transition"] = candidates["anchor_class"] + "->" + candidates["gpu_meta_class"]
    return candidates


def apply_rows(base: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    patched = base.copy()
    mapping = rows.set_index("id")["gpu_meta_class"]
    mask = patched["id"].isin(mapping.index)
    if int(mask.sum()) != len(mapping):
        missing = sorted(set(mapping.index) - set(patched.loc[mask, "id"]))
        raise ValueError(f"missing candidate ids: {missing}")
    patched.loc[mask, TARGET] = patched.loc[mask, "id"].map(mapping)
    return patched


def save(name: str, frame: pd.DataFrame, base: pd.DataFrame, top8: pd.DataFrame, top13: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = frame[["id", TARGET]].copy()
    path = OUT_DIR / f"{name}.csv"
    out.to_csv(path, index=False)
    print(
        f"{name}: "
        f"diff_vs_nina={int(out[TARGET].ne(base[TARGET]).sum())} "
        f"diff_vs_top8={int(out[TARGET].ne(top8[TARGET]).sum())} "
        f"diff_vs_top13={int(out[TARGET].ne(top13[TARGET]).sum())} "
        f"path={path}"
    )


def main() -> None:
    base = read_submission(NINA_VOTE2)
    top8 = read_submission(AMRY_TOP8)
    top13 = read_submission(AMRY_TOP13)
    for name, frame in [("top8", top8), ("top13", top13)]:
        if not frame["id"].equals(base["id"]):
            raise ValueError(f"{name}: id order mismatch")

    candidates = load_ranked_candidates()

    for n in [6, 9, 10, 11, 12]:
        save(f"jun08_amry_top{n}", apply_rows(base, candidates.head(n)), base, top8, top13)

    for rank in [6, 7, 8]:
        rows = candidates.head(8)
        rows = rows[rows["rank"] != rank]
        save(f"jun08_amry_top8_drop_r{rank}", apply_rows(base, rows), base, top8, top13)

    for rank in [9, 10, 11, 12, 13]:
        rows = candidates.head(13)
        rows = rows[rows["rank"] != rank]
        save(f"jun08_amry_top13_drop_r{rank}", apply_rows(base, rows), base, top8, top13)

    for rank in [14, 17]:
        rows = pd.concat([candidates.head(13), candidates[candidates["rank"] == rank]], ignore_index=True)
        save(f"jun08_amry_top13_plus_r{rank}", apply_rows(base, rows), base, top8, top13)

    summary_cols = ["rank", "id", "anchor_class", "gpu_meta_class", "transition", "score"]
    print("\nTop candidate rows:")
    print(candidates[summary_cols].head(17).to_string(index=False))


if __name__ == "__main__":
    main()
