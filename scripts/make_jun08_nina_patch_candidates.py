#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "jun08"
TARGET = "class"

NINA_97122 = ROOT / "references/kaggle_outputs/jun08_nina_ps_s6e6/0.97122.csv"
AMRY_CANDIDATES = ROOT / "references/kaggle_outputs/jun07_amry_meta_patch_97108/gpu_patch_candidates.csv"


def save(name: str, frame: pd.DataFrame, base: pd.DataFrame, rows: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.csv"
    frame[["id", TARGET]].to_csv(path, index=False)
    print(f"{name}: changed={int(frame[TARGET].ne(base[TARGET]).sum())} path={path}")
    print(rows[["rank", "id", "current_class", "gpu_meta_class", "score"]].to_string(index=False))


def apply_rows(base: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    mapping = rows.set_index("id")["gpu_meta_class"]
    mask = out["id"].isin(mapping.index)
    out.loc[mask, TARGET] = out.loc[mask, "id"].map(mapping)
    return out


def main() -> None:
    base = pd.read_csv(NINA_97122)
    if list(base.columns) != ["id", TARGET]:
        raise ValueError(f"{NINA_97122}: unexpected columns {list(base.columns)}")

    candidates = pd.read_csv(AMRY_CANDIDATES)
    candidates = candidates[~candidates["already_tried"].astype(bool)].copy()
    candidates = candidates.sort_values("score", ascending=False).reset_index(drop=True)
    candidates["rank"] = candidates.index + 1
    current = base.set_index("id")[TARGET]
    candidates["current_class"] = candidates["id"].map(current)
    candidates = candidates[candidates["current_class"] != candidates["gpu_meta_class"]].copy()

    rank46 = candidates[candidates["rank"] == 46]
    save("jun08_nina97122_amry_rank46", apply_rows(base, rank46), base, rank46)

    top100_new = candidates[candidates["rank"] <= 100]
    save("jun08_nina97122_amry_top100_new4", apply_rows(base, top100_new), base, top100_new)


if __name__ == "__main__":
    main()
