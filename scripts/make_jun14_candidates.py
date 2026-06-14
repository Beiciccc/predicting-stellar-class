#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "jun14"
TARGET = "class"

SAMPLE = ROOT / "data/raw/sample_submission.csv"
BEST7 = ROOT / "outputs/jun12/jun12_mehran_7row.csv"
RIDGE_DIR = ROOT / "references/kaggle_outputs/jun14_nithin_ridge_flip"

SOURCES = {
    "jun14_nina_vote4_refresh": ROOT / "references/kaggle_outputs/jun14_nina_vote4_refresh/submission.csv",
    "jun14_vlad_097186_final": ROOT / "references/kaggle_outputs/jun14_vlad_final_dataset/submission.csv",
    "jun14_ridge_submission": RIDGE_DIR / "submission.csv",
    "jun14_meenal_refresh": ROOT / "references/kaggle_outputs/jun14_meenal_refresh/submission.csv",
    "jun14_ridge_top20": RIDGE_DIR / "ridge_flip_candidates/ridge_top20.csv",
    "jun14_ridge_top35": RIDGE_DIR / "ridge_flip_candidates/ridge_top35.csv",
    "jun14_ridge_top75": RIDGE_DIR / "ridge_flip_candidates/ridge_top75.csv",
    "jun14_ridge_top100": RIDGE_DIR / "ridge_flip_candidates/ridge_top100.csv",
    "jun14_ridge_top150": RIDGE_DIR / "ridge_flip_candidates/ridge_top150.csv",
    "jun14_ridge_top150_tail10": RIDGE_DIR / "ridge_flip_candidates/ridge_top150_tail10.csv",
    "jun14_ridge_top150_tail20": RIDGE_DIR / "ridge_flip_candidates/ridge_top150_tail20.csv",
    "jun14_ridge_top180": RIDGE_DIR / "ridge_flip_candidates/ridge_top180.csv",
    "jun14_ridge_top180_tail10": RIDGE_DIR / "ridge_flip_candidates/ridge_top180_tail10.csv",
    "jun14_ridge_top180_tail20": RIDGE_DIR / "ridge_flip_candidates/ridge_top180_tail20.csv",
    "jun14_ridge_top220": RIDGE_DIR / "ridge_flip_candidates/ridge_top220.csv",
    "jun14_ridge_top260": RIDGE_DIR / "ridge_flip_candidates/ridge_top260.csv",
}

CUSTOM_CUTOFFS = {
    "jun14_ridge_top90": {
        "base": "jun14_ridge_top75",
        "json": RIDGE_DIR / "ridge_flip_candidates/ridge_top100.json",
        "start": 76,
        "end": 90,
    },
    "jun14_ridge_top110": {
        "base": "jun14_ridge_top100",
        "json": RIDGE_DIR / "ridge_flip_candidates/ridge_top150.json",
        "start": 101,
        "end": 110,
    },
}


def validate(frame: pd.DataFrame, sample: pd.DataFrame, name: str) -> None:
    if list(frame.columns) != ["id", TARGET]:
        raise ValueError(f"{name}: unexpected columns {list(frame.columns)}")
    if not frame["id"].equals(sample["id"]):
        raise ValueError(f"{name}: id order mismatch")


def save(name: str, frame: pd.DataFrame, reference: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.csv"
    frame[["id", TARGET]].to_csv(path, index=False)
    diff = int(frame[TARGET].ne(reference[TARGET]).sum())
    counts = frame[TARGET].value_counts().to_dict()
    print(f"{name}: diff_vs_best7={diff} counts={counts} path={path}")


def apply_rank_slice(base: pd.DataFrame, json_path: Path, start: int, end: int) -> pd.DataFrame:
    with json_path.open() as handle:
        flips = json.load(handle)["flips"]
    patch = {row["id"]: row[TARGET] for row in flips[start - 1 : end]}
    out = base.copy()
    mask = out["id"].isin(patch)
    out.loc[mask, TARGET] = out.loc[mask, "id"].map(patch)
    return out


def main() -> None:
    sample = pd.read_csv(SAMPLE)
    best7 = pd.read_csv(BEST7)
    validate(best7, sample, "best7")
    seen: dict[tuple[str, ...], str] = {}
    for name, source in SOURCES.items():
        frame = pd.read_csv(source)
        validate(frame, sample, name)
        key = tuple(frame[TARGET].astype(str))
        duplicate = seen.get(key)
        save(name, frame, best7)
        if duplicate:
            print(f"{name}: duplicate_of={duplicate}")
        else:
            seen[key] = name
    for name, config in CUSTOM_CUTOFFS.items():
        base = pd.read_csv(OUT_DIR / f"{config['base']}.csv")
        frame = apply_rank_slice(base, config["json"], config["start"], config["end"])
        validate(frame, sample, name)
        key = tuple(frame[TARGET].astype(str))
        duplicate = seen.get(key)
        save(name, frame, best7)
        if duplicate:
            print(f"{name}: duplicate_of={duplicate}")
        else:
            seen[key] = name


if __name__ == "__main__":
    main()
