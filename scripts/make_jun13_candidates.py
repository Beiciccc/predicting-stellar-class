#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "jun13"
TARGET = "class"

SAMPLE = ROOT / "data/raw/sample_submission.csv"
BEST7 = ROOT / "outputs/jun12/jun12_mehran_7row.csv"
CORE3 = ROOT / "outputs/jun12/jun12_mehran_7row_drop_tie4.csv"
MEENAL = ROOT / "references/kaggle_outputs/jun13_meenal_ensemble/submission.csv"
NINA4 = ROOT / "references/kaggle_outputs/jun13_nina_vote4/submission.csv"

MEENAL4 = {
    672821: "GALAXY",
    735431: "GALAXY",
    785671: "GALAXY",
    795768: "GALAXY",
}
MEENAL_STAR2 = {
    672821: "GALAXY",
    795768: "GALAXY",
}
MEENAL_QSO2 = {
    735431: "GALAXY",
    785671: "GALAXY",
}
AMRY_NEUTRAL = {
    600202: "GALAXY",
    580482: "GALAXY",
}
QSO_NEUTRAL4 = {
    659588: "GALAXY",
    742100: "GALAXY",
    749913: "GALAXY",
    817047: "GALAXY",
}


def validate(frame: pd.DataFrame, sample: pd.DataFrame, name: str) -> None:
    if list(frame.columns) != ["id", TARGET]:
        raise ValueError(f"{name}: unexpected columns {list(frame.columns)}")
    if not frame["id"].equals(sample["id"]):
        raise ValueError(f"{name}: id order mismatch")


def apply_rows(frame: pd.DataFrame, patch: dict[int, str]) -> pd.DataFrame:
    out = frame.copy()
    mask = out["id"].isin(patch)
    out.loc[mask, TARGET] = out.loc[mask, "id"].map(patch)
    return out


def save(name: str, frame: pd.DataFrame, reference: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.csv"
    frame[["id", TARGET]].to_csv(path, index=False)
    diff = int(frame[TARGET].ne(reference[TARGET]).sum())
    counts = frame[TARGET].value_counts().to_dict()
    print(f"{name}: diff_vs_best7={diff} counts={counts} path={path}")


def main() -> None:
    sample = pd.read_csv(SAMPLE)
    best7 = pd.read_csv(BEST7)
    core3 = pd.read_csv(CORE3)
    meenal = pd.read_csv(MEENAL)
    nina4 = pd.read_csv(NINA4)
    for name, frame in {
        "best7": best7,
        "core3": core3,
        "meenal": meenal,
        "nina4": nina4,
    }.items():
        validate(frame, sample, name)

    save("jun13_meenal4_direct", meenal, best7)
    for row_id, label in MEENAL4.items():
        save(f"jun13_best7_plus_meenal_{row_id}", apply_rows(best7, {row_id: label}), best7)
    save("jun13_best7_plus_meenal_star2", apply_rows(best7, MEENAL_STAR2), best7)
    save("jun13_best7_plus_meenal_qso2", apply_rows(best7, MEENAL_QSO2), best7)
    save("jun13_best7_drop_676483", nina4, best7)
    save("jun13_core3_plus_meenal4", apply_rows(core3, MEENAL4), best7)
    save("jun13_best7_plus_r10_r13", apply_rows(best7, AMRY_NEUTRAL), best7)
    save("jun13_best7_plus_qso_neutral4", apply_rows(best7, QSO_NEUTRAL4), best7)


if __name__ == "__main__":
    main()
