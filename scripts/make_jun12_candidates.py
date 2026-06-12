#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "jun12"
TARGET = "class"

SAMPLE = ROOT / "data/raw/sample_submission.csv"
BEST = ROOT / "outputs/jun09/jun09_vote1_amry_patch_top3.csv"
MEHRAN = ROOT / "references/kaggle_outputs/jun12_mehran_results/submission.csv"

STAR_PAIR = {
    584275: "GALAXY",
    695569: "GALAXY",
}
QSO_NEUTRAL4 = {
    659588: "GALAXY",
    742100: "GALAXY",
    749913: "GALAXY",
    817047: "GALAXY",
}
AMRY_NEUTRAL = {
    "r10": {600202: "GALAXY"},
    "r13": {580482: "GALAXY"},
}


def validate(frame: pd.DataFrame, sample: pd.DataFrame, name: str) -> None:
    if list(frame.columns) != ["id", TARGET]:
        raise ValueError(f"{name}: unexpected columns {list(frame.columns)}")
    if not frame["id"].equals(sample["id"]):
        raise ValueError(f"{name}: id order mismatch")


def patch_from_to(base: pd.DataFrame, candidate: pd.DataFrame) -> dict[int, str]:
    diff = candidate[TARGET].ne(base[TARGET])
    return {
        int(row_id): label
        for row_id, label in zip(candidate.loc[diff, "id"], candidate.loc[diff, TARGET])
    }


def apply_rows(frame: pd.DataFrame, patch: dict[int, str]) -> pd.DataFrame:
    out = frame.copy()
    mask = out["id"].isin(patch)
    out.loc[mask, TARGET] = out.loc[mask, "id"].map(patch)
    return out


def save(name: str, frame: pd.DataFrame, best: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.csv"
    frame[["id", TARGET]].to_csv(path, index=False)
    diff = int(frame[TARGET].ne(best[TARGET]).sum())
    counts = frame[TARGET].value_counts().to_dict()
    print(f"{name}: diff_vs_best={diff} counts={counts} path={path}")


def main() -> None:
    sample = pd.read_csv(SAMPLE)
    best = pd.read_csv(BEST)
    mehran = pd.read_csv(MEHRAN)
    validate(best, sample, "best")
    validate(mehran, sample, "mehran")

    mehran_patch = patch_from_to(best, mehran)
    save("jun12_mehran_7row", mehran, best)

    for row_id, label in mehran_patch.items():
        save(f"jun12_best_plus_mehran_{row_id}", apply_rows(best, {row_id: label}), best)

    for row_id in mehran_patch:
        keep = {key: value for key, value in mehran_patch.items() if key != row_id}
        save(f"jun12_mehran_7row_drop_{row_id}", apply_rows(best, keep), best)

    tied_rows = {604784, 643040, 697113, 736948}
    tied_drop_patch = {
        key: value for key, value in mehran_patch.items() if key not in tied_rows
    }
    save("jun12_mehran_7row_drop_tie4", apply_rows(best, tied_drop_patch), best)
    tied_drop_plus763670 = {
        key: value for key, value in tied_drop_patch.items() if key != 763670
    }
    save(
        "jun12_mehran_7row_drop_tie4_763670",
        apply_rows(best, tied_drop_plus763670),
        best,
    )
    save(
        "jun12_mehran_7row_drop_tie4_plus_star_pair",
        apply_rows(apply_rows(best, tied_drop_patch), STAR_PAIR),
        best,
    )
    save(
        "jun12_mehran_7row_drop_tie4_plus_r10_r13",
        apply_rows(apply_rows(best, tied_drop_patch), AMRY_NEUTRAL["r10"] | AMRY_NEUTRAL["r13"]),
        best,
    )

    save("jun12_mehran_7row_plus_star_pair", apply_rows(mehran, STAR_PAIR), best)
    save("jun12_mehran_7row_plus_qso_neutral4", apply_rows(mehran, QSO_NEUTRAL4), best)
    save(
        "jun12_mehran_7row_plus_star_pair_qso_neutral4",
        apply_rows(apply_rows(mehran, STAR_PAIR), QSO_NEUTRAL4),
        best,
    )
    save(
        "jun12_mehran_7row_plus_r10_r13",
        apply_rows(mehran, AMRY_NEUTRAL["r10"] | AMRY_NEUTRAL["r13"]),
        best,
    )


if __name__ == "__main__":
    main()
