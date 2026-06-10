#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "jun11"
TARGET = "class"

BEST = ROOT / "outputs/jun09/jun09_vote1_amry_patch_top3.csv"
SAMPLE = ROOT / "data/raw/sample_submission.csv"
NINA_JUN11 = ROOT / "references/kaggle_outputs/jun11_nina_ps_s6e6"
NINA_JUN10 = ROOT / "references/kaggle_outputs/jun10_nina_ps_s6e6"

MICRO_TOP3 = {
    755752: "GALAXY",
    676483: "GALAXY",
    665223: "GALAXY",
}

ROW_PATCHES = {
    "r4": {795768: "GALAXY"},
    "r5": {776857: "GALAXY"},
    "r6": {646787: "GALAXY"},
    "r7": {595581: "GALAXY"},
    "r10": {600202: "GALAXY"},
    "r13": {580482: "GALAXY"},
    "nina_742100": {742100: "GALAXY"},
    "nina_584275": {584275: "GALAXY"},
}

NINA5_QSO_GAL_NOVEL5 = {
    659588: "GALAXY",
    736763: "GALAXY",
    742100: "GALAXY",
    749913: "GALAXY",
    817047: "GALAXY",
}
NINA5_STAR_GAL_NOVEL2 = {
    584275: "GALAXY",
    695569: "GALAXY",
}
NINA5_OTHER_NOVEL3 = {
    643556: "QSO",
    717110: "STAR",
    806582: "STAR",
}
NINA5_GAL_SUPPORT_PLUS3 = {
    605444: "GALAXY",
    608417: "GALAXY",
    616567: "GALAXY",
}


def validate(frame: pd.DataFrame, sample: pd.DataFrame, name: str) -> None:
    if list(frame.columns) != ["id", TARGET]:
        raise ValueError(f"{name}: unexpected columns {list(frame.columns)}")
    if not frame["id"].equals(sample["id"]):
        raise ValueError(f"{name}: id order mismatch")


def apply_patch_rows(frame: pd.DataFrame, patch: dict[int, str]) -> pd.DataFrame:
    out = frame.copy()
    mask = out["id"].isin(patch)
    out.loc[mask, TARGET] = out.loc[mask, "id"].map(patch)
    return out


def majority_vote(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = frames[0][["id", TARGET]].copy()
    labels = pd.concat([frame[TARGET].rename(str(i)) for i, frame in enumerate(frames)], axis=1)
    voted = []
    for row in labels.itertuples(index=False, name=None):
        counts = Counter(row)
        voted.append(counts.most_common(1)[0][0])
    out[TARGET] = voted
    return out


def unanimous_patch(
    base: pd.DataFrame,
    frames: list[pd.DataFrame],
    protected_ids: set[int],
    target_label: str | None = None,
) -> pd.DataFrame:
    patch: dict[int, str] = {}
    labels = pd.concat([frame[TARGET].rename(str(i)) for i, frame in enumerate(frames)], axis=1)
    for row_id, current, row in zip(base["id"], base[TARGET], labels.itertuples(index=False, name=None)):
        if int(row_id) in protected_ids:
            continue
        if len(set(row)) != 1:
            continue
        label = row[0]
        if label == current:
            continue
        if target_label is not None and label != target_label:
            continue
        patch[int(row_id)] = label
    return apply_patch_rows(base, patch)


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
    validate(best, sample, "best")

    nina_files = {
        "97130": pd.read_csv(NINA_JUN11 / "0.97130.csv"),
        "97129": pd.read_csv(NINA_JUN11 / "0.97129.csv"),
        "97126b": pd.read_csv(NINA_JUN11 / "0.97126.b.csv"),
        "97124": pd.read_csv(NINA_JUN11 / "0.97124.csv"),
    }
    nina_majority_files = {
        "97135": pd.read_csv(NINA_JUN11 / "0.97135.csv"),
        **nina_files,
    }
    nina_consensus_files = {
        "97120": pd.read_csv(NINA_JUN10 / "0.97120.csv"),
        "97126": pd.read_csv(NINA_JUN10 / "0.97126.csv"),
        "97129": nina_files["97129"],
        "97126b": nina_files["97126b"],
        "97124": nina_files["97124"],
    }
    for name, frame in nina_files.items():
        validate(frame, sample, name)
    for name, frame in nina_majority_files.items():
        validate(frame, sample, name)
    for name, frame in nina_consensus_files.items():
        validate(frame, sample, name)

    save("jun11_vote1_top3_plus_r10", apply_patch_rows(best, ROW_PATCHES["r10"]), best)
    save("jun11_vote1_top3_plus_r13", apply_patch_rows(best, ROW_PATCHES["r13"]), best)
    save(
        "jun11_vote1_top3_plus_r10_r13",
        apply_patch_rows(best, ROW_PATCHES["r10"] | ROW_PATCHES["r13"]),
        best,
    )
    save(
        "jun11_vote1_top3_plus_r4_r10_r13",
        apply_patch_rows(best, ROW_PATCHES["r4"] | ROW_PATCHES["r10"] | ROW_PATCHES["r13"]),
        best,
    )
    save("jun11_vote1_top3_plus_742100", apply_patch_rows(best, ROW_PATCHES["nina_742100"]), best)
    save("jun11_vote1_top3_plus_584275", apply_patch_rows(best, ROW_PATCHES["nina_584275"]), best)
    save(
        "jun11_vote1_top3_plus_742100_584275",
        apply_patch_rows(best, ROW_PATCHES["nina_742100"] | ROW_PATCHES["nina_584275"]),
        best,
    )
    top7_patch = ROW_PATCHES["r4"] | ROW_PATCHES["r5"] | ROW_PATCHES["r6"] | ROW_PATCHES["r7"]
    nina5_gal_novel7 = NINA5_QSO_GAL_NOVEL5 | NINA5_STAR_GAL_NOVEL2
    save("jun11_top3_new5_all5_qso_gal_novel5", apply_patch_rows(best, NINA5_QSO_GAL_NOVEL5), best)
    save("jun11_top3_new5_all5_star_gal_novel2", apply_patch_rows(best, NINA5_STAR_GAL_NOVEL2), best)
    save("jun11_top3_new5_all5_gal_novel7", apply_patch_rows(best, nina5_gal_novel7), best)
    save("jun11_top7_drop_r5", apply_patch_rows(best, ROW_PATCHES["r4"] | ROW_PATCHES["r6"] | ROW_PATCHES["r7"]), best)
    save("jun11_top7_drop_r7", apply_patch_rows(best, ROW_PATCHES["r4"] | ROW_PATCHES["r5"] | ROW_PATCHES["r6"]), best)
    save("jun11_top7_drop_r4_r6", apply_patch_rows(best, ROW_PATCHES["r5"] | ROW_PATCHES["r7"]), best)
    save("jun11_top7_new5_all5_gal_novel7_no_r8", apply_patch_rows(best, top7_patch | nina5_gal_novel7), best)
    save(
        "jun11_top3_new5_all5_all_novel10",
        apply_patch_rows(best, nina5_gal_novel7 | NINA5_OTHER_NOVEL3),
        best,
    )
    save(
        "jun11_top3_new5_gal_support5_plus3",
        apply_patch_rows(best, nina5_gal_novel7 | NINA5_GAL_SUPPORT_PLUS3),
        best,
    )
    save(
        "jun11_vote1_top3_plus_nina5_unanimous",
        unanimous_patch(best, list(nina_consensus_files.values()), set(MICRO_TOP3)),
        best,
    )
    save(
        "jun11_vote1_top3_plus_nina5_unanimous_galaxy_only",
        unanimous_patch(best, list(nina_consensus_files.values()), set(MICRO_TOP3), target_label="GALAXY"),
        best,
    )

    for name, frame in nina_files.items():
        save(f"jun11_nina{name}_micro_top3", apply_patch_rows(frame, MICRO_TOP3), best)
        save(
            f"jun11_nina{name}_micro_top7",
            apply_patch_rows(frame, MICRO_TOP3 | ROW_PATCHES["r4"] | {776857: "GALAXY", 646787: "GALAXY", 595581: "GALAXY"}),
            best,
        )

    majority = majority_vote(list(nina_majority_files.values()))
    save("jun11_nina_new5_majority_micro_top3", apply_patch_rows(majority, MICRO_TOP3), best)
    save(
        "jun11_nina_new5_majority_micro_top7",
        apply_patch_rows(majority, MICRO_TOP3 | ROW_PATCHES["r4"] | {776857: "GALAXY", 646787: "GALAXY", 595581: "GALAXY"}),
        best,
    )


if __name__ == "__main__":
    main()
