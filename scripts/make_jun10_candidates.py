#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "jun10"
TARGET = "class"

SAMPLE = ROOT / "data/raw/sample_submission.csv"
NINA_DATA_JUN10 = ROOT / "references/kaggle_outputs/jun10_nina_ps_s6e6"
NINA_DATA_JUN08 = ROOT / "references/kaggle_outputs/jun08_nina_ps_s6e6"
VOTE1 = ROOT / "references/kaggle_outputs/jun09_nina_simple_vote1/submission.csv"
AMRY_NOTEBOOK = (
    ROOT
    / "references/kaggle_code/jun09_amry_97123_micro_patch"
    / "s6e6-0-97123-hard-proba-micro-patch-explained.ipynb"
)

MICRO_TOP3 = {
    755752: "GALAXY",
    676483: "GALAXY",
    665223: "GALAXY",
}

GRID_PATCH_ROWS = {
    614916: "GALAXY",
    703838: "QSO",
}

FILES = {
    "047b": "0.97047.b.csv",
    "101": "0.97101.csv",
    "108": "0.97108.csv",
    "111": "0.97111.csv",
    "120": "0.97120.csv",
    "122": "0.97122.csv",
    "126": "0.97126.csv",
}


def read_submission_file(filename: str) -> pd.DataFrame:
    for base in [NINA_DATA_JUN10, NINA_DATA_JUN08]:
        path = base / filename
        if path.exists():
            frame = pd.read_csv(path)
            if list(frame.columns) != ["id", TARGET]:
                raise ValueError(f"{path}: unexpected columns {list(frame.columns)}")
            return frame
    raise FileNotFoundError(filename)


def vote5(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = frames[-1][["id", TARGET]].copy()
    labels = pd.concat([frame[TARGET].rename(str(i)) for i, frame in enumerate(frames)], axis=1)
    voted = []
    for row in labels.itertuples(index=False, name=None):
        value_counts = pd.Series(row).value_counts()
        if int(value_counts.iloc[0]) >= 4:
            voted.append(value_counts.index[0])
        else:
            voted.append(row[-1])
    out[TARGET] = voted
    return out


def vote6_overlay(base: pd.DataFrame, frames: list[pd.DataFrame], threshold: int) -> pd.DataFrame:
    out = base[["id", TARGET]].copy()
    labels = pd.concat([frame[TARGET].rename(str(i)) for i, frame in enumerate(frames)], axis=1)
    voted = []
    for current, row in zip(out[TARGET], labels.itertuples(index=False, name=None)):
        value_counts = pd.Series(row).value_counts()
        if int(value_counts.iloc[0]) >= threshold:
            voted.append(value_counts.index[0])
        else:
            voted.append(current)
    out[TARGET] = voted
    return out


def protected_anchor(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = frames[-1][["id", TARGET]].copy()
    companions = pd.concat([frame[TARGET].rename(str(i)) for i, frame in enumerate(frames[:-1])], axis=1)
    unanimous = companions.nunique(axis=1).eq(1)
    out.loc[unanimous, TARGET] = companions.loc[unanimous, "3"].to_numpy()
    return out


def apply_patch_rows(frame: pd.DataFrame, patch: dict[int, str]) -> pd.DataFrame:
    out = frame.copy()
    mask = out["id"].isin(patch)
    out.loc[mask, TARGET] = out.loc[mask, "id"].map(patch)
    return out


def load_amry_vote1_patch_rows(base: pd.DataFrame) -> pd.DataFrame:
    notebook = json.loads(AMRY_NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    module = ast.parse(source)
    rows = None
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "PATCH_ROWS":
                rows = ast.literal_eval(node.value)
                break
    if rows is None:
        raise ValueError("PATCH_ROWS not found")
    patch = pd.DataFrame(rows)
    current = base.set_index("id")[TARGET]
    patch["current_vote1"] = patch["id"].map(current)
    patch = patch[patch["current_vote1"] != patch["patched_class"]].copy()
    patch = patch.sort_values(
        ["proba_strong_count", "mean_margin", "min_margin"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return patch


def apply_rows(frame: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    mapping = rows.set_index("id")["patched_class"]
    mask = out["id"].isin(mapping.index)
    out.loc[mask, TARGET] = out.loc[mask, "id"].map(mapping)
    return out


def save(name: str, frame: pd.DataFrame, best: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.csv"
    frame[["id", TARGET]].to_csv(path, index=False)
    print(
        f"{name}: diff_vs_best={int(frame[TARGET].ne(best[TARGET]).sum())} "
        f"counts={frame[TARGET].value_counts().to_dict()} path={path}"
    )


def main() -> None:
    sample = pd.read_csv(SAMPLE)
    base_files = {key: read_submission_file(filename) for key, filename in FILES.items()}
    for key, frame in base_files.items():
        if not frame["id"].equals(sample["id"]):
            raise ValueError(f"{key}: id order mismatch")

    best = pd.read_csv(ROOT / "outputs/jun09/jun09_vote1_amry_patch_top3.csv")
    vote1 = pd.read_csv(VOTE1)
    patch_rows = load_amry_vote1_patch_rows(vote1)

    original_frames = [base_files[key] for key in ["047b", "101", "108", "111", "122"]]
    replacements = {
        "replace_047b_with_126": ["126", "101", "108", "111", "122"],
        "replace_101_with_126": ["047b", "126", "108", "111", "122"],
        "replace_108_with_126": ["047b", "101", "126", "111", "122"],
        "replace_111_with_126": ["047b", "101", "108", "126", "122"],
        "anchor_126": ["047b", "101", "108", "111", "126"],
    }
    for name, keys in replacements.items():
        frame = vote5([base_files[key] for key in keys])
        save(f"jun10_vote5_{name}_micro_top3", apply_patch_rows(frame, MICRO_TOP3), best)

    save(
        "jun10_vote6_add126_threshold4_micro_top3",
        apply_patch_rows(vote6_overlay(vote1, original_frames + [base_files["126"]], threshold=4), MICRO_TOP3),
        best,
    )
    save(
        "jun10_vote7_add120_126_threshold5_micro_top3",
        apply_patch_rows(
            vote6_overlay(vote1, original_frames + [base_files["120"], base_files["126"]], threshold=5),
            MICRO_TOP3,
        ),
        best,
    )
    save(
        "jun10_protected_anchor_original_micro_top3",
        apply_patch_rows(protected_anchor(original_frames), MICRO_TOP3),
        best,
    )

    top3 = patch_rows.head(3)
    top7 = patch_rows.head(7)
    for row_number in [4, 5, 6, 7]:
        rows = pd.concat([top3, patch_rows.iloc[[row_number - 1]]], ignore_index=True)
        save(f"jun10_vote1_top3_plus_r{row_number}", apply_rows(vote1, rows), best)
    rows = pd.concat([top7, patch_rows.iloc[[8]], patch_rows.iloc[[9]]], ignore_index=True)
    save("jun10_vote1_top7_plus_r9_r10", apply_rows(vote1, rows), best)
    rows = pd.concat([top7, patch_rows.iloc[[8]], patch_rows.iloc[[9]], patch_rows.iloc[[12]], patch_rows.iloc[[15]]], ignore_index=True)
    save("jun10_vote1_top7_plus_r9_r10_r13_r16", apply_rows(vote1, rows), best)

    for row_number in [4, 6]:
        rows = top7.drop(index=patch_rows.index[row_number - 1]).reset_index(drop=True)
        save(f"jun10_vote1_top7_drop_r{row_number}", apply_rows(vote1, rows), best)

    top7_frame = apply_rows(vote1, top7)
    save("jun10_vote1_top7_plus_grid614916", apply_patch_rows(top7_frame, {614916: GRID_PATCH_ROWS[614916]}), best)
    save("jun10_vote1_top7_plus_grid614916_grid703838", apply_patch_rows(top7_frame, GRID_PATCH_ROWS), best)


if __name__ == "__main__":
    main()
