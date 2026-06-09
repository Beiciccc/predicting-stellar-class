#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "jun09"
TARGET = "class"

VOTE1 = ROOT / "references/kaggle_outputs/jun09_nina_simple_vote1/submission.csv"
AMRY_NOTEBOOK = (
    ROOT
    / "references/kaggle_code/jun09_amry_97123_micro_patch"
    / "s6e6-0-97123-hard-proba-micro-patch-explained.ipynb"
)


def load_patch_rows() -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def save(name: str, base: pd.DataFrame, rows: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = base.copy()
    mapping = rows.set_index("id")["patched_class"]
    mask = out["id"].isin(mapping.index)
    out.loc[mask, TARGET] = out.loc[mask, "id"].map(mapping)
    path = OUT_DIR / f"{name}.csv"
    out.to_csv(path, index=False)
    print(f"{name}: changed={int(out[TARGET].ne(base[TARGET]).sum())} path={path}")
    print(
        rows[
            [
                "id",
                "current_vote1",
                "patched_class",
                "proba_strong_count",
                "mean_margin",
                "min_margin",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    base = pd.read_csv(VOTE1)
    if list(base.columns) != ["id", TARGET]:
        raise ValueError(f"{VOTE1}: unexpected columns {list(base.columns)}")

    patch = load_patch_rows()
    current = base.set_index("id")[TARGET]
    patch["current_vote1"] = patch["id"].map(current)
    patch = patch[patch["current_vote1"] != patch["patched_class"]].copy()
    patch = patch.sort_values(
        ["proba_strong_count", "mean_margin", "min_margin"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    for n in [1, 3, 5, 7, 10, 13, 15, 16]:
        save(f"jun09_vote1_amry_patch_top{n}", base, patch.head(n))

    top7 = patch.head(7)
    for row_number in [8, 9, 10]:
        rows = pd.concat([top7, patch.iloc[[row_number - 1]]], ignore_index=True)
        save(f"jun09_vote1_amry_patch_top7_plus_r{row_number}", base, rows)


if __name__ == "__main__":
    main()
