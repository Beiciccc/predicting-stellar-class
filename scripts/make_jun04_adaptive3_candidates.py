#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "jun04_adaptive3"
TARGET = "class"

BASE = ROOT / "outputs/public_blends/qso_strict3_lr_flex_nina_lgbm_ours.csv"
ANCHOR = ROOT / "outputs/jun04_adaptive2/s37_plus_fachri_nina_deep_lgbm.csv"
CANDIDATES = {
    "new5": ROOT / "outputs/jun04_public_patches/new5_unanimous_patch.csv",
    "six_vote": ROOT / "outputs/jun04_micro/six_vote_realmlp_added.csv",
    "star_overlay": ROOT / "outputs/jun04_micro/star_bias_overlay_lr_rows.csv",
    "mild_vote": ROOT / "outputs/jun04_micro/mild_bias_vote.csv",
    "oof_vote": ROOT / "outputs/jun04_micro/oofbest_bias_vote.csv",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE)
    anchor = pd.read_csv(ANCHOR)

    if not anchor["id"].equals(base["id"]):
        raise ValueError("Anchor id order mismatch")

    for name, path in CANDIDATES.items():
        candidate = pd.read_csv(path)
        if list(candidate.columns) != ["id", TARGET]:
            raise ValueError(f"{name}: unexpected columns {list(candidate.columns)}")
        if not candidate["id"].equals(base["id"]):
            raise ValueError(f"{name}: id order mismatch")

        labels = anchor[TARGET].copy()
        mask = candidate[TARGET].ne(base[TARGET])
        labels.loc[mask] = candidate.loc[mask, TARGET]

        out = pd.DataFrame({"id": anchor["id"], TARGET: labels})
        out_path = OUT_DIR / f"s38_plus_{name}.csv"
        out.to_csv(out_path, index=False)
        print(
            f"{out_path.name}: diff_vs_anchor={int(labels.ne(anchor[TARGET]).sum())} "
            f"diff_vs_base={int(labels.ne(base[TARGET]).sum())} "
            f"counts={labels.value_counts().to_dict()}"
        )


if __name__ == "__main__":
    main()
