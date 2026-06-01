from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CAT_COLS = ["spectral_type", "galaxy_population"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    color_pairs = [
        ("u", "g"),
        ("g", "r"),
        ("r", "i"),
        ("i", "z"),
        ("u", "r"),
        ("g", "i"),
        ("r", "z"),
        ("u", "z"),
    ]
    for left, right in color_pairs:
        out[f"{left}_minus_{right}"] = out[left] - out[right]

    out["redshift_abs"] = out["redshift"].abs()
    out["redshift_log1p_abs"] = np.log1p(out["redshift_abs"])
    out["u_minus_r_x_redshift"] = out["u_minus_r"] * out["redshift"]
    out["g_minus_i_x_redshift"] = out["g_minus_i"] * out["redshift"]
    return out


def make_design_matrices(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = add_features(train.drop(columns=["class"], errors="ignore"))
    test_x = add_features(test.drop(columns=["class"], errors="ignore"))
    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=CAT_COLS, dummy_na=False)
    combined = combined.drop(columns=["id"], errors="ignore")
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)
    train_out = combined.iloc[: len(train)].reset_index(drop=True)
    test_out = combined.iloc[len(train) :].reset_index(drop=True)
    return train_out, test_out


def reconstructed_spectral_type(g: pd.Series, r: pd.Series) -> pd.Series:
    return pd.cut(
        r - g,
        [-np.inf, -1, -0.5, 0, np.inf],
        labels=["M", "G/K", "A/F", "O/B"],
    ).astype(str)


def reconstructed_galaxy_population(u: pd.Series, r: pd.Series) -> pd.Series:
    return pd.cut(
        u - r,
        [-np.inf, 2.2, np.inf],
        labels=["Blue_Cloud", "Red_Sequence"],
    ).astype(str)

