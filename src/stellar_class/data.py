from __future__ import annotations

from pathlib import Path

import pandas as pd

COMPETITION = "playground-series-s6e6"
ID_COL = "id"
TARGET = "class"
CLASSES = ["GALAXY", "QSO", "STAR"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    kaggle_dir = Path("/kaggle/input/playground-series-s6e6")
    if kaggle_dir.exists():
        return kaggle_dir
    return project_root() / "data" / "raw"


def load_data(base_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(base_dir) if base_dir is not None else data_dir()
    train = pd.read_csv(base / "train.csv")
    test = pd.read_csv(base / "test.csv")
    sample_submission = pd.read_csv(base / "sample_submission.csv")
    return train, test, sample_submission

