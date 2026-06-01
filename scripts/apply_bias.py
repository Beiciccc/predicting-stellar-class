#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["GALAXY", "QSO", "STAR"]


def load_bias(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("bias", data)
    return np.array([float(values[label]) for label in CLASSES], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_proba", type=Path)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, default=Path("data/raw/sample_submission.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    test = pd.read_csv(args.test_proba)
    sample = pd.read_csv(args.sample_submission)
    bias = load_bias(args.bias)
    proba = test[[f"proba_{label}" for label in CLASSES]].to_numpy()
    pred = (np.log(np.clip(proba, 1e-12, 1.0)) + bias).argmax(axis=1)

    submission = sample.copy()
    if not submission["id"].equals(test["id"]):
        raise ValueError("test_proba id order does not match sample_submission")
    submission["class"] = [CLASSES[idx] for idx in pred]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()

