#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

CLASSES = ["GALAXY", "QSO", "STAR"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oof", type=Path)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.oof)
    y = df["y_true"].map({label: idx for idx, label in enumerate(CLASSES)}).to_numpy()
    proba = df[[f"proba_{label}" for label in CLASSES]].to_numpy()
    logp = np.log(np.clip(proba, 1e-12, 1.0))

    rng = np.random.default_rng(args.seed)
    best_score = balanced_accuracy_score(y, proba.argmax(axis=1))
    best_bias = np.zeros(len(CLASSES), dtype=float)

    for _ in range(args.trials):
        bias = rng.normal(0.0, args.scale, size=len(CLASSES))
        bias -= bias[0]
        score = balanced_accuracy_score(y, (logp + bias).argmax(axis=1))
        if score > best_score:
            best_score = score
            best_bias = bias

    payload = {
        "classes": CLASSES,
        "best_balanced_accuracy": float(best_score),
        "bias": {label: float(value) for label, value in zip(CLASSES, best_bias)},
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

