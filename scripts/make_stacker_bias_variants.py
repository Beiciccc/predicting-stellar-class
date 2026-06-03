#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "references/kaggle_outputs/cdeotte_gpu_lr_stacker"
OUT_DIR = ROOT / "outputs" / "stacker_bias"
CLASSES = ["GALAXY", "QSO", "STAR"]
TARGET = "class"


def score_bias(oof: np.ndarray, y: np.ndarray, bias: np.ndarray) -> float:
    pred = (np.log(np.clip(oof, 1e-12, 1.0)) + bias).argmax(axis=1)
    return balanced_accuracy_score(y, pred)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data/raw/train.csv")
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    base_submission = pd.read_csv(BASE_DIR / "submission.csv")
    oof = np.load(BASE_DIR / "oof_lr_stacker_v1.npy")
    pred = np.load(BASE_DIR / "pred_lr_stacker_v1.npy")
    y = train[TARGET].map({label: idx for idx, label in enumerate(CLASSES)}).to_numpy()

    if not base_submission["id"].equals(sample["id"]):
        raise ValueError("Base submission id order mismatch")

    candidates: list[tuple[float, float, float]] = []
    for qso in np.linspace(-0.08, 0.08, 65):
        for star in np.linspace(-0.08, 0.08, 65):
            bias = np.array([0.0, qso, star])
            candidates.append((score_bias(oof, y, bias), qso, star))
    candidates.sort(reverse=True)

    print("base_oof", score_bias(oof, y, np.zeros(3)))
    print("top_biases")
    seen: set[tuple[int, int]] = set()
    saved = 0
    for score, qso, star in candidates[:200]:
        key = (round(qso * 1000), round(star * 1000))
        if key in seen:
            continue
        seen.add(key)
        bias = np.array([0.0, qso, star])
        labels = [CLASSES[idx] for idx in (np.log(np.clip(pred, 1e-12, 1.0)) + bias).argmax(axis=1)]
        diff = pd.Series(labels).ne(base_submission[TARGET]).sum()
        print(f"score={score:.9f} qso={qso:.4f} star={star:.4f} diff_vs_lr={diff}")
        if saved < 8:
            out = sample.copy()
            out[TARGET] = labels
            out.to_csv(OUT_DIR / f"stacker_bias_q{qso:+.4f}_s{star:+.4f}.csv", index=False)
            saved += 1


if __name__ == "__main__":
    main()
