#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

CLASSES = ["GALAXY", "QSO", "STAR"]


def proba_cols() -> list[str]:
    return [f"proba_{label}" for label in CLASSES]


def read_oof(path: Path) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    df = pd.read_csv(path)
    y = df["y_true"].map({label: idx for idx, label in enumerate(CLASSES)}).to_numpy()
    return y, df[proba_cols()].to_numpy(), df["id"]


def read_test(path: Path) -> tuple[np.ndarray, pd.Series]:
    df = pd.read_csv(path)
    return df[proba_cols()].to_numpy(), df["id"]


def random_weights(n: int, trials: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(n), size=trials)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", nargs="+", type=Path, required=True)
    parser.add_argument("--test", nargs="+", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, default=Path("data/raw/sample_submission.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if len(args.oof) != len(args.test):
        raise ValueError("--oof and --test must have the same length")

    y_ref = None
    oof_arrays = []
    for path in args.oof:
        y, proba, ids = read_oof(path)
        if y_ref is None:
            y_ref = y
            id_ref = ids
        elif not np.array_equal(y_ref, y) or not id_ref.equals(ids):
            raise ValueError(f"OOF alignment mismatch: {path}")
        oof_arrays.append(proba)

    test_arrays = []
    test_id_ref = None
    for path in args.test:
        proba, ids = read_test(path)
        if test_id_ref is None:
            test_id_ref = ids
        elif not test_id_ref.equals(ids):
            raise ValueError(f"Test alignment mismatch: {path}")
        test_arrays.append(proba)

    oof_stack = np.stack(oof_arrays)
    test_stack = np.stack(test_arrays)
    best_score = -1.0
    best_w = None
    for weights in random_weights(len(args.oof), args.trials, args.seed):
        blended = np.tensordot(weights, oof_stack, axes=(0, 0))
        score = balanced_accuracy_score(y_ref, blended.argmax(axis=1))
        if score > best_score:
            best_score = score
            best_w = weights

    blended_test = np.tensordot(best_w, test_stack, axes=(0, 0))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.DataFrame(blended_test, columns=proba_cols())
    test_df.insert(0, "id", test_id_ref.to_numpy())
    test_df.to_csv(args.output_dir / "test_proba.csv", index=False)

    sample = pd.read_csv(args.sample_submission)
    if not sample["id"].equals(test_id_ref):
        raise ValueError("sample_submission id order does not match test proba")
    submission = sample.copy()
    submission["class"] = [CLASSES[idx] for idx in blended_test.argmax(axis=1)]
    submission.to_csv(args.output_dir / "submission.csv", index=False)

    metrics = {
        "oof_balanced_accuracy": float(best_score),
        "weights": [float(x) for x in best_w],
        "oof_files": [str(path) for path in args.oof],
        "test_files": [str(path) for path in args.test],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

