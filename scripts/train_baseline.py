#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stellar_class.data import CLASSES, ID_COL, TARGET, load_data
from stellar_class.features import make_design_matrices


def build_model(name: str, seed: int, n_estimators: int):
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            num_class=len(CLASSES),
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=96,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=0.2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASSES),
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "cat":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="MultiClass",
            iterations=n_estimators,
            learning_rate=0.05,
            depth=8,
            random_seed=seed,
            verbose=False,
        )
    raise ValueError(f"Unknown model: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgbm", "xgb", "cat"], default="lgbm")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=0, help="Use a stratified sample for smoke tests.")
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lgbm_baseline"))
    return parser.parse_args()


def stratified_sample(train: pd.DataFrame, sample: int, seed: int) -> pd.DataFrame:
    if sample <= 0 or sample >= len(train):
        return train
    parts = []
    for _, group in train.groupby(TARGET):
        n = max(1, int(round(sample * len(group) / len(train))))
        parts.append(group.sample(n=min(n, len(group)), random_state=seed))
    return pd.concat(parts, axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, test, sample_submission = load_data(args.data_dir)
    train = stratified_sample(train, args.sample, args.seed)

    train_x, test_x = make_design_matrices(train, test)
    y = train[TARGET].map({label: idx for idx, label in enumerate(CLASSES)}).to_numpy()

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(train), len(CLASSES)), dtype=float)
    test_proba = np.zeros((len(test), len(CLASSES)), dtype=float)
    fold_scores = []
    fold_ids = np.full(len(train), -1, dtype=int)

    for fold, (trn_idx, val_idx) in enumerate(splitter.split(train_x, y), start=1):
        model = build_model(args.model, args.seed + fold, args.n_estimators)
        model.fit(train_x.iloc[trn_idx], y[trn_idx])
        val_proba = model.predict_proba(train_x.iloc[val_idx])
        fold_ids[val_idx] = fold
        oof[val_idx] = val_proba
        test_proba += model.predict_proba(test_x) / args.folds
        score = balanced_accuracy_score(y[val_idx], val_proba.argmax(axis=1))
        fold_scores.append(float(score))
        print(f"fold={fold} balanced_accuracy={score:.6f}")

    pred = oof.argmax(axis=1)
    oof_score = balanced_accuracy_score(y, pred)
    recalls = recall_score(y, pred, average=None, labels=list(range(len(CLASSES))))
    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES))))

    proba_cols = [f"proba_{label}" for label in CLASSES]
    oof_df = pd.DataFrame(oof, columns=proba_cols)
    oof_df.insert(0, "pred", [CLASSES[idx] for idx in pred])
    oof_df.insert(0, "fold", fold_ids)
    oof_df.insert(0, "y_true", train[TARGET].to_numpy())
    oof_df.insert(0, ID_COL, train[ID_COL].to_numpy())
    oof_df.to_csv(args.output_dir / "oof_predictions.csv", index=False)

    test_df = pd.DataFrame(test_proba, columns=proba_cols)
    test_df.insert(0, ID_COL, test[ID_COL].to_numpy())
    test_df.to_csv(args.output_dir / "test_proba.csv", index=False)

    submission = sample_submission.copy()
    submission[TARGET] = [CLASSES[idx] for idx in test_proba.argmax(axis=1)]
    submission.to_csv(args.output_dir / "submission.csv", index=False)

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "seed": args.seed,
        "folds": args.folds,
        "n_estimators": args.n_estimators,
        "sample_rows": len(train),
        "feature_count": int(train_x.shape[1]),
        "fold_balanced_accuracy": fold_scores,
        "oof_balanced_accuracy": float(oof_score),
        "recall_by_class": {label: float(value) for label, value in zip(CLASSES, recalls)},
        "confusion_matrix": cm.tolist(),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

