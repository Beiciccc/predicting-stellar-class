#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stellar_class.data import CLASSES, ID_COL, TARGET, load_data
from stellar_class.features import (
    CAT_COLS,
    add_features,
    reconstructed_galaxy_population,
    reconstructed_spectral_type,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--external-weight", type=float, default=0.25)
    parser.add_argument("--external-max-rows", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--external-path", type=Path, default=Path("data/external/sdss17/star_classification.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_external(path: Path, max_rows: int, seed: int) -> pd.DataFrame:
    ext = pd.read_csv(path)
    keep = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift", TARGET]
    ext = ext[keep].copy()
    ext = ext[ext[TARGET].isin(CLASSES)].reset_index(drop=True)
    ext["spectral_type"] = reconstructed_spectral_type(ext["g"], ext["r"])
    ext["galaxy_population"] = reconstructed_galaxy_population(ext["u"], ext["r"])
    ext.insert(0, ID_COL, -np.arange(1, len(ext) + 1))
    if max_rows > 0 and max_rows < len(ext):
        ext = ext.sample(max_rows, random_state=seed).reset_index(drop=True)
    return ext


def design_matrices(comp: pd.DataFrame, ext: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comp_x = add_features(comp.drop(columns=[TARGET], errors="ignore"))
    ext_x = add_features(ext.drop(columns=[TARGET], errors="ignore"))
    test_x = add_features(test.drop(columns=[TARGET], errors="ignore"))
    combined = pd.concat([comp_x, ext_x, test_x], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=CAT_COLS, dummy_na=False)
    combined = combined.drop(columns=[ID_COL], errors="ignore")
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)
    n_comp = len(comp)
    n_ext = len(ext)
    return (
        combined.iloc[:n_comp].reset_index(drop=True),
        combined.iloc[n_comp : n_comp + n_ext].reset_index(drop=True),
        combined.iloc[n_comp + n_ext :].reset_index(drop=True),
    )


def build_model(seed: int, n_estimators: int) -> LGBMClassifier:
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comp, test, sample_submission = load_data(args.data_dir)
    ext = load_external(args.external_path, args.external_max_rows, args.seed)

    comp_x, ext_x, test_x = design_matrices(comp, ext, test)
    class_to_idx = {label: idx for idx, label in enumerate(CLASSES)}
    y = comp[TARGET].map(class_to_idx).to_numpy()
    y_ext = ext[TARGET].map(class_to_idx).to_numpy()

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros((len(comp), len(CLASSES)), dtype=float)
    test_proba = np.zeros((len(test), len(CLASSES)), dtype=float)
    fold_ids = np.full(len(comp), -1, dtype=int)
    fold_scores = []

    for fold, (trn_idx, val_idx) in enumerate(splitter.split(comp_x, y), start=1):
        train_x = pd.concat([comp_x.iloc[trn_idx], ext_x], axis=0, ignore_index=True)
        train_y = np.concatenate([y[trn_idx], y_ext])
        sample_weight = np.concatenate([
            np.ones(len(trn_idx), dtype=float),
            np.full(len(ext), args.external_weight, dtype=float),
        ])
        model = build_model(args.seed + fold, args.n_estimators)
        model.fit(train_x, train_y, sample_weight=sample_weight)
        val_proba = model.predict_proba(comp_x.iloc[val_idx])
        oof[val_idx] = val_proba
        fold_ids[val_idx] = fold
        test_proba += model.predict_proba(test_x) / args.folds
        score = balanced_accuracy_score(y[val_idx], val_proba.argmax(axis=1))
        fold_scores.append(float(score))
        print(f"fold={fold} balanced_accuracy={score:.6f}", flush=True)

    pred = oof.argmax(axis=1)
    oof_score = balanced_accuracy_score(y, pred)
    recalls = recall_score(y, pred, average=None, labels=list(range(len(CLASSES))))
    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES))))
    proba_cols = [f"proba_{label}" for label in CLASSES]

    oof_df = pd.DataFrame(oof, columns=proba_cols)
    oof_df.insert(0, "pred", [CLASSES[idx] for idx in pred])
    oof_df.insert(0, "fold", fold_ids)
    oof_df.insert(0, "y_true", comp[TARGET].to_numpy())
    oof_df.insert(0, ID_COL, comp[ID_COL].to_numpy())
    oof_df.to_csv(args.output_dir / "oof_predictions.csv", index=False)

    test_df = pd.DataFrame(test_proba, columns=proba_cols)
    test_df.insert(0, ID_COL, test[ID_COL].to_numpy())
    test_df.to_csv(args.output_dir / "test_proba.csv", index=False)

    submission = sample_submission.copy()
    submission[TARGET] = [CLASSES[idx] for idx in test_proba.argmax(axis=1)]
    submission.to_csv(args.output_dir / "submission.csv", index=False)

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "lgbm_sdss17",
        "seed": args.seed,
        "folds": args.folds,
        "n_estimators": args.n_estimators,
        "external_rows": len(ext),
        "external_weight": args.external_weight,
        "feature_count": int(comp_x.shape[1]),
        "fold_balanced_accuracy": fold_scores,
        "oof_balanced_accuracy": float(oof_score),
        "recall_by_class": {label: float(value) for label, value in zip(CLASSES, recalls)},
        "confusion_matrix": cm.tolist(),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

