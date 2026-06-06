from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

CLASSES = ["GALAXY", "QSO", "STAR"]
TARGET = "class"
MODEL_CONFIGS = [
    {"model": "lgbm", "seed": 2024, "n_estimators": 1200, "weight": 1.0},
]
CLASS_BIAS = np.array([0.0, 0.53, 0.92])
PUBLIC_SUBMISSION_SLUGS = {
    "lr": "gpu-logistic-regression-stacker",
    "flex": "blender-is-all-you-need",
    "nina": "ps-s6e6-simple-voting-subsystem",
    "lgbm_cal": "single-lightgbm-lb-0-96728",
    "fachri": "weighted-consensus-patch-0-97047",
    "nina_simple": "ps-s6e6-simple-vote",
    "deeplearn": "attack-of-blenders-on-stellar",
    "ektarr": "ensemble-and-tuning-predicting-stellar-class",
    "stpete": "stellar-class-logistic-stacking",
}


def add_features(df):
    out = df.copy()
    for left, right in [
        ("u", "g"),
        ("g", "r"),
        ("r", "i"),
        ("i", "z"),
        ("u", "r"),
        ("g", "i"),
        ("r", "z"),
        ("u", "z"),
    ]:
        out[f"{left}_minus_{right}"] = out[left] - out[right]
    out["redshift_abs"] = out["redshift"].abs()
    out["redshift_log1p_abs"] = np.log1p(out["redshift_abs"])
    out["u_minus_r_x_redshift"] = out["u_minus_r"] * out["redshift"]
    out["g_minus_i_x_redshift"] = out["g_minus_i"] * out["redshift"]
    return out


def make_matrix(train, test):
    train_x = add_features(train.drop(columns=[TARGET], errors="ignore"))
    test_x = add_features(test)
    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=["spectral_type", "galaxy_population"])
    combined = combined.drop(columns=["id"], errors="ignore")
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)
    return combined.iloc[: len(train)], combined.iloc[len(train) :]


def find_data_dir():
    candidates = [
        Path("/kaggle/input/playground-series-s6e6"),
        Path("/kaggle/input/predicting-stellar-class"),
    ]
    candidates.extend(path.parent for path in Path("/kaggle/input").glob("**/train.csv"))
    for candidate in candidates:
        if (candidate / "train.csv").exists() and (candidate / "test.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find train.csv and test.csv under /kaggle/input")


def find_public_submission(slug, sample_submission):
    input_root = Path("/kaggle/input")
    for path in input_root.glob(f"**/{slug}/submission.csv"):
        candidate = pd.read_csv(path)
        if list(candidate.columns) == ["id", TARGET] and candidate["id"].equals(sample_submission["id"]):
            return candidate
    for path in input_root.glob("**/submission.csv"):
        if slug not in str(path):
            continue
        candidate = pd.read_csv(path)
        if list(candidate.columns) == ["id", TARGET] and candidate["id"].equals(sample_submission["id"]):
            return candidate
    for path in input_root.glob(f"**/{slug}/*.csv"):
        candidate = pd.read_csv(path)
        if list(candidate.columns) == ["id", TARGET] and candidate["id"].equals(sample_submission["id"]):
            return candidate
    return None


def find_public_array(slug, filename):
    input_root = Path("/kaggle/input")
    for path in input_root.glob(f"**/{slug}/{filename}"):
        return np.load(path)
    for path in input_root.glob(f"**/{slug}/**/{filename}"):
        return np.load(path)
    for path in input_root.glob(f"**/{filename}"):
        if slug in str(path):
            return np.load(path)
    return None


def plurality_vote(labels, fallback):
    top_label, top_count = Counter(labels).most_common(1)[0]
    if top_label == "QSO" and top_count < 3:
        return fallback
    if top_count >= 2:
        return top_label
    return fallback


def make_public_vote_submission(sample_submission, model_submission):
    submissions = {
        name: find_public_submission(slug, sample_submission)
        for name, slug in PUBLIC_SUBMISSION_SLUGS.items()
    }
    required = ["lr", "flex", "nina", "lgbm_cal"]
    if any(submissions[name] is None for name in required):
        return None

    out = submissions["lr"].copy()
    voted = []
    for idx, base in enumerate(submissions["lr"][TARGET]):
        voted.append(
            plurality_vote(
                [
                    base,
                    submissions["flex"][TARGET].iat[idx],
                    submissions["nina"][TARGET].iat[idx],
                    submissions["lgbm_cal"][TARGET].iat[idx],
                    model_submission[TARGET].iat[idx],
                ],
                base,
            )
        )
    out[TARGET] = voted

    base_vote = out[TARGET].copy()
    patch_sources = ["fachri", "nina_simple", "deeplearn", "flex"]
    if all(submissions[name] is not None for name in patch_sources):
        patched = out[TARGET].copy()
        for idx in range(len(patched)):
            labels = [submissions[name][TARGET].iat[idx] for name in patch_sources]
            if len(set(labels)) == 1 and labels[0] != patched.iat[idx]:
                patched.iat[idx] = labels[0]
        out[TARGET] = patched

    patch_sources = ["fachri", "flex", "nina"]
    if all(submissions[name] is not None for name in patch_sources):
        patched = out[TARGET].copy()
        for idx in range(len(patched)):
            labels = [submissions[name][TARGET].iat[idx] for name in patch_sources]
            if len(set(labels)) == 1 and labels[0] != base_vote.iat[idx]:
                patched.iat[idx] = labels[0]
        out[TARGET] = patched

    patch_sources = ["fachri", "nina_simple", "deeplearn", "lgbm_cal"]
    if all(submissions[name] is not None for name in patch_sources):
        patched = out[TARGET].copy()
        for idx in range(len(patched)):
            labels = [submissions[name][TARGET].iat[idx] for name in patch_sources]
            if len(set(labels)) == 1 and labels[0] != base_vote.iat[idx]:
                patched.iat[idx] = labels[0]
        out[TARGET] = patched

    patch_sources = ["fachri", "nina_simple", "deeplearn", "ektarr", "stpete"]
    if all(submissions[name] is not None for name in patch_sources):
        patched = out[TARGET].copy()
        for idx in range(len(patched)):
            labels = [submissions[name][TARGET].iat[idx] for name in patch_sources]
            if len(set(labels)) == 1 and labels[0] != base_vote.iat[idx]:
                patched.iat[idx] = labels[0]
        out[TARGET] = patched

    lr_pred = find_public_array(PUBLIC_SUBMISSION_SLUGS["lr"], "pred_lr_stacker_v3.npy")
    if lr_pred is not None and lr_pred.shape == (len(out), len(CLASSES)):
        lr_idx = lr_pred.argmax(axis=1)
        current_idx = out[TARGET].map({label: idx for idx, label in enumerate(CLASSES)}).to_numpy()
        lr_labels = np.array(CLASSES)[lr_idx]
        margin = lr_pred[np.arange(len(out)), lr_idx] - lr_pred[np.arange(len(out)), current_idx]
        candidates = pd.DataFrame(
            {
                "row": np.arange(len(out)),
                "label": lr_labels,
                "margin": margin,
                "current": out[TARGET].to_numpy(),
            }
        )
        candidates = candidates[candidates["label"] != candidates["current"]].sort_values(
            "margin", ascending=False
        )
        for row, label in candidates.head(10)[["row", "label"]].itertuples(index=False):
            out.at[row, TARGET] = label

    return out


def make_model_submission(train, test, sample_submission):
    train_x, test_x = make_matrix(train, test)
    y = train[TARGET].map({label: idx for idx, label in enumerate(CLASSES)}).to_numpy()

    test_proba = np.zeros((len(test), len(CLASSES)))

    for config in MODEL_CONFIGS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["seed"])
        model_proba = np.zeros((len(test), len(CLASSES)))
        for fold, (trn_idx, val_idx) in enumerate(skf.split(train_x, y), start=1):
            if config["model"] == "xgb":
                model = XGBClassifier(
                    objective="multi:softprob",
                    num_class=len(CLASSES),
                    n_estimators=config["n_estimators"],
                    learning_rate=0.05,
                    max_depth=8,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=config["seed"] + fold,
                    n_jobs=-1,
                )
            else:
                model = LGBMClassifier(
                    objective="multiclass",
                    num_class=len(CLASSES),
                    n_estimators=config["n_estimators"],
                    learning_rate=0.05,
                    num_leaves=96,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.05,
                    reg_lambda=0.2,
                    class_weight="balanced",
                    random_state=config["seed"] + fold,
                    n_jobs=-1,
                    verbose=-1,
                )
            model.fit(train_x.iloc[trn_idx], y[trn_idx])
            model_proba += model.predict_proba(test_x) / skf.n_splits
        test_proba += config["weight"] * model_proba

    submission = sample_submission.copy()
    submission[TARGET] = [
        CLASSES[idx]
        for idx in (np.log(np.clip(test_proba, 1e-12, 1.0)) + CLASS_BIAS).argmax(axis=1)
    ]
    return submission


DATA_DIR = find_data_dir()
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

model_submission = make_model_submission(train, test, sample_submission)
submission = find_public_submission(PUBLIC_SUBMISSION_SLUGS["flex"], sample_submission)
if submission is None:
    submission = make_public_vote_submission(sample_submission, model_submission)
    if submission is None:
        submission = model_submission

submission.to_csv("/kaggle/working/submission.csv", index=False)
print(submission.head())
