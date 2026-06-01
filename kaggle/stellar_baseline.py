import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "/kaggle/input/playground-series-s6e6"
CLASSES = ["GALAXY", "QSO", "STAR"]
TARGET = "class"
SEED = 777
CLASS_BIAS = np.array([0.0, 0.7013898538339098, 0.6982218073099267])


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


train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

train_x, test_x = make_matrix(train, test)
y = train[TARGET].map({label: idx for idx, label in enumerate(CLASSES)}).to_numpy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
test_proba = np.zeros((len(test), len(CLASSES)))

for fold, (trn_idx, val_idx) in enumerate(skf.split(train_x, y), start=1):
    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(CLASSES),
        n_estimators=900,
        learning_rate=0.05,
        num_leaves=96,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.2,
        class_weight="balanced",
        random_state=SEED + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(train_x.iloc[trn_idx], y[trn_idx])
    test_proba += model.predict_proba(test_x) / skf.n_splits

submission = sample_submission.copy()
submission[TARGET] = [CLASSES[idx] for idx in (np.log(np.clip(test_proba, 1e-12, 1.0)) + CLASS_BIAS).argmax(axis=1)]
submission.to_csv("/kaggle/working/submission.csv", index=False)
print(submission.head())
