# Predicting Stellar Class

Kaggle Playground Series S6E6 project for predicting stellar object class labels:
`GALAXY`, `STAR`, and `QSO`.

- Competition: <https://www.kaggle.com/competitions/playground-series-s6e6>
- Metric: balanced accuracy
- Submission format: `id,class`
- Daily submission limit: 10
- Final submission deadline: 2026-06-30 23:59 UTC

## Repository Contents

```text
configs/        Reproducible experiment configuration
docs/           Data notes, results log, and competition notes
kaggle/         Kaggle Code entry point
notebooks/      Local exploration notebooks
scripts/        Utility commands for status checks and submissions
src/            Shared training and inference code
```

The Kaggle competition data is not included in this repository. Download it
from the competition page or with the Kaggle API:

```bash
kaggle competitions download -c playground-series-s6e6 -p data/raw
unzip -n data/raw/playground-series-s6e6.zip -d data/raw
```

Expected local data files:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The local environment used for preparation already has `pandas`, `scikit-learn`,
`lightgbm`, `xgboost`, `catboost`, and `torch` available.

## Baseline

Run a quick smoke test:

```bash
python scripts/train_baseline.py --sample 30000 --n-estimators 80 --output-dir outputs/smoke_lgbm
```

Run the full LightGBM baseline:

```bash
python scripts/train_baseline.py --model lgbm --folds 5 --seed 42 --output-dir outputs/lgbm_seed42
```

Generated outputs include OOF predictions, test probabilities, metrics, and a
submission file. These generated artifacts are ignored by git.

## Results

Experiment results are tracked in [docs/experiments.md](docs/experiments.md).

