# Data Notes

Source: <https://www.kaggle.com/competitions/playground-series-s6e6>

The dataset was inspired by the Stellar Classification Dataset - SDSS17. The
feature distributions are close to, but not identical to, the source dataset.

## Files

| file | rows | columns | notes |
| --- | ---: | ---: | --- |
| `train.csv` | 577347 | 12 | Includes target column `class` |
| `test.csv` | 247435 | 11 | Predict `class` for each `id` |
| `sample_submission.csv` | 247435 | 2 | Required submission format |

## Columns

- `id`: row identifier
- `alpha`, `delta`: sky coordinates
- `u`, `g`, `r`, `i`, `z`: photometric bands
- `redshift`: redshift value
- `spectral_type`: categorical feature
- `galaxy_population`: categorical feature
- `class`: target label in the training set

## Target Distribution

| class | count |
| --- | ---: |
| `GALAXY` | 377480 |
| `QSO` | 117143 |
| `STAR` | 82724 |

No missing values were found in the downloaded train or test files.

## Derived Feature Checks

The current public discussion indicates that two categorical fields can be
reconstructed from photometric color differences:

- `spectral_type` from `r - g`
- `galaxy_population` from `u - r`

This matters when evaluating whether the original SDSS-style dataset can be
aligned with the competition columns.

