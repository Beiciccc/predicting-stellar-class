# Experiment Log

| run | date | commit | model | folds | local balanced accuracy | public score | notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| prep | 2026-06-01 | pending | none | 0 | n/a | n/a | Project setup, data download, baseline scaffold |
| smoke_lgbm | 2026-06-01 | pending | LightGBM | 5 | 0.953786 | n/a | 30000-row stratified smoke test; submission schema verified |
| s6e6-01 | 2026-06-01 | pending | LightGBM seed 42, 700 estimators | 5 | 0.964407 | 0.96551 | Raw full-data baseline |
| s6e6-02 | 2026-06-01 | pending | LightGBM seed 42, 700 estimators + class bias | 5 | 0.965440 | 0.96632 | OOF log-probability bias: QSO +0.6043, STAR +0.4650 |
| s6e6-03 | 2026-06-01 | pending | LightGBM seed 42, 700 estimators + refined class bias | 5 | 0.965457 | 0.96626 | Bias neighborhood test: QSO +0.5793, STAR +0.4650 |
| s6e6-04 | 2026-06-01 | pending | LightGBM seed 777, 900 estimators + class bias | 5 | 0.965279 | 0.96641 | New best public score; OOF bias QSO +0.7014, STAR +0.6982 |
| s6e6-05 | 2026-06-01 | pending | 50/50 LightGBM seed 42 + seed 777 blend + class bias | 5 | 0.965781 | 0.96630 | Highest OOF so far, but public score below single seed 777 |
| s6e6-06 | 2026-06-01 | pending | LightGBM seed 2024, 1200 estimators + class bias | 5 | 0.965480 | 0.96657 | New best public score; stronger STAR bias |
| s6e6-07 | 2026-06-01 | pending | LightGBM 3-seed weighted blend + class bias | 5 | 0.965916 | 0.96630 | OOF-best blend underperformed public LB versus seed 2024 single model |
| s6e6-08 | 2026-06-01 | pending | XGBoost seed 42, 700 estimators + class bias | 5 | 0.965115 | 0.96624 | XGBoost required large QSO/STAR bias; useful as diversity but weaker public score |
| s6e6-09 | 2026-06-01 | pending | LightGBM seed 2024, 1200 estimators + alternate class bias | 5 | 0.965500 | 0.96669 | New best public score; alternate OOF bias QSO +0.4433, STAR +0.7458 |
| s6e6-10 | 2026-06-01 | pending | LightGBM seed 2024, 1200 estimators + fine class bias | 5 | 0.965522 | 0.96679 | Best score of this run; fine OOF bias QSO +0.4733, STAR +0.8658 |

## Calibration Notes

The smoke OOF file supports class-bias search. A 3000-trial random search
improved OOF balanced accuracy from `0.953786` to `0.957266` on the smoke run.
This should be repeated on full OOF predictions before any calibrated
submission.
