# Experiment Log

| run | date | commit | model | folds | local balanced accuracy | public score | notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| prep | 2026-06-01 | pending | none | 0 | n/a | n/a | Project setup, data download, baseline scaffold |
| smoke_lgbm | 2026-06-01 | pending | LightGBM | 5 | 0.953786 | n/a | 30000-row stratified smoke test; submission schema verified |
| s6e6-01 | 2026-06-01 | pending | LightGBM seed 42, 700 estimators | 5 | 0.964407 | 0.96551 | Raw full-data baseline |

## Calibration Notes

The smoke OOF file supports class-bias search. A 3000-trial random search
improved OOF balanced accuracy from `0.953786` to `0.957266` on the smoke run.
This should be repeated on full OOF predictions before any calibrated
submission.
