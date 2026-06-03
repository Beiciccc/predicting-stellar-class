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
| s6e6-11 | 2026-06-02 | pending | LightGBM seed 2024, 1200 estimators + tighter grid bias | 5 | 0.965525 | 0.96677 | Tiny OOF gain did not improve public score |
| s6e6-12 | 2026-06-02 | pending | LightGBM seed 2024, 1200 estimators + lower STAR bias | 5 | 0.965508 | 0.96672 | STAR bias reduction hurt public score |
| s6e6-13 | 2026-06-02 | pending | LightGBM seed 2024, 1200 estimators + higher STAR bias | 5 | 0.965494 | 0.96701 | New best public score; public split favors higher STAR bias |
| s6e6-14 | 2026-06-02 | pending | LightGBM seed 2024, 1200 estimators + higher QSO bias | 5 | 0.965490 | 0.96666 | Increasing QSO bias hurt public score |
| s6e6-15 | 2026-06-02 | pending | LightGBM seed 2024, 1200 estimators + STAR 1.00 bias | 5 | 0.965336 | 0.96682 | STAR bias above 0.92 lost public score |
| s6e6-16 | 2026-06-02 | pending | LightGBM seed 2024, 1200 estimators + QSO 0.50, STAR 0.92 bias | 5 | 0.965453 | 0.96692 | Lower QSO than s6e6-13 underperformed |
| s6e6-17 | 2026-06-02 | pending | LightGBM seed 2024, 1200 estimators + QSO 0.55, STAR 0.92 bias | 5 | 0.965464 | 0.96699 | Slightly higher QSO than s6e6-13 underperformed by 0.00002 |
| s6e6-18 | 2026-06-02 | pending | LightGBM seed 2024, 1200 estimators + QSO 0.53, STAR 0.94 bias | 5 | 0.965464 | 0.96686 | STAR above 0.92 underperformed |
| s6e6-19 | 2026-06-02 | pending | LightGBM seed 2024, 1200 estimators + QSO 0.52, STAR 0.92 bias | 5 | 0.965483 | 0.96697 | Local QSO adjustment below s6e6-13 underperformed |
| s6e6-20 | 2026-06-02 | pending | LightGBM seed 2024, 1200 estimators + QSO 0.535, STAR 0.92 bias | 5 | 0.965485 | 0.96698 | Local QSO midpoint underperformed s6e6-13 |
| s6e6-21 | 2026-06-03 | pending | Public GPU logistic-regression stacker | n/a | 0.969656 | 0.97021 | Public output from `cdeotte/gpu-logistic-regression-stacker`; schema verified |
| s6e6-22 | 2026-06-03 | pending | Public weighted submission blender | n/a | n/a | 0.97015 | Public output from `flexonafft/blender-is-all-you-need`; slightly below s6e6-21 |
| s6e6-23 | 2026-06-03 | pending | LR stacker anchored public unanimous overwrite | n/a | n/a | 0.97051 | Base `cdeotte` LR stacker; overwrite 441 rows where `flex`, `nina`, and public calibrated LGBM agree |
| s6e6-24 | 2026-06-03 | pending | LR stacker anchored RealMLP unanimous overwrite | n/a | n/a | 0.97018 | Base `cdeotte` LR stacker; overwrite 939 rows where `flex`, `nina`, and public RealMLP agree; lower than s6e6-23 |
| s6e6-25 | 2026-06-03 | pending | Public stacker/blender/LGBM plus local LGBM hard vote | n/a | n/a | 0.97064 | Five-way plurality vote over `cdeotte`, `flex`, `nina`, public calibrated LGBM, and local best LGBM; new best |
| s6e6-26 | 2026-06-03 | pending | Five-way vote with QSO strict-three rule | n/a | n/a | 0.97065 | Same as s6e6-25, but QSO needs at least three votes; changed 3 rows and became new best |
| s6e6-27 | 2026-06-03 | pending | Five-way vote with QSO two-vote fallback to non-QSO | n/a | n/a | 0.97064 | Changed 3 rows versus s6e6-26; fallback to non-QSO underperformed LR fallback |
| s6e6-28 | 2026-06-03 | pending | Five-way vote with strict three-vote rule | n/a | n/a | 0.97060 | Changed 2 rows versus s6e6-26; strict fallback for non-QSO labels hurt public score |
| s6e6-29 | 2026-06-03 | pending | Five-way vote with STAR tie preference | n/a | n/a | 0.97064 | Changed 4 rows versus s6e6-26; STAR tie preference did not improve public score |
| s6e6-30 | 2026-06-03 | pending | External 3-of-5 vote with STAR-safe and QSO-all rule | n/a | n/a | 0.97058 | Changed 714 rows versus s6e6-26; strong QSO suppression was too aggressive |
| sdss17_w025 | 2026-06-02 | pending | LightGBM seed 2024 + SDSS17 external weight 0.25 | 5 | 0.963139 | n/a | Not submitted; external-data OOF was materially weaker |

## Calibration Notes

The smoke OOF file supports class-bias search. A 3000-trial random search
improved OOF balanced accuracy from `0.953786` to `0.957266` on the smoke run.
This should be repeated on full OOF predictions before any calibrated
submission.
