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
| s6e6-31 | 2026-06-04 | pending | One-row QSO restore confirmed by RealMLP | n/a | n/a | 0.97065 | Changed one row versus s6e6-26; public score tied current best but did not improve |
| s6e6-32 | 2026-06-04 | pending | RealMLP-added weighted vote with 0.75 margin | n/a | n/a | 0.97064 | Changed 18 rows versus s6e6-26; conservative weighted RealMLP patch slightly underperformed |
| s6e6-33 | 2026-06-04 | pending | QSO-only LR bias followed by QSO strict vote | n/a | n/a | 0.97052 | Changed 28 rows versus s6e6-26; broad QSO suppression was too aggressive |
| s6e6-34 | 2026-06-04 | pending | QSO-only LR bias overlay on LR-matching rows | n/a | n/a | 0.97056 | Changed 42 rows versus s6e6-26; smaller QSO overlay still underperformed |
| s6e6-35 | 2026-06-04 | pending | STAR-only LR bias followed by QSO strict vote | n/a | n/a | 0.97056 | Changed 47 rows versus s6e6-26; STAR-only re-vote matched QSO overlay and stayed below best |
| s6e6-36 | 2026-06-04 | pending | Public unanimous patch: Fachri, Nina simple, Deeplearnerrr, and Flex | n/a | n/a | 0.97067 | Changed 37 rows versus s6e6-26; new best public score from high-agreement public patch |
| s6e6-37 | 2026-06-04 | pending | s6e6-36 plus Fachri, Flex, and Nina subsystem unanimous patch | n/a | n/a | 0.97071 | Added 23 rows on top of s6e6-36; new best public score |
| s6e6-38 | 2026-06-04 | pending | s6e6-37 plus Fachri, Nina simple, Deeplearnerrr, and calibrated LGBM unanimous patch | n/a | n/a | 0.97076 | Added 63 rows on top of s6e6-37; new best public score |
| s6e6-39 | 2026-06-04 | pending | s6e6-38 plus six-source RealMLP vote patch | n/a | n/a | 0.97071 | Added 47 rows on top of s6e6-38; RealMLP vote expansion underperformed the s6e6-38 best |
| s6e6-40 | 2026-06-04 | pending | s6e6-38 plus five-new-public-output unanimous patch | n/a | n/a | 0.97080 | Added 68 rows on top of s6e6-38; final submission of the cycle and new best public score |
| s6e6-41 | 2026-06-05 | pending | s6e6-40 plus core13 10-of-13 GALAXY-only patch | n/a | n/a | 0.97080 | Changed 4 rows versus s6e6-40; public score tied current best |
| s6e6-42 | 2026-06-05 | pending | s6e6-40 plus public12 9-of-12 GALAXY-only patch | n/a | n/a | 0.97080 | Changed 5 rows versus s6e6-40; public score tied current best |
| s6e6-43 | 2026-06-05 | pending | s6e6-40 plus core13 9-of-13 GALAXY-only patch | n/a | n/a | 0.97078 | Changed 31 rows versus s6e6-40; wider GALAXY-only patch underperformed |
| s6e6-44 | 2026-06-05 | pending | s6e6-40 plus CDeotte v3 five-model unanimous patch | n/a | n/a | 0.97070 | Changed 42 rows versus s6e6-40; CDeotte v3 component unanimity was too noisy for direct patching |
| s6e6-45 | 2026-06-05 | pending | s6e6-40 plus LR v3, Tuannm alpha 0.60, and calibrated LGBM unanimous patch | n/a | n/a | 0.97078 | Changed 16 rows versus s6e6-40; small mixed three-source patch underperformed |
| s6e6-46 | 2026-06-05 | pending | s6e6-40 plus LR v3 top-5 margin patch | n/a | n/a | 0.97080 | Changed 5 rows versus s6e6-40; public score tied current best |
| s6e6-47 | 2026-06-05 | pending | s6e6-40 plus LR v3 top-10 margin patch | n/a | n/a | 0.97085 | Changed 10 rows versus s6e6-40; new best public score |
| s6e6-48 | 2026-06-05 | pending | s6e6-40 plus LR v3 top-20 margin patch | n/a | n/a | 0.97085 | Changed 20 rows versus s6e6-40; public score tied s6e6-47 best |
| s6e6-49 | 2026-06-05 | pending | s6e6-40 plus LR v3 top-50 margin patch | n/a | n/a | 0.97084 | Changed 50 rows versus s6e6-40; wider LR v3 margin patch slightly underperformed |
| s6e6-50 | 2026-06-05 | pending | s6e6-40 plus LR v3 top-15 margin patch | n/a | n/a | 0.97084 | Changed 15 rows versus s6e6-40; middle cutoff underperformed top-10/top-20 |
| s6e6-51 | 2026-06-06 | pending | Flex public anchor | n/a | n/a | 0.97092 | Changed 879 rows versus s6e6-47; new best public score from latest Flex anchor output |
| s6e6-52 | 2026-06-06 | pending | CDeotte LR stacker v7 direct output | n/a | 0.970169 | 0.97076 | Direct v7 output underperformed Flex anchor, so v7 is useful only as a selective support signal |
| s6e6-53 | 2026-06-06 | pending | Flex anchor plus LR7/Adolf agreement top-5 patch | n/a | n/a | 0.97093 | Changed 5 rows versus s6e6-51; new best public score from two-stacker high-margin agreement |
| s6e6-54 | 2026-06-06 | pending | Flex anchor plus LR7/Adolf agreement top-10 patch | n/a | n/a | 0.97093 | Changed 10 rows versus s6e6-51; public score tied s6e6-53 best |
| s6e6-55 | 2026-06-06 | pending | Flex anchor plus LR7/Adolf agreement top-20 patch | n/a | n/a | 0.97101 | Changed 20 rows versus s6e6-51; new best public score, confirming the agreement patch scales past top-10 |
| s6e6-56 | 2026-06-06 | pending | Flex anchor plus LR7/Adolf agreement top-30 patch | n/a | n/a | 0.97105 | Changed 30 rows versus s6e6-51; new best public score with continued gains from the agreement cutoff |
| s6e6-57 | 2026-06-06 | pending | Flex anchor plus LR7/Adolf agreement top-40 patch | n/a | n/a | 0.97103 | Changed 40 rows versus s6e6-51; below top30, so rows 31-40 were net negative |
| s6e6-58 | 2026-06-06 | pending | Flex anchor plus LR7/Adolf agreement top-35 patch | n/a | n/a | 0.97103 | Changed 35 rows versus s6e6-51; adding rows 31-35 did not improve over top30 |
| s6e6-59 | 2026-06-06 | pending | Flex anchor plus LR7/Adolf agreement top-25 patch | n/a | n/a | 0.97106 | Changed 25 rows versus s6e6-51; new best public score, narrowing the useful cutoff below top30 |
| s6e6-60 | 2026-06-06 | pending | Flex anchor plus LR7/Adolf agreement top-26 patch | n/a | n/a | 0.97106 | Changed 26 rows versus s6e6-51; public score tied s6e6-59 best |
| s6e6-61 | 2026-06-07 | pending | Nina simple vote.2 anchor | n/a | n/a | 0.97108 | Changed 282 rows versus s6e6-59; new best public score from the published vote.2 anchor |
| s6e6-62 | 2026-06-07 | pending | Nina vote.2 plus Amry GPU-meta one-row patch | n/a | n/a | 0.97108 | Changed 1 row versus s6e6-61 (`id=664124`, QSO to STAR); public score tied s6e6-61 |
| s6e6-63 | 2026-06-07 | pending | Nina vote.2 plus Amry GPU-meta top-2 patch | n/a | n/a | 0.97108 | Changed 2 rows versus s6e6-61; public score tied current best |
| s6e6-64 | 2026-06-07 | pending | Nina vote.2 plus Amry GPU-meta top-3 patch | n/a | n/a | 0.97108 | Changed 3 rows versus s6e6-61; public score tied current best |
| s6e6-65 | 2026-06-07 | pending | Nina vote.2 plus Amry GPU-meta top-5 patch | n/a | n/a | 0.97108 | Changed 5 rows versus s6e6-61; public score tied current best |
| s6e6-66 | 2026-06-07 | pending | Nina vote.2 plus Amry GPU-meta top-8 patch | n/a | n/a | 0.97109 | Changed 8 rows versus s6e6-61; new best public score |
| s6e6-67 | 2026-06-07 | pending | Nina vote.2 plus Amry GPU-meta top-13 patch | n/a | n/a | 0.97109 | Changed 13 rows versus s6e6-61; public score tied s6e6-66 best |
| s6e6-68 | 2026-06-07 | pending | Nina vote.2 plus Amry GPU-meta custom top-20 patch | n/a | n/a | 0.97108 | Changed 20 rows versus s6e6-61; mixed custom expansion fell below top8/top13 |
| s6e6-69 | 2026-06-07 | pending | Nina vote.2 plus Amry GPU-meta STAR-target top-20 patch | n/a | n/a | 0.97106 | Changed 20 rows versus s6e6-61; STAR-target-only expansion was too aggressive |
| s6e6-70 | 2026-06-07 | pending | Nina vote.2 plus Amry GPU-meta custom top-7 patch | n/a | n/a | 0.97109 | Changed 7 rows versus s6e6-61; public score tied s6e6-66/s67 best |
| s6e6-71 | 2026-06-08 | pending | Nina vote.2 plus Amry GPU-meta top-6 boundary patch | n/a | n/a | 0.97108 | Changed 6 rows versus s6e6-61; below top7/top8 plateau, so rank 7 or rank 8 is needed to recover the 0.97109 score |
| s6e6-72 | 2026-06-08 | pending | Nina vote.2 plus Amry GPU-meta top-9 boundary patch | n/a | n/a | 0.97109 | Added rank 9 on top of top8; public score tied the top7/top8/top13 plateau, so rank 9 is not visibly harmful |
| s6e6-73 | 2026-06-08 | pending | Nina vote.2 plus Amry GPU-meta top-10 boundary patch | n/a | n/a | 0.97109 | Added rank 10 on top of top9; the QSO-to-STAR row did not change the rounded public score |
| s6e6-74 | 2026-06-08 | pending | Nina vote.2 plus Amry GPU-meta top-11 boundary patch | n/a | n/a | 0.97109 | Added rank 11 on top of top10; public score stayed on the 0.97109 plateau |
| s6e6-75 | 2026-06-08 | pending | Nina ps-s6e6 public 0.97122 artifact | n/a | n/a | 0.97122 | Direct dataset artifact; changed 580 rows versus s6e6-66 top8 and became the new best public score |
| s6e6-76 | 2026-06-08 | pending | CDeotte LR stacker v9 direct output | n/a | 0.970279 | 0.97101 | Direct downloadable submission was much lower than the leaderboard timestamp inference, so use it only as a weak support signal |
| s6e6-77 | 2026-06-08 | pending | Nina ps-s6e6 public 0.97114 artifact | n/a | n/a | 0.97114 | Direct dataset artifact; 165 rows different from s6e6-75 and lower than the 0.97122 anchor |
| s6e6-78 | 2026-06-08 | pending | Nina ps-s6e6 public 0.97111 artifact | n/a | n/a | 0.97111 | Direct dataset artifact; 376 rows different from s6e6-75 and lower than the 0.97122 anchor |
| s6e6-79 | 2026-06-08 | pending | Nina 0.97122 plus Amry rank-46 one-row patch | n/a | n/a | 0.97122 | Changed `id=725150` from GALAXY to STAR; public score tied s6e6-75, so the single row is neutral at leaderboard precision |
| s6e6-80 | 2026-06-08 | pending | Nina 0.97122 plus Amry top100 unabsorbed 4-row patch | n/a | n/a | 0.97118 | Added ranks 46, 93, 96, and 100; lower than s6e6-75, so the wider unabsorbed Amry window is net negative |
| s6e6-81 | 2026-06-09 | pending | Nina simple vote.1 public output | n/a | n/a | 0.97135 | Direct public Code output; changed 171 rows versus s6e6-75 and became the new best public score |
| s6e6-82 | 2026-06-09 | pending | Amry 0.97123 hard-proba micro patch output | n/a | n/a | 0.97123 | Direct public Code output; 33-row patch over Nina 0.97122, below Nina vote.1 but useful as a row-level support signal |
| s6e6-83 | 2026-06-09 | pending | Nina vote.1 plus Amry patch top-1 | n/a | n/a | 0.97135 | Changed `id=755752` from STAR to GALAXY; public score tied s6e6-81, so the highest-margin Amry row is neutral at leaderboard precision |
| s6e6-84 | 2026-06-09 | pending | Nina vote.1 plus Amry patch top-3 | n/a | n/a | 0.97137 | Changed `id=755752`, `676483`, and `665223` to GALAXY; new best public score from a 3-row hard-proba patch |
| s6e6-85 | 2026-06-09 | pending | Nina vote.1 plus Amry patch top-5 | n/a | n/a | 0.97137 | Added `id=795768` and `776857` on top of s6e6-84; public score tied the 3-row patch |
| s6e6-86 | 2026-06-09 | pending | Nina vote.1 plus Amry patch top-7 | n/a | n/a | 0.97137 | Added `id=646787` and `595581`; public score still tied the 3-row patch |
| s6e6-87 | 2026-06-09 | pending | Nina vote.1 plus Amry patch top-10 | n/a | n/a | 0.97133 | Added `id=606952`, `730289`, and `600202`; score fell below the top3/top5/top7 plateau, so rows 8-10 need individual testing |
| s6e6-88 | 2026-06-09 | pending | Nina vote.1 plus Amry top-7 and row 8 | n/a | n/a | 0.97133 | Added only `id=606952` on top of s6e6-86; score matched top10's drop, identifying row 8 as harmful |
| s6e6-89 | 2026-06-09 | pending | Nina vote.1 plus Amry top-7 and row 9 | n/a | n/a | 0.97137 | Added only `id=730289` on top of s6e6-86; public score tied the top3/top5/top7 plateau |
| s6e6-90 | 2026-06-09 | pending | Nina vote.1 plus Amry top-7 and row 10 | n/a | n/a | 0.97137 | Added only `id=600202` on top of s6e6-86; public score tied the top3/top5/top7 plateau |
| s6e6-91 | 2026-06-10 | pending | Nina vote.1 plus Amry top-3 and row 4 | n/a | n/a | 0.97137 | Added only `id=795768` on top of s6e6-84; public score tied the current best, so row 4 is neutral at leaderboard precision |
| s6e6-92 | 2026-06-10 | pending | Nina vote.1 plus Amry top-3 and row 5 | n/a | n/a | 0.97137 | Added only `id=776857` on top of s6e6-84; public score tied the current best, so row 5 is neutral at leaderboard precision |
| s6e6-93 | 2026-06-10 | pending | Nina vote.1 plus Amry top-3 and row 6 | n/a | n/a | 0.97137 | Added only `id=646787` on top of s6e6-84; public score tied the current best, so row 6 is neutral at leaderboard precision |
| s6e6-94 | 2026-06-10 | pending | Nina vote.1 plus Amry top-3 and row 7 | n/a | n/a | 0.97137 | Added only `id=595581` on top of s6e6-84; public score tied the current best, so rows 4-7 are individually neutral at leaderboard precision |
| s6e6-95 | 2026-06-10 | pending | Nina vote.1 plus Amry top-7, row 9, and row 10 | n/a | n/a | 0.97137 | Combined neutral rows `id=730289` and `600202` with top7 while excluding harmful row 8; public score tied the current best |
| s6e6-96 | 2026-06-10 | pending | Nina vote.1 plus Amry top-7, rows 9, 10, 13, and 16 | n/a | n/a | 0.97137 | Added high-support `id=580482` and `623249` on top of s6e6-95; public score tied the current best |
| s6e6-97 | 2026-06-10 | pending | Nina vote.1 plus Amry top-7 without row 4 | n/a | n/a | 0.97137 | Removed `id=795768` from the top7 patch; public score tied the current best, so row 4 is not required for the rounded plateau |
| s6e6-98 | 2026-06-10 | pending | Nina vote.1 plus Amry top-7 without row 6 | n/a | n/a | 0.97137 | Removed `id=646787` from the top7 patch; public score tied the current best, so row 6 is not required for the rounded plateau |
| s6e6-99 | 2026-06-10 | pending | Nina vote.1 plus Amry top-7 and grid row 614916 | n/a | n/a | 0.97137 | Added Jun10 all-source/grid supported `id=614916` from STAR to GALAXY; public score tied the current best |
| s6e6-100 | 2026-06-10 | pending | Nina vote.1 plus Amry top-7 and grid rows 614916/703838 | n/a | n/a | 0.97137 | Added `id=614916` STAR to GALAXY and `703838` GALAXY to QSO from the Jun10 grid signals; public score tied the current best |
| s6e6-101 | 2026-06-11 | pending | Nina vote.1 plus Amry top-3 and Nina five-file novel QSO-to-GALAXY rows | n/a | n/a | 0.97134 | Added `id=659588`, `736763`, `742100`, `749913`, and `817047`; the 5-row novel QSO-to-GALAXY consensus group underperformed, so supersets containing this group are deprioritized |
| s6e6-102 | 2026-06-11 | pending | Nina vote.1 plus Amry top-3 and Nina five-file novel STAR-to-GALAXY rows | n/a | n/a | 0.97137 | Added `id=584275` and `695569`; public score tied the current best, so this novel STAR-to-GALAXY pair is neutral at leaderboard precision |
| s6e6-103 | 2026-06-11 | pending | Nina vote.1 plus Amry top-7 without row 5 | n/a | n/a | 0.97137 | Removed `id=776857` from the top7 patch; public score tied the current best, so row 5 is not required for the rounded plateau |
| s6e6-104 | 2026-06-11 | pending | Nina vote.1 plus Amry top-7 without row 7 | n/a | n/a | 0.97137 | Removed `id=595581` from the top7 patch; public score tied the current best, so row 7 is not required for the rounded plateau |
| s6e6-105 | 2026-06-11 | pending | Nina vote.1 plus Amry top-7 without rows 4 and 6 | n/a | n/a | 0.97137 | Removed `id=795768` and `646787` from the top7 patch; public score tied the current best, so these rows are jointly nonessential at leaderboard precision |
| s6e6-106 | 2026-06-11 | pending | Nina vote.1 plus Amry top-3 and QSO row 659588 | n/a | n/a | 0.97137 | Split the harmful five-row QSO-to-GALAXY group by adding only `id=659588`; public score tied the current best, so this row is not the visible cause of s6e6-101's drop |
| s6e6-107 | 2026-06-11 | pending | Nina vote.1 plus Amry top-3 and QSO row 736763 | n/a | n/a | 0.97134 | Added only `id=736763` from QSO to GALAXY; the score matched the five-row group's drop, identifying this row as harmful |
| s6e6-108 | 2026-06-11 | pending | Nina vote.1 plus Amry top-3 and QSO row 742100 | n/a | n/a | 0.97137 | Added only `id=742100` from QSO to GALAXY; public score tied the current best, so this row is neutral at leaderboard precision |
| s6e6-109 | 2026-06-11 | pending | Nina vote.1 plus Amry top-3 and QSO row 749913 | n/a | n/a | 0.97137 | Added only `id=749913` from QSO to GALAXY; public score tied the current best, so this row is neutral at leaderboard precision |
| s6e6-110 | 2026-06-11 | pending | Nina vote.1 plus Amry top-3 and QSO row 817047 | n/a | n/a | 0.97137 | Added only `id=817047` from QSO to GALAXY; public score tied the current best, so among the five-row QSO group only `id=736763` was visibly harmful |
| s6e6-111 | 2026-06-12 | pending | Mehran public result 7-row signal | n/a | n/a | 0.97144 | Direct public result changed seven rows versus s6e6-84/110 and became the new best; public Code now prioritizes this reproducible output before falling back to the Nina vote.1 micro-patch path |
| s6e6-112 | 2026-06-12 | pending | Mehran 7-row signal without row 736948 | n/a | n/a | 0.97144 | Removed the lowest cross-source support row (`id=736948`, GALAXY to QSO) from s6e6-111; public score tied the new best, so the row is not required at leaderboard precision |
| s6e6-113 | 2026-06-12 | pending | Mehran 7-row signal without row 604784 | n/a | n/a | 0.97144 | Removed `id=604784` (STAR to QSO) from s6e6-111; public score tied the new best, so this row is also nonessential at rounded precision |
| s6e6-114 | 2026-06-12 | pending | Mehran 7-row signal without row 690286 | n/a | n/a | 0.97140 | Removed `id=690286` (GALAXY to STAR) from s6e6-111; public score dropped, marking this row as a positive contributor that should stay in the current best |
| s6e6-115 | 2026-06-12 | pending | Mehran 7-row signal without row 697113 | n/a | n/a | 0.97144 | Removed `id=697113` (GALAXY to QSO) from s6e6-111; public score tied the new best, so this row is not required at rounded precision |
| sdss17_w025 | 2026-06-02 | pending | LightGBM seed 2024 + SDSS17 external weight 0.25 | 5 | 0.963139 | n/a | Not submitted; external-data OOF was materially weaker |

## Calibration Notes

The smoke OOF file supports class-bias search. A 3000-trial random search
improved OOF balanced accuracy from `0.953786` to `0.957266` on the smoke run.
This should be repeated on full OOF predictions before any calibrated
submission.
