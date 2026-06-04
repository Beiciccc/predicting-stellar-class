#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "jun04_micro"
TARGET = "class"
CLASSES = ["GALAXY", "QSO", "STAR"]

FILES = {
    "best": ROOT / "outputs/public_blends/qso_strict3_lr_flex_nina_lgbm_ours.csv",
    "s25": ROOT / "outputs/public_blends/hard_vote_lr_flex_nina_lgbm_ours.csv",
    "lr": ROOT / "references/kaggle_outputs/cdeotte_gpu_lr_stacker/submission.csv",
    "flex": ROOT / "references/kaggle_outputs/flex_blender/submission.csv",
    "nina": ROOT / "references/kaggle_outputs/nina_vote/submission.csv",
    "lgbm_cal": ROOT / "references/kaggle_outputs/lgbm_single_96728/submission.csv",
    "realmlp": ROOT / "references/kaggle_outputs/realmlp_single/submission.csv",
    "ours13": ROOT / "outputs/lgbm_seed2024_e1200/submission_high_star_q53_s92.csv",
}

MARGIN_WEIGHTS = {
    "lr": 1.00,
    "flex": 0.88,
    "nina": 0.88,
    "realmlp": 0.70,
    "lgbm_cal": 0.45,
    "ours13": 0.35,
}


def read_frames() -> dict[str, pd.DataFrame]:
    frames = {name: pd.read_csv(path) for name, path in FILES.items()}
    sample_ids = frames["best"]["id"]
    for name, frame in frames.items():
        if list(frame.columns) != ["id", TARGET]:
            raise ValueError(f"{name}: unexpected columns {list(frame.columns)}")
        if not frame["id"].equals(sample_ids):
            raise ValueError(f"{name}: id order mismatch")
    return frames


def biased_lr_labels(qso: float = 0.0, star: float = 0.0) -> pd.Series:
    pred = np.load(ROOT / "references/kaggle_outputs/cdeotte_gpu_lr_stacker/pred_lr_stacker_v1.npy")
    bias = np.array([0.0, qso, star])
    labels = [CLASSES[idx] for idx in (np.log(np.clip(pred, 1e-12, 1.0)) + bias).argmax(axis=1)]
    return pd.Series(labels, name=TARGET)


def qso_strict_vote(labels: list[str], fallback: str) -> str:
    counts = Counter(labels)
    top_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == top_count]
    if len(winners) != 1:
        return fallback
    top_label = winners[0]
    if top_label == "QSO" and top_count < 3:
        return fallback
    return top_label


def six_vote_with_tie_fallback(labels: list[str], fallback: str) -> str:
    counts = Counter(labels)
    top_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == top_count]
    if len(winners) != 1:
        return fallback
    top_label = winners[0]
    if top_label == "QSO" and top_count < 3:
        return fallback
    return top_label


def weighted_label(labels: dict[str, str], fallback: str, min_margin: float) -> str:
    scores: dict[str, float] = {}
    for name, label in labels.items():
        scores[label] = scores.get(label, 0.0) + MARGIN_WEIGHTS[name]
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2:
        return ranked[0][0]
    if ranked[0][1] - ranked[1][1] > min_margin:
        return ranked[0][0]
    return fallback


def save(name: str, ids: pd.Series, labels: pd.Series | list[str], anchor: pd.Series) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.Series(labels, name=TARGET)
    out = pd.DataFrame({"id": ids, TARGET: labels})
    path = OUT_DIR / f"{name}.csv"
    out.to_csv(path, index=False)
    diff = int(labels.ne(anchor).sum())
    counts = labels.value_counts().to_dict()
    print(f"{name}: diff_vs_best={diff} counts={counts} path={path}")


def main() -> None:
    frames = read_frames()
    ids = frames["best"]["id"]
    best = frames["best"][TARGET]
    s25 = frames["s25"][TARGET]
    lr = frames["lr"][TARGET]
    flex = frames["flex"][TARGET]
    nina = frames["nina"][TARGET]
    lgbm = frames["lgbm_cal"][TARGET]
    realmlp = frames["realmlp"][TARGET]
    ours = frames["ours13"][TARGET]

    qso_bias = biased_lr_labels(qso=-0.0425)
    star_bias = biased_lr_labels(star=-0.0375)
    mild_bias = biased_lr_labels(qso=-0.0350, star=-0.0300)
    oof_best_bias = biased_lr_labels(qso=-0.0425, star=-0.0375)

    micro_qso = best.copy()
    mask = best.ne(s25) & s25.eq("QSO") & flex.eq("QSO") & nina.eq("QSO") & realmlp.eq("QSO")
    micro_qso.loc[mask] = "QSO"
    save("micro_qso2_realmlp_confirm", ids, micro_qso, best)

    weighted = []
    for idx, fallback in enumerate(best):
        weighted.append(
            weighted_label(
                {
                    "lr": lr.iat[idx],
                    "flex": flex.iat[idx],
                    "nina": nina.iat[idx],
                    "lgbm_cal": lgbm.iat[idx],
                    "realmlp": realmlp.iat[idx],
                    "ours13": ours.iat[idx],
                },
                fallback,
                min_margin=0.75,
            )
        )
    save("weighted_realmlp_margin075", ids, weighted, best)

    for name, lr_variant in [
        ("qso_only_bias_vote", qso_bias),
        ("star_only_bias_vote", star_bias),
        ("mild_bias_vote", mild_bias),
        ("oofbest_bias_vote", oof_best_bias),
    ]:
        voted = [
            qso_strict_vote(
                [
                    lr_variant.iat[idx],
                    flex.iat[idx],
                    nina.iat[idx],
                    lgbm.iat[idx],
                    ours.iat[idx],
                ],
                lr_variant.iat[idx],
            )
            for idx in range(len(best))
        ]
        save(name, ids, voted, best)

    qso_overlay = best.copy()
    mask = best.eq(lr) & qso_bias.ne(lr)
    qso_overlay.loc[mask] = qso_bias.loc[mask]
    save("qso_only_bias_overlay", ids, qso_overlay, best)

    star_overlay = best.copy()
    mask = best.eq(lr) & lr.eq("STAR") & star_bias.eq("GALAXY")
    star_overlay.loc[mask] = star_bias.loc[mask]
    save("star_bias_overlay_lr_rows", ids, star_overlay, best)

    mild_overlay = best.copy()
    mask = best.eq(lr) & mild_bias.ne(lr)
    mild_overlay.loc[mask] = mild_bias.loc[mask]
    save("mild_bias_overlay_lr_rows", ids, mild_overlay, best)

    six_vote = [
        six_vote_with_tie_fallback(
            [
                lr.iat[idx],
                flex.iat[idx],
                nina.iat[idx],
                lgbm.iat[idx],
                realmlp.iat[idx],
                ours.iat[idx],
            ],
            lr.iat[idx],
        )
        for idx in range(len(best))
    ]
    save("six_vote_realmlp_added", ids, six_vote, best)


if __name__ == "__main__":
    main()
