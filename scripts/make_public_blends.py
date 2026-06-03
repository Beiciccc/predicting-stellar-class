#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "public_blends"
TARGET = "class"

FILES = {
    "ours13": ROOT / "outputs/lgbm_seed2024_e1200/submission_high_star_q53_s92.csv",
    "lr": ROOT / "references/kaggle_outputs/cdeotte_gpu_lr_stacker/submission.csv",
    "flex": ROOT / "references/kaggle_outputs/flex_blender/submission.csv",
    "nina": ROOT / "references/kaggle_outputs/nina_vote/submission.csv",
    "lgbm_cal": ROOT / "references/kaggle_outputs/lgbm_single_96728/submission.csv",
    "realmlp": ROOT / "references/kaggle_outputs/realmlp_single/submission.csv",
}

PUBLIC_WEIGHTS = {
    "lr": 0.97021,
    "flex": 0.97015,
    "nina": 0.97009,
    "realmlp": 0.96973,
    "lgbm_cal": 0.96728,
    "ours13": 0.96701,
}


def read_submissions() -> dict[str, pd.DataFrame]:
    frames = {name: pd.read_csv(path) for name, path in FILES.items()}
    sample = frames["lr"][["id"]]
    for name, frame in frames.items():
        if list(frame.columns) != ["id", TARGET]:
            raise ValueError(f"{name}: unexpected columns {list(frame.columns)}")
        if not frame["id"].equals(sample["id"]):
            raise ValueError(f"{name}: id order mismatch")
    return frames


def save(name: str, ids: pd.Series, classes: list[str], anchor: pd.Series) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": ids, TARGET: classes})
    path = OUT_DIR / f"{name}.csv"
    out.to_csv(path, index=False)
    counts = out[TARGET].value_counts().to_dict()
    diff = out[TARGET].ne(anchor).sum()
    print(f"{path} diff_vs_lr={diff} counts={counts}")


def majority_vote(labels: list[str], fallback: str) -> str:
    counts = Counter(labels)
    top_label, top_count = counts.most_common(1)[0]
    if top_count >= 2:
        return top_label
    return fallback


def weighted_vote(labels: dict[str, str], fallback: str) -> str:
    scores: dict[str, float] = {}
    for name, label in labels.items():
        scores[label] = scores.get(label, 0.0) + PUBLIC_WEIGHTS[name]
    best_score = max(scores.values())
    winners = [label for label, score in scores.items() if score == best_score]
    if len(winners) == 1:
        return winners[0]
    return fallback


def weighted_vote_with_weights(labels: dict[str, tuple[str, float]], fallback: str) -> str:
    scores: dict[str, float] = {}
    for label, weight in labels.values():
        scores[label] = scores.get(label, 0.0) + weight
    best_score = max(scores.values())
    winners = [label for label, score in scores.items() if score == best_score]
    if len(winners) == 1:
        return winners[0]
    return fallback


def vote_count(labels: list[str]) -> tuple[str, int]:
    top_label, top_count = Counter(labels).most_common(1)[0]
    return top_label, top_count


def main() -> None:
    frames = read_submissions()
    ids = frames["lr"]["id"]
    lr = frames["lr"][TARGET]

    unanimous = []
    flex_nina = []
    flex_nina_realmlp = []
    top3 = []
    lr_flex_nina_realmlp = []
    lr_lgbm_ours = []
    lr_flex_nina_lgbm_ours = []
    weighted = []
    weighted_core = []
    weighted_diverse_star = []
    star_safe = []
    strict3 = []
    qso_strict3 = []
    qso_no_two_vote = []
    star_tie_prefer = []
    realmlp_flex_nina = []

    for idx in range(len(lr)):
        base = lr.iat[idx]
        flex = frames["flex"][TARGET].iat[idx]
        nina = frames["nina"][TARGET].iat[idx]
        lgbm = frames["lgbm_cal"][TARGET].iat[idx]
        realmlp = frames["realmlp"][TARGET].iat[idx]
        ours = frames["ours13"][TARGET].iat[idx]

        unanimous.append(flex if flex == nina == lgbm else base)
        flex_nina.append(flex if flex == nina else base)
        flex_nina_realmlp.append(flex if flex == nina == realmlp else base)
        top3.append(majority_vote([base, flex, nina], base))
        lr_flex_nina_realmlp.append(majority_vote([base, flex, nina, realmlp], base))
        lr_lgbm_ours.append(majority_vote([base, lgbm, ours], base))
        lr_flex_nina_lgbm_ours.append(majority_vote([base, flex, nina, lgbm, ours], base))
        realmlp_flex_nina.append(majority_vote([realmlp, flex, nina], base))
        weighted.append(
            weighted_vote(
                {
                    "lr": base,
                    "flex": flex,
                    "nina": nina,
                    "realmlp": realmlp,
                    "lgbm_cal": lgbm,
                    "ours13": ours,
                },
                base,
            )
        )
        weighted_core.append(
            weighted_vote_with_weights(
                {
                    "lr": (base, 1.00),
                    "flex": (flex, 0.88),
                    "nina": (nina, 0.88),
                    "realmlp": (realmlp, 0.70),
                    "lgbm_cal": (lgbm, 0.45),
                },
                base,
            )
        )
        weighted_diverse_star.append(
            weighted_vote_with_weights(
                {
                    "lr": (base, 1.00),
                    "flex": (flex, 0.75),
                    "nina": (nina, 0.75),
                    "realmlp": (realmlp, 0.65),
                    "lgbm_cal": (lgbm, 0.55),
                    "ours13": (ours, 0.35),
                },
                base,
            )
        )
        external_votes = [flex, nina, realmlp, lgbm, ours]
        top_label, top_count = Counter(external_votes).most_common(1)[0]
        if top_count >= 3 and not (top_label == "QSO" and top_count < 5):
            star_safe.append(top_label)
        else:
            star_safe.append(base)
        five_votes = [base, flex, nina, lgbm, ours]
        plurality_label, plurality_count = vote_count(five_votes)
        strict3.append(plurality_label if plurality_count >= 3 else base)
        qso_strict3.append(
            base if plurality_label == "QSO" and plurality_count < 3 else plurality_label
        )
        if plurality_label == "QSO" and plurality_count < 3:
            without_qso = [label for label in five_votes if label != "QSO"]
            fallback_label, fallback_count = vote_count(without_qso)
            qso_no_two_vote.append(fallback_label if fallback_count >= 2 else base)
        else:
            qso_no_two_vote.append(plurality_label)
        counts = Counter(five_votes)
        max_count = max(counts.values())
        tied = {label for label, count in counts.items() if count == max_count}
        if "STAR" in tied and max_count >= 2:
            star_tie_prefer.append("STAR")
        elif plurality_count >= 2:
            star_tie_prefer.append(plurality_label)
        else:
            star_tie_prefer.append(base)

    save("lr_anchor_flex_nina_lgbm_unanimous", ids, unanimous, lr)
    save("lr_anchor_flex_nina_majority", ids, flex_nina, lr)
    save("lr_anchor_flex_nina_realmlp_unanimous", ids, flex_nina_realmlp, lr)
    save("top3_lr_flex_nina_majority", ids, top3, lr)
    save("hard_vote_lr_flex_nina_realmlp", ids, lr_flex_nina_realmlp, lr)
    save("lr_lgbm_ours_majority", ids, lr_lgbm_ours, lr)
    save("hard_vote_lr_flex_nina_lgbm_ours", ids, lr_flex_nina_lgbm_ours, lr)
    save("realmlp_flex_nina_majority", ids, realmlp_flex_nina, lr)
    save("weighted_public_vote", ids, weighted, lr)
    save("weighted_core_vote", ids, weighted_core, lr)
    save("weighted_diverse_star_vote", ids, weighted_diverse_star, lr)
    save("anchor_3of5_star_safe", ids, star_safe, lr)
    save("strict3_lr_flex_nina_lgbm_ours", ids, strict3, lr)
    save("qso_strict3_lr_flex_nina_lgbm_ours", ids, qso_strict3, lr)
    save("qso_no_two_vote_lr_flex_nina_lgbm_ours", ids, qso_no_two_vote, lr)
    save("star_tie_prefer_lr_flex_nina_lgbm_ours", ids, star_tie_prefer, lr)


if __name__ == "__main__":
    main()
