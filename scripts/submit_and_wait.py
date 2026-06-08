#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import time
from pathlib import Path


def run(args: list[str], retries: int = 5) -> str:
    delay = 30
    for attempt in range(retries + 1):
        result = subprocess.run(args, text=True, capture_output=True)
        if result.returncode == 0:
            return result.stdout.strip()
        message = result.stderr.strip() or result.stdout.strip()
        transient = any(
            token in message
            for token in [
                "429",
                "Connection timed out",
                "ConnectTimeoutError",
                "Max retries exceeded",
                "Read timed out",
                "Remote end closed connection",
                "Temporary failure",
            ]
        )
        if not transient or attempt == retries:
            raise RuntimeError(message)
        print(f"transient_kaggle_error: sleeping {delay}s before retry")
        time.sleep(delay)
        delay = min(delay * 2, 180)
    raise RuntimeError("Unexpected retry loop exit")


def submissions_table(competition: str) -> str:
    return run(["kaggle", "competitions", "submissions", "-c", competition])


def submissions_csv(competition: str) -> list[dict[str, str]]:
    text = run(["kaggle", "competitions", "submissions", "-c", competition, "-v"])
    if not text or "No submissions found" in text:
        return []
    return list(csv.DictReader(io.StringIO(text)))


def count_submission_rows(text: str) -> int:
    if not text or "No submissions found" in text:
        return 0
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= 2:
        return 0
    return max(0, len(lines) - 2)


def public_score(row: dict[str, str]) -> str:
    for key, value in row.items():
        normalized = key.lower().replace("_", "")
        if normalized in {"publicscore", "publicleaderboardscore", "score"}:
            cleaned = (value or "").strip()
            if cleaned and cleaned.lower() not in {"none", "null", "nan"}:
                return cleaned
    return ""


def submission_ref(row: dict[str, str] | None) -> str:
    if not row:
        return ""
    return (row.get("ref") or row.get("id") or row.get("submissionId") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("-m", "--message", required=True)
    parser.add_argument("-c", "--competition", default="playground-series-s6e6")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-minutes", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait-score", action="store_true")
    args = parser.parse_args()

    if not args.submission.exists():
        raise FileNotFoundError(args.submission)

    before_table = submissions_table(args.competition)
    before_rows = submissions_csv(args.competition)
    before_count = len(before_rows) if before_rows else count_submission_rows(before_table)
    before_ref = submission_ref(before_rows[0]) if before_rows else ""
    print(f"submissions_before: {before_count}")
    if before_ref:
        print(f"latest_ref_before: {before_ref}")

    if args.dry_run:
        print("dry_run: submission not sent")
        return

    print(run([
        "kaggle",
        "competitions",
        "submit",
        "-c",
        args.competition,
        "-f",
        str(args.submission),
        "-m",
        args.message,
    ]))

    deadline = time.time() + args.timeout_minutes * 60
    last = ""
    while time.time() < deadline:
        time.sleep(args.poll_seconds)
        current_table = submissions_table(args.competition)
        current_rows = submissions_csv(args.competition)
        current_count = len(current_rows) if current_rows else count_submission_rows(current_table)
        current_ref = submission_ref(current_rows[0]) if current_rows else ""
        new_record = (current_ref and current_ref != before_ref) or current_count > before_count
        if current_table != last:
            print(current_table)
            last = current_table
        if new_record and args.no_wait_score:
            print("submission_record_created: true")
            return
        if new_record and current_rows:
            score = public_score(current_rows[0])
            if score:
                print(f"public_score: {score}")
                return

    raise TimeoutError("Timed out waiting for a new scored submission record.")


if __name__ == "__main__":
    main()
