#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from datetime import datetime, timezone


def run_kaggle(args: list[str]) -> str:
    result = subprocess.run(args, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def parse_submissions(text: str) -> list[dict[str, str]]:
    if not text or "No submissions found" in text:
        return []
    try:
        return list(csv.DictReader(io.StringIO(text)))
    except csv.Error:
        return []


def submission_datetime(row: dict[str, str]) -> datetime | None:
    for key in ["date", "submissionDate", "submitted", "Date"]:
        value = row.get(key)
        if not value:
            continue
        value = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default="playground-series-s6e6")
    parser.add_argument("--daily-limit", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    output = run_kaggle(["kaggle", "competitions", "submissions", "-c", args.competition, "-v"])
    rows = parse_submissions(output)
    today = datetime.now(timezone.utc).date()
    today_rows = [row for row in rows if (submission_datetime(row) or datetime.min.replace(tzinfo=timezone.utc)).date() == today]
    payload = {
        "competition": args.competition,
        "utc_date": str(today),
        "daily_limit": args.daily_limit,
        "submissions_today": len(today_rows),
        "estimated_remaining": max(0, args.daily_limit - len(today_rows)),
        "total_submissions_listed": len(rows),
        "authoritative_check": "Submit acceptance/rejection and a new submission-list record determine actual capacity.",
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    for key, value in payload.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

