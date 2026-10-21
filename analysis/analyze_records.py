#!/usr/bin/env python3
"""Reproduce the freshness-risk analysis using only Python's standard library."""

import csv
import hashlib
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "ai_eval_records.csv"
ALERT_LAG_MINUTES = 15
REJECT_COST_MINUTES = 90
ALERT_COST_MINUTES = 10


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_rows():
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        required = {
            "record_id",
            "query_time",
            "kb_synced_at",
            "tenant_group",
            "domain_category",
            "confidence_score",
            "answer_token_len",
            "is_valid",
        }
        if fields is None or not required.issubset(fields):
            raise ValueError("CSV is missing one or more required columns")
        raw_rows = list(reader)

    unique_rows = []
    seen_rows = set()
    id_fingerprints = defaultdict(set)
    for row in raw_rows:
        fingerprint = tuple(row[field] for field in fields)
        id_fingerprints[row["record_id"]].add(fingerprint)
        if fingerprint in seen_rows:
            continue
        seen_rows.add(fingerprint)
        row["lag_minutes"] = (
            parse_time(row["query_time"]) - parse_time(row["kb_synced_at"])
        ).total_seconds() / 60
        if row["lag_minutes"] < 0:
            raise ValueError(f"KB sync occurs after query for record {row['record_id']}")
        if row["is_valid"] not in {"", "0", "1"}:
            raise ValueError(f"Unexpected is_valid value: {row['is_valid']!r}")
        unique_rows.append(row)

    conflicting_ids = sum(len(fingerprints) > 1 for fingerprints in id_fingerprints.values())
    return raw_rows, unique_rows, conflicting_ids


def labeled(rows):
    return [row for row in rows if row["is_valid"] in {"0", "1"}]


def reject_count(rows):
    return sum(row["is_valid"] == "0" for row in rows)


def pct(numerator, denominator):
    return f"{100 * numerator / denominator:.1f}%" if denominator else "n/a"


def alert_metrics(rows, threshold=ALERT_LAG_MINUTES):
    alerts = [row for row in rows if row["lag_minutes"] > threshold]
    known_alerts = labeled(alerts)
    rejects = reject_count(known_alerts)
    known_rejects = reject_count(labeled(rows))
    pending_alerts = sum(row["is_valid"] == "" for row in alerts)
    gross_minutes = rejects * REJECT_COST_MINUTES
    marking_minutes = len(alerts) * ALERT_COST_MINUTES
    return {
        "alerts": len(alerts),
        "known_alerts": len(known_alerts),
        "pending_alerts": pending_alerts,
        "captured_rejects": rejects,
        "valid_alerts": sum(row["is_valid"] == "1" for row in alerts),
        "precision": pct(rejects, len(known_alerts)),
        "recall": pct(rejects, known_rejects),
        "flag_rate": pct(len(alerts), len(rows)),
        "known_rejection_rate": pct(reject_count(labeled(rows)), len(labeled(rows))),
        "gross_avoided_minutes_if_all_captured_rejects_prevented": gross_minutes,
        "alert_marking_minutes": marking_minutes,
        "net_minutes_if_all_captured_rejects_prevented": gross_minutes - marking_minutes,
        "break_even_prevented_share_of_captured_rejects": pct(
            math.ceil(marking_minutes / REJECT_COST_MINUTES), rejects
        ),
    }


def print_metrics(name, rows):
    result = alert_metrics(rows)
    print(f"\n{name}")
    print(f"  rows={len(rows)} labeled={len(labeled(rows))} rejects={reject_count(labeled(rows))}")
    print(f"  baseline_rejection_rate={result['known_rejection_rate']}")
    print(
        "  alerts={alerts} known={known_alerts} pending={pending_alerts} "
        "valid={valid_alerts} captured_rejects={captured_rejects}".format(**result)
    )
    print(
        "  precision={precision} recall={recall} flag_rate={flag_rate}".format(**result)
    )
    print(
        "  potential_gross_saved={gross_avoided_minutes_if_all_captured_rejects_prevented} "
        "min alert_cost={alert_marking_minutes} min "
        "potential_net={net_minutes_if_all_captured_rejects_prevented} min".format(**result)
    )
    print(
        "  break_even_prevented_share={break_even_prevented_share_of_captured_rejects}".format(
            **result
        )
    )


def main():
    raw_rows, rows, conflicting_ids = load_rows()
    source_hash = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    labels = Counter(row["is_valid"] or "pending" for row in rows)
    dates = sorted({row["query_time"][:10] for row in rows})
    split_date = dates[-2]
    train = [row for row in rows if row["query_time"][:10] < split_date]
    test = [row for row in rows if row["query_time"][:10] >= split_date]

    print("SOURCE AND DATA QUALITY")
    print(f"  source={CSV_PATH.relative_to(ROOT)}")
    print(f"  sha256={source_hash}")
    print(f"  raw_rows={len(raw_rows)} exact_duplicate_rows_removed={len(raw_rows) - len(rows)}")
    print(f"  distinct_rows={len(rows)} conflicting_record_ids={conflicting_ids}")
    print(f"  labels={dict(sorted(labels.items()))}")
    print(f"  query_dates={dates[0]}..{dates[-1]} time_split_date={split_date}")

    invalid = [row for row in rows if row["is_valid"] == "0"]
    valid = [row for row in rows if row["is_valid"] == "1"]
    print("\nFRESHNESS BY LABEL (minutes between query and last KB sync)")
    for name, group in (("rejected", invalid), ("valid", valid)):
        lags = sorted(row["lag_minutes"] for row in group)
        median = lags[len(lags) // 2]
        mean = sum(lags) / len(lags)
        print(f"  {name}: n={len(group)} mean={mean:.1f} median={median:.1f}")

    print("\nREJECTION RATE BY FRESHNESS BAND")
    bands = (
        ("<=15", lambda lag: lag <= 15),
        ("15<lag<=30", lambda lag: 15 < lag <= 30),
        (">30", lambda lag: lag > 30),
    )
    for name, predicate in bands:
        group = [row for row in rows if predicate(row["lag_minutes"])]
        known = labeled(group)
        print(
            f"  {name}: rows={len(group)} labeled={len(known)} pending={len(group) - len(known)} "
            f"rejects={reject_count(known)} rate={pct(reject_count(known), len(known))}"
        )

    print("\nFRESHNESS ASSOCIATION WITHIN TENANT GROUP")
    for tenant in sorted({row["tenant_group"] for row in rows}):
        print(f"  {tenant}")
        for name, predicate in (("<=15", lambda lag: lag <= 15), (">15", lambda lag: lag > 15)):
            group = labeled(
                [
                    row
                    for row in rows
                    if row["tenant_group"] == tenant and predicate(row["lag_minutes"])
                ]
            )
            print(
                f"    {name}: n={len(group)} rejects={reject_count(group)} "
                f"rate={pct(reject_count(group), len(group))}"
            )

    print("\nCONFIDENCE SCORE CHECK (0-9, 10-19, 20-30)")
    for name, predicate in (
        ("0-9", lambda score: 0 <= score <= 9),
        ("10-19", lambda score: 10 <= score <= 19),
        ("20-30", lambda score: 20 <= score <= 30),
    ):
        group = labeled(
            [row for row in rows if predicate(int(row["confidence_score"]))]
        )
        print(
            f"  {name}: n={len(group)} rejects={reject_count(group)} "
            f"rate={pct(reject_count(group), len(group))}"
        )

    sweep = [
        (alert_metrics(train, threshold)["net_minutes_if_all_captured_rejects_prevented"], threshold)
        for threshold in range(66)
    ]
    best_net = max(net for net, _ in sweep)
    best_thresholds = [threshold for net, threshold in sweep if net == best_net]
    print("\nTRAIN-ONLY THRESHOLD SWEEP (0..65 MIN, NET COUNTS PENDING ALERT COST)")
    print(f"  best_potential_net={best_net} min at strict thresholds={best_thresholds}")
    print("  selected operational threshold=15 min, close to the training optimum")

    print_metrics("ALL DATES, ALERT WHEN KB LAG > 15 MIN", rows)
    print_metrics("TRAIN PERIOD (ALL BUT LAST TWO DATES)", train)
    print_metrics("TIME-ORDERED CHECK (LAST TWO DATES)", test)

    print("\nCOST ASSUMPTIONS")
    print(f"  one_alert_mark={ALERT_COST_MINUTES} minutes")
    print(f"  one_rejection_rework={REJECT_COST_MINUTES} minutes")
    print(f"  break_even_rejection_risk={pct(ALERT_COST_MINUTES, REJECT_COST_MINUTES)}")
    print("  potential savings assume the intervention prevents a captured rejection.")
    print("  The time-ordered check is exploratory, not an untouched prospective trial.")


if __name__ == "__main__":
    main()
