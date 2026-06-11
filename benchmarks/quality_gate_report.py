from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QualityReport:
    total_cases: int
    passed_cases: int
    pass_rate: float
    mean_utility_gain: float
    mean_label_confidence: float
    low_confidence_cases: int
    negative_gain_cases: int
    verdict: str


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_quality(rows: list[dict], min_pass_rate: float = 0.75, min_mean_gain: float = 1.0, min_label_conf: float = 0.65) -> QualityReport:
    total = len(rows)
    if total == 0:
        return QualityReport(0, 0, 0.0, 0.0, 0.0, 0, 0, "FAIL_EMPTY_OR_MISSING")

    passed = sum(1 for row in rows if row.get("pass_fail") == "PASS")
    gains = [as_float(row.get("utility_gain", "0")) for row in rows]
    confidences = [as_float(row.get("evidence_confidence_mean", "0.5"), 0.5) for row in rows]
    pass_rate = passed / total
    mean_gain = sum(gains) / total
    mean_conf = sum(confidences) / total
    low_conf = sum(1 for value in confidences if value < min_label_conf)
    negative_gain = sum(1 for value in gains if value <= 0)

    if pass_rate >= min_pass_rate and mean_gain >= min_mean_gain and mean_conf >= min_label_conf and negative_gain == 0:
        verdict = "PASS_STRONG"
    elif pass_rate >= min_pass_rate and mean_gain > 0:
        verdict = "PASS_PROMISING"
    else:
        verdict = "FAIL_NEEDS_REVIEW"

    return QualityReport(total, passed, pass_rate, mean_gain, mean_conf, low_conf, negative_gain, verdict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize whether a benchmark run is strong enough to cite.")
    parser.add_argument("summary_csv", type=Path, nargs="?", default=Path("benchmark_results/real_github_issue_summary.csv"))
    parser.add_argument("--min-pass-rate", type=float, default=0.75)
    parser.add_argument("--min-mean-gain", type=float, default=1.0)
    parser.add_argument("--min-label-confidence", type=float, default=0.65)
    args = parser.parse_args()

    rows = load_rows(args.summary_csv)
    report = summarize_quality(
        rows,
        min_pass_rate=args.min_pass_rate,
        min_mean_gain=args.min_mean_gain,
        min_label_conf=args.min_label_confidence,
    )

    print("Benchmark Quality Gate")
    print("======================")
    print(f"Input CSV: {args.summary_csv}")
    if not rows:
        print("No benchmark summary rows found. The benchmark probably did not complete, often because GitHub rate-limited comment fetches.")
        print("Run a smaller benchmark or set GITHUB_TOKEN, then run this report again.")
    print(f"Cases: {report.total_cases}")
    print(f"Passed: {report.passed_cases}")
    print(f"Pass rate: {report.pass_rate:.1%}")
    print(f"Mean utility gain: {report.mean_utility_gain:.2f}")
    print(f"Mean label confidence: {report.mean_label_confidence:.2f}")
    print(f"Low-confidence cases: {report.low_confidence_cases}")
    print(f"Negative/zero-gain cases: {report.negative_gain_cases}")
    print(f"Verdict: {report.verdict}")


if __name__ == "__main__":
    main()
