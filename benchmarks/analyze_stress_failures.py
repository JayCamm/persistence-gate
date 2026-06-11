from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def classify_fault(row: dict) -> str:
    scenario = row.get("scenario", "")
    gain = as_float(row.get("utility_gain"))
    risky_prevented = as_int(row.get("risky_prevented"))
    stale_prevented = as_int(row.get("stale_prevented"))
    gated_risky = as_int(row.get("gated_risky"))
    gated_stale = as_int(row.get("gated_stale"))
    ordinary_risky = as_int(row.get("ordinary_risky"))
    helpful_lost = as_int(row.get("helpful_lost"))

    if scenario == "clean_control":
        if gain <= 0:
            return "clean_control_expected_no_gain"
        return "clean_control_gain"

    if gated_risky > 0 or gated_stale > 0:
        return "gate_allowed_risky_or_stale"
    if ordinary_risky == 0 and risky_prevented == 0 and stale_prevented == 0:
        return "ordinary_baseline_did_not_select_bad_item"
    if helpful_lost > 0:
        return "gate_lost_helpful_evidence"
    if gain <= 0:
        return "no_positive_net_gain"
    return "passed_or_nonfault"


def summarize(rows: list[dict]) -> None:
    failures = [row for row in rows if row.get("pass_fail") != "PASS"]
    nonpositive = [row for row in rows if as_float(row.get("utility_gain")) <= 0]

    print("Stress Failure Analysis")
    print("=======================")
    print(f"Rows: {len(rows)}")
    print(f"Weak/failing rows: {len(failures)}")
    print(f"Negative/zero-gain rows: {len(nonpositive)}")

    by_fault = Counter(classify_fault(row) for row in rows)
    print("\nFault classes")
    for label, count in by_fault.most_common():
        print(f"{label}: {count}")

    print("\nFailures by domain/scenario")
    table: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in failures:
        table[(row.get("domain", ""), row.get("scenario", ""))].append(row)
    for (domain, scenario), group in sorted(table.items()):
        mean_gain = sum(as_float(row.get("utility_gain")) for row in group) / max(1, len(group))
        fault_counts = Counter(classify_fault(row) for row in group)
        fault_text = ", ".join(f"{name}={count}" for name, count in fault_counts.most_common())
        print(f"{domain}/{scenario}: {len(group)} failures, mean_gain={mean_gain:.2f}, {fault_text}")

    print("\nImplementation interpretation")
    risky_failures = [row for row in failures if row.get("scenario") == "risky_workaround"]
    if risky_failures:
        domains = Counter(row.get("domain") for row in risky_failures)
        print("Risky-workaround failures are concentrated in:")
        for domain, count in domains.most_common():
            print(f"- {domain}: {count}")
        print("Likely logical fix: add a conservative safety profile that hard-blocks high-risk/high-harm workaround items even when lexical relevance is high, while keeping balanced mode unchanged.")


def write_fault_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["fault_class"]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            enriched = dict(row)
            enriched["fault_class"] = classify_fault(row)
            writer.writerow(enriched)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze weak and nonpositive-gain rows from the multi-domain stress benchmark.")
    parser.add_argument("summary_csv", type=Path, nargs="?", default=Path("benchmark_results/multi_domain_stress_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/multi_domain_stress_faults.csv"))
    args = parser.parse_args()

    rows = load_rows(args.summary_csv)
    summarize(rows)
    write_fault_csv(rows, args.out)
    print(f"\nSaved fault CSV: {args.out}")


if __name__ == "__main__":
    main()
