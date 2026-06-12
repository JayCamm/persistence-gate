from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.comparative_benchmark import load_cases, run_case, summarize_results, write_csv
from benchmarks.generate_gray_zone_comparative_cases import generate_cases, write_jsonl


@dataclass(frozen=True)
class ProfileSummary:
    profile: str
    cases: int
    false_allows: int
    false_blocks: int
    flagged_answers: int
    safe_answers: int
    clean_cases: int
    false_allow_rate: float
    false_block_rate: float
    flagged_answer_rate: float
    clean_rate: float


def write_profile_summary(path: Path, rows: list[ProfileSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gray-zone generated cases across gate profiles.")
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--profiles", default="permissive,balanced,conservative")
    parser.add_argument("--case-out", type=Path, default=Path("benchmark_data/generated_gray_zone_cases.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_results/gray_zone_profile_sensitivity"))
    args = parser.parse_args()

    cases = generate_cases(args.count, args.seed)
    write_jsonl(args.case_out, cases)
    loaded_cases = load_cases(args.case_out)
    profiles = [profile.strip() for profile in args.profiles.split(",") if profile.strip()]

    all_case_rows = []
    profile_rows: list[ProfileSummary] = []
    for profile in profiles:
        case_rows = [
            run_case(case, method="persistence_gate", top_k=args.top_k, profile=profile)
            for case in loaded_cases
        ]
        labeled_case_rows = [replace(row, method=f"persistence_gate:{profile}") for row in case_rows]
        all_case_rows.extend(labeled_case_rows)
        summary = summarize_results(case_rows)[0]
        profile_rows.append(
            ProfileSummary(
                profile=profile,
                cases=summary.cases,
                false_allows=summary.false_allows,
                false_blocks=summary.false_blocks,
                flagged_answers=summary.unsafe_answers,
                safe_answers=summary.safe_answers,
                clean_cases=summary.clean_cases,
                false_allow_rate=summary.false_allows / summary.cases if summary.cases else 0.0,
                false_block_rate=summary.false_blocks / summary.cases if summary.cases else 0.0,
                flagged_answer_rate=summary.unsafe_answers / summary.cases if summary.cases else 0.0,
                clean_rate=summary.clean_cases / summary.cases if summary.cases else 0.0,
            )
        )

    write_csv(args.out_dir / "gray_zone_case_results.csv", all_case_rows)
    write_profile_summary(args.out_dir / "gray_zone_summary.csv", profile_rows)

    print("Gray-Zone Profile Sensitivity Benchmark")
    print("=======================================")
    print(f"Cases: {len(loaded_cases)}")
    print(f"Seed: {args.seed}")
    print(f"Profiles: {', '.join(profiles)}")
    print(f"Top-k: {args.top_k}")
    print("\nSummary:")
    for row in profile_rows:
        print(
            f"{row.profile}: "
            f"false_allows={row.false_allows} ({row.false_allow_rate:.1%}), "
            f"false_blocks={row.false_blocks} ({row.false_block_rate:.1%}), "
            f"flagged_answers={row.flagged_answers} ({row.flagged_answer_rate:.1%}), "
            f"safe_answers={row.safe_answers}, "
            f"clean_cases={row.clean_cases}/{row.cases} ({row.clean_rate:.1%})"
        )
    print(f"\nSaved cases: {args.case_out}")
    print(f"Saved: {args.out_dir / 'gray_zone_case_results.csv'}")
    print(f"Saved: {args.out_dir / 'gray_zone_summary.csv'}")


if __name__ == "__main__":
    main()
