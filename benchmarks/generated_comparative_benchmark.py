from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.comparative_benchmark import load_cases, run_case, summarize_results, write_csv
from benchmarks.generate_synthetic_comparative_cases import generate_cases, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and run a large synthetic comparative benchmark.")
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--profile", choices=["permissive", "balanced", "conservative"], default="balanced")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--methods",
        default="ordinary_top_k,recency_filter,metadata_filter,prompt_warning_only,persistence_gate",
        help="Comma-separated method list.",
    )
    parser.add_argument("--case-out", type=Path, default=Path("benchmark_data/generated_comparative_cases.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_results/generated_comparative"))
    args = parser.parse_args()

    cases = generate_cases(args.count, args.seed)
    write_jsonl(args.case_out, cases)

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    loaded_cases = load_cases(args.case_out)
    rows = [run_case(case, method=method, top_k=args.top_k, profile=args.profile) for case in loaded_cases for method in methods]
    summaries = summarize_results(rows)

    write_csv(args.out_dir / "generated_case_results.csv", rows)
    write_csv(args.out_dir / "generated_summary.csv", summaries)

    print("Generated Comparative Benchmark")
    print("===============================")
    print(f"Cases: {len(loaded_cases)}")
    print(f"Seed: {args.seed}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Profile: {args.profile}")
    print(f"Top-k: {args.top_k}")
    print("\nSummary:")
    for summary in summaries:
        fa_rate = summary.false_allows / summary.cases if summary.cases else 0.0
        fb_rate = summary.false_blocks / summary.cases if summary.cases else 0.0
        bad_answer_rate = summary.unsafe_answers / summary.cases if summary.cases else 0.0
        clean_rate = summary.clean_cases / summary.cases if summary.cases else 0.0
        print(
            f"{summary.method}: "
            f"false_allows={summary.false_allows} ({fa_rate:.1%}), "
            f"false_blocks={summary.false_blocks} ({fb_rate:.1%}), "
            f"flagged_answers={summary.unsafe_answers} ({bad_answer_rate:.1%}), "
            f"safe_answers={summary.safe_answers}, "
            f"clean_cases={summary.clean_cases}/{summary.cases} ({clean_rate:.1%})"
        )
    print(f"\nSaved cases: {args.case_out}")
    print(f"Saved: {args.out_dir / 'generated_case_results.csv'}")
    print(f"Saved: {args.out_dir / 'generated_summary.csv'}")


if __name__ == "__main__":
    main()
