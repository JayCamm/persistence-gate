from pathlib import Path

from benchmarks.snapshot_benchmark import load_cases, run_case


def test_snapshot_cases_load():
    cases = load_cases(Path("benchmark_data/snapshot_cases.jsonl"))
    assert len(cases) >= 5
    assert all("query" in case for case in cases)
    assert all("documents" in case for case in cases)


def test_snapshot_cases_pass_balanced_profile():
    cases = load_cases(Path("benchmark_data/snapshot_cases.jsonl"))
    results = [run_case(case, profile="balanced", top_k=4) for case in cases]
    assert all(result.passed for result in results)
    assert sum(result.false_allows for result in results) == 0
    assert sum(result.false_blocks for result in results) == 0
    assert sum(result.blocked_from_ordinary_top_k for result in results) >= 1
