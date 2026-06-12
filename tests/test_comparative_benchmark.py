from pathlib import Path

from benchmarks.comparative_benchmark import load_cases, run_case, summarize_results


def test_comparative_benchmark_loads_snapshot_cases():
    cases = load_cases(Path("benchmark_data/snapshot_cases.jsonl"))
    assert len(cases) >= 5
    assert all("documents" in case for case in cases)


def test_persistence_gate_has_no_false_allows_on_snapshot_cases():
    cases = load_cases(Path("benchmark_data/snapshot_cases.jsonl"))
    rows = [run_case(case, method="persistence_gate", top_k=4, profile="balanced") for case in cases]
    assert sum(row.false_allows for row in rows) == 0
    assert sum(row.false_blocks for row in rows) == 0


def test_ordinary_top_k_allows_some_known_blocked_evidence():
    cases = load_cases(Path("benchmark_data/snapshot_cases.jsonl"))
    rows = [run_case(case, method="ordinary_top_k", top_k=4, profile="balanced") for case in cases]
    assert sum(row.false_allows for row in rows) >= 1


def test_summary_groups_methods():
    cases = load_cases(Path("benchmark_data/snapshot_cases.jsonl"))[:1]
    rows = [
        run_case(cases[0], method="ordinary_top_k", top_k=4, profile="balanced"),
        run_case(cases[0], method="persistence_gate", top_k=4, profile="balanced"),
    ]
    summaries = summarize_results(rows)
    assert {summary.method for summary in summaries} == {"ordinary_top_k", "persistence_gate"}


def test_comprehensive_cases_load_and_cover_many_domains():
    cases = load_cases(Path("benchmark_data/comprehensive_comparative_cases.jsonl"))
    assert len(cases) >= 20
    assert len({case["domain"] for case in cases}) >= 8


def test_comprehensive_benchmark_exposes_baseline_tradeoffs():
    cases = load_cases(Path("benchmark_data/comprehensive_comparative_cases.jsonl"))
    ordinary_rows = [run_case(case, method="ordinary_top_k", top_k=3, profile="balanced") for case in cases]
    recency_rows = [run_case(case, method="recency_filter", top_k=3, profile="balanced") for case in cases]
    metadata_rows = [run_case(case, method="metadata_filter", top_k=3, profile="balanced") for case in cases]
    gate_rows = [run_case(case, method="persistence_gate", top_k=3, profile="balanced") for case in cases]

    assert sum(row.false_allows for row in ordinary_rows) >= 10
    assert sum(row.false_blocks for row in recency_rows) >= 5
    assert sum(row.false_allows for row in metadata_rows) >= 10
    assert sum(row.false_allows for row in gate_rows) == 0
    assert sum(row.false_blocks for row in gate_rows) == 0
