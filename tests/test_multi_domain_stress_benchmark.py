from benchmarks.multi_domain_stress_benchmark import build_case, run_stress_case, summarize_group
import random


def test_stress_case_builder_returns_memories():
    rng = random.Random(1)
    name, query, memories = build_case(1, "software_policy", "stale_contradiction", rng)
    assert "software_policy" in name
    assert "current answer" in query
    assert len(memories) >= 5
    assert any(memory.metadata.get("label_stale") for memory in memories)
    assert all("label_confidence" in memory.metadata for memory in memories)


def test_stress_case_can_run_with_refined_verdict():
    rng = random.Random(1)
    name, query, memories = build_case(1, "enterprise_policy", "risky_workaround", rng)
    summary = run_stress_case(name, query, memories, top_k=4, profile="balanced")
    assert summary.case_id == name
    assert summary.profile == "balanced"
    assert summary.utility_gain is not None
    assert summary.evidence_confidence_mean > 0.5
    assert summary.verdict in {
        "PASS_IMPROVED",
        "PASS_CLEAN_NO_HARM",
        "PASS_NO_RISK_EXPOSURE",
        "FAIL_ALLOWED_RISK",
        "FAIL_LOST_HELPFUL",
        "FAIL_NET_HARM",
        "WEAK_NO_GAIN",
    }
    assert summary.pass_fail in {"PASS", "FAIL", "WEAK"}


def test_group_summary_reports_failure_count():
    rng = random.Random(1)
    rows = []
    for scenario in ["clean_control", "stale_contradiction", "risky_workaround", "ambiguous_mixed"]:
        name, query, memories = build_case(1, "support_knowledge", scenario, rng)
        rows.append(run_stress_case(name, query, memories, top_k=4))
    summary = summarize_group("domain:support_knowledge", rows)
    assert summary.cases == 4
    assert 0 <= summary.pass_rate <= 1
    assert 0 <= summary.failed <= 4
    assert summary.mean_confidence > 0.5


def test_conservative_profile_blocks_high_risk_workaround():
    rng = random.Random(1)
    name, query, memories = build_case(1, "software_policy", "risky_workaround", rng)
    summary = run_stress_case(name, query, memories, top_k=4, profile="conservative")
    assert summary.gated_risky == 0
    assert summary.gated_stale == 0
