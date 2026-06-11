from benchmarks.quality_gate_report import summarize_quality


def test_quality_gate_passes_strong_run():
    rows = [
        {"pass_fail": "PASS", "utility_gain": "2.0", "evidence_confidence_mean": "0.8"},
        {"pass_fail": "PASS", "utility_gain": "1.5", "evidence_confidence_mean": "0.7"},
    ]
    report = summarize_quality(rows)
    assert report.verdict == "PASS_STRONG"
    assert report.total_cases == 2
    assert report.negative_gain_cases == 0


def test_quality_gate_flags_weak_run():
    rows = [
        {"pass_fail": "WEAK", "utility_gain": "-0.1", "evidence_confidence_mean": "0.5"},
        {"pass_fail": "PASS", "utility_gain": "0.4", "evidence_confidence_mean": "0.6"},
    ]
    report = summarize_quality(rows)
    assert report.verdict == "FAIL_NEEDS_REVIEW"
    assert report.negative_gain_cases == 1
