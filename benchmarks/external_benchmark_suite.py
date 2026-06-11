from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from persistence_memory import MemoryItem, TaskContext
from persistence_memory.benchmark import evaluate_gate_vs_topk


@dataclass
class BenchmarkCase:
    track: str
    name: str
    query: str
    items: list[MemoryItem]
    top_k: int = 5
    expected: str = ""


@dataclass
class CaseSummary:
    track: str
    name: str
    expected: str
    ordinary_net: float
    gated_net: float
    utility_gain: float
    ordinary_helpful: int
    gated_helpful: int
    ordinary_risky: int
    gated_risky: int
    ordinary_stale: int
    gated_stale: int
    ordinary_uncertain: int
    gated_uncertain: int
    risky_prevented: int
    stale_prevented: int
    helpful_lost: int
    pass_fail: str
    pass_reason: str


def memory(
    memory_id: str,
    text: str,
    *,
    kind: str,
    helpful: bool = False,
    risky: bool = False,
    stale: bool = False,
    risk: float = 0.08,
    harm: float = 0.0,
    usefulness: float = 0.35,
    burden: float = 0.12,
    context_scope: str = "project",
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        text=text,
        source=f"external_benchmark:{kind}",
        context_scope=context_scope,
        risk=risk,
        harm_score=harm,
        usefulness_score=usefulness,
        burden=burden,
        metadata={
            "kind": kind,
            "label_helpful": helpful,
            "label_risky": risky,
            "label_stale": stale,
        },
    )


def clean_beir_style_cases() -> BenchmarkCase:
    items = [
        memory("clean_0", "Current documentation says Persistence Gate retrieves candidates, scores them, then gates influence before use.", kind="document", helpful=True, usefulness=0.75),
        memory("clean_1", "The controller test confirms helpful memory can become validated after repeated positive feedback.", kind="document", helpful=True, usefulness=0.70),
        memory("clean_2", "The benchmark module compares ordinary top-k against Persistence Gate using the same retrieved candidates.", kind="document", helpful=True, usefulness=0.72),
        memory("clean_3", "The retriever module provides lexical relevance ranking so the prototype can run without external embedding dependencies.", kind="document", helpful=True, usefulness=0.66),
        memory("clean_4", "The clean control benchmark verifies Persistence Gate should not overblock safe useful evidence.", kind="document", helpful=True, usefulness=0.74),
        memory("clean_5", "Repository metadata can be transformed into memory items for small demonstrations.", kind="document", helpful=True, usefulness=0.60),
        memory("clean_distractor", "A weather forecast API is not direct evidence for Persistence Gate memory governance.", kind="document", usefulness=0.05, context_scope="general"),
    ]
    return BenchmarkCase(
        track="clean_control",
        name="clean_beir_style_control",
        query="What safe current information explains how Persistence Gate works?",
        items=items,
        top_k=5,
        expected="gate_does_not_overblock",
    )


def trec_covid_style_temporal_case() -> BenchmarkCase:
    items = [
        memory(
            "old_preliminary_claim",
            "Early preliminary report: the treatment should be assumed effective based on a small uncontrolled sample. This old claim was later contradicted.",
            kind="document",
            risky=True,
            stale=True,
            risk=0.84,
            harm=0.76,
            usefulness=-0.35,
        ),
        memory(
            "current_review_claim",
            "Current systematic review: later controlled evidence did not support the preliminary treatment claim, so the old conclusion should not guide decisions.",
            kind="document",
            helpful=True,
            risk=0.04,
            usefulness=0.85,
        ),
        memory(
            "current_guideline",
            "Current guideline: rely on updated controlled evidence and avoid using superseded preliminary findings as decision support.",
            kind="document",
            helpful=True,
            risk=0.04,
            usefulness=0.82,
        ),
        memory(
            "old_press_summary",
            "Old news summary: relevant early evidence suggested strong benefit, but the article predates later corrections.",
            kind="document",
            risky=True,
            stale=True,
            risk=0.65,
            harm=0.55,
            usefulness=-0.10,
        ),
        memory("background", "Background document explains study design, sample size, confounding, and why uncontrolled evidence is weak.", kind="document", helpful=True, usefulness=0.58),
        memory("irrelevant", "A database indexing note mentions retrieval latency but does not answer the evidence question.", kind="document", usefulness=0.05, context_scope="general"),
    ]
    return BenchmarkCase(
        track="temporal_evidence",
        name="trec_covid_style_stale_evidence",
        query="What current evidence should guide the decision about the treatment claim?",
        items=items,
        top_k=5,
        expected="gate_beats_ordinary",
    )


def wiki_revision_style_case() -> BenchmarkCase:
    items = [
        memory(
            "reverted_bad_edit",
            "Reverted article version: the project is production ready and compression is proven reliable. This vandalized or incorrect edit was later reverted.",
            kind="document",
            risky=True,
            stale=True,
            risk=0.82,
            harm=0.80,
            usefulness=-0.45,
        ),
        memory(
            "current_article",
            "Current article version: Persistence Gate is a research prototype and compression has not been proven reliable yet.",
            kind="document",
            helpful=True,
            risk=0.03,
            usefulness=0.82,
        ),
        memory(
            "revert_note",
            "Revision note: reverted unsupported production-ready claim and restored cautious prototype wording.",
            kind="document",
            helpful=True,
            risk=0.04,
            usefulness=0.70,
        ),
        memory(
            "old_claim",
            "Old claim: use every relevant memory because relevance is equivalent to trust. This claim is stale and contradicted.",
            kind="document",
            risky=True,
            stale=True,
            risk=0.88,
            harm=0.82,
            usefulness=-0.50,
        ),
        memory("supporting", "Supporting note: the gate should preserve useful evidence while blocking risky or stale influence.", kind="document", helpful=True, usefulness=0.72),
    ]
    return BenchmarkCase(
        track="revision_contradiction",
        name="wiki_style_reverted_claim",
        query="Which current article claim should influence the answer about Persistence Gate status?",
        items=items,
        top_k=4,
        expected="gate_beats_ordinary",
    )


def public_issue_style_case() -> BenchmarkCase:
    items = [
        memory(
            "old_workaround_comment",
            "Old issue comment: workaround is to disable the gate and always use top-k retrieval. This was before the bug fix and is now outdated.",
            kind="issue",
            risky=True,
            stale=True,
            risk=0.82,
            harm=0.78,
            usefulness=-0.40,
        ),
        memory(
            "fix_commit",
            "Fix commit: controller now captures ordinary baseline before mutating memory states and separates blocked items from top-k overflow.",
            kind="commit",
            helpful=True,
            risk=0.04,
            usefulness=0.82,
        ),
        memory(
            "current_test",
            "Current passing test: harmful relevant memory appears in ordinary top-k but not in Persistence Gate allowed results.",
            kind="test",
            helpful=True,
            risk=0.03,
            usefulness=0.80,
        ),
        memory(
            "closed_issue",
            "Closed issue resolution: old workaround should no longer influence decisions after the controller fix landed.",
            kind="issue",
            helpful=True,
            risk=0.04,
            usefulness=0.70,
        ),
        memory("irrelevant_commit", "Commit message: update image assets and plot formatting for unrelated simulation demo.", kind="commit", usefulness=0.05, context_scope="history"),
    ]
    return BenchmarkCase(
        track="software_history",
        name="public_issue_style_old_workaround",
        query="Should the old workaround or the current fix influence the software decision?",
        items=items,
        top_k=4,
        expected="gate_beats_ordinary",
    )


def all_cases() -> list[BenchmarkCase]:
    return [
        clean_beir_style_cases(),
        trec_covid_style_temporal_case(),
        wiki_revision_style_case(),
        public_issue_style_case(),
    ]


def pass_rule(case: BenchmarkCase, result) -> tuple[bool, str]:
    """Return pass/fail and a concise explanation for the benchmark track."""
    if case.expected == "gate_does_not_overblock":
        # Clean controls should not require a win. They should prove the gate does
        # not introduce risk or discard a meaningful amount of safe useful evidence.
        if result.gated.risky_selected > 0:
            return False, "clean_control_failed_added_risk"
        if result.helpful_items_lost > 1:
            return False, "clean_control_failed_overblocked_helpful_items"
        if result.utility_gain < -0.75:
            return False, "clean_control_failed_large_utility_loss"
        return True, "clean_control_passed_no_overblocking"

    if result.utility_gain > 0 and result.risky_items_prevented > 0 and result.stale_items_prevented >= 0:
        return True, "gate_prevented_risky_or_stale_influence"
    return False, "gate_did_not_outperform_on_risky_or_stale_track"


def summarize_case(case: BenchmarkCase) -> CaseSummary:
    task = TaskContext(query=case.query, context_scope="project", need=0.95, risk_tolerance=0.40, abstention_score=0.04)
    result = evaluate_gate_vs_topk(case.items, task=task, top_k=case.top_k)

    ordinary_net = result.ordinary.net_utility()
    gated_net = result.gated.net_utility()
    passed, reason = pass_rule(case, result)

    return CaseSummary(
        track=case.track,
        name=case.name,
        expected=case.expected,
        ordinary_net=ordinary_net,
        gated_net=gated_net,
        utility_gain=result.utility_gain,
        ordinary_helpful=result.ordinary.helpful_selected,
        gated_helpful=result.gated.helpful_selected,
        ordinary_risky=result.ordinary.risky_selected,
        gated_risky=result.gated.risky_selected,
        ordinary_stale=result.ordinary.stale_selected,
        gated_stale=result.gated.stale_selected,
        ordinary_uncertain=result.ordinary.uncertain_selected,
        gated_uncertain=result.gated.uncertain_selected,
        risky_prevented=result.risky_items_prevented,
        stale_prevented=result.stale_items_prevented,
        helpful_lost=result.helpful_items_lost,
        pass_fail="PASS" if passed else "FAIL",
        pass_reason=reason,
    )


def write_csv(path: Path, summaries: list[CaseSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external-style benchmark suite for Persistence Gate.")
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/external_benchmark_summary.csv"))
    parser.add_argument("--json", type=Path, default=Path("benchmark_results/external_benchmark_summary.json"))
    args = parser.parse_args()

    summaries = [summarize_case(case) for case in all_cases()]

    print("External Benchmark Suite")
    print("========================")
    for summary in summaries:
        print(
            f"{summary.pass_fail} | {summary.track} | {summary.name} | "
            f"ordinary={summary.ordinary_net:.2f} gated={summary.gated_net:.2f} gain={summary.utility_gain:.2f} | "
            f"risk_prevented={summary.risky_prevented} stale_prevented={summary.stale_prevented} helpful_lost={summary.helpful_lost} | "
            f"reason={summary.pass_reason}"
        )

    passed = sum(1 for summary in summaries if summary.pass_fail == "PASS")
    print(f"\nPassed {passed}/{len(summaries)} tracks")

    write_csv(args.out, summaries)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps([asdict(summary) for summary in summaries], indent=2), encoding="utf-8")
    print(f"Saved CSV: {args.out}")
    print(f"Saved JSON: {args.json}")


if __name__ == "__main__":
    main()
