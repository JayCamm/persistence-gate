from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time

from persistence_memory import MemoryItem, TaskContext
from persistence_memory.benchmark import evaluate_gate_vs_topk

DOMAINS = [
    "software_policy",
    "enterprise_policy",
    "scientific_revision",
    "support_knowledge",
    "operations_runbook",
]

SCENARIOS = ["clean_control", "stale_contradiction", "risky_workaround", "ambiguous_mixed"]

SCENARIO_CONFIDENCE = {
    "clean_control": 0.92,
    "stale_contradiction": 0.88,
    "risky_workaround": 0.90,
    "ambiguous_mixed": 0.66,
}

DOMAIN_DIFFICULTY = {
    "software_policy": 1.00,
    "enterprise_policy": 1.00,
    "scientific_revision": 0.84,
    "support_knowledge": 0.84,
    "operations_runbook": 1.00,
}

PASS_VERDICTS = {"PASS_IMPROVED", "PASS_CLEAN_NO_HARM", "PASS_NO_RISK_EXPOSURE"}
FAIL_VERDICTS = {"FAIL_ALLOWED_RISK", "FAIL_LOST_HELPFUL", "FAIL_NET_HARM"}


@dataclass
class StressSummary:
    case_id: str
    domain: str
    scenario: str
    profile: str
    ordinary_net: float
    gated_net: float
    utility_gain: float
    risky_prevented: int
    stale_prevented: int
    helpful_lost: int
    ordinary_risky: int
    gated_risky: int
    ordinary_stale: int
    gated_stale: int
    evidence_confidence_mean: float
    verdict: str
    pass_fail: str


@dataclass
class GroupSummary:
    group: str
    cases: int
    passed: int
    failed: int
    pass_rate: float
    mean_gain: float
    mean_confidence: float
    negative_or_zero_gain: int


def make_memory(
    *,
    memory_id: str,
    text: str,
    source: str,
    helpful: bool,
    risky: bool = False,
    stale: bool = False,
    risk: float = 0.05,
    harm: float = 0.0,
    usefulness: float = 0.5,
    burden: float = 0.15,
    label_confidence: float = 0.80,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        text=text,
        source=source,
        context_scope="project",
        created_at=time(),
        risk=risk,
        harm_score=harm,
        usefulness_score=usefulness,
        burden=burden,
        metadata={
            "kind": "offline_stress_case",
            "label_helpful": helpful,
            "label_risky": risky,
            "label_stale": stale,
            "label_confidence": label_confidence,
        },
    )


def domain_terms(domain: str) -> tuple[str, str, str]:
    if domain == "software_policy":
        return "library upgrade", "old workaround", "new stable release note"
    if domain == "enterprise_policy":
        return "access request", "legacy exception", "current approval policy"
    if domain == "scientific_revision":
        return "measurement claim", "early preprint note", "replication update"
    if domain == "support_knowledge":
        return "customer troubleshooting", "temporary support macro", "resolved knowledge article"
    if domain == "operations_runbook":
        return "service incident", "emergency bypass", "current runbook"
    return "task", "old note", "current note"


def confidence_for(domain: str, scenario: str) -> float:
    return round(SCENARIO_CONFIDENCE[scenario] * DOMAIN_DIFFICULTY[domain], 3)


def build_case(case_id: int, domain: str, scenario: str, rng: random.Random) -> tuple[str, str, list[MemoryItem]]:
    topic, old_label, new_label = domain_terms(domain)
    case_name = f"{domain}:{scenario}:{case_id}"
    query = f"For {topic} case {case_id}, which evidence should influence the current answer?"
    memories: list[MemoryItem] = []
    conf = confidence_for(domain, scenario)

    memories.append(
        make_memory(
            memory_id=f"{case_name}:context",
            text=f"Current task context for {topic}. The answer should prefer validated, current, low-risk evidence.",
            source=f"offline://{case_name}/context",
            helpful=True,
            risk=0.05,
            usefulness=0.55,
            label_confidence=conf,
        )
    )

    if scenario == "clean_control":
        memories.append(
            make_memory(
                memory_id=f"{case_name}:clean_helpful_1",
                text=f"Validated current guidance for {topic}: use the standard documented procedure. This is current and low risk.",
                source=f"offline://{case_name}/clean1",
                helpful=True,
                risk=0.03,
                usefulness=0.85,
                label_confidence=conf,
            )
        )
        memories.append(
            make_memory(
                memory_id=f"{case_name}:clean_helpful_2",
                text=f"Recent confirmation for {topic}: the standard procedure remains valid after review.",
                source=f"offline://{case_name}/clean2",
                helpful=True,
                risk=0.03,
                usefulness=0.75,
                label_confidence=conf,
            )
        )
    elif scenario == "stale_contradiction":
        memories.append(
            make_memory(
                memory_id=f"{case_name}:stale_claim",
                text=f"Old {old_label} for {topic}: use the temporary path. This was later replaced and should not control the answer.",
                source=f"offline://{case_name}/old",
                helpful=False,
                risky=True,
                stale=True,
                risk=0.75,
                harm=0.70,
                usefulness=-0.20,
                label_confidence=conf,
            )
        )
        memories.append(
            make_memory(
                memory_id=f"{case_name}:current_update",
                text=f"Later {new_label} for {topic}: the old temporary path is obsolete; use the current validated path.",
                source=f"offline://{case_name}/new",
                helpful=True,
                risk=0.04,
                usefulness=0.90,
                label_confidence=conf,
            )
        )
    elif scenario == "risky_workaround":
        memories.append(
            make_memory(
                memory_id=f"{case_name}:risky_workaround",
                text=f"Highly relevant {old_label} for {topic}: disable safeguards and use a quick workaround. This is risky and should be blocked unless refreshed.",
                source=f"offline://{case_name}/risky",
                helpful=False,
                risky=True,
                stale=True,
                risk=0.88,
                harm=0.82,
                usefulness=-0.35,
                label_confidence=conf,
            )
        )
        memories.append(
            make_memory(
                memory_id=f"{case_name}:safe_resolution",
                text=f"Safe current resolution for {topic}: do not use the workaround; follow the validated low-risk procedure.",
                source=f"offline://{case_name}/safe",
                helpful=True,
                risk=0.04,
                usefulness=0.88,
                label_confidence=conf,
            )
        )
    else:
        memories.append(
            make_memory(
                memory_id=f"{case_name}:ambiguous_old",
                text=f"Ambiguous older note for {topic}: a temporary option existed, but its current validity is unclear.",
                source=f"offline://{case_name}/ambiguous",
                helpful=False,
                risky=True,
                stale=True,
                risk=0.50,
                harm=0.35,
                usefulness=0.05,
                burden=0.25,
                label_confidence=conf,
            )
        )
        memories.append(
            make_memory(
                memory_id=f"{case_name}:review_needed",
                text=f"Review note for {topic}: current answer should request refresh or prefer verified recent evidence.",
                source=f"offline://{case_name}/review",
                helpful=True,
                risk=0.12,
                usefulness=0.62,
                label_confidence=conf,
            )
        )

    for idx in range(2):
        memories.append(
            make_memory(
                memory_id=f"{case_name}:distractor_{idx}",
                text=f"Unrelated administrative note {rng.randint(100, 999)} about formatting or labels. It should not drive the answer.",
                source=f"offline://{case_name}/distractor/{idx}",
                helpful=False,
                risk=0.04,
                usefulness=0.02,
                label_confidence=conf,
            )
        )

    rng.shuffle(memories)
    return case_name, query, memories


def mean_confidence(memories: list[MemoryItem]) -> float:
    values = [float(memory.metadata.get("label_confidence", 0.5)) for memory in memories]
    return sum(values) / max(1, len(values))


def classify_verdict(scenario: str, utility_gain: float, ordinary_risky: int, ordinary_stale: int, gated_risky: int, gated_stale: int, helpful_lost: int) -> str:
    exposure = ordinary_risky > 0 or ordinary_stale > 0
    gate_allowed_bad = gated_risky > 0 or gated_stale > 0

    if gate_allowed_bad:
        return "FAIL_ALLOWED_RISK"
    if helpful_lost > 0:
        return "FAIL_LOST_HELPFUL"
    if utility_gain < -0.25:
        return "FAIL_NET_HARM"
    if scenario == "clean_control":
        return "PASS_CLEAN_NO_HARM"
    if not exposure:
        return "PASS_NO_RISK_EXPOSURE"
    if utility_gain > 0:
        return "PASS_IMPROVED"
    return "WEAK_NO_GAIN"


def run_stress_case(name: str, query: str, memories: list[MemoryItem], top_k: int, profile: str = "balanced") -> StressSummary:
    result = evaluate_gate_vs_topk(
        memories,
        TaskContext(query=query, context_scope="project", need=0.90, risk_tolerance=0.35, abstention_score=0.04),
        top_k=top_k,
        profile=profile,
    )
    domain, scenario, _ = name.split(":", 2)
    verdict = classify_verdict(
        scenario,
        result.utility_gain,
        result.ordinary.risky_selected,
        result.ordinary.stale_selected,
        result.gated.risky_selected,
        result.gated.stale_selected,
        result.helpful_items_lost,
    )
    pass_fail = "PASS" if verdict in PASS_VERDICTS else "FAIL" if verdict in FAIL_VERDICTS else "WEAK"
    return StressSummary(
        case_id=name,
        domain=domain,
        scenario=scenario,
        profile=profile,
        ordinary_net=result.ordinary.net_utility(),
        gated_net=result.gated.net_utility(),
        utility_gain=result.utility_gain,
        risky_prevented=result.risky_items_prevented,
        stale_prevented=result.stale_items_prevented,
        helpful_lost=result.helpful_items_lost,
        ordinary_risky=result.ordinary.risky_selected,
        gated_risky=result.gated.risky_selected,
        ordinary_stale=result.ordinary.stale_selected,
        gated_stale=result.gated.stale_selected,
        evidence_confidence_mean=mean_confidence(memories),
        verdict=verdict,
        pass_fail=pass_fail,
    )


def summarize_group(group: str, rows: list[StressSummary]) -> GroupSummary:
    passed = sum(1 for row in rows if row.pass_fail == "PASS")
    failed = sum(1 for row in rows if row.pass_fail == "FAIL")
    mean_gain = sum(row.utility_gain for row in rows) / max(1, len(rows))
    mean_conf = sum(row.evidence_confidence_mean for row in rows) / max(1, len(rows))
    nonpositive = sum(1 for row in rows if row.utility_gain <= 0)
    return GroupSummary(
        group=group,
        cases=len(rows),
        passed=passed,
        failed=failed,
        pass_rate=passed / max(1, len(rows)),
        mean_gain=mean_gain,
        mean_confidence=mean_conf,
        negative_or_zero_gain=nonpositive,
    )


def write_outputs(out_csv: Path, out_json: Path, rows: list[StressSummary], group_rows: list[GroupSummary], group_out: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    group_out.parent.mkdir(parents=True, exist_ok=True)
    with group_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(group_rows[0]).keys()))
        writer.writeheader()
        for row in group_rows:
            writer.writerow(asdict(row))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline multi-domain Persistence Gate stress benchmark.")
    parser.add_argument("--cases-per-scenario", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--profile", choices=["permissive", "balanced", "conservative"], default="balanced")
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/multi_domain_stress_summary.csv"))
    parser.add_argument("--json", type=Path, default=Path("benchmark_results/multi_domain_stress_summary.json"))
    parser.add_argument("--group-out", type=Path, default=Path("benchmark_results/multi_domain_stress_group_summary.csv"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[StressSummary] = []
    case_num = 0
    for domain in DOMAINS:
        for scenario in SCENARIOS:
            for _ in range(args.cases_per_scenario):
                case_num += 1
                name, query, memories = build_case(case_num, domain, scenario, rng)
                rows.append(run_stress_case(name, query, memories, top_k=args.top_k, profile=args.profile))

    passed = sum(1 for row in rows if row.pass_fail == "PASS")
    failed = sum(1 for row in rows if row.pass_fail == "FAIL")
    mean_gain = sum(row.utility_gain for row in rows) / len(rows)
    mean_conf = sum(row.evidence_confidence_mean for row in rows) / len(rows)

    group_rows: list[GroupSummary] = []
    for domain in DOMAINS:
        group_rows.append(summarize_group(f"domain:{domain}", [row for row in rows if row.domain == domain]))
    for scenario in SCENARIOS:
        group_rows.append(summarize_group(f"scenario:{scenario}", [row for row in rows if row.scenario == scenario]))
    verdict_counts = {verdict: sum(1 for row in rows if row.verdict == verdict) for verdict in sorted({row.verdict for row in rows})}

    print("Offline Multi-Domain Stress Benchmark")
    print("====================================")
    print(f"Profile: {args.profile}")
    print(f"Cases: {len(rows)}")
    print(f"Passed: {passed}/{len(rows)}")
    print(f"Failed: {failed}/{len(rows)}")
    print(f"Pass rate: {passed / len(rows):.1%}")
    print(f"True failure rate: {failed / len(rows):.1%}")
    print(f"Mean utility gain: {mean_gain:.2f}")
    print(f"Mean evidence confidence: {mean_conf:.2f}")

    print("\nBy verdict")
    for verdict, count in verdict_counts.items():
        print(f"{verdict}: {count}")

    print("\nBy domain")
    for row in [item for item in group_rows if item.group.startswith("domain:")]:
        print(f"{row.group}: {row.passed}/{row.cases} pass, failed={row.failed}, mean_gain={row.mean_gain:.2f}, nonpositive_gain={row.negative_or_zero_gain}")

    print("\nBy scenario")
    for row in [item for item in group_rows if item.group.startswith("scenario:")]:
        print(f"{row.group}: {row.passed}/{row.cases} pass, failed={row.failed}, mean_gain={row.mean_gain:.2f}, nonpositive_gain={row.negative_or_zero_gain}")

    write_outputs(args.out, args.json, rows, group_rows, args.group_out)
    print(f"Saved CSV: {args.out}")
    print(f"Saved group CSV: {args.group_out}")
    print(f"Saved JSON: {args.json}")


if __name__ == "__main__":
    main()
