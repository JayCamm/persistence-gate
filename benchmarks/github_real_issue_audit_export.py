from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from persistence_memory import TaskContext
from persistence_memory.audit import detect_circularity, shuffle_for_blind_review, stable_blind_id
from persistence_memory.benchmark import evaluate_gate_vs_topk
from persistence_memory.labeling import label_memory

from benchmarks.github_real_issue_benchmark import (
    DEFAULT_REPOS,
    build_case_from_issue,
    fetch_issue_comments,
    search_real_issues,
)


def build_audit_rows(case_name: str, query: str, memories, top_k: int) -> list[dict]:
    result = evaluate_gate_vs_topk(
        memories,
        TaskContext(query=query, context_scope="project", need=0.95, risk_tolerance=0.40, abstention_score=0.04),
        top_k=top_k,
    )

    role_by_id: dict[str, set[str]] = {}
    for item in result.report.ordinary_top_k:
        role_by_id.setdefault(item.id, set()).add("ordinary_topk")
    for scored in result.report.allowed:
        role_by_id.setdefault(scored.memory.id, set()).add("gate_allowed")
    for scored in result.report.blocked:
        role_by_id.setdefault(scored.memory.id, set()).add("gate_blocked")
    for scored in result.report.not_selected or []:
        role_by_id.setdefault(scored.memory.id, set()).add("gate_passed_not_selected")

    circularity = detect_circularity(memories)
    warnings_by_id: dict[str, list[str]] = {}
    for warning in circularity:
        warnings_by_id.setdefault(warning.memory_id, []).append(f"{warning.warning_type}: {warning.detail}")

    rows: list[dict] = []
    for item in memories:
        labels = label_memory(item)
        roles = sorted(role_by_id.get(item.id, []))
        rows.append(
            {
                "blind_id": stable_blind_id(case_name, item.id),
                "case_name": case_name,
                "query": query,
                "system_roles_hidden_for_blind_review": ";".join(roles),
                "source_url": item.source,
                "memory_id": item.id,
                "memory_text": item.text,
                "heuristic_helpful": labels.helpful,
                "heuristic_risky": labels.risky,
                "heuristic_stale": labels.stale,
                "heuristic_uncertain": labels.uncertain,
                "heuristic_reasons": ";".join(labels.reasons),
                "circularity_warnings": ";".join(warnings_by_id.get(item.id, [])),
                "human_label_helpful": "",
                "human_label_risky": "",
                "human_label_stale": "",
                "human_label_irrelevant": "",
                "human_notes": "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export blind human-review CSV for real GitHub issue benchmark.")
    parser.add_argument("--repos", nargs="*", default=DEFAULT_REPOS)
    parser.add_argument("--per-repo", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/real_issue_blind_audit.csv"))
    parser.add_argument("--unblinded-out", type=Path, default=Path("benchmark_results/real_issue_unblinded_audit.csv"))
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    found_issues = search_real_issues(args.repos, per_repo=args.per_repo)
    cases = []
    for issue in found_issues:
        if len(cases) >= args.max_cases:
            break
        repo = issue.get("_repo")
        number = issue.get("number")
        if not repo or not number:
            continue
        comments = fetch_issue_comments(repo, int(number), limit=30)
        built = build_case_from_issue(issue, comments)
        if built:
            cases.append(built)

    if not cases:
        raise SystemExit("No usable cases found. Try --per-repo 20 or set GITHUB_TOKEN.")

    unblinded_rows: list[dict] = []
    for case_name, query, memories in cases:
        unblinded_rows.extend(build_audit_rows(case_name, query, memories, top_k=args.top_k))

    blind_rows = []
    for row in unblinded_rows:
        blind = dict(row)
        # Hide system outcome/heuristic labels in the blind file, but keep source and text.
        blind.pop("system_roles_hidden_for_blind_review", None)
        blind.pop("heuristic_helpful", None)
        blind.pop("heuristic_risky", None)
        blind.pop("heuristic_stale", None)
        blind.pop("heuristic_uncertain", None)
        blind.pop("heuristic_reasons", None)
        blind.pop("circularity_warnings", None)
        blind_rows.append(blind)

    write_csv(args.unblinded_out, unblinded_rows)
    write_csv(args.out, shuffle_for_blind_review(blind_rows, seed=args.seed))

    warnings = sum(1 for row in unblinded_rows if row["circularity_warnings"])
    print("Real Issue Blind Audit Export")
    print("=============================")
    print(f"Cases: {len(cases)}")
    print(f"Rows: {len(unblinded_rows)}")
    print(f"Circularity warning rows: {warnings}")
    print(f"Blind CSV: {args.out}")
    print(f"Unblinded CSV: {args.unblinded_out}")
    print("Use the blind CSV for human review. Use the unblinded CSV only after labeling is complete.")


if __name__ == "__main__":
    main()
