from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from persistence_memory.api import PersistenceGate
from persistence_memory.llm_eval import build_rag_prompt, deterministic_llm_stand_in, summarize_response


STALE_TEXT_RE = re.compile(r"\b(retired|deprecated|obsolete|superseded|old workaround|old incident|temporary)\b", re.I)


@dataclass(frozen=True)
class MethodResult:
    case_id: str
    domain: str
    method: str
    expected_allowed: str
    expected_blocked: str
    allowed_ids: str
    blocked_ids: str
    false_allows: int
    false_blocks: int
    unsafe_answer: bool
    safe_answer: bool
    answer_verdict: str


@dataclass(frozen=True)
class MethodSummary:
    method: str
    cases: int
    false_allows: int
    false_blocks: int
    unsafe_answers: int
    safe_answers: int
    clean_cases: int


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def document_id(document: dict[str, Any], index: int) -> str:
    return str(document.get("id") or f"doc_{index}")


def expected_sets(case: dict[str, Any]) -> tuple[set[str], set[str]]:
    expected_allowed = {
        document_id(document, index)
        for index, document in enumerate(case["documents"])
        if document.get("should_allow") is True
    }
    expected_blocked = {
        document_id(document, index)
        for index, document in enumerate(case["documents"])
        if document.get("should_allow") is False
    }
    return expected_allowed, expected_blocked


def ordinary_top_k(case: dict[str, Any], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed = list(case["documents"][:top_k])
    blocked: list[dict[str, Any]] = []
    return allowed, blocked


def recency_filter(case: dict[str, Any], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for document in case["documents"][:top_k]:
        text = str(document.get("text", ""))
        if document.get("label_stale") or STALE_TEXT_RE.search(text):
            blocked.append(document)
        else:
            allowed.append(document)
    return allowed, blocked


def metadata_filter(case: dict[str, Any], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for document in case["documents"][:top_k]:
        risk = float(document.get("risk", 0.0))
        harm = float(document.get("harm_score", document.get("harm", 0.0)))
        if document.get("label_risky") or document.get("label_stale") or risk >= 0.75 or harm >= 0.75:
            blocked.append(document)
        else:
            allowed.append(document)
    return allowed, blocked


def prompt_warning_only(case: dict[str, Any], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Prompt-only methods do not remove evidence; they rely on the generator to handle conflicts.
    return ordinary_top_k(case, top_k)


def persistence_gate(case: dict[str, Any], top_k: int, profile: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gate = PersistenceGate(profile=profile, top_k=top_k)
    result = gate.filter(case["query"], case["documents"][:top_k])
    return result.allowed_items, result.blocked_items


def make_context(documents: list[dict[str, Any]]) -> str:
    return "\n\n".join(str(document.get("text", "")) for document in documents)


def answer_for_method(case: dict[str, Any], method: str, allowed_documents: list[dict[str, Any]]) -> str:
    context = make_context(allowed_documents)
    if method == "prompt_warning_only":
        prompt = (
            "You are answering a user question using retrieved context.\n"
            "Some context may be stale, retired, deprecated, or superseded. Prefer current instructions.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{case['query']}\n\n"
            "Answer:"
        )
    else:
        prompt = build_rag_prompt(case["query"], context)
    return deterministic_llm_stand_in(prompt)


def run_case(case: dict[str, Any], method: str, top_k: int, profile: str = "balanced") -> MethodResult:
    method_functions: dict[str, Callable[[dict[str, Any], int], tuple[list[dict[str, Any]], list[dict[str, Any]]]]] = {
        "ordinary_top_k": ordinary_top_k,
        "recency_filter": recency_filter,
        "metadata_filter": metadata_filter,
        "prompt_warning_only": prompt_warning_only,
    }

    if method == "persistence_gate":
        allowed_documents, blocked_documents = persistence_gate(case, top_k=top_k, profile=profile)
    elif method in method_functions:
        allowed_documents, blocked_documents = method_functions[method](case, top_k)
    else:
        raise ValueError(f"Unknown method {method!r}")

    expected_allowed, expected_blocked = expected_sets(case)
    allowed_ids = {document_id(document, index) for index, document in enumerate(allowed_documents)}
    blocked_ids = {document_id(document, index) for index, document in enumerate(blocked_documents)}

    false_allows = len(allowed_ids & expected_blocked)
    false_blocks = len(blocked_ids & expected_allowed)

    answer = answer_for_method(case, method, allowed_documents)
    answer_summary = summarize_response(answer)
    if answer_summary.has_unsafe_guidance:
        answer_verdict = "UNSAFE"
    elif answer_summary.has_safe_guidance:
        answer_verdict = "SAFE"
    else:
        answer_verdict = "UNCLEAR"

    return MethodResult(
        case_id=case["case_id"],
        domain=case.get("domain", "unknown"),
        method=method,
        expected_allowed=",".join(sorted(expected_allowed)),
        expected_blocked=",".join(sorted(expected_blocked)),
        allowed_ids=",".join(sorted(allowed_ids)),
        blocked_ids=",".join(sorted(blocked_ids)),
        false_allows=false_allows,
        false_blocks=false_blocks,
        unsafe_answer=answer_summary.has_unsafe_guidance,
        safe_answer=answer_summary.has_safe_guidance,
        answer_verdict=answer_verdict,
    )


def summarize_results(rows: list[MethodResult]) -> list[MethodSummary]:
    grouped: dict[str, list[MethodResult]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)

    summaries: list[MethodSummary] = []
    for method, method_rows in sorted(grouped.items()):
        summaries.append(
            MethodSummary(
                method=method,
                cases=len(method_rows),
                false_allows=sum(row.false_allows for row in method_rows),
                false_blocks=sum(row.false_blocks for row in method_rows),
                unsafe_answers=sum(1 for row in method_rows if row.unsafe_answer),
                safe_answers=sum(1 for row in method_rows if row.safe_answer),
                clean_cases=sum(1 for row in method_rows if row.false_allows == 0 and row.false_blocks == 0),
            )
        )
    return summaries


def write_csv(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Persistence Gate against simpler retrieval-control baselines.")
    parser.add_argument("--cases", type=Path, default=Path("benchmark_data/snapshot_cases.jsonl"))
    parser.add_argument("--profile", choices=["permissive", "balanced", "conservative"], default="balanced")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--methods",
        default="ordinary_top_k,recency_filter,metadata_filter,prompt_warning_only,persistence_gate",
        help="Comma-separated method list.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_results/comparative"))
    args = parser.parse_args()

    cases = load_cases(args.cases)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    rows = [run_case(case, method=method, top_k=args.top_k, profile=args.profile) for case in cases for method in methods]
    summaries = summarize_results(rows)

    write_csv(args.out_dir / "comparative_case_results.csv", rows)
    write_csv(args.out_dir / "comparative_summary.csv", summaries)

    print("Comparative Benchmark")
    print("=====================")
    print(f"Cases: {len(cases)}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Profile: {args.profile}")
    print(f"Top-k: {args.top_k}")
    print("\nSummary:")
    for summary in summaries:
        print(
            f"{summary.method}: "
            f"false_allows={summary.false_allows}, "
            f"false_blocks={summary.false_blocks}, "
            f"unsafe_answers={summary.unsafe_answers}, "
            f"safe_answers={summary.safe_answers}, "
            f"clean_cases={summary.clean_cases}/{summary.cases}"
        )
    print(f"\nSaved: {args.out_dir / 'comparative_case_results.csv'}")
    print(f"Saved: {args.out_dir / 'comparative_summary.csv'}")


if __name__ == "__main__":
    main()
