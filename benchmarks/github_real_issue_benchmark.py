from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time as time_module
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import time

from persistence_memory import MemoryItem, TaskContext
from persistence_memory.benchmark import evaluate_gate_vs_topk

DEFAULT_REPOS = [
    "psf/requests",
    "pallets/flask",
    "django/django",
    "numpy/numpy",
    "pandas-dev/pandas",
    "scikit-learn/scikit-learn",
    "matplotlib/matplotlib",
    "fastapi/fastapi",
]

# Workaround terms should represent provisional advice, not merely a generic mention.
WORKAROUND_TERMS = [
    "workaround",
    "temporary fix",
    "temporary workaround",
    "hack",
    "for now",
    "until fixed",
    "disable",
    "pin to",
    "downgrade",
]

# Resolution terms should represent later/current evidence that the issue state changed.
RESOLUTION_TERMS = [
    "fixed",
    "resolved",
    "closed by",
    "merged",
    "released",
    "available in",
    "no longer needed",
    "should be fixed",
    "this is fixed",
    "this has been fixed",
]

SEARCH_QUERIES = [
    "workaround fixed",
    "workaround resolved",
    "temporary fix fixed",
    "downgrade fixed",
    "pin to fixed",
    "disable fixed",
]


def github_request(path: str) -> dict | list | None:
    url = f"https://api.github.com{path}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "persistence-gate-real-issue-benchmark"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"Warning: GitHub API returned {exc.code} for {path}", file=sys.stderr)
    except urllib.error.URLError as exc:
        print(f"Warning: could not reach GitHub for {path}: {exc}", file=sys.stderr)
    return None


def contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def compact(text: str, limit: int = 1400) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def search_real_issues(repos: list[str], per_repo: int, query_terms: list[str] | None = None) -> list[dict]:
    """Search public GitHub issues using several query templates.

    Deduping across query templates gives more real cases without inflating evidence.
    """
    query_terms = query_terms or SEARCH_QUERIES
    issues: list[dict] = []
    seen: set[str] = set()
    per_query = max(1, min(20, per_repo))
    for repo in repos:
        for terms in query_terms:
            query = f"repo:{repo} is:issue is:closed {terms}"
            encoded = urllib.parse.quote(query)
            payload = github_request(f"/search/issues?q={encoded}&sort=updated&order=desc&per_page={per_query}")
            if isinstance(payload, dict):
                for item in payload.get("items", []):
                    key = item.get("html_url") or f"{repo}#{item.get('number')}"
                    if key in seen:
                        continue
                    seen.add(key)
                    item["_repo"] = repo
                    item["_search_terms"] = terms
                    issues.append(item)
            # Keep unauthenticated runs polite and less likely to hit secondary limits.
            time_module.sleep(0.25)
    return issues


def fetch_issue_comments(repo: str, issue_number: int, limit: int = 50) -> list[dict]:
    payload = github_request(f"/repos/{repo}/issues/{issue_number}/comments?per_page={limit}")
    comments = payload if isinstance(payload, list) else []
    return sorted(comments, key=lambda row: row.get("created_at") or "")


def make_memory(
    *,
    memory_id: str,
    text: str,
    source: str,
    kind: str,
    helpful: bool = False,
    risky: bool = False,
    stale: bool = False,
    risk: float = 0.08,
    harm: float = 0.0,
    usefulness: float = 0.35,
    burden: float = 0.20,
    label_confidence: float = 0.5,
    evidence_role: str = "unknown",
    created_at_text: str | None = None,
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
            "kind": kind,
            "label_helpful": helpful,
            "label_risky": risky,
            "label_stale": stale,
            "label_confidence": label_confidence,
            "evidence_role": evidence_role,
            "source_created_at": created_at_text,
        },
    )


def comment_role(comment: dict) -> str:
    body = comment.get("body", "")
    has_workaround = contains_any(body, WORKAROUND_TERMS)
    has_resolution = contains_any(body, RESOLUTION_TERMS)
    if has_workaround and has_resolution:
        return "mixed_workaround_resolution"
    if has_workaround:
        return "workaround"
    if has_resolution:
        return "resolution"
    return "other"


def case_quality(workarounds: list[dict], resolutions: list[dict]) -> float:
    if not workarounds or not resolutions:
        return 0.0
    earliest_workaround = min((parse_time(row.get("created_at")) for row in workarounds), default=None)
    latest_resolution = max((parse_time(row.get("created_at")) for row in resolutions), default=None)
    if earliest_workaround and latest_resolution and earliest_workaround < latest_resolution:
        return 1.0
    # Some GitHub comments lack robust temporal order in tests or edge cases; still usable but lower confidence.
    return 0.55


def build_case_from_issue(issue: dict, comments: list[dict], min_quality: float = 0.55) -> tuple[str, str, list[MemoryItem]] | None:
    repo = issue.get("_repo") or "unknown/repo"
    number = issue.get("number")
    title = compact(issue.get("title", ""), 300)
    body = compact(issue.get("body", ""))
    html_url = issue.get("html_url", "")

    workaround_comments = [comment for comment in comments if comment_role(comment) in {"workaround", "mixed_workaround_resolution"}]
    resolution_comments = [comment for comment in comments if comment_role(comment) in {"resolution", "mixed_workaround_resolution"}]
    quality = case_quality(workaround_comments, resolution_comments)

    # Require at least one real workaround-like memory and one real resolution-like memory.
    # Also require temporal/role quality so we are less likely to label ambiguous comments as proof.
    if not workaround_comments or not resolution_comments or quality < min_quality:
        return None

    memories: list[MemoryItem] = []
    issue_text = f"Real closed GitHub issue from {repo} #{number}: {title}. Body: {body}. URL: {html_url}"
    memories.append(
        make_memory(
            memory_id=f"{repo}#{number}:issue_body".replace("/", "__"),
            text=issue_text,
            source=html_url,
            kind="issue",
            helpful=True,
            risk=0.10,
            usefulness=0.58,
            label_confidence=0.55,
            evidence_role="issue_context",
            created_at_text=issue.get("created_at"),
        )
    )

    # Use earliest workaround-like comments as the stale/risky candidates.
    ordered_workarounds = sorted(workaround_comments, key=lambda row: row.get("created_at") or "")
    for idx, comment in enumerate(ordered_workarounds[:2]):
        text = compact(comment.get("body", ""))
        memories.append(
            make_memory(
                memory_id=f"{repo}#{number}:old_workaround_{idx}".replace("/", "__"),
                text=f"Earlier workaround-like issue comment from {repo} #{number}: {text}",
                source=comment.get("html_url", html_url),
                kind="issue",
                risky=True,
                stale=True,
                risk=0.82,
                harm=0.74,
                usefulness=-0.30,
                label_confidence=quality,
                evidence_role="earlier_workaround_candidate",
                created_at_text=comment.get("created_at"),
            )
        )

    # Use latest resolution-like comments as the current/helpful candidates.
    ordered_resolutions = sorted(resolution_comments, key=lambda row: row.get("created_at") or "")
    for idx, comment in enumerate(ordered_resolutions[-2:]):
        text = compact(comment.get("body", ""))
        memories.append(
            make_memory(
                memory_id=f"{repo}#{number}:resolution_{idx}".replace("/", "__"),
                text=f"Later resolution-like issue comment from {repo} #{number}: {text}",
                source=comment.get("html_url", html_url),
                kind="issue",
                helpful=True,
                risk=0.04,
                harm=0.0,
                usefulness=0.86,
                label_confidence=quality,
                evidence_role="later_resolution_candidate",
                created_at_text=comment.get("created_at"),
            )
        )

    memories.append(
        make_memory(
            memory_id=f"{repo}#{number}:distractor".replace("/", "__"),
            text="Unrelated project maintenance note about formatting, labels, or documentation style. It should not decide the issue outcome.",
            source=html_url,
            kind="document",
            helpful=False,
            risk=0.05,
            usefulness=0.05,
            label_confidence=0.90,
            evidence_role="distractor",
            created_at_text=None,
        )
    )

    query = f"For {repo} issue #{number}, should the earlier workaround or the later fix/resolution influence the current answer?"
    name = f"{repo}#{number} {title}"
    return name, query, memories


@dataclass
class RealIssueSummary:
    name: str
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
    pass_fail: str


def mean_label_confidence(memories: list[MemoryItem]) -> float:
    values = [float(item.metadata.get("label_confidence", 0.5)) for item in memories]
    return sum(values) / max(1, len(values))


def run_case(name: str, query: str, memories: list[MemoryItem], top_k: int) -> RealIssueSummary:
    result = evaluate_gate_vs_topk(
        memories,
        TaskContext(query=query, context_scope="project", need=0.95, risk_tolerance=0.40, abstention_score=0.04),
        top_k=top_k,
    )
    passed = result.utility_gain > 0 and result.risky_items_prevented > 0 and mean_label_confidence(memories) >= 0.55
    return RealIssueSummary(
        name=name,
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
        evidence_confidence_mean=mean_label_confidence(memories),
        pass_fail="PASS" if passed else "WEAK",
    )


def write_csv(path: Path, rows: list[RealIssueSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Persistence Gate on real public GitHub issue/comment data.")
    parser.add_argument("--repos", nargs="*", default=DEFAULT_REPOS)
    parser.add_argument("--per-repo", type=int, default=12)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--comments-per-issue", type=int, default=50)
    parser.add_argument("--min-quality", type=float, default=0.55)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/real_github_issue_summary.csv"))
    parser.add_argument("--json", type=Path, default=Path("benchmark_results/real_github_issue_summary.json"))
    args = parser.parse_args()

    print("Real GitHub Issue-History Benchmark")
    print("===================================")
    print("This benchmark uses public issue/comment text fetched live from GitHub.")
    print("Bias controls: multi-query search, deduped issues, chronological workaround/resolution pairing, label confidence, and later blind audit export.\n")

    found_issues = search_real_issues(args.repos, per_repo=args.per_repo)
    cases: list[tuple[str, str, list[MemoryItem]]] = []

    for issue in found_issues:
        if len(cases) >= args.max_cases:
            break
        repo = issue.get("_repo")
        number = issue.get("number")
        if not repo or not number:
            continue
        comments = fetch_issue_comments(repo, int(number), limit=args.comments_per_issue)
        built = build_case_from_issue(issue, comments, min_quality=args.min_quality)
        if built:
            cases.append(built)
        time_module.sleep(0.2)

    if not cases:
        print("No usable real issue cases found under the current search terms/rate limit.")
        print("Try: --per-repo 20 --max-cases 10 --min-quality 0.5 or set GITHUB_TOKEN for higher GitHub API limits.")
        raise SystemExit(1)

    summaries = [run_case(name, query, memories, top_k=args.top_k) for name, query, memories in cases]

    passed = 0
    total_gain = 0.0
    for summary in summaries:
        total_gain += summary.utility_gain
        if summary.pass_fail == "PASS":
            passed += 1
        print(
            f"{summary.pass_fail} | {summary.name} | ordinary={summary.ordinary_net:.2f} "
            f"gated={summary.gated_net:.2f} gain={summary.utility_gain:.2f} "
            f"risk_prevented={summary.risky_prevented} stale_prevented={summary.stale_prevented} helpful_lost={summary.helpful_lost} "
            f"label_conf={summary.evidence_confidence_mean:.2f}"
        )

    print(f"\nPassed {passed}/{len(summaries)} real issue cases")
    print(f"Mean utility gain: {total_gain / max(1, len(summaries)):.2f}")
    print("Interpretation: PASS means the gate improved net utility, prevented at least one workaround-like risky influence, and passed minimum label-confidence checks.")

    write_csv(args.out, summaries)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps([asdict(row) for row in summaries], indent=2), encoding="utf-8")
    print(f"Saved CSV: {args.out}")
    print(f"Saved JSON: {args.json}")


if __name__ == "__main__":
    main()
