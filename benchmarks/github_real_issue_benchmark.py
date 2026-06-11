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
from pathlib import Path
from time import time

from persistence_memory import MemoryItem, TaskContext
from persistence_memory.benchmark import evaluate_gate_vs_topk

DEFAULT_REPOS = [
    "psf/requests",
    "pallets/flask",
    "django/django",
    "numpy/numpy",
]

WORKAROUND_TERMS = [
    "workaround",
    "temporary fix",
    "hack",
    "for now",
    "until fixed",
    "disable",
    "pin to",
    "downgrade",
]

RESOLUTION_TERMS = [
    "fixed",
    "resolved",
    "closed by",
    "merged",
    "released",
    "available in",
    "no longer needed",
    "should be fixed",
]

RISK_TERMS = [
    "workaround",
    "hack",
    "temporary",
    "disable",
    "deprecated",
    "downgrade",
    "pin to",
    "old version",
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


def search_real_issues(repos: list[str], per_repo: int) -> list[dict]:
    issues: list[dict] = []
    for repo in repos:
        query = f"repo:{repo} is:issue is:closed workaround fixed"
        encoded = urllib.parse.quote(query)
        payload = github_request(f"/search/issues?q={encoded}&sort=updated&order=desc&per_page={per_repo}")
        if isinstance(payload, dict):
            for item in payload.get("items", []):
                item["_repo"] = repo
                issues.append(item)
        # Keep unauthenticated runs polite and less likely to hit secondary limits.
        time_module.sleep(0.4)
    return issues


def fetch_issue_comments(repo: str, issue_number: int, limit: int = 20) -> list[dict]:
    payload = github_request(f"/repos/{repo}/issues/{issue_number}/comments?per_page={limit}")
    return payload if isinstance(payload, list) else []


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
        },
    )


def build_case_from_issue(issue: dict, comments: list[dict]) -> tuple[str, str, list[MemoryItem]] | None:
    repo = issue.get("_repo") or "unknown/repo"
    number = issue.get("number")
    title = compact(issue.get("title", ""), 300)
    body = compact(issue.get("body", ""))
    html_url = issue.get("html_url", "")

    workaround_comments = [comment for comment in comments if contains_any(comment.get("body", ""), WORKAROUND_TERMS)]
    resolution_comments = [comment for comment in comments if contains_any(comment.get("body", ""), RESOLUTION_TERMS)]

    # Require at least one real workaround-like memory and one real resolution-like memory.
    # Otherwise the case is too ambiguous for this benchmark.
    if not workaround_comments or not resolution_comments:
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
            risk=0.12,
            usefulness=0.60,
        )
    )

    for idx, comment in enumerate(workaround_comments[:2]):
        text = compact(comment.get("body", ""))
        memories.append(
            make_memory(
                memory_id=f"{repo}#{number}:old_workaround_{idx}".replace("/", "__"),
                text=f"Older workaround-like issue comment from {repo} #{number}: {text}",
                source=comment.get("html_url", html_url),
                kind="issue",
                risky=True,
                stale=True,
                risk=0.78,
                harm=0.70,
                usefulness=-0.25,
            )
        )

    for idx, comment in enumerate(resolution_comments[-2:]):
        text = compact(comment.get("body", ""))
        memories.append(
            make_memory(
                memory_id=f"{repo}#{number}:resolution_{idx}".replace("/", "__"),
                text=f"Resolution-like issue comment from {repo} #{number}: {text}",
                source=comment.get("html_url", html_url),
                kind="issue",
                helpful=True,
                risk=0.06,
                harm=0.0,
                usefulness=0.80,
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
        )
    )

    query = f"For {repo} issue #{number}, should the old workaround or the later fix/resolution influence the current answer?"
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
    pass_fail: str


def run_case(name: str, query: str, memories: list[MemoryItem], top_k: int) -> RealIssueSummary:
    result = evaluate_gate_vs_topk(
        memories,
        TaskContext(query=query, context_scope="project", need=0.95, risk_tolerance=0.40, abstention_score=0.04),
        top_k=top_k,
    )
    passed = result.utility_gain > 0 and result.risky_items_prevented > 0
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
    parser.add_argument("--per-repo", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/real_github_issue_summary.csv"))
    parser.add_argument("--json", type=Path, default=Path("benchmark_results/real_github_issue_summary.json"))
    args = parser.parse_args()

    print("Real GitHub Issue-History Benchmark")
    print("===================================")
    print("This benchmark uses public issue/comment text fetched live from GitHub.")
    print("Labels are heuristic: workaround-like comments are treated as stale/risky candidates; later fixed/resolved comments are treated as current helpful candidates.\n")

    found_issues = search_real_issues(args.repos, per_repo=args.per_repo)
    cases: list[tuple[str, str, list[MemoryItem]]] = []

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
        time_module.sleep(0.3)

    if not cases:
        print("No usable real issue cases found under the current search terms/rate limit.")
        print("Try: --per-repo 20 --max-cases 10 or set GITHUB_TOKEN for higher GitHub API limits.")
        raise SystemExit(1)

    summaries = [run_case(name, query, memories, top_k=args.top_k) for name, query, memories in cases]

    passed = 0
    for summary in summaries:
        if summary.pass_fail == "PASS":
            passed += 1
        print(
            f"{summary.pass_fail} | {summary.name} | ordinary={summary.ordinary_net:.2f} "
            f"gated={summary.gated_net:.2f} gain={summary.utility_gain:.2f} "
            f"risk_prevented={summary.risky_prevented} stale_prevented={summary.stale_prevented} helpful_lost={summary.helpful_lost}"
        )

    print(f"\nPassed {passed}/{len(summaries)} real issue cases")
    print("Interpretation: PASS means the gate improved net utility and prevented at least one workaround-like risky influence in that real public issue case.")

    write_csv(args.out, summaries)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps([asdict(row) for row in summaries], indent=2), encoding="utf-8")
    print(f"Saved CSV: {args.out}")
    print(f"Saved JSON: {args.json}")


if __name__ == "__main__":
    main()
