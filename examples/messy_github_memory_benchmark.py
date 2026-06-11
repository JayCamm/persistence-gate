from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from time import time

from persistence_memory import MemoryItem, TaskContext
from persistence_memory.benchmark import evaluate_gate_vs_topk, item_is_risky, item_is_stale
from persistence_memory.labeling import label_memory

DEFAULT_REPOS = [
    "JayCamm/persistence-gate",
    "JayCamm/error-catastrophe-sim",
    "JayCamm/pattern-sweep",
    "JayCamm/universe-simulator",
]

TEXT_EXTENSIONS = {
    ".md",
    ".py",
    ".txt",
    ".toml",
    ".json",
    ".jsonl",
    ".yml",
    ".yaml",
}

RISK_TERMS = [
    "deprecated",
    "obsolete",
    "stale",
    "outdated",
    "old claim",
    "failed approach",
    "broken assumption",
    "incorrect",
    "contradicted",
    "do not use",
    "always retrieve top-k",
    "use immediately",
]

HELPFUL_TERMS = [
    "test",
    "tests",
    "pytest",
    "usage",
    "install",
    "readme",
    "validated",
    "confirmed",
    "controller",
    "scorer",
    "benchmark",
    "evaluation",
    "architecture",
]

STALE_TERMS = [
    "deprecated",
    "obsolete",
    "legacy",
    "stale",
    "outdated",
    "previous conclusion",
    "superseded",
    "contradicted",
]


def github_request(path: str) -> dict | list | None:
    url = f"https://api.github.com{path}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "persistence-gate-messy-benchmark"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"Warning: GitHub API returned {exc.code} for {path}", file=sys.stderr)
    except urllib.error.URLError as exc:
        print(f"Warning: could not reach GitHub for {path}: {exc}", file=sys.stderr)
    return None


def decode_base64_content(payload: dict | None) -> str:
    if not isinstance(payload, dict) or "content" not in payload:
        return ""
    try:
        return base64.b64decode(payload["content"]).decode("utf-8", errors="replace")
    except Exception:
        return ""


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def count_terms(text: str, terms: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def parse_github_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def age_bucket(value: str | None) -> str:
    parsed = parse_github_time(value)
    if parsed is None:
        return "unknown"
    days = (datetime.now(timezone.utc) - parsed).days
    if days < 30:
        return "fresh"
    if days < 180:
        return "recent"
    if days < 730:
        return "aging"
    return "old"


def age_risk(value: str | None) -> float:
    bucket = age_bucket(value)
    return {"fresh": 0.02, "recent": 0.10, "aging": 0.25, "old": 0.45}.get(bucket, 0.25)


def chunk_text(text: str, max_chars: int = 1100) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    return [clean[start : start + max_chars] for start in range(0, len(clean), max_chars)]


def get_repo_tree(repo: str, branch: str) -> list[dict]:
    encoded_branch = urllib.parse.quote(branch, safe="")
    payload = github_request(f"/repos/{repo}/git/trees/{encoded_branch}?recursive=1")
    if isinstance(payload, dict) and isinstance(payload.get("tree"), list):
        return [entry for entry in payload["tree"] if entry.get("type") == "blob"]
    return []


def get_file_text(repo: str, path: str, branch: str) -> str:
    encoded_path = urllib.parse.quote(path)
    encoded_branch = urllib.parse.quote(branch, safe="")
    payload = github_request(f"/repos/{repo}/contents/{encoded_path}?ref={encoded_branch}")
    return decode_base64_content(payload if isinstance(payload, dict) else None)


def get_commits(repo: str, limit: int = 25) -> list[dict]:
    payload = github_request(f"/repos/{repo}/commits?per_page={limit}")
    return payload if isinstance(payload, list) else []


def risk_for_kind(kind: str, text: str, date_value: str | None) -> float:
    risk_hits = count_terms(text, RISK_TERMS)
    stale_hits = count_terms(text, STALE_TERMS)
    base = age_risk(date_value)
    if kind in {"source", "test"}:
        return clamp(base * 0.35 + 0.05 * risk_hits)
    if kind == "commit":
        return clamp(base + 0.06 * risk_hits)
    return clamp(base + 0.16 * risk_hits + (0.10 if stale_hits else 0.0))


def helpfulness_for_kind(kind: str, text: str) -> float:
    helpful_hits = count_terms(text, HELPFUL_TERMS)
    bonus = 0.18 if kind in {"readme", "test"} else 0.10 if kind == "source" else 0.0
    return clamp(0.10 + 0.12 * helpful_hits + bonus)


def make_memory(
    *,
    memory_id: str,
    text: str,
    repo: str,
    kind: str,
    path: str = "",
    date_value: str | None = None,
    context_scope: str = "project",
) -> MemoryItem:
    bucket = age_bucket(date_value)
    risk = risk_for_kind(kind, text, date_value)
    helpfulness = helpfulness_for_kind(kind, text)
    burden = clamp(len(text) / 2400)

    return MemoryItem(
        id=memory_id,
        text=text,
        source=f"github:{repo}:{path or kind}",
        context_scope=context_scope,
        created_at=time(),
        confidence=0.70 if kind in {"readme", "source", "test"} else 0.50,
        importance=clamp(0.45 + helpfulness * 0.35),
        burden=burden,
        risk=risk,
        usefulness_score=helpfulness,
        harm_score=clamp(max(0.0, risk - 0.20)),
        metadata={
            "repo": repo,
            "kind": kind,
            "path": path,
            "date": date_value,
            "age_bucket": bucket,
        },
    )


def build_repo_memories(repo: str, max_files: int, max_commits: int) -> list[MemoryItem]:
    repo_payload = github_request(f"/repos/{repo}")
    if not isinstance(repo_payload, dict) or "full_name" not in repo_payload:
        return []
    branch = repo_payload.get("default_branch") or "main"
    pushed_at = repo_payload.get("pushed_at")

    memories: list[MemoryItem] = []
    repo_text = (
        f"Repository {repo_payload.get('full_name')}. Description: {repo_payload.get('description') or 'No description provided.'}. "
        f"Language: {repo_payload.get('language') or 'unknown'}. Last pushed: {pushed_at}. Default branch: {branch}."
    )
    memories.append(make_memory(memory_id=f"{repo}::repo_meta".replace("/", "__"), text=repo_text, repo=repo, kind="repo_meta", date_value=pushed_at))

    tree = get_repo_tree(repo, branch)
    text_files = []
    for entry in tree:
        path = entry.get("path", "")
        suffix = Path(path).suffix.lower()
        if suffix in TEXT_EXTENSIONS and not any(part in path for part in [".git/", "__pycache__", ".venv", "node_modules"]):
            text_files.append(path)
    text_files = text_files[:max_files]

    for path in text_files:
        text = get_file_text(repo, path, branch)
        if not text.strip():
            continue
        kind = "test" if path.startswith("tests/") or "/test" in path else "readme" if "readme" in path.lower() else "source" if path.endswith(".py") else "sample" if path.endswith(".jsonl") else "document"
        for index, chunk in enumerate(chunk_text(text)):
            memory_id = f"{repo}::{path}::chunk{index}".replace("/", "__")
            decorated = f"File {path} from {repo}. {chunk}"
            memories.append(make_memory(memory_id=memory_id, text=decorated, repo=repo, kind=kind, path=path, date_value=pushed_at))

    commits = get_commits(repo, limit=max_commits)
    for index, commit in enumerate(commits):
        commit_data = commit.get("commit", {}) if isinstance(commit, dict) else {}
        message = commit_data.get("message", "")
        date_value = (commit_data.get("committer") or {}).get("date")
        kind = "old_commit" if age_bucket(date_value) in {"aging", "old"} else "commit"
        text = f"Commit from {repo}. Date: {date_value}. Message: {message}"
        memory_id = f"{repo}::commit::{index}".replace("/", "__")
        memories.append(make_memory(memory_id=memory_id, text=text, repo=repo, kind=kind, date_value=date_value, context_scope="history"))

    return memories


def print_metrics(name: str, metrics) -> None:
    print(
        f"{name}: selected={metrics.selected}, helpful={metrics.helpful_selected}, risky={metrics.risky_selected}, "
        f"stale={metrics.stale_selected}, uncertain={metrics.uncertain_selected}, burden={metrics.burden:.2f}, net={metrics.net_utility():.2f}"
    )


def print_items(title: str, items) -> None:
    print("\n" + title)
    print("=" * len(title))
    if not items:
        print("None")
        return
    for item in items:
        memory = item.memory if hasattr(item, "memory") else item
        score = f", score={item.score:.3f}, decision={item.decision.value}, reasons={item.reasons}" if hasattr(item, "score") else ""
        labels = label_memory(memory)
        flags = []
        if labels.risky:
            flags.append("risky")
        if labels.stale:
            flags.append("stale")
        if labels.helpful:
            flags.append("helpful")
        if labels.uncertain:
            flags.append("uncertain")
        print(f"- {memory.id}{score} flags={flags} label_reasons={list(labels.reasons)}")
        print(f"  {memory.text[:260]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bias-audited messy GitHub memory benchmark for Persistence Gate.")
    parser.add_argument("--repos", nargs="*", default=DEFAULT_REPOS)
    parser.add_argument("--query", default="What current evidence and code should influence the next development step for Persistence Gate?")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-files", type=int, default=24)
    parser.add_argument("--max-commits", type=int, default=25)
    parser.add_argument("--save-corpus", type=Path, default=None)
    args = parser.parse_args()

    print("Building bias-audited messy file-level GitHub memory corpus")
    print("===========================================================")
    corpus: list[MemoryItem] = []
    for repo in args.repos:
        print(f"Fetching repo/files/commits for {repo}...")
        corpus.extend(build_repo_memories(repo, max_files=args.max_files, max_commits=args.max_commits))

    if not corpus:
        raise SystemExit("No corpus items were created.")

    if args.save_corpus:
        with args.save_corpus.open("w", encoding="utf-8") as handle:
            for item in corpus:
                row = item.__dict__.copy()
                row["state"] = item.state.value
                handle.write(json.dumps(row, default=str) + "\n")
        print(f"Saved {len(corpus)} memory items to {args.save_corpus}")

    task = TaskContext(query=args.query, context_scope="project", need=0.9, risk_tolerance=0.50, abstention_score=0.04)
    result = evaluate_gate_vs_topk(corpus, task=task, top_k=args.top_k)

    print("\nBenchmark summary")
    print("=================")
    print(f"Corpus items: {len(corpus)}")
    print_metrics("Ordinary top-k", result.ordinary)
    print_metrics("Persistence Gate", result.gated)
    print(f"Utility gain: {result.utility_gain:.2f}")
    print(f"Risky items prevented: {result.risky_items_prevented}")
    print(f"Stale items prevented: {result.stale_items_prevented}")
    print(f"Helpful items lost: {result.helpful_items_lost}")

    print_items("Ordinary top-k selected", result.report.ordinary_top_k)
    print_items("Persistence Gate allowed", result.report.allowed)
    print_items("Persistence Gate blocked", result.report.blocked[:30])
    print_items("Allowed but not selected due to top-k budget", (result.report.not_selected or [])[:20])

    print("\nInterpretation")
    print("==============")
    if result.utility_gain > 0 and result.risky_items_prevented > 0:
        print("Persistence Gate improved net utility and prevented risky/stale influence under bias-aware labels.")
    elif result.utility_gain > 0:
        print("Persistence Gate improved net utility, mostly by reshaping the context budget rather than hard-blocking risk.")
    else:
        print("Persistence Gate did not beat ordinary top-k under this stricter scoring setup. That is useful pressure for improving the gate.")


if __name__ == "__main__":
    main()
