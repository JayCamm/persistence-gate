from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .models import MemoryItem

RISK_CLAIM_TERMS = {
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
}

HELPFUL_TERMS = {
    "test",
    "tests",
    "pytest",
    "passed",
    "validated",
    "confirmed",
    "current",
    "install",
    "usage",
    "readme",
    "controller",
    "scorer",
    "benchmark",
    "evaluation",
    "architecture",
}

STALE_TERMS = {
    "deprecated",
    "obsolete",
    "stale",
    "outdated",
    "old claim",
    "previous conclusion",
    "superseded",
    "contradicted",
}

CODE_KINDS = {"source", "test"}
DOCUMENT_KINDS = {"readme", "document", "sample", "issue", "commit", "old_commit", "repo_meta"}


@dataclass(frozen=True)
class MemoryLabels:
    helpful: bool = False
    risky: bool = False
    stale: bool = False
    uncertain: bool = False
    reasons: tuple[str, ...] = ()


def age_bucket_from_date(value: str | None) -> str:
    if not value:
        return "unknown"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    age_days = (datetime.now(timezone.utc) - parsed).days
    if age_days < 30:
        return "fresh"
    if age_days < 180:
        return "recent"
    if age_days < 730:
        return "aging"
    return "old"


def count_terms(text: str, terms: set[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def label_memory(item: MemoryItem) -> MemoryLabels:
    """Bias-aware heuristic labels for benchmark evaluation.

    This intentionally treats code differently from claims/docs. A source file that
    contains a function called `item_is_risky` should not automatically be labeled
    risky. Risk is strongest when a document or sample memory makes a claim that is
    old, contradicted, deprecated, or unsafe to use.
    """
    explicit_helpful = item.metadata.get("label_helpful")
    explicit_risky = item.metadata.get("label_risky")
    explicit_stale = item.metadata.get("label_stale")
    if explicit_helpful is not None or explicit_risky is not None or explicit_stale is not None:
        reasons = []
        if explicit_helpful:
            reasons.append("explicit_helpful")
        if explicit_risky:
            reasons.append("explicit_risky")
        if explicit_stale:
            reasons.append("explicit_stale")
        return MemoryLabels(bool(explicit_helpful), bool(explicit_risky), bool(explicit_stale), False, tuple(reasons))

    kind = str(item.metadata.get("kind", "")).lower()
    path = str(item.metadata.get("path", "")).lower()
    text = item.text.lower()
    bucket = item.metadata.get("age_bucket") or age_bucket_from_date(item.metadata.get("date"))

    reasons: list[str] = []
    helpful_hits = count_terms(text, HELPFUL_TERMS)
    stale_hits = count_terms(text, STALE_TERMS)
    risk_hits = count_terms(text, RISK_CLAIM_TERMS)

    is_code = kind in CODE_KINDS or path.endswith(".py")
    is_test = kind == "test" or "/test" in path or path.startswith("tests/")
    is_doc_or_claim = kind in DOCUMENT_KINDS or path.endswith((".md", ".txt", ".jsonl"))

    helpful = False
    risky = False
    stale = False

    if is_test or helpful_hits >= 2 or item.usefulness_score >= 0.55:
        helpful = True
        reasons.append("helpful_evidence")

    if bucket == "old" or kind == "old_commit":
        stale = True
        reasons.append("old_history")

    # Claims/docs can be risky because the text may be advice or stored belief.
    # Code only becomes risky if it is both high-harm/high-risk or explicitly a TODO/broken item.
    if is_doc_or_claim and (risk_hits >= 1 or item.risk >= 0.55 or item.harm_score >= 0.45):
        risky = True
        reasons.append("risky_claim_or_doc")
    elif is_code and (item.risk >= 0.70 or item.harm_score >= 0.65):
        risky = True
        reasons.append("high_risk_code")

    if stale_hits >= 1 and is_doc_or_claim:
        stale = True
        reasons.append("stale_claim_term")

    uncertain = not helpful and not risky and not stale
    if uncertain:
        reasons.append("uncertain")

    return MemoryLabels(helpful=helpful, risky=risky, stale=stale, uncertain=uncertain, reasons=tuple(reasons))
