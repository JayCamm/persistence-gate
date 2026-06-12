from __future__ import annotations

from .models import EvidenceRole, QueryIntent


def coerce_query_intent(value: str | QueryIntent | None) -> QueryIntent:
    if isinstance(value, QueryIntent):
        return value
    if value is None:
        return QueryIntent.GENERAL_LOOKUP
    try:
        return QueryIntent(str(value))
    except ValueError:
        return QueryIntent.GENERAL_LOOKUP


def coerce_evidence_role(value: str | EvidenceRole | None) -> EvidenceRole:
    if isinstance(value, EvidenceRole):
        return value
    if value is None:
        return EvidenceRole.UNCERTAIN
    try:
        return EvidenceRole(str(value))
    except ValueError:
        return EvidenceRole.UNCERTAIN


def infer_query_intent(query: str) -> QueryIntent:
    q = query.lower()
    history_terms = (
        "what changed",
        "changed from",
        "history",
        "historical",
        "older",
        "old ",
        "legacy",
        "migration",
        "compare",
        "comparison",
    )
    audit_terms = ("audit", "review", "why did", "trace", "evidence trail")
    current_terms = (
        "current",
        "should",
        "now",
        "today",
        "guide",
        "enforce",
        "process",
        "procedure",
        "how do i",
        "how should",
    )
    if any(term in q for term in history_terms):
        return QueryIntent.HISTORY_COMPARISON
    if any(term in q for term in audit_terms):
        return QueryIntent.AUDIT_REVIEW
    if any(term in q for term in current_terms):
        return QueryIntent.CURRENT_ACTION
    return QueryIntent.GENERAL_LOOKUP


def infer_evidence_role(text: str, metadata: dict | None = None) -> EvidenceRole:
    metadata = metadata or {}
    explicit = metadata.get("evidence_role") or metadata.get("influence_role")
    if explicit:
        return coerce_evidence_role(str(explicit))

    t = text.lower()
    if "current warning" in t or "do not treat" in t or "do not use" in t or "warning against" in t:
        return EvidenceRole.WARNING_AGAINST_LEGACY
    if "historical comparison" in t or "what changed" in t or "history" in t:
        return EvidenceRole.HISTORICAL_CONTEXT
    if "legacy instruction" in t or "legacy guidance" in t or "removed" in t or "deprecated" in t:
        return EvidenceRole.LEGACY_INSTRUCTION
    if "current authoritative" in t or "current guidance" in t or "validated" in t:
        return EvidenceRole.CURRENT_GUIDANCE
    return EvidenceRole.UNCERTAIN
