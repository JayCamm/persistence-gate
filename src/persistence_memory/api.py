from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .controller import GateReport, MemoryController
from .models import MemoryItem, ScoredMemory, TaskContext
from .profiles import GateProfile
from .scorer import PersistenceScorer
from .store import InMemoryStore


@dataclass
class GateFilterResult:
    """Implementation-facing result from PersistenceGate.filter."""

    query: str
    profile: str
    report: GateReport
    audit_log: list[dict[str, Any]]

    @property
    def allowed(self) -> list[ScoredMemory]:
        return self.report.allowed

    @property
    def blocked(self) -> list[ScoredMemory]:
        return self.report.blocked

    @property
    def not_selected(self) -> list[ScoredMemory]:
        return self.report.not_selected or []

    @property
    def allowed_items(self) -> list[MemoryItem]:
        return [scored.memory for scored in self.allowed]

    @property
    def blocked_items(self) -> list[MemoryItem]:
        return [scored.memory for scored in self.blocked]

    @property
    def allowed_ids(self) -> list[str]:
        return self.report.allowed_ids

    @property
    def blocked_ids(self) -> list[str]:
        return self.report.blocked_ids

    @property
    def warnings(self) -> list[str]:
        warning_rows = [row for row in self.audit_log if row["decision"] == "allow_with_warning"]
        return [f"{row['id']}: {', '.join(row['reasons']) or 'allowed with warning'}" for row in warning_rows]

    @property
    def allowed_context(self) -> str:
        return "\n\n".join(item.text for item in self.allowed_items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "profile": self.profile,
            "allowed_ids": self.allowed_ids,
            "blocked_ids": self.blocked_ids,
            "not_selected_ids": self.report.not_selected_ids,
            "ordinary_top_k_ids": self.report.ordinary_top_k_ids,
            "blocked_from_ordinary_top_k": self.report.blocked_from_ordinary_top_k,
            "warnings": self.warnings,
            "audit_log": self.audit_log,
        }


class PersistenceGate:
    """Small implementation API for influence-gating retrieved memory.

    This class is designed to wrap an existing retriever/vector database/search
    system. It accepts retrieved items, scores them with the selected profile,
    and returns allowed evidence plus an audit trail.
    """

    def __init__(
        self,
        profile: str | GateProfile = "balanced",
        top_k: int = 6,
        context_scope: str = "project",
        need: float = 0.90,
        risk_tolerance: float = 0.35,
        abstention_score: float = 0.04,
    ) -> None:
        self.profile = profile
        self.top_k = top_k
        self.context_scope = context_scope
        self.need = need
        self.risk_tolerance = risk_tolerance
        self.abstention_score = abstention_score

    def filter(
        self,
        query: str,
        retrieved_items: Iterable[MemoryItem | dict[str, Any]],
        *,
        profile: str | GateProfile | None = None,
        top_k: int | None = None,
        context_scope: str | None = None,
        need: float | None = None,
        risk_tolerance: float | None = None,
        abstention_score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GateFilterResult:
        active_profile = profile if profile is not None else self.profile
        active_top_k = top_k if top_k is not None else self.top_k
        active_context_scope = context_scope if context_scope is not None else self.context_scope

        items = [self._coerce_item(item, default_context_scope=active_context_scope) for item in retrieved_items]
        task = TaskContext(
            query=query,
            context_scope=active_context_scope,
            need=self.need if need is None else need,
            risk_tolerance=self.risk_tolerance if risk_tolerance is None else risk_tolerance,
            abstention_score=self.abstention_score if abstention_score is None else abstention_score,
            metadata=metadata or {},
        )

        controller = MemoryController(InMemoryStore(items), scorer=PersistenceScorer(profile=active_profile))
        report = controller.retrieve_report(task, candidates=items, top_k=active_top_k)
        profile_name = active_profile.name if isinstance(active_profile, GateProfile) else str(active_profile)
        return GateFilterResult(
            query=query,
            profile=profile_name,
            report=report,
            audit_log=self._audit_log(report),
        )

    def _coerce_item(self, item: MemoryItem | dict[str, Any], default_context_scope: str) -> MemoryItem:
        if isinstance(item, MemoryItem):
            return item
        if not isinstance(item, dict):
            raise TypeError(f"Retrieved item must be MemoryItem or dict, got {type(item)!r}")
        if "text" not in item:
            raise ValueError("Retrieved item dict must include a 'text' field")

        metadata = dict(item.get("metadata") or {})
        for key in ["label_helpful", "label_risky", "label_stale", "label_uncertain", "label_confidence"]:
            if key in item and key not in metadata:
                metadata[key] = item[key]

        return MemoryItem(
            id=str(item.get("id") or item.get("source") or abs(hash(item["text"]))),
            text=str(item["text"]),
            source=str(item.get("source") or "retrieved"),
            context_scope=str(item.get("context_scope") or default_context_scope),
            relevance=float(item.get("relevance", 0.0)),
            confidence=float(item.get("confidence", 0.5)),
            importance=float(item.get("importance", 0.5)),
            burden=float(item.get("burden", 0.15)),
            risk=float(item.get("risk", 0.05)),
            usefulness_score=float(item.get("usefulness_score", item.get("usefulness", 0.5))),
            harm_score=float(item.get("harm_score", item.get("harm", 0.0))),
            metadata=metadata,
        )

    def _audit_log(self, report: GateReport) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for bucket, scored_items in [
            ("allowed", report.allowed),
            ("blocked", report.blocked),
            ("not_selected", report.not_selected or []),
        ]:
            for scored in scored_items:
                item = scored.memory
                rows.append(
                    {
                        "bucket": bucket,
                        "id": item.id,
                        "source": item.source,
                        "decision": scored.decision.value,
                        "score": scored.score,
                        "reasons": list(scored.reasons),
                        "risk": item.risk,
                        "harm_score": item.harm_score,
                        "burden": item.burden,
                        "usefulness_score": item.usefulness_score,
                        "state": item.state.value,
                        "text_preview": item.text[:160],
                    }
                )
        return rows
