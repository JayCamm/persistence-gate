from __future__ import annotations

from dataclasses import dataclass
from time import time

from .models import FeedbackEvent, GateDecision, MemoryItem, MemoryState, ScoredMemory, TaskContext
from .scorer import PersistenceScorer
from .store import InMemoryStore


@dataclass
class GateReport:
    """Full result of a persistence-gating pass.

    `allowed` is what downstream software may use.
    `blocked` is memory rejected by the gate.
    `not_selected` passed the gate but did not fit into the top-k context budget.
    `ordinary_top_k` is the baseline: what a naive relevance-only retriever would have used.
    """

    allowed: list[ScoredMemory]
    blocked: list[ScoredMemory]
    ordinary_top_k: list[MemoryItem]
    not_selected: list[ScoredMemory] | None = None

    @property
    def allowed_ids(self) -> list[str]:
        return [item.memory.id for item in self.allowed]

    @property
    def blocked_ids(self) -> list[str]:
        return [item.memory.id for item in self.blocked]

    @property
    def not_selected_ids(self) -> list[str]:
        return [item.memory.id for item in (self.not_selected or [])]

    @property
    def ordinary_top_k_ids(self) -> list[str]:
        return [item.id for item in self.ordinary_top_k]

    @property
    def blocked_from_ordinary_top_k(self) -> list[str]:
        ordinary = set(self.ordinary_top_k_ids)
        return [item.memory.id for item in self.blocked if item.memory.id in ordinary]


class MemoryController:
    """Retrieve-score-gate-feedback controller.

    This is the core software object. It does not do embeddings itself; for now it
    accepts candidate MemoryItems whose relevance has already been set by any retriever.
    """

    def __init__(self, store: InMemoryStore | None = None, scorer: PersistenceScorer | None = None) -> None:
        self.store = store or InMemoryStore()
        self.scorer = scorer or PersistenceScorer()

    def ingest(self, item: MemoryItem) -> None:
        self.store.add(item)

    def ordinary_top_k(self, candidates: list[MemoryItem], top_k: int = 6) -> list[MemoryItem]:
        """Naive baseline: use the most relevant active memories without persistence gating."""
        active = [item for item in candidates if item.is_active()]
        return sorted(active, key=lambda item: item.relevance, reverse=True)[:top_k]

    def evaluate_candidates(self, candidates: list[MemoryItem], task: TaskContext, top_k: int = 6) -> GateReport:
        """Score every candidate and return allowed, blocked, and ordinary baseline items."""
        # Capture the ordinary baseline before gating mutates any states.
        ordinary = self.ordinary_top_k(candidates, top_k=top_k)

        scored = [self.scorer.score(candidate, task) for candidate in candidates if candidate.state != MemoryState.DELETED]
        scored.sort(key=lambda s: s.score, reverse=True)

        allowed: list[ScoredMemory] = []
        blocked: list[ScoredMemory] = []
        not_selected: list[ScoredMemory] = []

        for item in scored:
            if item.decision in {GateDecision.ALLOW, GateDecision.ALLOW_WITH_WARNING}:
                if len(allowed) < top_k:
                    allowed.append(item)
                else:
                    not_selected.append(item)
            else:
                blocked.append(item)
                if item.decision == GateDecision.QUARANTINE:
                    item.memory.state = MemoryState.QUARANTINED
                    self.store.update(item.memory)

        return GateReport(allowed=allowed, blocked=blocked, ordinary_top_k=ordinary, not_selected=not_selected)

    def gate_candidates(self, candidates: list[MemoryItem], task: TaskContext, top_k: int = 6) -> list[ScoredMemory]:
        return self.evaluate_candidates(candidates, task, top_k=top_k).allowed

    def retrieve_and_gate(self, task: TaskContext, candidates: list[MemoryItem] | None = None, top_k: int = 6) -> list[ScoredMemory]:
        # Prototype fallback: use active store items as candidates.
        candidates = candidates if candidates is not None else self.store.active()
        return self.gate_candidates(candidates, task, top_k=top_k)

    def retrieve_report(self, task: TaskContext, candidates: list[MemoryItem] | None = None, top_k: int = 6) -> GateReport:
        candidates = candidates if candidates is not None else self.store.all()
        return self.evaluate_candidates(candidates, task, top_k=top_k)

    def allowed_context(self, scored: list[ScoredMemory]) -> str:
        return "\n\n".join(item.memory.text for item in scored if item.decision in {GateDecision.ALLOW, GateDecision.ALLOW_WITH_WARNING})

    def apply_feedback(self, event: FeedbackEvent) -> None:
        memory = self.store.get(event.memory_id)
        if memory is None:
            return

        if event.helped:
            memory.help_count += 1
            memory.usefulness_score = min(1.0, memory.usefulness_score + 0.12 * event.weight)
            memory.harm_score = max(0.0, memory.harm_score - 0.05 * event.weight)
            memory.risk = max(0.0, memory.risk - 0.03 * event.weight)
            if memory.help_count >= 3 and memory.harm_count == 0:
                memory.state = MemoryState.VALIDATED

        if event.harmed or event.contradicted:
            memory.harm_count += 1
            memory.harm_score = min(1.0, memory.harm_score + 0.16 * event.weight)
            memory.usefulness_score = max(-1.0, memory.usefulness_score - 0.10 * event.weight)
            memory.risk = min(1.0, memory.risk + 0.10 * event.weight)
            if memory.harm_count >= 2 or event.contradicted:
                memory.state = MemoryState.QUARANTINED

        memory.last_used_at = time()
        self.store.update(memory)
