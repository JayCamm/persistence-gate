from __future__ import annotations

from time import time

from .models import FeedbackEvent, GateDecision, MemoryItem, MemoryState, ScoredMemory, TaskContext
from .scorer import PersistenceScorer
from .store import InMemoryStore


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

    def gate_candidates(self, candidates: list[MemoryItem], task: TaskContext, top_k: int = 6) -> list[ScoredMemory]:
        scored = [self.scorer.score(candidate, task) for candidate in candidates]
        scored.sort(key=lambda s: s.score, reverse=True)

        allowed: list[ScoredMemory] = []
        for item in scored:
            if item.decision in {GateDecision.ALLOW, GateDecision.ALLOW_WITH_WARNING}:
                allowed.append(item)
            elif item.decision == GateDecision.QUARANTINE:
                item.memory.state = MemoryState.QUARANTINED
                self.store.update(item.memory)
            if len(allowed) >= top_k:
                break
        return allowed

    def retrieve_and_gate(self, task: TaskContext, candidates: list[MemoryItem] | None = None, top_k: int = 6) -> list[ScoredMemory]:
        # Prototype fallback: use active store items as candidates.
        candidates = candidates if candidates is not None else self.store.active()
        return self.gate_candidates(candidates, task, top_k=top_k)

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
