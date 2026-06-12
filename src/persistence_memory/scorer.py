from __future__ import annotations

from math import exp
from time import time

from .models import GateDecision, MemoryItem, MemoryState, ScoredMemory, TaskContext
from .profiles import GateProfile, get_profile


class PersistenceScorer:
    """Scores whether a memory should influence a current task.

    This is intentionally transparent and adjustable. It is not a learned model yet.
    """

    def __init__(
        self,
        allow_threshold: float | None = None,
        warning_threshold: float | None = None,
        quarantine_threshold: float | None = None,
        harm_weight: float | None = None,
        burden_weight: float | None = None,
        staleness_weight: float | None = None,
        risk_weight: float | None = None,
        profile: str | GateProfile = "balanced",
    ) -> None:
        self.profile = get_profile(profile)
        self.allow_threshold = self.profile.allow_threshold if allow_threshold is None else allow_threshold
        self.warning_threshold = self.profile.warning_threshold if warning_threshold is None else warning_threshold
        self.quarantine_threshold = self.profile.quarantine_threshold if quarantine_threshold is None else quarantine_threshold
        self.harm_weight = self.profile.harm_weight if harm_weight is None else harm_weight
        self.burden_weight = self.profile.burden_weight if burden_weight is None else burden_weight
        self.staleness_weight = self.profile.staleness_weight if staleness_weight is None else staleness_weight
        self.risk_weight = self.profile.risk_weight if risk_weight is None else risk_weight

    def score(self, memory: MemoryItem, task: TaskContext, now: float | None = None) -> ScoredMemory:
        now = now or time()
        reasons: list[str] = []

        relevance = memory.relevance
        context_fit = 1.0 if memory.context_scope in {"global", task.context_scope} else 0.25
        validity = self._validity(memory, now)
        usefulness = memory.usefulness_score + 0.12 * memory.help_count
        harm = memory.harm_score + 0.16 * memory.harm_count
        burden = memory.burden
        risk = memory.risk
        validated_bonus = 0.15 if memory.state == MemoryState.VALIDATED else 0.0
        need = task.need

        raw_score = (
            0.34 * relevance
            + 0.20 * context_fit
            + 0.18 * validity
            + 0.22 * usefulness
            + 0.12 * need
            + validated_bonus
            - self.harm_weight * harm
            - self.burden_weight * burden
            - self.risk_weight * risk
        )

        # Abstention-aware adjustment. If task risk tolerance is low, memory must earn more influence.
        score = raw_score - (1.0 - task.risk_tolerance) * 0.12 - task.abstention_score

        if validity < 0.35:
            reasons.append("stale_or_expired")
        if context_fit < 0.5:
            reasons.append("context_mismatch")
        if harm > 0.5:
            reasons.append("high_harm_history")
        if burden > 0.5:
            reasons.append("high_burden")
        if self.profile.high_risk_block_threshold is not None and risk >= self.profile.high_risk_block_threshold:
            reasons.append("profile_high_risk_block")
        if self.profile.high_harm_block_threshold is not None and harm >= self.profile.high_harm_block_threshold:
            reasons.append("profile_high_harm_block")
        if memory.state == MemoryState.VALIDATED:
            reasons.append("validated")

        decision = self._decision(score, memory, reasons)
        return ScoredMemory(memory=memory, score=score, decision=decision, reasons=reasons)

    def _validity(self, memory: MemoryItem, now: float) -> float:
        if memory.valid_until is None:
            return 0.75
        remaining = memory.valid_until - now
        if remaining >= 0:
            return 1.0 / (1.0 + exp(-remaining / 86_400.0))
        return exp(remaining / 86_400.0)

    def _decision(self, score: float, memory: MemoryItem, reasons: list[str]) -> GateDecision:
        if memory.state in {MemoryState.DELETED, MemoryState.ARCHIVED}:
            return GateDecision.IGNORE
        if memory.state == MemoryState.QUARANTINED:
            return GateDecision.QUARANTINE
        if "profile_high_risk_block" in reasons or "profile_high_harm_block" in reasons or "high_harm_history" in reasons:
            return GateDecision.QUARANTINE
        if score >= self.allow_threshold:
            return GateDecision.ALLOW
        if score >= self.warning_threshold:
            return GateDecision.ALLOW_WITH_WARNING
        if score <= self.quarantine_threshold:
            return GateDecision.QUARANTINE
        if "stale_or_expired" in reasons:
            return GateDecision.REFRESH_REQUIRED
        return GateDecision.IGNORE
