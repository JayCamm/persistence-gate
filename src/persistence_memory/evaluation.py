from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    total: int = 0
    allowed: int = 0
    blocked: int = 0
    harmful_allowed: int = 0
    helpful_allowed: int = 0
    burden: float = 0.0

    @property
    def harmful_rate(self) -> float:
        return self.harmful_allowed / max(1, self.allowed)

    @property
    def allow_rate(self) -> float:
        return self.allowed / max(1, self.total)

    def net_utility(self, harm_weight: float = 0.10, burden_weight: float = 0.03) -> float:
        # Prototype utility: helpful allowed minus harmful allowed and burden.
        return self.helpful_allowed - harm_weight * self.harmful_allowed - burden_weight * self.burden
