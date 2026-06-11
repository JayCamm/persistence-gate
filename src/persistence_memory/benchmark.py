from __future__ import annotations

from dataclasses import dataclass

from .controller import GateReport, MemoryController
from .models import MemoryItem, TaskContext
from .retriever import rank_by_relevance
from .store import InMemoryStore


@dataclass
class StrategyMetrics:
    selected: int = 0
    helpful_selected: int = 0
    risky_selected: int = 0
    stale_selected: int = 0
    burden: float = 0.0

    def net_utility(self, helpful_weight: float = 1.0, risk_weight: float = 1.0, stale_weight: float = 0.7, burden_weight: float = 0.25) -> float:
        return (
            helpful_weight * self.helpful_selected
            - risk_weight * self.risky_selected
            - stale_weight * self.stale_selected
            - burden_weight * self.burden
        )


@dataclass
class BenchmarkResult:
    ordinary: StrategyMetrics
    gated: StrategyMetrics
    report: GateReport

    @property
    def utility_gain(self) -> float:
        return self.gated.net_utility() - self.ordinary.net_utility()

    @property
    def risky_items_prevented(self) -> int:
        return self.ordinary.risky_selected - self.gated.risky_selected

    @property
    def stale_items_prevented(self) -> int:
        return self.ordinary.stale_selected - self.gated.stale_selected


def item_is_helpful(item: MemoryItem) -> bool:
    return bool(item.metadata.get("label_helpful")) or item.usefulness_score >= 0.45


def item_is_risky(item: MemoryItem) -> bool:
    return bool(item.metadata.get("label_risky")) or item.risk >= 0.45 or item.harm_score >= 0.35


def item_is_stale(item: MemoryItem) -> bool:
    return bool(item.metadata.get("label_stale")) or item.metadata.get("age_bucket") == "old" or item.metadata.get("kind") == "old_commit"


def compute_metrics(items: list[MemoryItem]) -> StrategyMetrics:
    return StrategyMetrics(
        selected=len(items),
        helpful_selected=sum(1 for item in items if item_is_helpful(item)),
        risky_selected=sum(1 for item in items if item_is_risky(item)),
        stale_selected=sum(1 for item in items if item_is_stale(item)),
        burden=sum(item.burden for item in items),
    )


def evaluate_gate_vs_topk(items: list[MemoryItem], task: TaskContext, top_k: int = 8) -> BenchmarkResult:
    """Compare ordinary relevance-only top-k against Persistence Gate.

    Relevance is computed from the task query for every item. Ordinary top-k uses
    only that relevance. Persistence Gate sees the same relevance scores but also
    uses risk, harm, usefulness, burden, context, and state.
    """
    ranked = rank_by_relevance(task.query, items, copy_items=True)
    ordinary_items = ranked[:top_k]

    controller = MemoryController(InMemoryStore(ranked))
    report = controller.retrieve_report(task, top_k=top_k)
    gated_items = [scored.memory for scored in report.allowed]

    return BenchmarkResult(
        ordinary=compute_metrics(ordinary_items),
        gated=compute_metrics(gated_items),
        report=report,
    )
