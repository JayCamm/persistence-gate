"""Persistence-aware memory controller.

This package provides a small software prototype for influence-gated retrieval:
retrieve candidates, score whether they should influence the current task, gate them,
then update memory state from feedback.
"""

from .models import MemoryItem, MemoryState, TaskContext, GateDecision, FeedbackEvent
from .controller import GateReport, MemoryController
from .store import InMemoryStore
from .scorer import PersistenceScorer
from .retriever import lexical_relevance, rank_by_relevance
from .benchmark import BenchmarkResult, StrategyMetrics, evaluate_gate_vs_topk

__all__ = [
    "MemoryItem",
    "MemoryState",
    "TaskContext",
    "GateDecision",
    "FeedbackEvent",
    "GateReport",
    "MemoryController",
    "InMemoryStore",
    "PersistenceScorer",
    "lexical_relevance",
    "rank_by_relevance",
    "BenchmarkResult",
    "StrategyMetrics",
    "evaluate_gate_vs_topk",
]
