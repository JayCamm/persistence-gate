"""Persistence-aware memory controller.

This package provides a small software prototype for influence-gated retrieval:
retrieve candidates, score whether they should influence the current task, gate them,
then update memory state from feedback.
"""

from .models import MemoryItem, MemoryState, TaskContext, GateDecision, FeedbackEvent
from .controller import GateReport, MemoryController
from .store import InMemoryStore
from .scorer import PersistenceScorer

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
]
