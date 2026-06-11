from __future__ import annotations

from .models import MemoryItem, MemoryState


class InMemoryStore:
    """Small in-memory store for prototype and tests."""

    def __init__(self, items: list[MemoryItem] | None = None) -> None:
        self._items: dict[str, MemoryItem] = {item.id: item for item in (items or [])}

    def add(self, item: MemoryItem) -> None:
        self._items[item.id] = item

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._items.get(memory_id)

    def all(self, include_deleted: bool = False) -> list[MemoryItem]:
        items = list(self._items.values())
        if include_deleted:
            return items
        return [item for item in items if item.state != MemoryState.DELETED]

    def active(self) -> list[MemoryItem]:
        return [item for item in self._items.values() if item.is_active()]

    def update(self, item: MemoryItem) -> None:
        self._items[item.id] = item

    def __len__(self) -> int:
        return len(self._items)
