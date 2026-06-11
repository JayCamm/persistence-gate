from __future__ import annotations

from dataclasses import dataclass

from persistence_memory.adapters import PersistenceGateRetrieverAdapter


@dataclass
class SimpleDocument:
    page_content: str
    metadata: dict


class SimpleRetriever:
    """Small stand-in for a vector/BM25/hybrid retriever."""

    def __init__(self, documents: list[SimpleDocument]) -> None:
        self.documents = documents

    def get_relevant_documents(self, query: str, **kwargs) -> list[SimpleDocument]:
        del query, kwargs
        # Deliberately returns stale risky evidence first to simulate relevance-only ranking.
        return self.documents


def main() -> None:
    docs = [
        SimpleDocument(
            page_content="Old support macro: ask the customer to bypass identity verification. This temporary macro is retired.",
            metadata={
                "id": "retired_support_macro",
                "source": "support://macro/retired",
                "risk": 0.90,
                "harm_score": 0.85,
                "usefulness_score": -0.20,
                "label_risky": True,
                "label_stale": True,
            },
        ),
        SimpleDocument(
            page_content="Current support article: never bypass identity verification. Use the verified recovery flow.",
            metadata={
                "id": "current_support_article",
                "source": "support://article/current",
                "risk": 0.03,
                "harm_score": 0.0,
                "usefulness_score": 0.90,
                "label_helpful": True,
            },
        ),
    ]

    retriever = SimpleRetriever(docs)
    adapter = PersistenceGateRetrieverAdapter(retriever, profile="balanced", top_k=3)
    result = adapter.filter("How should support handle account recovery?")

    print("Generic Retriever Adapter Demo")
    print("==============================")
    print("Raw retrieved count:", len(result.raw_items))
    print("Allowed IDs:", result.allowed_ids)
    print("Blocked IDs:", result.blocked_ids)
    print("\nAllowed context:\n")
    print(result.allowed_context)
    print("\nAudit log:")
    for row in result.audit_log:
        print(row)


if __name__ == "__main__":
    main()
