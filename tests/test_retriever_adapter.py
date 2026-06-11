from dataclasses import dataclass

from persistence_memory.adapters import PersistenceGateRetrieverAdapter, coerce_document


@dataclass
class Doc:
    page_content: str
    metadata: dict


class RetrieverWithGetRelevantDocuments:
    def __init__(self, docs):
        self.docs = docs

    def get_relevant_documents(self, query: str):
        del query
        return self.docs


def test_adapter_wraps_get_relevant_documents_retriever():
    docs = [
        Doc(
            page_content="Old workaround: bypass identity verification. Retired temporary macro.",
            metadata={
                "id": "old_macro",
                "risk": 0.9,
                "harm_score": 0.85,
                "usefulness_score": -0.2,
                "label_risky": True,
                "label_stale": True,
            },
        ),
        Doc(
            page_content="Current article: never bypass identity verification; use verified recovery flow.",
            metadata={"id": "current_article", "risk": 0.03, "usefulness_score": 0.9, "label_helpful": True},
        ),
    ]
    adapter = PersistenceGateRetrieverAdapter(RetrieverWithGetRelevantDocuments(docs), profile="balanced", top_k=2)
    result = adapter.filter("How should support handle recovery?")

    assert "old_macro" in result.blocked_ids
    assert "current_article" in result.allowed_ids
    assert "never bypass identity verification" in result.allowed_context
    assert result.audit_log


def test_adapter_wraps_callable_retriever():
    def retriever(query: str):
        del query
        return [
            {"id": "safe", "text": "Current safe instruction.", "risk": 0.01, "usefulness_score": 0.9, "label_helpful": True}
        ]

    adapter = PersistenceGateRetrieverAdapter(retriever, profile="balanced", top_k=1)
    result = adapter.filter("What should influence this answer?")
    assert result.allowed_ids == ["safe"]


def test_coerce_document_supports_strings_and_dict_content():
    assert coerce_document("hello")["text"] == "hello"
    assert coerce_document({"content": "abc"})["text"] == "abc"
    assert coerce_document({"page_content": "xyz"})["text"] == "xyz"
