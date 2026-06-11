from persistence_memory.adapters import PersistenceGateRetrieverAdapter
from persistence_memory.local_vector import LocalVectorStore, cosine_similarity, bag_of_words_vector


def test_local_vector_similarity():
    left = bag_of_words_vector("payment service recovery")
    right = bag_of_words_vector("payment service failure recovery")
    unrelated = bag_of_words_vector("documentation headings")
    assert cosine_similarity(left, right) > cosine_similarity(left, unrelated)


def test_local_vector_store_returns_ranked_documents():
    store = LocalVectorStore(
        [
            {"id": "unrelated", "text": "documentation headings and formatting"},
            {"id": "related", "text": "payment service recovery safeguards"},
        ]
    )
    results = store.search("payment service recovery", top_k=2)
    assert results[0].metadata["id"] == "related"
    assert results[0].metadata["retrieval_score"] > results[1].metadata["retrieval_score"]


def test_local_vector_retriever_adapter_blocks_risky_nearest_neighbor():
    docs = [
        {
            "id": "old_bypass",
            "text": "payment service failure recovery disable safeguards emergency bypass retired",
            "risk": 0.9,
            "harm_score": 0.85,
            "usefulness_score": -0.2,
            "label_risky": True,
            "label_stale": True,
        },
        {
            "id": "current_runbook",
            "text": "payment service failure recovery do not disable safeguards validated recovery path",
            "risk": 0.03,
            "harm_score": 0.0,
            "usefulness_score": 0.9,
            "label_helpful": True,
        },
    ]
    store = LocalVectorStore(docs)
    retriever = store.as_retriever(top_k=2)
    raw = retriever.retrieve("payment service failure recovery emergency bypass")
    assert raw[0].metadata["id"] == "old_bypass"

    adapter = PersistenceGateRetrieverAdapter(retriever, profile="balanced", top_k=2)
    result = adapter.filter("payment service failure recovery emergency bypass")
    assert "old_bypass" in result.blocked_ids
    assert "current_runbook" in result.allowed_ids
    assert "do not disable safeguards" in result.allowed_context
