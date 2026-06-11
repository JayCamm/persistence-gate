from persistence_memory.adapters import PersistenceGateRetrieverAdapter
from persistence_memory.chroma_adapter import ChromaRetriever, hash_embedding


class FakeChromaCollection:
    def query(self, query_embeddings, n_results, include):
        del query_embeddings, include
        documents = [
            "Old macro: bypass identity verification. This retired macro is unsafe.",
            "Current article: never bypass identity verification; use verified recovery.",
        ][:n_results]
        metadatas = [
            {
                "id": "old_macro",
                "source": "fake://old",
                "risk": 0.9,
                "harm_score": 0.85,
                "usefulness_score": -0.2,
                "label_risky": True,
                "label_stale": True,
            },
            {
                "id": "current_article",
                "source": "fake://current",
                "risk": 0.03,
                "harm_score": 0.0,
                "usefulness_score": 0.9,
                "label_helpful": True,
            },
        ][:n_results]
        return {"documents": [documents], "metadatas": [metadatas], "distances": [[0.1, 0.2][:n_results]]}


def test_hash_embedding_is_deterministic_and_normalized():
    first = hash_embedding("payment service recovery", dimensions=16)
    second = hash_embedding("payment service recovery", dimensions=16)
    assert first == second
    assert len(first) == 16
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9


def test_chroma_retriever_can_be_gated_with_fake_collection():
    retriever = ChromaRetriever(FakeChromaCollection(), top_k=2)
    adapter = PersistenceGateRetrieverAdapter(retriever, profile="balanced", top_k=2)
    result = adapter.filter("How should support handle recovery?")

    assert "old_macro" in result.blocked_ids
    assert "current_article" in result.allowed_ids
    assert "never bypass identity verification" in result.allowed_context
