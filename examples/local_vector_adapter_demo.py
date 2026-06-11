from __future__ import annotations

from persistence_memory.adapters import PersistenceGateRetrieverAdapter
from persistence_memory.local_vector import LocalVectorStore


DOCUMENTS = [
    {
        "id": "old_payment_bypass",
        "text": "Payment service failure recovery: disable safeguards and use the emergency bypass. This old temporary incident note is retired.",
        "source": "local-vector://ops/old-bypass",
        "risk": 0.88,
        "harm_score": 0.82,
        "usefulness_score": -0.25,
        "label_risky": True,
        "label_stale": True,
    },
    {
        "id": "current_payment_runbook",
        "text": "Payment service failure recovery: do not disable safeguards. Use the validated recovery path and verify health checks.",
        "source": "local-vector://ops/current-runbook",
        "risk": 0.04,
        "harm_score": 0.0,
        "usefulness_score": 0.92,
        "label_helpful": True,
    },
    {
        "id": "postmortem_payment_fix",
        "text": "Payment service postmortem: the emergency bypass was retired after the stable fix. The current runbook supersedes old incident notes.",
        "source": "local-vector://ops/postmortem",
        "risk": 0.05,
        "harm_score": 0.0,
        "usefulness_score": 0.84,
        "label_helpful": True,
    },
    {
        "id": "unrelated_formatting_note",
        "text": "Documentation formatting note: use consistent headings in runbooks.",
        "source": "local-vector://docs/formatting",
        "risk": 0.02,
        "harm_score": 0.0,
        "usefulness_score": 0.05,
    },
]


def main() -> None:
    query = "payment service failure recovery safeguards emergency bypass"
    store = LocalVectorStore(DOCUMENTS)
    retriever = store.as_retriever(top_k=4)
    raw_results = retriever.retrieve(query)

    adapter = PersistenceGateRetrieverAdapter(retriever, profile="balanced", top_k=4)
    result = adapter.filter(query)

    print("Local Vector Adapter Demo")
    print("=========================")
    print("\nRaw vector retrieval order:")
    for doc in raw_results:
        print(doc.metadata.get("id"), "score=", round(float(doc.metadata.get("retrieval_score", 0.0)), 4))

    print("\nPersistence Gate output:")
    print("Allowed IDs:", result.allowed_ids)
    print("Blocked IDs:", result.blocked_ids)

    print("\nAllowed context:\n")
    print(result.allowed_context)

    print("\nAudit log:")
    for row in result.audit_log:
        print(row)


if __name__ == "__main__":
    main()
