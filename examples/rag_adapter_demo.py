from __future__ import annotations

from persistence_memory.api import PersistenceGate


CORPUS = [
    {
        "id": "old_workaround",
        "text": "Old incident note: for payment service failures, disable safeguards and use the emergency bypass. This was a temporary workaround.",
        "source": "kb://incident/old-workaround",
        "risk": 0.88,
        "harm_score": 0.82,
        "usefulness_score": -0.25,
        "label_risky": True,
        "label_stale": True,
    },
    {
        "id": "current_runbook",
        "text": "Current payment service runbook: do not disable safeguards. Use the validated low-risk recovery path and verify health checks.",
        "source": "kb://runbook/current-payment-service",
        "risk": 0.04,
        "harm_score": 0.0,
        "usefulness_score": 0.92,
        "label_helpful": True,
    },
    {
        "id": "postmortem_update",
        "text": "Postmortem update: the emergency bypass was retired after the stable fix. The current runbook supersedes the old incident note.",
        "source": "kb://postmortem/payment-service-fix",
        "risk": 0.05,
        "harm_score": 0.0,
        "usefulness_score": 0.84,
        "label_helpful": True,
    },
    {
        "id": "formatting_note",
        "text": "Documentation note: use consistent headings and labels in runbooks.",
        "source": "kb://docs/formatting",
        "risk": 0.02,
        "harm_score": 0.0,
        "usefulness_score": 0.05,
    },
]


def fake_retriever(query: str, top_k: int = 4) -> list[dict]:
    """Tiny deterministic retriever stand-in.

    A real RAG system would replace this with a vector store, BM25, hybrid
    search, or tool output. We deliberately rank the old workaround high to
    show the failure mode of relevance-only retrieval.
    """
    del query
    ordered_ids = ["old_workaround", "current_runbook", "postmortem_update", "formatting_note"]
    by_id = {doc["id"]: doc for doc in CORPUS}
    return [by_id[doc_id] for doc_id in ordered_ids[:top_k]]


def build_prompt(query: str, context: str) -> str:
    return f"Answer using only this allowed context.\n\nContext:\n{context}\n\nQuestion:\n{query}\n"


def deterministic_answer(context: str) -> str:
    """A stand-in for an LLM so the demo is reproducible without API keys."""
    if "disable safeguards" in context and "do not disable safeguards" not in context:
        return "Use the emergency bypass and disable safeguards."
    if "do not disable safeguards" in context:
        return "Do not disable safeguards. Use the current validated recovery path and verify health checks."
    return "Insufficient safe context; request a refresh."


def main() -> None:
    query = "How should we handle the current payment service failure?"
    retrieved = fake_retriever(query)

    ordinary_context = "\n\n".join(item["text"] for item in retrieved)
    ordinary_prompt = build_prompt(query, ordinary_context)
    ordinary_answer = deterministic_answer(ordinary_context)

    gate = PersistenceGate(profile="balanced", top_k=4)
    gated = gate.filter(query, retrieved)
    gated_prompt = build_prompt(query, gated.allowed_context)
    gated_answer = deterministic_answer(gated.allowed_context)

    print("Deterministic RAG Adapter Demo")
    print("==============================")
    print("\nRetrieved IDs:")
    print([item["id"] for item in retrieved])

    print("\n--- Ordinary relevance-only RAG context ---")
    print(ordinary_context)
    print("\nOrdinary answer:")
    print(ordinary_answer)

    print("\n--- Persistence Gate output ---")
    print("Allowed IDs:", gated.allowed_ids)
    print("Blocked IDs:", gated.blocked_ids)
    print("Blocked from ordinary top-k:", gated.report.blocked_from_ordinary_top_k)

    print("\n--- Gated RAG context ---")
    print(gated.allowed_context)
    print("\nGated answer:")
    print(gated_answer)

    print("\n--- Audit log ---")
    for row in gated.audit_log:
        print(row)

    print("\n--- Prompt shape ordinary ---")
    print(ordinary_prompt)
    print("\n--- Prompt shape gated ---")
    print(gated_prompt)


if __name__ == "__main__":
    main()
