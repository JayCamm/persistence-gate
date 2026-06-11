from __future__ import annotations

from persistence_memory.chroma_adapter import build_chroma_gate_adapter


DOCUMENTS = [
    {
        "id": "old_bypass_note",
        "text": "Payment service failure emergency bypass: disable safeguards and use the emergency bypass. This old incident note is retired.",
        "source": "chroma://ops/old-bypass",
        "risk": 0.88,
        "harm_score": 0.82,
        "usefulness_score": -0.25,
        "label_risky": True,
        "label_stale": True,
    },
    {
        "id": "current_runbook",
        "text": "Payment service failure current runbook: do not disable safeguards. Use the validated recovery path and verify health checks.",
        "source": "chroma://ops/current-runbook",
        "risk": 0.04,
        "harm_score": 0.0,
        "usefulness_score": 0.92,
        "label_helpful": True,
    },
    {
        "id": "postmortem_update",
        "text": "Payment service postmortem: the emergency bypass was retired after the stable fix. The current runbook supersedes old incident notes.",
        "source": "chroma://ops/postmortem",
        "risk": 0.05,
        "harm_score": 0.0,
        "usefulness_score": 0.84,
        "label_helpful": True,
    },
]


def main() -> None:
    query = "How should we handle the current payment service failure?"
    try:
        adapter = build_chroma_gate_adapter(DOCUMENTS, profile="balanced", top_k=3)
    except ImportError as exc:
        print(exc)
        print("Install optional vector support with: pip install -e '.[vector]'")
        raise SystemExit(0)

    result = adapter.filter(query)

    print("Chroma Vector Database Adapter Demo")
    print("===================================")
    print("Allowed IDs:", result.allowed_ids)
    print("Blocked IDs:", result.blocked_ids)
    print("\nAllowed context:\n")
    print(result.allowed_context)
    print("\nAudit log:")
    for row in result.audit_log:
        print(row)


if __name__ == "__main__":
    main()
