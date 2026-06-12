from persistence_memory.api import PersistenceGate


def test_legacy_instruction_blocked_for_current_action_even_when_useful():
    gate = PersistenceGate(profile="conservative", top_k=3)
    result = gate.filter(
        "What should guide the current process now?",
        [
            {
                "id": "legacy_high_usefulness",
                "text": "Legacy instruction: use the old shortcut procedure.",
                "relevance": 1.0,
                "risk": 0.70,
                "harm_score": 0.45,
                "usefulness_score": 0.95,
                "evidence_role": "legacy_instruction",
            },
            {
                "id": "current_guidance",
                "text": "Current guidance: use the approved current process.",
                "relevance": 0.82,
                "risk": 0.03,
                "harm_score": 0.0,
                "usefulness_score": 0.85,
                "evidence_role": "current_guidance",
            },
        ],
        query_intent="current_action",
    )

    assert "current_guidance" in result.allowed_ids
    assert "legacy_high_usefulness" in result.blocked_ids
    legacy_audit = next(row for row in result.audit_log if row["id"] == "legacy_high_usefulness")
    assert "legacy_instruction_for_current_action" in legacy_audit["reasons"]


def test_warning_against_legacy_is_allowed_for_current_action():
    gate = PersistenceGate(profile="conservative", top_k=3)
    result = gate.filter(
        "What should guide the current process now?",
        [
            {
                "id": "warning_doc",
                "text": "Current warning memo: do not use the old unsafe shortcut. Use the approved process.",
                "relevance": 1.0,
                "risk": 0.55,
                "harm_score": 0.0,
                "usefulness_score": 0.90,
                "evidence_role": "warning_against_legacy",
            }
        ],
        query_intent="current_action",
    )

    assert "warning_doc" in result.allowed_ids
    warning_audit = next(row for row in result.audit_log if row["id"] == "warning_doc")
    assert "warning_against_legacy_not_instruction" in warning_audit["reasons"]


def test_historical_context_allowed_for_history_question():
    gate = PersistenceGate(profile="conservative", top_k=3)
    result = gate.filter(
        "What changed from the old process to the current process?",
        [
            {
                "id": "history_doc",
                "text": "Historical comparison note: this old procedure explains what changed.",
                "relevance": 1.0,
                "risk": 0.28,
                "harm_score": 0.0,
                "usefulness_score": 0.84,
                "evidence_role": "historical_context",
            }
        ],
        query_intent="history_comparison",
    )

    assert "history_doc" in result.allowed_ids
    history_audit = next(row for row in result.audit_log if row["id"] == "history_doc")
    assert "history_requested" in history_audit["reasons"]
