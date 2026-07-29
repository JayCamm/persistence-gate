# Persistence Gate

**Persistence Gate** is provider-neutral evidence-control infrastructure for AI systems. It sits between retrieval or memory and generation/action, then determines which information is allowed to influence the downstream answer, recommendation, or workflow.

```text
query -> retrieval / memory -> source-bound candidates -> Persistence Gate
      -> allowed / held / blocked / quarantined evidence
      -> answer permission + trace -> LLM / agent / human review
```

## Repository status

This public repository contains the original installable reference implementation and demonstrations under `src/persistence_memory/`.

The current canonical private-alpha engine used by Constellation and the BCV shadow pilot is **Persistence Gate Core v1.4.2**. It is materially broader than this original reference package and adds:

- source-bound evidence objects and exact coordinates;
- lifecycle state, effective-time, scope, authority, and proposition normalization;
- correction and supersession handling;
- explicit unknown/review states;
- deterministic evidence admission and answer permission;
- retrieved-content security quarantine;
- statement verification;
- decision trace and replay;
- provider-neutral model and retrieval adapters.

The canonical v1.4.2 source is not published in this public repository. This repository is being retained as a transparent reference implementation and public research record. See `docs/CANONICAL_V1_4_2_STATUS.md` for the current architecture and evidence boundary.

## Why this exists

Most retrieval-augmented systems follow this path:

```text
find relevant information -> put it in context -> generate an answer
```

Relevance is not enough. Information can be stale, superseded, draft-only, outside scope, low-authority, contradictory, malicious, or incomplete while remaining highly relevant to the query.

Persistence Gate adds a separate influence decision:

```text
find information -> bind it to its source -> assess state / authority / scope / risk
-> allow, hold, block, quarantine, or request review
```

## Public reference implementation

Install and run the original public package:

```bash
git clone https://github.com/JayCamm/persistence-gate.git
cd persistence-gate
pip install -e ".[dev]"
pytest
```

Primary API:

```python
from persistence_memory.api import PersistenceGate

gate = PersistenceGate(profile="balanced", top_k=6)
result = gate.filter(
    query="Which guidance should influence the answer?",
    retrieved_items=retrieved_items,
)

print(result.allowed_ids)
print(result.blocked_ids)
print(result.allowed_context)
print(result.audit_log)
```

The public package remains useful for understanding post-retrieval influence control, profile tradeoffs, and deterministic RAG integration. It should not be confused with the full v1.4.2 private-alpha runtime.

## Current validation evidence

The current project evidence includes:

- a 120-case structured calibration benchmark identifying a balanced authority range of 70–90 within that fixture;
- a 240-case synthetic messy-evidence normalization benchmark with a frozen wording holdout;
- source-binding validation across 34 pages and 868 extracted candidates from two clean real PDFs;
- 105 passing core regression tests for v1.4.2;
- a seven-test Microsoft Foundry adapter suite;
- BCV shadow integration in which the new engine remains observational rather than controlling.

These results are bounded. They do not establish universal document understanding, production security, independent human agreement, live tenant permission trimming, or enterprise ROI. Full details and checksums are under `docs/`.

## Design principle

Information should not influence an AI system merely because it was retrieved, remembered, generated, or stored in an approved system. It should earn permission for the specific answer or action.

## Supported claim

Persistence Gate has controlled-test evidence that explicit evidence admission, lifecycle handling, authority separation, and security quarantine can reduce unsafe influence while preserving useful current evidence in bounded fixtures.

## Not yet proven

Persistence Gate is not yet production-proven. The next proof requirements are independently annotated heterogeneous documents, live Microsoft identity and permission tests, adversarial retrieval testing, head-to-head comparisons against ordinary RAG/assistants, and measured reduction in human verification burden.
