# Persistence Gate

**Persistence Gate** is AI memory-governance middleware. It sits between retrieval and generation/action, then decides which retrieved information is allowed to influence the downstream system.

It does **not** replace search, embeddings, vector databases, documents, or LLMs. It controls influence after retrieval.

```text
query -> retriever/vector DB/search -> candidate memories -> Persistence Gate -> allowed context + audit log -> LLM/agent/action
```

## Why this exists

Most retrieval-augmented generation systems do this:

```text
find relevant information -> put it in context -> generate an answer
```

That is risky because information can be relevant and still be harmful to use. Old workarounds, contradicted claims, retired policies, temporary fixes, and stale instructions can stay highly relevant long after they should stop influencing the system.

Persistence Gate adds a missing step:

```text
find relevant information -> check validity/risk/usefulness/context -> allow, warn, block, quarantine, or refresh
```

## What Persistence Gate does

Persistence Gate scores each retrieved memory item using transparent signals:

- relevance
- current validity / staleness
- usefulness
- risk
- harm history
- burden
- context fit
- task need
- abstention baseline
- selected operating profile

It returns:

- allowed items
- blocked items
- allowed context string for downstream prompts
- warnings
- ordinary top-k baseline IDs
- blocked-from-baseline IDs
- audit log with scores, reasons, and decisions

## What it is not

Persistence Gate is not RAM, not a vector database, not a search engine, not a model, and not a replacement for RAG. It is an influence-control layer for retrieved information.

## Current status

Working implementation prototype.

Current evidence:

- 29 automated tests passing
- implementation API working
- deterministic RAG adapter demo working
- corrected multi-domain stress benchmark: 2,000/2,000 pass, 0 true failures
- profile sensitivity benchmark shows meaningful safety/recall tradeoff
- real GitHub issue-history benchmark was promising, but live API limits make it less stable than offline benchmarks

This is not production-ready yet. It is ready for implementation experiments, review, and integration testing.

## Install

```bash
git clone https://github.com/JayCamm/persistence-gate.git
cd persistence-gate
pip install -e ".[dev]"
pytest
```

## Quickstart: gate retrieved items

```python
from persistence_memory.api import PersistenceGate

retrieved_items = [
    {
        "id": "old_workaround",
        "text": "Old workaround: disable safeguards and use emergency bypass.",
        "risk": 0.88,
        "harm_score": 0.82,
        "usefulness_score": -0.25,
        "label_risky": True,
        "label_stale": True,
    },
    {
        "id": "current_runbook",
        "text": "Current runbook: do not disable safeguards. Use the validated recovery path.",
        "risk": 0.04,
        "usefulness_score": 0.90,
        "label_helpful": True,
    },
]

gate = PersistenceGate(profile="balanced", top_k=3)
result = gate.filter(
    query="Which runbook guidance should influence the answer?",
    retrieved_items=retrieved_items,
)

print(result.allowed_ids)
print(result.blocked_ids)
print(result.allowed_context)
print(result.audit_log)
```

Expected behavior:

```text
allowed: current safe evidence
blocked: old risky workaround
audit: score + decision reasons for each item
```

## RAG adapter demo

Run:

```bash
python examples/rag_adapter_demo.py
```

This demo shows the core failure mode and the fix:

```text
same retrieved documents
ordinary RAG context includes obsolete emergency-bypass guidance
Persistence Gate blocks that item
allowed context contains current runbook + postmortem update
downstream answer becomes safe
```

## Implementation API

Main import:

```python
from persistence_memory.api import PersistenceGate
```

Primary call:

```python
result = gate.filter(
    query=query,
    retrieved_items=retrieved_items,
    profile="balanced",
    top_k=6,
    context_scope="project",
    need=0.90,
    risk_tolerance=0.35,
    abstention_score=0.04,
)
```

Accepted item formats:

- `MemoryItem` objects
- dictionaries with at least a `text` field

Useful result fields:

```python
result.allowed
result.blocked
result.not_selected
result.allowed_items
result.blocked_items
result.allowed_ids
result.blocked_ids
result.warnings
result.allowed_context
result.audit_log
result.to_dict()
```

## Operating profiles

Persistence Gate supports three profiles:

| Profile | Use case | Behavior |
|---|---|---|
| `permissive` | low-stakes recall-first systems | allows more borderline evidence |
| `balanced` | default implementation profile | blocks risky/stale evidence while preserving useful context |
| `conservative` | safety-sensitive systems | stronger risk/harm/staleness penalties and hard blocks |

Use `balanced` by default. Use `conservative` when bad memory influence is costly. Use `permissive` only when humans review outputs or risk is low.

## Benchmarks and demos

### Run all tests

```bash
pytest
```

### Multi-domain stress benchmark

```bash
python benchmarks/multi_domain_stress_benchmark.py \
  --cases-per-scenario 100 \
  --top-k 4 \
  --seed 20260611 \
  --profile balanced
```

### Profile sensitivity benchmark

```bash
python benchmarks/profile_sensitivity_benchmark.py
```

### External-style benchmark

```bash
python benchmarks/external_benchmark_suite.py
```

### RAG adapter demo

```bash
python examples/rag_adapter_demo.py
```

### Implementation API demo

```bash
python examples/implementation_api_demo.py
```

## Package layout

```text
src/persistence_memory/
  api.py          # implementation-facing PersistenceGate API
  models.py       # MemoryItem, TaskContext, states, decisions
  profiles.py     # permissive / balanced / conservative profiles
  scorer.py       # transparent scoring logic
  controller.py   # retrieve-score-gate-feedback controller
  store.py        # in-memory store
  retriever.py    # simple lexical baseline retriever
  benchmark.py    # ordinary top-k vs gated evaluation helpers
  labeling.py     # benchmark labels and heuristics
  audit.py        # blind-review and circularity-audit helpers

examples/
  implementation_api_demo.py
  rag_adapter_demo.py
  github_real_data_demo.py
  github_corpus_demo.py

benchmarks/
  multi_domain_stress_benchmark.py
  profile_sensitivity_benchmark.py
  external_benchmark_suite.py
  github_real_issue_benchmark.py
  quality_gate_report.py
  github_real_issue_audit_export.py
  score_human_audit.py

tests/
  test_implementation_api.py
  test_rag_adapter_demo.py
  test_multi_domain_stress_benchmark.py
  test_external_benchmark_suite.py
  test_quality_gate_report.py
  ...
```

## Design principle

Data should not merely be retrieved because it is relevant. It should be allowed to influence the system only when its expected current value exceeds its risk, burden, staleness, and abstention baseline.

## Supported claim

Persistence Gate has evidence that it can reduce unsafe influence from stale, risky, or contradicted retrieved information while preserving useful current context in controlled benchmarks and deterministic demos.

## Not yet proven

Persistence Gate is not yet proven production-ready. It has not yet been validated across large real-world deployments, human-labeled audit sets, medical/legal/high-stakes domains, or multiple live vector databases. The next validation steps are snapshot real-data benchmarks, adversarial tests, blind human review, and real RAG integrations.
