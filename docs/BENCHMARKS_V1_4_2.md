# Persistence Gate v1.4.2 — Benchmark Summary

## 1. Structured calibration benchmark

A 120-case generic fixture exercised authority thresholds, lifecycle state, supersession, scope, identity, missing evidence, corroboration, and retrieved-instruction cases.

At thresholds 70–90 within this fixture:

- decision accuracy: 100%;
- evidence-admission precision: 100%;
- admissible-evidence recall: 100%;
- unsafe release cases: 0;
- local p95 gate latency: approximately 5–11 ms.

Latency excludes retrieval, document processing, network calls, model inference, and statement verification. Below 70, lower-authority material began leaking into consequential results. At 95, useful evidence was overblocked.

## 2. Synthetic messy-evidence normalization benchmark

A 240-case benchmark used 150 calibration cases and a 90-case frozen wording holdout across drafts, revoked versions, conflicting states, misleading metadata, scope mismatches, ambiguous dates, negation, paraphrases, and selected prompt-injection patterns.

At a selected field-confidence threshold of 0.85 within the covered synthetic scenarios:

- final disposition accuracy: 100%;
- unsafe acceptance rate: 0%;
- required valid evidence retained: 100%;
- covered-field accuracy: 100%.

The naive metadata-trusting baseline achieved 30% disposition accuracy and a 95.5% unsafe-acceptance rate in the same fixture.

This benchmark is synthetic. It validates contracts and rule behavior, not universal language understanding.

## 3. Real-document source-binding validation

Two clean project PDFs totaling 34 pages produced 868 extracted evidence candidates.

The source-binding adapter achieved:

- valid source coordinates: 100%;
- exact excerpt round-trip: 100%;
- normalized-text round-trip: 100%;
- source-hash reproducibility: 100%;
- repeated-run determinism: 100%;
- required anchor recovery: 100%;
- no authority inferred merely from extraction: 100%.

This validates extraction fidelity and source navigation. It does not validate every semantic interpretation.

## 4. Required next benchmark

The next decisive evaluation must use heterogeneous real records—scans, tables, emails, amendments, conflicting editions, and incomplete files—with at least two independent reviewers, adjudicated ground truth, calibration curves, and end-to-end comparison against ordinary RAG or Copilot.
