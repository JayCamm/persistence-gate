# Dirty Metadata Bias Audit

This audit checks whether the dirty-metadata result depends on hidden answer labels or unfair tuning.

It runs the dirty metadata benchmark under several input-signal variants:

1. `full_signals_role_hidden`
   - The gate sees risk, harm, and usefulness signals.
   - The answer-key role field is removed from the gate input metadata.

2. `warning_risk_stress`
   - Useful warning documents receive moderate risk.
   - This tests whether simple risk-threshold filters overblock useful warning evidence.

3. `neutral_signals`
   - Risk, harm, and usefulness values are neutralized across the workspace.
   - The gate should not win strongly if the influence signals are removed.

4. `shuffled_signals`
   - Risk, harm, and usefulness values are randomly shuffled across documents.
   - If gate performance degrades, the result depends on meaningful governance signals rather than hidden answer labels.

## Run

```bash
python benchmarks/dirty_metadata_bias_audit.py \
  --profile conservative \
  --top-k 3 \
  --candidate-pool 8 \
  --shuffle-replicates 10 \
  --refresh \
  --out-dir benchmark_results/dirty_metadata_bias_audit
```

## Outputs

```text
benchmark_results/dirty_metadata_bias_audit/source_manifest.csv
benchmark_results/dirty_metadata_bias_audit/case_details.csv
benchmark_results/dirty_metadata_bias_audit/gate_audit_log.csv
benchmark_results/dirty_metadata_bias_audit/summary.csv
benchmark_results/dirty_metadata_bias_audit/bias_audit_interpretation.md
```

## Interpretation

A stronger result looks like this:

- Full signals with role hidden: Persistence Gate performs well.
- Warning risk stress: Persistence Gate retains useful warning evidence better than simple risk filters.
- Neutral signals: Persistence Gate performance drops.
- Shuffled signals: Persistence Gate performance drops or varies.

That pattern supports the claim that Persistence Gate is not reading hidden answer labels. It is using governance signals.

The audit still does not prove automatic risk/usefulness extraction from raw text. It tests whether the gate can fairly use such signals once available.
