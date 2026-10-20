# Forward Data Supply Protocol

## Scope

This is an operational/governance control, not return evidence. Mutable forward data
lives only under `forward_*`; frozen historical research inputs remain at their original hashes.

## Protocol

| experiment_id            | pipeline_id     | evidence_class   | claims_independent_evidence   | historical_cutoff   | source   | forward_table_prefix   | natural_key_merge   | historical_namespace_mutation_allowed   | strategy_logic_changed   | performance_results_included   |
|:-------------------------|:----------------|:-----------------|:------------------------------|:--------------------|:---------|:-----------------------|:--------------------|:----------------------------------------|:-------------------------|:-------------------------------|
| forward_data_pipeline_v1 | forward_data_v1 | governance_only  | False                         | 2026-09-30          | baostock | forward_               | True                | False                                   | False                    | False                          |

## Implementation checks

| check                            | status   | detail                                                                                                                                                                                                                      |
|:---------------------------------|:---------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| governance_only                  | PASS     | governance_only                                                                                                                                                                                                             |
| separate_forward_namespace       | PASS     | forward_calendar,forward_daily_bars_baostock,forward_daily_bars_validated,forward_factors_raw,forward_industry_snapshot_baostock,forward_labels_raw,forward_quality_report,forward_security_snapshot,forward_security_union |
| historical_cutoff_explicit       | PASS     | 2026-09-30                                                                                                                                                                                                                  |
| vendor_errors_fail_closed        | PASS     | True                                                                                                                                                                                                                        |
| weekly_partial_period_blocked    | PASS     | calendar_lookahead_days=10                                                                                                                                                                                                  |
| factor_warmup_sufficient         | PASS     | factor_lookback_rows=150                                                                                                                                                                                                    |
| strategy_configuration_unchanged | PASS     | paper_trading.yaml and frozen model bundle are not rewritten                                                                                                                                                                |

