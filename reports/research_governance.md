# Phase 4：Historical OOS 污染治理与未来证据冻结

## 决策

2021–2026（当前数据截止 2026-09-30）已经被反复查看，永久归类为 consumed historical OOS。Phase 4 没有重新选模；当前候选包仅用于未来 lockbox 与真实时间 paper tracking。

## 证据分层

| period_name          | evidence_class       | start_date   | end_date   | already_observed_at_freeze   | consumed_at_freeze   | selection_allowed   | disclosure                                                  |
|:---------------------|:---------------------|:-------------|:-----------|:-----------------------------|:---------------------|:--------------------|:------------------------------------------------------------|
| training             | training             | 2006-01-01   | 2015-12-31 | True                         | False                | True                |                                                             |
| validation_selection | validation_selection | 2016-01-01   | 2020-12-31 | True                         | False                | True                |                                                             |
| historical_oos       | historical_oos       | 2021-01-01   | 2026-09-30 | True                         | True                 | False               | inspected_repeatedly_during_research_not_a_pristine_lockbox |
| future_lockbox       | future_lockbox       | 2026-10-01   | 2027-09-30 | False                        | False                | False               |                                                             |
| paper_trading        | paper_trading        | nan          | nan        | False                        | False                | False               |                                                             |

## 当前候选冻结

| experiment_id          | bundle_id           | freeze_commit                            | working_tree_dirty_at_freeze   | source_tree_sha256                                               | dependency_snapshot_sha256                                       | config_bundle_sha256                                             | primary_model        | primary_alpha_pipeline   | primary_portfolio_construction   | selection_source          | target             | training_window     | primary_cost_scenario   |   primary_round_trip_cost_bps | execution_convention                                         | historical_oos_used_for_selection   | new_model_selected_in_phase4   | freeze_status              |
|:-----------------------|:--------------------|:-----------------------------------------|:-------------------------------|:-----------------------------------------------------------------|:-----------------------------------------------------------------|:-----------------------------------------------------------------|:---------------------|:-------------------------|:---------------------------------|:--------------------------|:-------------------|:--------------------|:------------------------|------------------------------:|:-------------------------------------------------------------|:------------------------------------|:-------------------------------|:---------------------------|
| research_governance_v1 | candidate_bundle_v1 | 978aedf9615482e79cbca9670c458e8663b5ebcc | False                          | ab3ed285efb03ada31073211dbc6d1431d1de5410f4a7483d190e462af8024db | 5421cbd1021acebd533e40e88aec3a47aae72688d515dd6b27f970dfaa0cf3b1 | 2ca3c8127a8f15790ef431294060ae1f1b84950369380e78ab02c3ab847ce8e1 | hgb_shallow15_locked | raw                      | risk_constrained                 | validation_2016_2020_only | fwd_return_5d_lag1 | expanding_from_2006 | C0_fixed_020bp          |                            20 | returns_accrue_before_close_trade_blocked_orders_retry_daily | False                               | False                          | FROZEN_FOR_FUTURE_EVIDENCE |

## Future lockbox

| lockbox_id                      | state        | start_date   | planned_end_date   |   minimum_daily_observations |   minimum_weekly_signal_observations |   minimum_calendar_months |   minimum_coverage_fraction | interim_performance_access_allowed   | operational_health_access_allowed   | consumed   | first_result_access_utc   | open_rule                                          | cryptographic_access_control_implemented   |
|:--------------------------------|:-------------|:-------------|:-------------------|-----------------------------:|-------------------------------------:|--------------------------:|----------------------------:|:-------------------------------------|:------------------------------------|:-----------|:--------------------------|:---------------------------------------------------|:-------------------------------------------|
| future_lockbox_2026q4_2027q3_v1 | SEALED_EMPTY | 2026-10-01   | 2027-09-30         |                          252 |                                   52 |                        12 |                        0.95 | False                                | True                                | False      |                           | planned_end_date_and_all_minimum_observation_gates | False                                      |

本地 hash-chain 能发现访问日志篡改，但不是密码学保密或独立托管；机构部署应把结果密钥和访问权限交给独立研究治理角色。

## Paper trading

| paper_id                  | state              | activation_rule                                              | retroactive_backfill_allowed   |   recorded_observations | performance_reporting_frequency   | operational_health_reporting_frequency   | outcome_driven_change_requires_new_version   | is_live_capital   |
|:--------------------------|:-------------------|:-------------------------------------------------------------|:-------------------------------|------------------------:|:----------------------------------|:-----------------------------------------|:---------------------------------------------|:------------------|
| paper_candidate_bundle_v1 | FROZEN_NOT_STARTED | first_eligible_signal_strictly_after_candidate_freeze_commit | False                          |                       0 | monthly                           | daily                                    | True                                         | False             |

Paper trading 不能回填。任何基于其结果的修改都必须产生新候选版本，旧版本继续保留。

## 冻结假设族

|   hypothesis_order | hypothesis                                                         | is_primary   | metric                                       | paired_clock       | bootstrap                |   block_length_trading_days |   bootstrap_samples | multiplicity_control   |   false_discovery_rate | frozen_before_future_data   |
|-------------------:|:-------------------------------------------------------------------|:-------------|:---------------------------------------------|:-------------------|:-------------------------|----------------------------:|--------------------:|:-----------------------|-----------------------:|:----------------------------|
|                  1 | hgb_raw_incremental_net_return_vs_style_matched_benchmark_positive | True         | incremental_net_return_annualized_arithmetic | common_daily_dates | circular_block_bootstrap |                          20 |                2000 | benjamini_hochberg     |                   0.05 | True                        |
|                  2 | hgb_industry_size_neutralized_incremental_net_return_positive      | False        | incremental_net_return_annualized_arithmetic | common_daily_dates | circular_block_bootstrap |                          20 |                2000 | benjamini_hochberg     |                   0.05 | True                        |
|                  3 | ridge_raw_incremental_net_return_positive                          | False        | incremental_net_return_annualized_arithmetic | common_daily_dates | circular_block_bootstrap |                          20 |                2000 | benjamini_hochberg     |                   0.05 | True                        |
|                  4 | ridge_industry_size_neutralized_incremental_net_return_positive    | False        | incremental_net_return_annualized_arithmetic | common_daily_dates | circular_block_bootstrap |                          20 |                2000 | benjamini_hochberg     |                   0.05 | True                        |

## 验收

| check                                     | status   | passed   |
|:------------------------------------------|:---------|:---------|
| historical_oos_explicitly_consumed        | PASS     | True     |
| historical_oos_reselection_prohibited     | PASS     | True     |
| candidate_commit_recorded                 | PASS     | True     |
| candidate_source_tree_clean_at_freeze     | PASS     | True     |
| dependency_hashes_complete                | PASS     | True     |
| future_lockbox_unconsumed                 | PASS     | True     |
| future_lockbox_interim_performance_sealed | PASS     | True     |
| paper_trading_not_backfilled              | PASS     | True     |
| access_log_hash_chain_valid               | PASS     | True     |
| phase4_did_not_select_new_model           | PASS     | True     |