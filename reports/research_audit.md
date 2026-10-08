# Research artifact audit

Generated at `2026-10-08T07:56:51.454945+00:00`.

Summary: **45 PASS / 2 WARN / 0 FAIL**

| Status | Check | Detail |
|---|---|---|
| PASS | `chronological_split_order` | train={'start_year': 2006, 'end_year': 2015}, validation={'start_year': 2016, 'end_year': 2020}, historical_oos={'start_year': 2021, 'end_year': 2026} |
| WARN | `historical_oos_is_not_pristine` | The 2021-2026 block has already been inspected during project development. It is historical out-of-sample evidence, not a never-seen production lockbox. |
| PASS | `artifact:research_coverage/a_share` | rows=4 |
| PASS | `historical_universe_download_coverage` | coverage=100.00%, missing_files=0 |
| PASS | `artifact:model_validation_predictions/ridge_v1` | rows=813,175 |
| PASS | `ridge_validation_prediction_keys_unique` | duplicate_rows=0 |
| PASS | `ridge_validation_prediction_years` | observed=2016-2020, expected=2016-2020 |
| PASS | `ridge_validation_availability_after_signal` | available_date must be strictly after asof_date |
| PASS | `ridge_validation_scores_finite` | rows=813,175, dates=254 |
| PASS | `artifact:model_oos_predictions/ridge_v1` | rows=1,400,025 |
| PASS | `ridge_oos_prediction_keys_unique` | duplicate_rows=0 |
| PASS | `ridge_oos_prediction_years` | observed=2021-2026, expected=2021-2026 |
| PASS | `ridge_oos_availability_after_signal` | available_date must be strictly after asof_date |
| PASS | `ridge_oos_scores_finite` | rows=1,400,025, dates=295 |
| PASS | `artifact:model_challenger_oos_predictions/hgb_v1` | rows=1,400,025 |
| PASS | `challenger_oos_prediction_keys_unique` | duplicate_rows=0 |
| PASS | `challenger_oos_prediction_years` | observed=2021-2026, expected=2021-2026 |
| PASS | `challenger_oos_availability_after_signal` | available_date must be strictly after asof_date |
| PASS | `challenger_oos_scores_finite` | rows=1,400,025, dates=295 |
| PASS | `artifact:model_selection/ridge_v1` | rows=1 |
| PASS | `selection_provenance:model_selection` | selection_data_end=2020-12-31; expected=2020-12-31 |
| PASS | `artifact:portfolio_selection/long_only_v1` | rows=1 |
| PASS | `selection_provenance:portfolio_selection` | selection_period=validation_2016_2020_only; expected=validation_2016_2020_only |
| PASS | `artifact:turnover_buffer_selection/v1` | rows=1 |
| PASS | `selection_provenance:turnover_buffer_selection` | selection_period=validation_2016_2020_only; expected=validation_2016_2020_only |
| PASS | `artifact:model_challenger_selection/hgb_v1` | rows=1 |
| PASS | `selection_provenance:model_challenger_selection` | selection_period=validation_2016_2020_only; expected=validation_2016_2020_only |
| PASS | `artifact:turnover_buffer_oos_weekly/v1` | rows=1,180 |
| PASS | `ridge_buffered_net_return_identity` | max_abs_error=0.000e+00 |
| PASS | `ridge_buffered_returns_finite_and_above_minus_one` | dates=291 |
| PASS | `artifact:challenger_portfolio_oos_weekly/v1` | rows=295 |
| PASS | `challenger_buffered_net_return_identity` | max_abs_error=0.000e+00 |
| PASS | `challenger_buffered_returns_finite_and_above_minus_one` | dates=292 |
| PASS | `artifact:turnover_buffer_oos_holdings/v1` | rows=59,000 |
| PASS | `ridge_weights_sum_to_one` | max_abs_error=0.000e+00 |
| PASS | `ridge_holding_keys_unique` | duplicate_rows=0 |
| PASS | `artifact:challenger_portfolio_oos_holdings/v1` | rows=59,000 |
| PASS | `challenger_weights_sum_to_one` | max_abs_error=0.000e+00 |
| PASS | `challenger_holding_keys_unique` | duplicate_rows=0 |
| PASS | `paired_model_dates_available` | common_dates=291, first=2021-01-08, last=2026-09-18 |
| PASS | `artifact:robustness_cost_grid/v1` | rows=8 |
| PASS | `artifact:robustness_block_bootstrap/v1` | rows=6 |
| PASS | `artifact:model_oos_annual/ridge_v1` | rows=6 |
| PASS | `ridge_rank_ic_positive_each_oos_year` | minimum_annual_rank_ic=0.0898 |
| PASS | `ridge_excess_bootstrap_interval_above_zero` | 95% CI=[5.17%, 34.13%] |
| PASS | `challenger_increment_not_overstated` | 95% CI=[-0.87%, 7.45%]; report as an economic signal, not proven superiority |
| WARN | `known_scope_limitations` | No point-in-time industry neutralization, no official historical CSI membership, and no broker-calibrated market-impact model. |

## Headline findings

- `common_oos_weeks`: 291
- `ridge_annualized_return_20bp`: 0.19908873296969398
- `challenger_annualized_return_20bp`: 0.22898196710494045
- `ridge_excess_vs_csi500_ci_low`: 0.05170363109106627
- `ridge_excess_vs_csi500_ci_high`: 0.34129704589057785
- `challenger_minus_ridge_ci_low`: -0.008666286294444084
- `challenger_minus_ridge_ci_high`: 0.07447269715754343
- `ridge_rank_ic_positive_every_oos_year`: True
