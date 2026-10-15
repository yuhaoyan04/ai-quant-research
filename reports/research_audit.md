# Research artifact audit

Generated at `2026-10-15T08:29:17.446019+00:00`.

Summary: **173 PASS / 6 WARN / 0 FAIL**

| Status | Check | Detail |
|---|---|---|
| PASS | `chronological_split_order` | train={'start_year': 2006, 'end_year': 2015}, validation={'start_year': 2016, 'end_year': 2020}, historical_oos={'start_year': 2021, 'end_year': 2026} |
| WARN | `historical_oos_is_not_pristine` | The 2021-2026 block has already been inspected during project development. It is historical out-of-sample evidence, not a never-seen production lockbox. |
| PASS | `frozen_snapshot_integrity:baseline_ridge_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:baseline_ridge_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:challenger_hgb_blend_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:challenger_hgb_blend_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:factor_preprocessing_ablation_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:factor_preprocessing_ablation_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:factor_preprocessing_ablation_v2` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:factor_preprocessing_ablation_v2` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:factor_preprocessing_oos_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:factor_preprocessing_oos_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:ic_decay_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:ic_decay_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:leave_one_factor_out_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:leave_one_factor_out_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:label_horizon_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:label_horizon_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:size_neutralization_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:size_neutralization_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:model_benchmark_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:model_benchmark_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:rf_hyperparameter_stability_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:rf_hyperparameter_stability_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:rf_execution_forensics_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:rf_execution_forensics_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:rf_daily_retry_execution_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:rf_daily_retry_execution_v1` | all registered inputs match their SHA-256 hashes |
| WARN | `invalidated_experiment:factor_preprocessing_ablation_v1` | severity=portfolio_metrics_invalid; superseded_by=factor_preprocessing_ablation_v2; reason=Validation portfolios were formed after filtering rows with non-null future returns. RankIC and model-selection metrics used the correct realized subset, but Top-200 portfolio metrics had outcome-availability leakage. |
| WARN | `invalidated_experiment:model_benchmark_v1` | severity=portfolio_metrics_execution_idealized; superseded_by=rf_daily_retry_execution_v1; reason=Model ranking and RankIC are valid, but the benchmark Top-200 portfolio assumes every next-day rebalance is filled. RandomForest disproportionately selected stocks blocked by next-day price limits or suspensions, materially inflating validation returns and modestly inflating historical-OOS returns. |
| WARN | `invalidated_experiment:rf_execution_forensics_v1` | severity=execution_timing_incomplete; superseded_by=rf_daily_retry_execution_v1; reason=The weekly constrained simulator applied a blocked next-day order only at the following weekly rebalance. It also evaluated every signal with a full five-day label even when holiday-week signal dates were fewer than five trading days apart, creating overlapping portfolio return intervals. The price-limit diagnosis remains valid, but its 19.09% validation and 24.19% historical-OOS returns are not the final executable estimates. |
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
| PASS | `artifact:factor_preprocessing_oos_predictions/v1` | rows=1,390,009 |
| PASS | `factor_preprocessing_oos_prediction_keys_unique` | duplicate_rows=0 |
| PASS | `factor_preprocessing_oos_prediction_years` | observed=2021-2026, expected=2021-2026 |
| PASS | `factor_preprocessing_oos_availability_after_signal` | available_date must be strictly after asof_date |
| PASS | `factor_preprocessing_oos_scores_finite` | rows=1,390,009, dates=293 |
| PASS | `artifact:model_selection/ridge_v1` | rows=1 |
| PASS | `selection_provenance:model_selection` | selection_data_end=2020-12-31; expected=2020-12-31 |
| PASS | `artifact:portfolio_selection/long_only_v1` | rows=1 |
| PASS | `selection_provenance:portfolio_selection` | selection_period=validation_2016_2020_only; expected=validation_2016_2020_only |
| PASS | `artifact:turnover_buffer_selection/v1` | rows=1 |
| PASS | `selection_provenance:turnover_buffer_selection` | selection_period=validation_2016_2020_only; expected=validation_2016_2020_only |
| PASS | `artifact:model_challenger_selection/hgb_v1` | rows=1 |
| PASS | `selection_provenance:model_challenger_selection` | selection_period=validation_2016_2020_only; expected=validation_2016_2020_only |
| PASS | `artifact:label_horizon_selection/v1` | rows=1 |
| PASS | `selection_provenance:label_horizon_selection` | selection_period=validation_2016_2020_only; expected=validation_2016_2020_only |
| PASS | `artifact:model_benchmark_selection/v1` | rows=1 |
| PASS | `selection_provenance:model_benchmark_selection` | selection_period=validation_2016_2020_only; expected=validation_2016_2020_only |
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
| PASS | `artifact:lofo_summary/v1` | rows=32 |
| PASS | `artifact:factor_preprocessing_oos_summary/v1` | rows=2 |
| PASS | `lofo_full_reproduces_preprocessing:mean_rank_ic` | lofo=0.0977156414, preprocessing_oos=0.0977156414 |
| PASS | `lofo_full_reproduces_preprocessing:annualized_icir` | lofo=4.5327174107, preprocessing_oos=4.5327174107 |
| PASS | `lofo_full_reproduces_preprocessing:portfolio_annualized_return` | lofo=0.2121601595, preprocessing_oos=0.2121601595 |
| PASS | `artifact:label_horizon_oos_predictions/v1` | rows=1,400,025 |
| PASS | `label_horizon_oos_prediction_keys_unique` | duplicate_rows=0 |
| PASS | `label_horizon_oos_prediction_years` | observed=2021-2026, expected=2021-2026 |
| PASS | `label_horizon_oos_availability_after_signal` | available_date must be strictly after asof_date |
| PASS | `label_horizon_oos_scores_finite` | rows=1,400,025, dates=295 |
| PASS | `artifact:label_horizon_oos_portfolio/v1` | rows=1,770 |
| PASS | `label_horizon_selected_net_return_identity` | max_abs_error=0.000e+00 |
| PASS | `label_horizon_selected_returns_finite_and_above_minus_one` | dates=289 |
| PASS | `artifact:label_horizon_panel_audit/v1` | rows=21 |
| PASS | `artifact:label_horizon_selection/v1` | rows=1 |
| PASS | `artifact:label_horizon_oos_summary/v1` | rows=6 |
| PASS | `artifact:label_horizon_oos_bootstrap/v1` | rows=5 |
| PASS | `label_panel_preserves_eligible_rows` | years=21, all_years_match=True |
| PASS | `label_panel_keys_unique` | duplicate_rows=0 |
| PASS | `label_coverage_monotone_by_horizon` | coverage(5d) >= coverage(10d) >= coverage(20d) for every year |
| PASS | `label_selection_excludes_historical_oos` | selected=h20_raw_return |
| PASS | `artifact:factor_preprocessing_oos_summary/v1` | rows=2 |
| PASS | `label_h5_rank_reproduces_preprocessing:mean_rank_ic` | label_study=0.0977156414, preprocessing_oos=0.0977156414 |
| PASS | `label_h5_rank_reproduces_preprocessing:portfolio_annualized_return` | label_study=0.2121601595, preprocessing_oos=0.2121601595 |
| PASS | `label_selected_increment_not_overstated` | selected=h20_raw_return; 95% CI=[-5.09%, 5.03%] |
| PASS | `artifact:size_neutralization_oos_portfolio/v1` | rows=586 |
| PASS | `size_neutralization_raw_net_return_identity` | max_abs_error=0.000e+00 |
| PASS | `size_neutralization_raw_returns_finite_and_above_minus_one` | dates=293 |
| PASS | `artifact:size_neutralization_oos_portfolio/v1` | rows=586 |
| PASS | `size_neutralization_neutral_net_return_identity` | max_abs_error=0.000e+00 |
| PASS | `size_neutralization_neutral_returns_finite_and_above_minus_one` | dates=293 |
| PASS | `artifact:pit_size_exposure_index/v1` | rows=21 |
| PASS | `artifact:size_neutralization_oos_summary/v1` | rows=2 |
| PASS | `artifact:size_neutralization_oos_bootstrap/v1` | rows=2 |
| PASS | `pit_size_coverage` | minimum_eligible_weekly_coverage=99.998965% |
| PASS | `pit_size_keys_unique` | duplicate_rows=0 |
| PASS | `size_neutral_score_orthogonality` | abs_mean_score_log_size_correlation=8.420e-18 |
| PASS | `artifact:factor_preprocessing_oos_summary/v1` | rows=2 |
| PASS | `size_raw_reproduces_preprocessing:mean_rank_ic` | size_study=0.0977156414, preprocessing_oos=0.0977156414 |
| PASS | `size_raw_reproduces_preprocessing:portfolio_annualized_return` | size_study=0.2121601595, preprocessing_oos=0.2121601595 |
| PASS | `small_size_return_dependence_detected` | neutral-minus-raw 95% CI=[-8.76%, -0.54%] |
| PASS | `artifact:model_benchmark_oos_portfolio/v1` | rows=1,758 |
| PASS | `model_benchmark_winner_net_return_identity` | max_abs_error=0.000e+00 |
| PASS | `model_benchmark_winner_returns_finite_and_above_minus_one` | dates=293 |
| PASS | `artifact:model_benchmark_selection/v1` | rows=1 |
| PASS | `artifact:model_benchmark_oos_summary/v1` | rows=6 |
| PASS | `artifact:model_benchmark_oos_annual/v1` | rows=36 |
| PASS | `artifact:model_benchmark_winner_pairwise/v1` | rows=4 |
| PASS | `artifact:model_benchmark_availability/v1` | rows=8 |
| PASS | `model_benchmark_excludes_historical_oos_selection` | winner=random_forest_depth6 |
| PASS | `model_winner_rank_ic_positive_each_oos_year` | minimum_annual_rank_ic=0.0910 |
| PASS | `artifact:factor_preprocessing_oos_summary/v1` | rows=2 |
| PASS | `benchmark_ridge_reproduces_preprocessing:mean_rank_ic` | benchmark=0.0977156414, preprocessing_oos=0.0977156414 |
| PASS | `benchmark_ridge_reproduces_preprocessing:portfolio_annualized_return` | benchmark=0.2121601595, preprocessing_oos=0.2121601595 |
| PASS | `model_winner_return_increment_vs_ridge` | 95% CI=[1.30%, 10.98%] |
| PASS | `model_winner_rank_ic_increment_not_overstated` | 95% CI=[-0.0027, 0.0078] |
| PASS | `random_forest_not_claimed_superior_to_hgb` | 95% CI=[-1.54%, 5.75%] |
| WARN | `model_dependencies_unavailable` | not executed=['lightgbm', 'xgboost'] |
| PASS | `artifact:rf_hyperparameter_validation_summary/v1` | rows=11 |
| PASS | `artifact:rf_hyperparameter_protocol/v1` | rows=1 |
| PASS | `artifact:model_benchmark_validation_grid/v1` | rows=16 |
| PASS | `rf_stability_excludes_historical_oos` | evaluation_period=validation_2016_2020_only |
| PASS | `rf_stability_protocol_arithmetic` | neighbors=10, pass_fraction=70.0% |
| PASS | `rf_stability_reproduces_frozen_validation_baseline` | RankIC=0.099082, annual_return=69.98% |
| PASS | `rf_stability_does_not_promote_posthoc_winner` | local_robustness_pass=False |
| PASS | `artifact:rf_forensic_summary/v1` | rows=2 |
| PASS | `artifact:rf_forensic_holdings_summary/v1` | rows=1 |
| PASS | `artifact:rf_forensic_label_audit/v1` | rows=250 |
| PASS | `artifact:rf_oos_execution_summary/v1` | rows=2 |
| PASS | `artifact:rf_oos_execution_diagnostics/v1` | rows=1 |
| PASS | `rf_forensic_sampled_labels_rebuild_exactly` | rows=250, max_abs_error=0.000e+00 |
| PASS | `rf_validation_execution_bias_detected` | ideal=69.98%, constrained=19.09% |
| PASS | `rf_oos_execution_comparison_uses_common_periods` | ideal=293, constrained=293 |
| PASS | `rf_oos_execution_bias_detected` | ideal=27.95%, constrained=24.19% |
| PASS | `artifact:rf_daily_retry_daily/validation_v1` | rows=1,210 |
| PASS | `artifact:rf_daily_ideal_daily/validation_v1` | rows=1,210 |
| PASS | `artifact:rf_daily_retry_summary/validation_v1` | rows=1 |
| PASS | `artifact:rf_daily_ideal_summary/validation_v1` | rows=1 |
| PASS | `artifact:rf_daily_timing_audit/validation_v1` | rows=1 |
| PASS | `rf_daily_validation_accounting_identity` | max_abs_error=9.770e-15 |
| PASS | `rf_daily_validation_return_identity` | max_abs_error=7.184e-15 |
| PASS | `rf_daily_validation_no_hidden_daily_rebalance` | ideal_non_rebalance_max_turnover=0.000e+00 |
| PASS | `rf_daily_validation_execution_haircut_detected` | ideal=57.46%, constrained=10.77% |
| PASS | `rf_daily_validation_overlap_removed` | duplicated_horizon_days=61 |
| PASS | `artifact:rf_daily_retry_daily/historical_oos_v1` | rows=1,387 |
| PASS | `artifact:rf_daily_ideal_daily/historical_oos_v1` | rows=1,387 |
| PASS | `artifact:rf_daily_retry_summary/historical_oos_v1` | rows=1 |
| PASS | `artifact:rf_daily_ideal_summary/historical_oos_v1` | rows=1 |
| PASS | `artifact:rf_daily_timing_audit/historical_oos_v1` | rows=1 |
| PASS | `rf_daily_historical_oos_accounting_identity` | max_abs_error=1.010e-14 |
| PASS | `rf_daily_historical_oos_return_identity` | max_abs_error=6.610e-15 |
| PASS | `rf_daily_historical_oos_no_hidden_daily_rebalance` | ideal_non_rebalance_max_turnover=0.000e+00 |
| PASS | `rf_daily_historical_oos_execution_haircut_detected` | ideal=25.38%, constrained=21.62% |
| PASS | `rf_daily_historical_oos_overlap_removed` | duplicated_horizon_days=79 |
| WARN | `known_scope_limitations` | No point-in-time industry neutralization, no official historical CSI membership, and no broker-calibrated market-impact model. |

## Headline findings

- `registered_frozen_experiments`: 13
- `invalidated_experiments`: 3
- `common_oos_weeks`: 291
- `ridge_annualized_return_20bp`: 0.19908873296969398
- `challenger_annualized_return_20bp`: 0.22898196710494045
- `ridge_excess_vs_csi500_ci_low`: 0.05170363109106627
- `ridge_excess_vs_csi500_ci_high`: 0.34129704589057785
- `challenger_minus_ridge_ci_low`: -0.008666286294444084
- `challenger_minus_ridge_ci_high`: 0.07447269715754343
- `ridge_rank_ic_positive_every_oos_year`: True
- `label_validation_selected_target`: h20_raw_return
- `label_selected_oos_increment_ci_low`: -0.05092803225074174
- `label_selected_oos_increment_ci_high`: 0.050271773638776183
- `size_neutralized_oos_annualized_return`: 0.16300301209294266
- `size_neutralized_oos_mean_rank_ic`: 0.0943606700512136
- `size_neutralized_minus_raw_return_ci_low`: -0.08762867391696517
- `size_neutralized_minus_raw_return_ci_high`: -0.005449060689559533
- `model_benchmark_validation_winner`: random_forest_depth6
- `model_winner_oos_mean_rank_ic`: 0.10015807551762398
- `model_winner_oos_annualized_return`: 0.2794914760062783
- `model_winner_minus_ridge_return_ci_low`: 0.013037015731000245
- `model_winner_minus_ridge_return_ci_high`: 0.10982521927696937
- `rf_neighbor_pass_fraction`: 0.7
- `rf_local_robustness_pass`: False
- `rf_stability_minimum_neighbor_rank_ic`: 0.09254318713639618
- `rf_stability_minimum_neighbor_sharpe`: 0.784310515723939
- `rf_validation_idealized_annualized_return`: 0.6997555143830458
- `rf_validation_execution_constrained_annualized_return`: 0.19087190628281858
- `rf_oos_idealized_annualized_return`: 0.2794914760062783
- `rf_oos_execution_constrained_annualized_return`: 0.24194049194086764
- `rf_oos_execution_constrained_sharpe`: 0.8748900492302454
- `rf_validation_mean_size_percentile`: 0.28788580033544275
- `rf_validation_mean_blocked_buy_weight`: 0.02590663168490841
- `rf_daily_validation_ideal_annualized_return`: 0.574591624186662
- `rf_daily_validation_constrained_annualized_return`: 0.10767466373732937
- `rf_daily_oos_ideal_annualized_return`: 0.2538294055667403
- `rf_daily_oos_constrained_annualized_return`: 0.2162386898373556
- `rf_daily_oos_constrained_sharpe`: 0.8738047854052745
- `rf_daily_oos_constrained_maximum_drawdown`: -0.4327391378303085
