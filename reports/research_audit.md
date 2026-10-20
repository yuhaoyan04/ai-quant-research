# Research artifact audit

Generated at `2026-10-20T13:38:43.739141+00:00`.

Summary: **482 PASS / 46 WARN / 0 FAIL**

| Status | Check | Detail |
|---|---|---|
| PASS | `chronological_split_order` | train={'start_year': 2006, 'end_year': 2015}, validation={'start_year': 2016, 'end_year': 2020}, historical_oos={'start_year': 2021, 'end_year': 2026} |
| WARN | `historical_oos_is_not_pristine` | The 2021-2026 block has already been inspected during project development. It is historical out-of-sample evidence, not a never-seen production lockbox. |
| PASS | `frozen_snapshot_integrity:baseline_ridge_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:baseline_ridge_v1` | mismatches=['model_features/2020', 'model_validation_predictions/ridge_v1', 'model_oos_predictions/ridge_v1', 'turnover_buffer_oos_holdings/v1', 'model_selection/ridge_v1', 'model_oos_annual/ridge_v1', 'turnover_buffer_oos_summary/v1', 'robustness_cost_grid/v1', 'robustness_block_bootstrap/v1'] |
| PASS | `frozen_snapshot_integrity:challenger_hgb_blend_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:challenger_hgb_blend_v1` | mismatches=['model_features/2020', 'model_challenger_validation_predictions/hgb_v1', 'model_challenger_oos_predictions/hgb_v1', 'challenger_portfolio_oos_holdings/v1', 'model_challenger_selection/hgb_v1', 'model_challenger_oos_annual/hgb_v1', 'challenger_portfolio_validation_summary/v1', 'challenger_portfolio_oos_summary/v1', 'robustness_cost_grid/v1', 'robustness_block_bootstrap/v1'] |
| PASS | `frozen_snapshot_integrity:factor_preprocessing_ablation_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:factor_preprocessing_ablation_v1` | mismatches=['eligible_factor_panel_staging/2006', 'eligible_factor_panel_staging/2007', 'eligible_factor_panel_staging/2008', 'eligible_factor_panel_staging/2009', 'eligible_factor_panel_staging/2010', 'eligible_factor_panel_staging/2011', 'eligible_factor_panel_staging/2012', 'eligible_factor_panel_staging/2013', 'eligible_factor_panel_staging/2014', 'eligible_factor_panel_staging/2015', 'eligible_factor_panel_staging/2016', 'eligible_factor_panel_staging/2017', 'eligible_factor_panel_staging/2018', 'eligible_factor_panel_staging/2019', 'eligible_factor_panel_staging/2020', 'factor_preprocessing_ablation_selection/v1', 'factor_preprocessing_ablation_grid/v1', 'factor_preprocessing_ablation_annual/v1', 'factor_preprocessing_ablation_factor_summary/v1'] |
| PASS | `frozen_snapshot_integrity:factor_preprocessing_ablation_v2` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:factor_preprocessing_ablation_v2` | mismatches=['eligible_factor_panel_staging/2006', 'eligible_factor_panel_staging/2007', 'eligible_factor_panel_staging/2008', 'eligible_factor_panel_staging/2009', 'eligible_factor_panel_staging/2010', 'eligible_factor_panel_staging/2011', 'eligible_factor_panel_staging/2012', 'eligible_factor_panel_staging/2013', 'eligible_factor_panel_staging/2014', 'eligible_factor_panel_staging/2015', 'eligible_factor_panel_staging/2016', 'eligible_factor_panel_staging/2017', 'eligible_factor_panel_staging/2018', 'eligible_factor_panel_staging/2019', 'eligible_factor_panel_staging/2020', 'factor_preprocessing_ablation_selection/v2', 'factor_preprocessing_ablation_grid/v2', 'factor_preprocessing_ablation_annual/v2', 'factor_preprocessing_ablation_factor_summary/v2'] |
| PASS | `frozen_snapshot_integrity:factor_preprocessing_oos_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:factor_preprocessing_oos_v1` | mismatches=['eligible_factor_panel_staging/2006', 'eligible_factor_panel_staging/2007', 'eligible_factor_panel_staging/2008', 'eligible_factor_panel_staging/2009', 'eligible_factor_panel_staging/2010', 'eligible_factor_panel_staging/2011', 'eligible_factor_panel_staging/2012', 'eligible_factor_panel_staging/2013', 'eligible_factor_panel_staging/2014', 'eligible_factor_panel_staging/2015', 'eligible_factor_panel_staging/2016', 'eligible_factor_panel_staging/2017', 'eligible_factor_panel_staging/2018', 'eligible_factor_panel_staging/2019', 'eligible_factor_panel_staging/2020', 'eligible_factor_panel_staging/2021', 'eligible_factor_panel_staging/2022', 'eligible_factor_panel_staging/2023', 'eligible_factor_panel_staging/2024', 'eligible_factor_panel_staging/2025', 'eligible_factor_panel_staging/2026', 'factor_preprocessing_oos_summary/v1', 'factor_preprocessing_oos_annual/v1', 'factor_preprocessing_oos_bootstrap/v1'] |
| PASS | `frozen_snapshot_integrity:ic_decay_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:ic_decay_v1` | mismatches=['factor_preprocessing_oos_predictions/v1', 'ic_decay_panel/v1', 'ic_decay_summary/v1', 'ic_decay_annual/v1'] |
| PASS | `frozen_snapshot_integrity:leave_one_factor_out_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:leave_one_factor_out_v1` | mismatches=['eligible_factor_panel_staging/2006', 'eligible_factor_panel_staging/2007', 'eligible_factor_panel_staging/2008', 'eligible_factor_panel_staging/2009', 'eligible_factor_panel_staging/2010', 'eligible_factor_panel_staging/2011', 'eligible_factor_panel_staging/2012', 'eligible_factor_panel_staging/2013', 'eligible_factor_panel_staging/2014', 'eligible_factor_panel_staging/2015', 'eligible_factor_panel_staging/2016', 'eligible_factor_panel_staging/2017', 'eligible_factor_panel_staging/2018', 'eligible_factor_panel_staging/2019', 'eligible_factor_panel_staging/2020', 'eligible_factor_panel_staging/2021', 'eligible_factor_panel_staging/2022', 'eligible_factor_panel_staging/2023', 'eligible_factor_panel_staging/2024', 'eligible_factor_panel_staging/2025', 'eligible_factor_panel_staging/2026', 'lofo_summary/v1', 'lofo_paired_bootstrap/v1', 'lofo_comparison/v1'] |
| PASS | `frozen_snapshot_integrity:label_horizon_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:label_horizon_v1` | mismatches=['multi_horizon_factor_panel/2006', 'multi_horizon_factor_panel/2007', 'multi_horizon_factor_panel/2008', 'multi_horizon_factor_panel/2009', 'multi_horizon_factor_panel/2010', 'multi_horizon_factor_panel/2011', 'multi_horizon_factor_panel/2012', 'multi_horizon_factor_panel/2013', 'multi_horizon_factor_panel/2014', 'multi_horizon_factor_panel/2015', 'multi_horizon_factor_panel/2016', 'multi_horizon_factor_panel/2017', 'multi_horizon_factor_panel/2018', 'multi_horizon_factor_panel/2019', 'multi_horizon_factor_panel/2020', 'multi_horizon_factor_panel/2021', 'multi_horizon_factor_panel/2022', 'multi_horizon_factor_panel/2023', 'multi_horizon_factor_panel/2024', 'multi_horizon_factor_panel/2025', 'multi_horizon_factor_panel/2026', 'label_horizon_panel_audit/v1', 'label_horizon_validation_grid/v1', 'label_horizon_validation_candidates/v1', 'label_horizon_selection/v1', 'label_horizon_oos_summary/v1', 'label_horizon_oos_annual/v1', 'label_horizon_oos_bootstrap/v1'] |
| PASS | `frozen_snapshot_integrity:size_neutralization_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:size_neutralization_v1` | mismatches=['pit_size_exposure_index/v1', 'multi_horizon_factor_panel/2006', 'multi_horizon_factor_panel/2007', 'multi_horizon_factor_panel/2008', 'multi_horizon_factor_panel/2009', 'multi_horizon_factor_panel/2010', 'multi_horizon_factor_panel/2011', 'multi_horizon_factor_panel/2012', 'multi_horizon_factor_panel/2013', 'multi_horizon_factor_panel/2014', 'multi_horizon_factor_panel/2015', 'multi_horizon_factor_panel/2016', 'multi_horizon_factor_panel/2017', 'multi_horizon_factor_panel/2018', 'multi_horizon_factor_panel/2019', 'multi_horizon_factor_panel/2020', 'multi_horizon_factor_panel/2021', 'multi_horizon_factor_panel/2022', 'multi_horizon_factor_panel/2023', 'multi_horizon_factor_panel/2024', 'multi_horizon_factor_panel/2025', 'multi_horizon_factor_panel/2026', 'pit_size_exposures/2006', 'pit_size_exposures/2007', 'pit_size_exposures/2008', 'pit_size_exposures/2009', 'pit_size_exposures/2010', 'pit_size_exposures/2011', 'pit_size_exposures/2012', 'pit_size_exposures/2013', 'pit_size_exposures/2014', 'pit_size_exposures/2015', 'pit_size_exposures/2016', 'pit_size_exposures/2017', 'pit_size_exposures/2018', 'pit_size_exposures/2019', 'pit_size_exposures/2020', 'pit_size_exposures/2021', 'pit_size_exposures/2022', 'pit_size_exposures/2023', 'pit_size_exposures/2024', 'pit_size_exposures/2025', 'pit_size_exposures/2026', 'size_neutralization_validation_summary/v1', 'size_neutralization_oos_summary/v1', 'size_neutralization_oos_annual/v1', 'size_neutralization_oos_bootstrap/v1', 'size_neutralization_oos_exposures/v1'] |
| PASS | `frozen_snapshot_integrity:model_benchmark_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:model_benchmark_v1` | mismatches=['multi_horizon_factor_panel/2006', 'multi_horizon_factor_panel/2007', 'multi_horizon_factor_panel/2008', 'multi_horizon_factor_panel/2009', 'multi_horizon_factor_panel/2010', 'multi_horizon_factor_panel/2011', 'multi_horizon_factor_panel/2012', 'multi_horizon_factor_panel/2013', 'multi_horizon_factor_panel/2014', 'multi_horizon_factor_panel/2015', 'multi_horizon_factor_panel/2016', 'multi_horizon_factor_panel/2017', 'multi_horizon_factor_panel/2018', 'multi_horizon_factor_panel/2019', 'multi_horizon_factor_panel/2020', 'multi_horizon_factor_panel/2021', 'multi_horizon_factor_panel/2022', 'multi_horizon_factor_panel/2023', 'multi_horizon_factor_panel/2024', 'multi_horizon_factor_panel/2025', 'multi_horizon_factor_panel/2026', 'model_benchmark_validation_grid/v1', 'model_benchmark_family_selection/v1', 'model_benchmark_selection/v1', 'model_benchmark_oos_summary/v1', 'model_benchmark_oos_annual/v1', 'model_benchmark_oos_bootstrap/v1', 'model_benchmark_winner_pairwise/v1', 'model_benchmark_fit_audit/v1', 'model_benchmark_availability/v1'] |
| PASS | `frozen_snapshot_integrity:rf_hyperparameter_stability_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:rf_hyperparameter_stability_v1` | mismatches=['multi_horizon_factor_panel/2006', 'multi_horizon_factor_panel/2007', 'multi_horizon_factor_panel/2008', 'multi_horizon_factor_panel/2009', 'multi_horizon_factor_panel/2010', 'multi_horizon_factor_panel/2011', 'multi_horizon_factor_panel/2012', 'multi_horizon_factor_panel/2013', 'multi_horizon_factor_panel/2014', 'multi_horizon_factor_panel/2015', 'multi_horizon_factor_panel/2016', 'multi_horizon_factor_panel/2017', 'multi_horizon_factor_panel/2018', 'multi_horizon_factor_panel/2019', 'multi_horizon_factor_panel/2020', 'rf_hyperparameter_validation_summary/v1', 'rf_hyperparameter_axis_summary/v1', 'rf_hyperparameter_protocol/v1'] |
| PASS | `frozen_snapshot_integrity:rf_execution_forensics_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:rf_execution_forensics_v1` | mismatches=['model_benchmark_validation_portfolio/v1', 'model_benchmark_oos_portfolio/v1', 'execution_flag_index/v1', 'pit_size_exposure_index/v1', 'rf_forensic_summary/v1', 'rf_forensic_yearly/v1', 'rf_forensic_concentration/v1', 'rf_forensic_holdings_summary/v1', 'rf_forensic_missingness/v1', 'rf_forensic_label_audit/v1', 'rf_oos_execution_summary/v1', 'rf_oos_execution_yearly/v1', 'rf_oos_execution_diagnostics/v1', 'rf_oos_execution_fit_audit/v1'] |
| PASS | `frozen_snapshot_integrity:rf_daily_retry_execution_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:rf_daily_retry_execution_v1` | mismatches=['rf_target_holdings/validation_v1', 'rf_target_holdings/historical_oos_v1', 'rf_daily_execution_market/validation_v1', 'rf_daily_execution_market/historical_oos_v1', 'rf_daily_retry_daily/validation_v1', 'rf_daily_retry_daily/historical_oos_v1', 'rf_daily_retry_summary/validation_v1', 'rf_daily_retry_summary/historical_oos_v1', 'rf_daily_retry_annual/validation_v1', 'rf_daily_retry_annual/historical_oos_v1', 'rf_daily_ideal_daily/validation_v1', 'rf_daily_ideal_daily/historical_oos_v1', 'rf_daily_ideal_summary/validation_v1', 'rf_daily_ideal_summary/historical_oos_v1', 'rf_daily_ideal_annual/validation_v1', 'rf_daily_ideal_annual/historical_oos_v1', 'rf_daily_timing_audit/validation_v1', 'rf_daily_timing_audit/historical_oos_v1'] |
| PASS | `frozen_snapshot_integrity:factor_neutralization_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:factor_neutralization_v1` | mismatches=['pit_industry_exposure_index/v1', 'multi_horizon_factor_panel/2023', 'multi_horizon_factor_panel/2025', 'pit_industry_exposures/2023', 'pit_industry_exposures/2025', 'factor_neutralization_summary/v1', 'factor_neutralization_annual/v1', 'factor_neutralization_bootstrap/v1', 'factor_neutralization_conclusions/v1'] |
| PASS | `frozen_snapshot_integrity:model_portfolio_neutralization_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:model_portfolio_neutralization_v1` | mismatches=['pit_industry_exposure_index/v1', 'multi_horizon_factor_panel/2023', 'multi_horizon_factor_panel/2025', 'pit_industry_exposures/2023', 'pit_industry_exposures/2025', 'model_neutralization_protocol/v1', 'model_neutralization_validation_summary/v1', 'model_neutralization_oos_summary/v1', 'model_neutralization_oos_annual/v1', 'model_neutralization_oos_bootstrap/v1', 'model_neutralization_conclusions/v1', 'model_neutralization_fit_audit/v1'] |
| PASS | `frozen_snapshot_integrity:factor_neutralization_v2` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:factor_neutralization_v2` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:factor_tear_sheet_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:factor_tear_sheet_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:capm_attribution_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:capm_attribution_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:style_factor_attribution_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:style_factor_attribution_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:model_portfolio_neutralization_v2` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:model_portfolio_neutralization_v2` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:constrained_portfolio_neutralization_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:constrained_portfolio_neutralization_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:constrained_portfolio_daily_execution_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:constrained_portfolio_daily_execution_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:multi_factor_risk_attribution_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:multi_factor_risk_attribution_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:risk_aware_portfolio_optimization_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:risk_aware_portfolio_optimization_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:risk_aware_portfolio_optimization_v2` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:risk_aware_portfolio_optimization_v2` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:factor_registry_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:factor_registry_v1` | mismatches=['factor_registry/v1'] |
| PASS | `frozen_snapshot_integrity:portfolio_sensitivity_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:portfolio_sensitivity_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:hgb_hyperparameter_stability_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:hgb_hyperparameter_stability_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:factor_visual_diagnostics_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:factor_visual_diagnostics_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:incremental_alpha_test_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:incremental_alpha_test_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:research_conclusion_audit_v2` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:research_conclusion_audit_v2` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:final_research_baseline_v4` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:final_research_baseline_v4` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:final_research_baseline_v1` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:final_research_baseline_v1` | mismatches=['factor_registry/v1'] |
| PASS | `frozen_snapshot_integrity:final_research_baseline_v2` | all frozen files match FROZEN.json |
| WARN | `frozen_live_inputs:final_research_baseline_v2` | mismatches=['factor_registry/v1'] |
| PASS | `frozen_snapshot_integrity:factor_registry_v2` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:factor_registry_v2` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:research_conclusion_audit_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:research_conclusion_audit_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:final_research_baseline_v3` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:final_research_baseline_v3` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:transaction_cost_model_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:transaction_cost_model_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:risk_free_rate_attribution_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:risk_free_rate_attribution_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:market_benchmark_attribution_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:market_benchmark_attribution_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:model_stability_validation_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:model_stability_validation_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:research_governance_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:research_governance_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:institutional_execution_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:institutional_execution_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:institutional_execution_v2` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:institutional_execution_v2` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:portfolio_construction_comparison_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:portfolio_construction_comparison_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:unified_account_ledger_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:unified_account_ledger_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:statistical_inference_protocol_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:statistical_inference_protocol_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:alpha_evidence_matrix_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:alpha_evidence_matrix_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:incremental_alpha_fill_execution_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:incremental_alpha_fill_execution_v1` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:alpha_evidence_matrix_v2` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:alpha_evidence_matrix_v2` | all registered inputs match their SHA-256 hashes |
| PASS | `frozen_snapshot_integrity:engineering_quality_v1` | all frozen files match FROZEN.json |
| PASS | `frozen_live_inputs:engineering_quality_v1` | all registered inputs match their SHA-256 hashes |
| WARN | `invalidated_experiment:institutional_execution_v1` | severity=registry_self_staging_dirty_flag; superseded_by=institutional_execution_v2; reason=The research outputs and hashes are valid, but the experiment registry created its temporary snapshot directory inside the repository before capturing Git status. That self-created untracked directory caused the metadata to report dirty=true even though registration began from clean commit d1b00aedd5b3dcbc660733fa64288c550a83e446. |
| WARN | `invalidated_experiment:portfolio_sensitivity_v1` | severity=cost_stress_repricing_formula_superseded; superseded_by=transaction_cost_model_v1; reason=The v1 fixed-fill execution cost stress subtracted turnover times the fee rate directly from gross return. The daily executor defines turnover over pre-trade NAV while return is measured over previous NAV, so exact C0 reproduction additionally requires the pre-trade-to-previous-NAV ratio. The difference is small but the v1 stress grid is not an exact 20bp regression of the executable baseline. |
| WARN | `invalidated_experiment:factor_preprocessing_ablation_v1` | severity=portfolio_metrics_invalid; superseded_by=factor_preprocessing_ablation_v2; reason=Validation portfolios were formed after filtering rows with non-null future returns. RankIC and model-selection metrics used the correct realized subset, but Top-200 portfolio metrics had outcome-availability leakage. |
| WARN | `invalidated_experiment:model_benchmark_v1` | severity=portfolio_metrics_execution_idealized; superseded_by=rf_daily_retry_execution_v1; reason=Model ranking and RankIC are valid, but the benchmark Top-200 portfolio assumes every next-day rebalance is filled. RandomForest disproportionately selected stocks blocked by next-day price limits or suspensions, materially inflating validation returns and modestly inflating historical-OOS returns. |
| WARN | `invalidated_experiment:rf_execution_forensics_v1` | severity=execution_timing_incomplete; superseded_by=rf_daily_retry_execution_v1; reason=The weekly constrained simulator applied a blocked next-day order only at the following weekly rebalance. It also evaluated every signal with a full five-day label even when holiday-week signal dates were fewer than five trading days apart, creating overlapping portfolio return intervals. The price-limit diagnosis remains valid, but its 19.09% validation and 24.19% historical-OOS returns are not the final executable estimates. |
| WARN | `invalidated_experiment:factor_neutralization_v1` | severity=upstream_snapshot_truncation; superseded_by=factor_neutralization_v2; reason=BaoStock security snapshots for 2023-06-16 and 2025-10-10 were silently truncated at pagination boundaries. The affected PIT universe, factor panel, and industry exposure partitions were repaired and the experiment was rerun. |
| WARN | `invalidated_experiment:model_portfolio_neutralization_v1` | severity=upstream_snapshot_truncation; superseded_by=model_portfolio_neutralization_v2; reason=BaoStock security snapshots for 2023-06-16 and 2025-10-10 were silently truncated at pagination boundaries. The affected PIT universe, feature, and industry exposure partitions were repaired and the locked models were rerun. |
| WARN | `invalidated_experiment:risk_aware_portfolio_optimization_v1` | severity=frozen_provenance_incomplete; superseded_by=risk_aware_portfolio_optimization_v2; reason=The first immutable registration accidentally reused the constrained daily execution input-list YAML anchor. Research tables and conclusions are valid, but the frozen metadata omitted the full PIT beta, panel, size, and industry dependency set consumed by the risk optimizer. |
| WARN | `invalidated_experiment:baseline_ridge_v1` | severity=superseded_live_baseline; superseded_by=final_research_baseline_v2; reason=The immutable snapshot remains valid, but its registered live model and portfolio artifacts predate the final PIT lake repair and consolidated research rebuild. It is retained only as historical provenance. |
| WARN | `invalidated_experiment:challenger_hgb_blend_v1` | severity=superseded_live_baseline; superseded_by=final_research_baseline_v2; reason=The original HGB/Ridge blend snapshot predates the repaired PIT lake and unified model benchmark. Its immutable result is historical, not current. |
| WARN | `invalidated_experiment:factor_preprocessing_ablation_v2` | severity=superseded_live_artifacts; superseded_by=final_research_baseline_v2; reason=The leakage-corrected frozen snapshot is intact, but its live derived tables were regenerated during the consolidated rebuild. The final baseline records the current hashes while preserving this historical run. |
| WARN | `invalidated_experiment:factor_preprocessing_oos_v1` | severity=superseded_live_artifacts; superseded_by=final_research_baseline_v2; reason=Current OOS preprocessing artifacts were rebuilt after upstream PIT data repairs. The old snapshot is immutable but no longer the live publication baseline. |
| WARN | `invalidated_experiment:ic_decay_v1` | severity=superseded_live_artifacts; superseded_by=final_research_baseline_v2; reason=IC-decay live tables were regenerated after the PIT panel repair. The old frozen result remains auditable and is superseded for publication. |
| WARN | `invalidated_experiment:leave_one_factor_out_v1` | severity=superseded_live_artifacts; superseded_by=final_research_baseline_v2; reason=LOFO live outputs were regenerated during the final PIT rebuild. The immutable run is retained as history but is not the current baseline. |
| WARN | `invalidated_experiment:label_horizon_v1` | severity=superseded_live_artifacts; superseded_by=final_research_baseline_v2; reason=Label-horizon artifacts were regenerated from the repaired panel. The historical frozen snapshot remains intact and its selection caveat remains valid. |
| WARN | `invalidated_experiment:size_neutralization_v1` | severity=superseded_live_artifacts; superseded_by=factor_neutralization_v2; reason=Size-neutralization outputs were rebuilt after upstream security snapshot repair and are now incorporated into the full industry-plus-size chain. |
| WARN | `invalidated_experiment:rf_hyperparameter_stability_v1` | severity=superseded_model_family; superseded_by=hgb_hyperparameter_stability_v1; reason=The RF local grid is preserved, but the repaired unified benchmark selects HGB and the final robustness work uses a validation-only HGB two-dimensional surface. |
| WARN | `invalidated_experiment:rf_daily_retry_execution_v1` | severity=superseded_execution_baseline; superseded_by=constrained_portfolio_daily_execution_v1; reason=The RF execution audit remains useful forensic evidence, but the final execution baseline uses locked Ridge/HGB constrained portfolios on the repaired lake. |
| WARN | `invalidated_experiment:final_research_baseline_v1` | severity=superseded_publication_baseline; superseded_by=final_research_baseline_v2; reason=The first final aggregate was frozen before the dashboard and publication auditor gained explicit gates for the new factor registry, HGB surface, portfolio sensitivity, and cost stress. Its research metrics are unchanged; v2 records the final source tree and publication configuration. |
| WARN | `invalidated_experiment:factor_registry_v1` | severity=corrected_metadata_convention; superseded_by=factor_registry_v2; reason=The v1 registry incorrectly described adjustflag=3 raw OHLC and ex-right-reference pct_chg as a forward-adjusted price history. Factor formulas and numerical outputs are unchanged; v2 corrects the metadata. |
| WARN | `invalidated_experiment:final_research_baseline_v2` | severity=superseded_publication_baseline; superseded_by=final_research_baseline_v3; reason=V3 adds an explicit five-question decision hierarchy, BH-FDR control for the single-factor family, corrected price-return metadata, and a formal boundary between completed historical research and future paper/live work. |
| WARN | `invalidated_experiment:research_conclusion_audit_v1` | severity=superseded_conclusion_baseline; superseded_by=research_conclusion_audit_v2; reason=V2 adds the executable industry-and-six-style matched Incremental Alpha Test, including paired block intervals, equal-volatility diagnostics, transaction efficiency, and BH-FDR control across the four locked strategy variants. |
| WARN | `invalidated_experiment:final_research_baseline_v3` | severity=superseded_publication_baseline; superseded_by=final_research_baseline_v4; reason=V4 adds a characteristic-matched executable style benchmark and distinguishes individual confidence intervals from cross-strategy multiplicity-controlled evidence. |
| PASS | `artifact:research_coverage/a_share` | rows=4 |
| PASS | `historical_universe_download_coverage` | coverage=100.00%, missing_files=0 |
| PASS | `artifact:model_validation_predictions/ridge_v1` | rows=818,333 |
| PASS | `ridge_validation_prediction_keys_unique` | duplicate_rows=0 |
| PASS | `ridge_validation_prediction_years` | observed=2016-2020, expected=2016-2020 |
| PASS | `ridge_validation_availability_after_signal` | available_date must be strictly after asof_date |
| PASS | `ridge_validation_scores_finite` | rows=818,333, dates=254 |
| PASS | `artifact:model_oos_predictions/ridge_v1` | rows=1,394,487 |
| PASS | `ridge_oos_prediction_keys_unique` | duplicate_rows=0 |
| PASS | `ridge_oos_prediction_years` | observed=2021-2026, expected=2021-2026 |
| PASS | `ridge_oos_availability_after_signal` | available_date must be strictly after asof_date |
| PASS | `ridge_oos_scores_finite` | rows=1,394,487, dates=295 |
| PASS | `artifact:model_challenger_oos_predictions/hgb_v1` | rows=1,394,487 |
| PASS | `challenger_oos_prediction_keys_unique` | duplicate_rows=0 |
| PASS | `challenger_oos_prediction_years` | observed=2021-2026, expected=2021-2026 |
| PASS | `challenger_oos_availability_after_signal` | available_date must be strictly after asof_date |
| PASS | `challenger_oos_scores_finite` | rows=1,394,487, dates=295 |
| PASS | `artifact:factor_preprocessing_oos_predictions/v1` | rows=1,384,470 |
| PASS | `factor_preprocessing_oos_prediction_keys_unique` | duplicate_rows=0 |
| PASS | `factor_preprocessing_oos_prediction_years` | observed=2021-2026, expected=2021-2026 |
| PASS | `factor_preprocessing_oos_availability_after_signal` | available_date must be strictly after asof_date |
| PASS | `factor_preprocessing_oos_scores_finite` | rows=1,384,470, dates=293 |
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
| PASS | `ridge_rank_ic_positive_each_oos_year` | minimum_annual_rank_ic=0.0897 |
| PASS | `ridge_excess_bootstrap_interval_above_zero` | 95% CI=[5.32%, 34.24%] |
| PASS | `challenger_increment_interval_valid` | 95% CI=[1.19%, 9.19%]; inference=positive; historical OOS is not pristine |
| PASS | `artifact:lofo_summary/v1` | rows=32 |
| PASS | `artifact:factor_preprocessing_oos_summary/v1` | rows=2 |
| PASS | `lofo_full_reproduces_preprocessing:mean_rank_ic` | lofo=0.0976365971, preprocessing_oos=0.0976365971 |
| PASS | `lofo_full_reproduces_preprocessing:annualized_icir` | lofo=4.5261332009, preprocessing_oos=4.5261332009 |
| PASS | `lofo_full_reproduces_preprocessing:portfolio_annualized_return` | lofo=0.2116277973, preprocessing_oos=0.2116277973 |
| PASS | `artifact:label_horizon_oos_predictions/v1` | rows=1,394,487 |
| PASS | `label_horizon_oos_prediction_keys_unique` | duplicate_rows=0 |
| PASS | `label_horizon_oos_prediction_years` | observed=2021-2026, expected=2021-2026 |
| PASS | `label_horizon_oos_availability_after_signal` | available_date must be strictly after asof_date |
| PASS | `label_horizon_oos_scores_finite` | rows=1,394,487, dates=295 |
| PASS | `artifact:label_horizon_oos_portfolio/v1` | rows=1,770 |
| PASS | `label_horizon_selected_net_return_identity` | max_abs_error=0.000e+00 |
| PASS | `label_horizon_selected_returns_finite_and_above_minus_one` | dates=290 |
| PASS | `artifact:label_horizon_panel_audit/v1` | rows=21 |
| PASS | `artifact:label_horizon_selection/v1` | rows=1 |
| PASS | `artifact:label_horizon_oos_summary/v1` | rows=6 |
| PASS | `artifact:label_horizon_oos_bootstrap/v1` | rows=5 |
| PASS | `label_panel_preserves_eligible_rows` | years=21, all_years_match=True |
| PASS | `label_panel_keys_unique` | duplicate_rows=0 |
| PASS | `label_coverage_monotone_by_horizon` | coverage(5d) >= coverage(10d) >= coverage(20d) for every year |
| PASS | `label_selection_excludes_historical_oos` | selected=h20_raw_return |
| PASS | `artifact:factor_preprocessing_oos_summary/v1` | rows=2 |
| PASS | `label_h5_rank_reproduces_preprocessing:mean_rank_ic` | label_study=0.0976365971, preprocessing_oos=0.0976365971 |
| PASS | `label_h5_rank_reproduces_preprocessing:portfolio_annualized_return` | label_study=0.2116277973, preprocessing_oos=0.2116277973 |
| PASS | `label_selected_increment_not_overstated` | selected=h20_raw_return; 95% CI=[-4.71%, 5.88%] |
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
| PASS | `size_neutral_score_orthogonality` | abs_mean_score_log_size_correlation=9.557e-19 |
| PASS | `artifact:factor_preprocessing_oos_summary/v1` | rows=2 |
| PASS | `size_raw_reproduces_preprocessing:mean_rank_ic` | size_study=0.0976365971, preprocessing_oos=0.0976365971 |
| PASS | `size_raw_reproduces_preprocessing:portfolio_annualized_return` | size_study=0.2116277973, preprocessing_oos=0.2116277973 |
| PASS | `small_size_return_dependence_detected` | neutral-minus-raw 95% CI=[-8.61%, -0.67%] |
| PASS | `artifact:pit_industry_exposure_index/v1` | rows=21 |
| PASS | `artifact:factor_neutralization_summary/v1` | rows=180 |
| PASS | `artifact:factor_neutralization_bootstrap/v1` | rows=180 |
| PASS | `pit_industry_coverage` | minimum_eligible_weekly_coverage=98.5487% |
| PASS | `pit_industry_keys_unique` | duplicate_rows=0 |
| PASS | `pit_industry_no_future_information` | future_or_invalid_timing_rows=0 |
| PASS | `industry_neutral_factor_orthogonality` | maximum_mean_abs_industry_group_mean=1.294e-14 |
| PASS | `industry_size_neutral_factor_orthogonality` | industry=1.177e-14, size=2.135e-15 |
| PASS | `artifact:factor_tear_sheet_protocol/v1` | rows=1 |
| PASS | `artifact:factor_tear_sheet_weekly/v1` | rows=8,790 |
| PASS | `artifact:factor_tear_sheet_buckets/v1` | rows=87,900 |
| PASS | `artifact:factor_tear_sheet_summary/v1` | rows=30 |
| PASS | `artifact:factor_tear_sheet_annual/v1` | rows=180 |
| PASS | `artifact:factor_tear_sheet_regime/v1` | rows=120 |
| PASS | `artifact:factor_tear_sheet_bootstrap/v1` | rows=90 |
| PASS | `factor_tear_sheet_paired_oos_panel` | dates=293, pipelines=2, factors=15 |
| PASS | `factor_tear_sheet_deciles_complete` | rows=87,900, expected=87,900 |
| PASS | `factor_tear_sheet_regime_is_pit` | benchmark_trade_date<=signal_date and protocol declares trailing-only state |
| PASS | `factor_tear_sheet_cost_identity` | cost_bps_per_leg_turnover=20.0 |
| PASS | `factor_tear_sheet_no_oos_reselection` | direction_rule=sign_of_raw_training_mean_rank_ic |
| PASS | `artifact:capm_protocol/v1` | rows=1 |
| PASS | `artifact:capm_aligned_daily/v1` | rows=11,096 |
| PASS | `artifact:capm_summary/v1` | rows=24 |
| PASS | `artifact:capm_annual/v1` | rows=144 |
| PASS | `artifact:capm_rolling/v1` | rows=448 |
| PASS | `capm_protocol_uses_executable_nonoverlap_returns` | dates=1387, return=daily_net_return |
| PASS | `capm_complete_paired_panel` | summary_rows=24, annual_rows=144, observations=1387-1387 |
| PASS | `capm_hac_inference_complete` | newey_west_lags=[5] |
| PASS | `capm_ols_intercept_identity` | maximum_abs_error=1.409e-18 |
| PASS | `artifact:style_factor_protocol/v1` | rows=1 |
| PASS | `artifact:style_factor_targets/v1` | rows=1,392,156 |
| PASS | `artifact:style_factor_exposure_audit/v1` | rows=1,770 |
| PASS | `artifact:style_factor_daily_returns/v1` | rows=8,322 |
| PASS | `artifact:style_factor_diagnostics/v1` | rows=6 |
| PASS | `artifact:style_factor_multicollinearity/v1` | rows=7 |
| PASS | `artifact:style_factor_regression_summary/v1` | rows=8 |
| PASS | `artifact:style_factor_regression_annual/v1` | rows=48 |
| PASS | `style_factor_protocol_is_pit_and_locked` | construction=residual_proportional_self_financing; explanatory factors explicitly gross of costs |
| PASS | `style_factor_self_financing_and_orthogonal` | maximum_neutrality_error=1.576e-14 |
| PASS | `style_factor_complete_daily_panel` | dates=1387-1387, factors=6 |
| PASS | `style_factor_regression_not_ill_conditioned` | maximum_vif=2.035, condition_number=2.741 |
| PASS | `style_factor_multifactor_inference_complete` | summary_rows=8, annual_rows=48 |
| PASS | `artifact:model_neutralization_protocol/v1` | rows=1 |
| PASS | `artifact:model_neutralization_oos_summary/v1` | rows=8 |
| PASS | `artifact:model_neutralization_oos_daily/v1` | rows=2,344 |
| PASS | `artifact:model_neutralization_oos_bootstrap/v1` | rows=36 |
| PASS | `model_neutralization_protocol_locked` | no historical-OOS selection; locked models; paired exposure-complete sample |
| PASS | `model_neutralization_paired_dates` | minimum_dates=293, maximum_dates=293 |
| PASS | `linear_model_neutral_score_residual_small` | industry=4.541e-16, size=5.263e-17 |
| PASS | `model_neutralization_inference_complete` | metrics=abs_score_log_size_correlation,industry_active_weight_l1,net_return_annualized_arithmetic,portfolio_industry_hhi,rank_ic,score_industry_max_abs_mean |
| PASS | `artifact:constrained_portfolio_protocol/v1` | rows=1 |
| PASS | `artifact:constrained_portfolio_oos_weekly/v1` | rows=2,344 |
| PASS | `artifact:constrained_portfolio_oos_summary/v1` | rows=8 |
| PASS | `artifact:constrained_portfolio_oos_bootstrap/v1` | rows=40 |
| PASS | `constrained_portfolio_protocol_locked` | no historical-OOS selection; prespecified constraints; repaired v2 source |
| PASS | `constrained_portfolio_paired_dates` | minimum_dates=293, maximum_dates=293 |
| PASS | `constrained_portfolio_risk_bounds` | stock=0.000e+00, size=2.498e-16, industry=4.718e-16 |
| PASS | `constrained_portfolio_turnover_bound` | maximum_excess=4.998e-08, relaxation_fraction=0.000% |
| PASS | `constrained_portfolio_inference_complete` | metrics=abs_portfolio_size_z,industry_active_weight_l1,net_return_annualized_arithmetic,rank_ic,turnover |
| PASS | `artifact:constrained_daily_execution_protocol/v1` | rows=1 |
| PASS | `artifact:constrained_daily_execution_daily/v1` | rows=11,096 |
| PASS | `artifact:constrained_daily_execution_summary/v1` | rows=8 |
| PASS | `artifact:constrained_daily_execution_bootstrap/v1` | rows=4 |
| PASS | `artifact:constrained_execution_targets/v1` | rows=245,467 |
| PASS | `constrained_daily_execution_protocol_locked` | frozen targets; unique daily clock; persistent blocked share orders |
| PASS | `constrained_daily_execution_paired_targets` | minimum_dates=293, maximum_dates=293 |
| PASS | `constrained_daily_execution_accounting_identity` | dates=1387-1387, accounting=8.327e-15, return=5.840e-15 |
| PASS | `constrained_daily_execution_no_hidden_rebalance` | ideal_non_target_max_turnover=0.000e+00 |
| PASS | `constrained_daily_execution_haircut_detected` | strategies_with_haircut=4/4, negative_upper_intervals=4/4 |
| PASS | `artifact:institutional_execution_protocol/v1` | rows=1 |
| PASS | `artifact:institutional_execution_daily/v1` | rows=2,774 |
| PASS | `artifact:institutional_execution_orders/v1` | rows=136,495 |
| PASS | `artifact:institutional_execution_events/v1` | rows=473,296 |
| PASS | `artifact:institutional_execution_fills/v1` | rows=161,215 |
| PASS | `artifact:institutional_execution_positions/v1` | rows=592,376 |
| PASS | `artifact:institutional_execution_corporate_actions/v1` | rows=2,068 |
| PASS | `artifact:institutional_execution_summary/v1` | rows=2 |
| PASS | `artifact:institutional_execution_invariants/v1` | rows=2 |
| PASS | `institutional_execution_evidence_disclosed` | historical OOS diagnostic; no independent-evidence claim or model selection |
| PASS | `institutional_execution_common_clock` | scenarios=['daily_close_institutional_constraints', 'market_constraints_removed_upper_bound'], dates={'daily_close_institutional_constraints': 1387, 'market_constraints_removed_upper_bound': 1387} |
| PASS | `institutional_execution_invariants` | [{'scenario': 'daily_close_institutional_constraints', 'max_abs_nav_reconciliation_error': 1.7881393432617188e-07, 'minimum_cash': 0.6862432244688534, 'max_abs_accounting_weight_error': 0.0, 'negative_position_count': 0, 'negative_cash_day_count': 0, 't_plus_one_violation_count': 0, 'max_abs_order_fill_reconciliation_error': 0.0, 'fee_reconciliation_error': 0.0, 'unreconciled_corporate_action_count': 0, 'all_invariants_pass': True}, {'scenario': 'market_constraints_removed_upper_bound', 'max_abs_nav_reconciliation_error': 2.086162567138672e-07, 'minimum_cash': 2805.7423940931476, 'max_abs_accounting_weight_error': 0.0, 'negative_position_count': 0, 'negative_cash_day_count': 0, 't_plus_one_violation_count': 0, 'max_abs_order_fill_reconciliation_error': 0.0, 'fee_reconciliation_error': 0.0, 'unreconciled_corporate_action_count': 0, 'all_invariants_pass': True}] |
| PASS | `institutional_execution_nav_cash_positions_reconcile` | accounting_error=0.000e+00, nav_error_cny=2.086e-07, minimum_cash=0.686243 |
| PASS | `institutional_execution_fill_fees_reconcile` | fill_minus_daily_fee_cny=0.000e+00 |
| PASS | `institutional_execution_t_plus_one_fill_gate` | sell_fills=67,275, violations=0 |
| PASS | `institutional_execution_order_state_machine` | event_statuses=['ACTION_ADJUSTED', 'BLOCKED', 'EXPIRED', 'FILLED', 'NEW', 'PARTIALLY_FILLED', 'REPLACED'], orders=136,495, new_events=136,495 |
| PASS | `institutional_execution_orders_terminal` | terminal_statuses=['EXPIRED', 'FILLED', 'REPLACED'] |
| PASS | `institutional_execution_corporate_action_proxy_reconciles` | actions=2,068, max_error_cny=4.657e-10; proxy_not_official |
| PASS | `institutional_execution_constraints_have_measurable_haircut` | annualized=0.206647 vs 0.220813; tracking_l1=0.023068 vs 0.019099 |
| PASS | `artifact:unified_account_protocol/v1` | rows=1 |
| PASS | `artifact:unified_account_daily_ledger/v1` | rows=1,387 |
| PASS | `artifact:unified_account_security_ledger/v1` | rows=296,714 |
| PASS | `artifact:unified_account_cash_journal/v1` | rows=1,387 |
| PASS | `artifact:unified_account_orders/v1` | rows=68,262 |
| PASS | `artifact:unified_account_events/v1` | rows=221,968 |
| PASS | `artifact:unified_account_fills/v1` | rows=81,650 |
| PASS | `artifact:unified_account_corporate_actions/v1` | rows=1,034 |
| PASS | `artifact:unified_account_daily_reconciliation/v1` | rows=1,387 |
| PASS | `artifact:unified_account_position_reconciliation/v1` | rows=313,728 |
| PASS | `artifact:unified_account_order_exceptions/v1` | rows=17,495 |
| PASS | `artifact:unified_account_failed_assertions/v1` | rows=1 |
| PASS | `artifact:unified_account_checks/v1` | rows=16 |
| PASS | `unified_account_protocol_is_read_only_diagnostic` | frozen Phase 5 outputs normalized without changing returns or claiming new evidence |
| PASS | `unified_account_all_hard_reconciliations_pass` | checks=16, pass=15, warn=1, fail=0 |
| PASS | `unified_account_common_daily_key` | daily=1,387, cash=1,387, reconciliation=1,387 |
| PASS | `unified_account_target_actual_cash_semantics` | security_rows=296,714; required_fields=['actual_shares', 'actual_weight', 'cash_weight', 'target_weight', 'weight_gap'] |
| PASS | `unified_account_cash_nav_position_replay` | cash_error_cny=1.185e-07, nav_error_cny=1.192e-07, share_error=0.000e+00 |
| PASS | `unified_account_assertion_log_clean` | [{'check_name': 'NO_FAILED_ASSERTIONS', 'status': 'PASS', 'observed': 'all hard Phase 7 accounting assertions passed'}] |
| WARN | `legacy_corporate_action_scenario_key_missing` | The frozen Phase 5 corporate-action table concatenated two scenarios without a scenario key. Phase 7 preserves it and reconstructs account-specific actions from prior actual holdings plus frozen market action fields. |
| PASS | `artifact:statistical_inference_protocol/v1` | rows=1 |
| PASS | `artifact:statistical_inference_series/v1` | rows=18,499 |
| PASS | `artifact:statistical_inference_alignment/v1` | rows=37 |
| PASS | `artifact:statistical_inference_hac/v1` | rows=148 |
| PASS | `artifact:statistical_inference_block_bootstrap/v1` | rows=155 |
| PASS | `artifact:statistical_inference_fdr/v1` | rows=37 |
| PASS | `artifact:statistical_inference_family_summary/v1` | rows=4 |
| PASS | `artifact:statistical_inference_block_sensitivity/v1` | rows=37 |
| PASS | `artifact:statistical_trial_registry/v1` | rows=172 |
| PASS | `artifact:statistical_inference_checks/v1` | rows=10 |
| PASS | `artifact:statistical_inference_failed_assertions/v1` | rows=1 |
| PASS | `statistical_inference_protocol_discloses_post_oos_status` | harmonized inference is post-historical-OOS and does not claim future probability |
| PASS | `statistical_inference_all_protocol_checks_pass` | checks=10, failures=0 |
| PASS | `statistical_inference_fdr_families_not_pooled` | family_hypotheses={'factor_gross_spread': 15, 'factor_rank_ic': 15, 'model_incremental_alpha': 4, 'portfolio_construction_increment': 3} |
| PASS | `statistical_inference_paired_daily_clocks` | paired_hypotheses=7, observations=[np.int64(1387)] |
| PASS | `statistical_inference_dependence_sensitivity_complete` | hac=148, block_bootstrap=155, sensitivity=37 |
| PASS | `statistical_trial_registry_complete` | trials=172, categories=11 |
| PASS | `statistical_primary_incremental_alpha_not_overclaimed` | [{'point_estimate': 0.02870154626718977, 'ci_low': -0.004378561383624404, 'ci_high': 0.06472945903992583, 'bh_fdr_adjusted_p': 0.10294852573713144, 'claim_status': 'positive_value_not_established_after_family_fdr'}] |
| PASS | `statistical_inference_assertion_log_clean` | [{'check_name': 'NO_FAILED_ASSERTIONS', 'status': 'PASS', 'observed': 'all Phase 8 inference protocol assertions passed'}] |
| PASS | `artifact:incremental_alpha_protocol/v1` | rows=1 |
| PASS | `artifact:incremental_alpha_style_targets/v1` | rows=674,443 |
| PASS | `artifact:incremental_alpha_exposure_audit/v1` | rows=1,172 |
| PASS | `artifact:incremental_alpha_daily/v1` | rows=11,096 |
| PASS | `artifact:incremental_alpha_summary/v1` | rows=8 |
| PASS | `artifact:incremental_alpha_bootstrap/v1` | rows=24 |
| PASS | `artifact:incremental_alpha_comparison/v1` | rows=4 |
| PASS | `incremental_alpha_protocol_locked` | frozen ML targets; PIT style peers; same 20bp executable clock |
| PASS | `incremental_alpha_target_constraints` | dates=293-293, max_weight=0.00500000 |
| PASS | `incremental_alpha_exposure_and_turnover_match` | turnover=0.300000, industry_gap=0.0169, overlap=0.0106, max_mean_style_gap=0.3314 |
| PASS | `incremental_alpha_common_executable_clock` | dates=1387-1387, accounting=7.772e-15, return=5.234e-15 |
| PASS | `artifact:constrained_daily_execution_summary/v1` | rows=8 |
| PASS | `incremental_alpha_reproduces_frozen_ml_path` | strategies=4, max_annual_return_error=5.329e-15 |
| PASS | `incremental_alpha_inference_and_multiplicity_complete` | comparisons=4, metrics=6 |
| PASS | `artifact:alpha_evidence_protocol/v2` | rows=1 |
| PASS | `artifact:alpha_evidence_aligned_daily/v2` | rows=33,288 |
| PASS | `artifact:alpha_evidence_alignment/v2` | rows=24 |
| PASS | `artifact:alpha_evidence_risk_adjusted_regression/v2` | rows=96 |
| PASS | `artifact:alpha_evidence_risk_adjusted_fdr/v2` | rows=4 |
| PASS | `artifact:alpha_evidence_risk_sensitivity/v2` | rows=4 |
| PASS | `artifact:alpha_evidence_execution_comparison/v2` | rows=8 |
| PASS | `artifact:alpha_evidence_exposure_match/v2` | rows=4 |
| PASS | `artifact:alpha_evidence_candidate_matrix/v2` | rows=4 |
| PASS | `artifact:alpha_evidence_measure_definitions/v2` | rows=5 |
| PASS | `artifact:alpha_evidence_gap_register/v2` | rows=4 |
| PASS | `artifact:alpha_evidence_checks/v2` | rows=12 |
| PASS | `artifact:alpha_evidence_failed_assertions/v2` | rows=1 |
| PASS | `artifact:incremental_alpha_fill_protocol/v1` | rows=1 |
| PASS | `artifact:incremental_alpha_fill_summary/v1` | rows=8 |
| PASS | `artifact:incremental_alpha_fill_invariants/v1` | rows=8 |
| PASS | `artifact:incremental_alpha_fill_comparison/v1` | rows=4 |
| PASS | `artifact:incremental_alpha_fill_checks/v1` | rows=10 |
| PASS | `artifact:incremental_alpha_fill_failed_assertions/v1` | rows=1 |
| PASS | `alpha_evidence_protocol_discloses_nonindependent_history` | phase_status=COMPLETE_SHARE_LEVEL_FILL_COMPARISON |
| PASS | `alpha_evidence_all_implemented_scope_checks_pass` | checks=12, failures=0 |
| PASS | `alpha_evidence_common_clock_complete` | alignment_rows=24, aligned_rows=33288 |
| PASS | `alpha_evidence_risk_model_sensitivity_complete` | regressions=96, rf=3, return_types=['price_return', 'total_return'] |
| PASS | `alpha_evidence_four_candidate_fdr_family` | survivors=0, q=[0.597059, 0.597059, 0.597059, 0.597059] |
| PASS | `alpha_evidence_does_not_call_capm_pure_alpha` | [{'model': 'hgb_shallow15_locked', 'alpha_pipeline': 'industry_size_neutralized', 'evidence_verdict': 'incremental_alpha_not_established_after_multiplicity_and_risk_adjustment'}, {'model': 'hgb_shallow15_locked', 'alpha_pipeline': 'raw', 'evidence_verdict': 'incremental_alpha_not_established_after_multiplicity_and_risk_adjustment'}, {'model': 'ridge_locked', 'alpha_pipeline': 'industry_size_neutralized', 'evidence_verdict': 'incremental_alpha_not_established_after_multiplicity_and_risk_adjustment'}, {'model': 'ridge_locked', 'alpha_pipeline': 'raw', 'evidence_verdict': 'incremental_alpha_not_established_after_multiplicity_and_risk_adjustment'}] |
| PASS | `alpha_evidence_local_proxy_and_exposure_error_disclosed` | exposure_rows=4 |
| PASS | `alpha_evidence_assertion_log_clean` | [{'check_name': 'NO_FAILED_ASSERTIONS', 'status': 'PASS', 'observed': 'all Phase 9 assertions passed; share-level fill comparison is complete'}] |
| PASS | `alpha_evidence_share_level_fill_comparison_complete` | paths=8, ratios=[0.987309, 0.931394, 0.988337, 0.891164, 0.860965, 0.954013, 0.756205, 0.99485] |
| PASS | `alpha_evidence_fill_accounting_invariants_pass` | paths=8, checks=10 |
| PASS | `artifact:engineering_quality_protocol/v1` | rows=1 |
| PASS | `artifact:engineering_implementation_inventory/v1` | rows=6 |
| PASS | `artifact:engineering_random_seed_inventory/v1` | rows=3 |
| PASS | `artifact:engineering_test_layer_inventory/v1` | rows=10 |
| PASS | `artifact:engineering_output_contracts/v1` | rows=5 |
| PASS | `artifact:engineering_gap_register/v1` | rows=4 |
| PASS | `artifact:engineering_quality_checks/v1` | rows=8 |
| PASS | `artifact:engineering_failed_assertions/v1` | rows=1 |
| PASS | `engineering_quality_governance_only` | canonical_module=src/quant_research/execution_market.py |
| PASS | `engineering_quality_all_checks_pass` | checks=8 |
| PASS | `engineering_quality_single_execution_market_implementation` | [{'logic': 'corporate_action_proxy_formula', 'path': 'src/quant_research/execution_market.py', 'line': 41, 'is_definition': True}, {'logic': 'execution_buy_limit_rule', 'path': 'src/quant_research/execution_market.py', 'line': 52, 'is_definition': True}, {'logic': 'execution_market_public_api', 'path': 'src/quant_research/execution_market.py', 'line': 64, 'is_definition': True}, {'logic': 'execution_market_public_api', 'path': 'src/quant_research/incremental_alpha_fill_execution.py', 'line': 69, 'is_definition': False}, {'logic': 'execution_market_public_api', 'path': 'src/quant_research/institutional_execution.py', 'line': 552, 'is_definition': False}, {'logic': 'execution_market_public_api', 'path': 'src/quant_research/portfolio_construction_comparison.py', 'line': 531, 'is_definition': False}] |
| PASS | `engineering_quality_tests_and_seeds_complete` | layers=5, random_generators=3 |
| PASS | `engineering_quality_output_contracts_frozen` | [{'stage': 'phase5_execution', 'frozen_experiment': 'institutional_execution_v2'}, {'stage': 'phase6_portfolio', 'frozen_experiment': 'portfolio_construction_comparison_v1'}, {'stage': 'phase7_ledger', 'frozen_experiment': 'unified_account_ledger_v1'}, {'stage': 'phase8_inference', 'frozen_experiment': 'statistical_inference_protocol_v1'}, {'stage': 'phase9_alpha', 'frozen_experiment': 'alpha_evidence_matrix_v2'}] |
| PASS | `engineering_quality_no_open_p0` | {('P1', 'CONTROLLED_BY_IMMUTABLE_REGISTRY'): 1, ('P1', 'OPEN'): 1, ('P2', 'OPEN'): 1, ('P2', 'OPEN_NONRESEARCH'): 1} |
| PASS | `engineering_quality_assertion_log_clean` | [{'check_name': 'NO_FAILED_ASSERTIONS', 'status': 'PASS', 'observed': 'all Phase 10 engineering gates passed'}] |
| PASS | `artifact:risk_model_protocol/v1` | rows=1 |
| PASS | `artifact:pit_beta_exposure_index/v1` | rows=19 |
| PASS | `artifact:risk_model_factor_returns/v1` | rows=26,028 |
| PASS | `artifact:risk_model_portfolio_attribution/v1` | rows=1,172 |
| PASS | `artifact:risk_model_portfolio_forecasts/v1` | rows=1,172 |
| PASS | `artifact:risk_model_attribution_summary/v1` | rows=4 |
| PASS | `artifact:risk_model_attribution_bootstrap/v1` | rows=20 |
| PASS | `risk_attribution_protocol_locked` | prespecified PIT factors; lagged covariance; exact identity required |
| PASS | `pit_beta_coverage_and_keys` | minimum_coverage=91.42%, duplicates=0 |
| PASS | `risk_attribution_exact_and_paired` | dates=293-293, max_error=2.220e-16 |
| PASS | `risk_forecast_lagged_finite_positive` | minimum_prior_weeks=669, forecasts=1172 |
| PASS | `risk_attribution_inference_complete` | metrics=active_gross_return,active_net_return,industry_active_contribution,specific_selection_contribution,style_contribution |
| PASS | `artifact:risk_aware_portfolio_protocol/v2` | rows=1 |
| PASS | `artifact:risk_aware_portfolio_weekly/v2` | rows=3,516 |
| PASS | `artifact:risk_aware_portfolio_constraint_audit/v2` | rows=1,172 |
| PASS | `artifact:risk_aware_portfolio_forecasts/v2` | rows=2,344 |
| PASS | `artifact:risk_aware_portfolio_summary/v2` | rows=12 |
| PASS | `artifact:risk_aware_portfolio_bootstrap/v2` | rows=32 |
| PASS | `risk_aware_protocol_lagged_and_locked` | lagged covariance/specific risk; outcome-independent candidate set |
| PASS | `risk_aware_paired_dates` | dates=293-293, strategies=12 |
| PASS | `risk_aware_constraints_hold` | maximum_violation=4.515e-08, turnover_relaxation_fraction=0.00% |
| PASS | `target_volatility_forecast_calibrated` | maximum_mean_target_error=0.060%, no_leverage=True |
| PASS | `risk_aware_te_reduction_confirmed` | delta=-3.639%, 95%CI=[-4.163%,-3.152%] |
| PASS | `risk_aware_return_cost_detected` | delta=-4.707%, 95%CI=[-8.242%,-0.804%] |
| WARN | `risk_aware_policy_rejected` | The v2-registered minimum-active-variance policy reduces ex-ante tracking error but causes a statistically negative return delta; retain as a failed experiment. |
| PASS | `artifact:model_benchmark_oos_portfolio/v1` | rows=1,758 |
| PASS | `model_benchmark_winner_net_return_identity` | max_abs_error=0.000e+00 |
| PASS | `model_benchmark_winner_returns_finite_and_above_minus_one` | dates=293 |
| PASS | `artifact:model_benchmark_selection/v1` | rows=1 |
| PASS | `artifact:model_benchmark_oos_summary/v1` | rows=6 |
| PASS | `artifact:model_benchmark_oos_annual/v1` | rows=36 |
| PASS | `artifact:model_benchmark_winner_pairwise/v1` | rows=2 |
| PASS | `artifact:model_benchmark_availability/v1` | rows=8 |
| PASS | `model_benchmark_excludes_historical_oos_selection` | winner=hgb_shallow15 |
| PASS | `model_winner_has_annual_oos_results` | winner=hgb_shallow15, annual_rows=6 |
| PASS | `model_winner_rank_ic_positive_each_oos_year` | minimum_annual_rank_ic=0.0930 |
| PASS | `artifact:factor_preprocessing_oos_summary/v1` | rows=2 |
| PASS | `benchmark_ridge_reproduces_preprocessing:mean_rank_ic` | benchmark=0.0976365971, preprocessing_oos=0.0976365971 |
| PASS | `benchmark_ridge_reproduces_preprocessing:portfolio_annualized_return` | benchmark=0.2116277973, preprocessing_oos=0.2116277973 |
| PASS | `model_benchmark_has_ridge_reference` | ridge_candidates=['ridge_1'] |
| PASS | `model_winner_return_pairwise_vs_ridge` | candidate=hgb_shallow15, reference=ridge_1, metric=net_return_annualized_arithmetic, rows=1 |
| PASS | `model_winner_rank_ic_pairwise_vs_ridge` | candidate=hgb_shallow15, reference=ridge_1, metric=rank_ic, rows=1 |
| PASS | `model_winner_return_interval_valid` | 95% CI=[-0.86%, 7.26%]; inference=inconclusive |
| PASS | `model_winner_rank_ic_increment_not_overstated` | 95% CI=[-0.0023, 0.0071] |
| PASS | `random_forest_not_claimed_superior_to_hgb` | validation-selected winner=hgb_shallow15 (hgb); random forest is not the selected winner |
| WARN | `model_dependencies_unavailable` | not executed=['lightgbm', 'xgboost'] |
| PASS | `model_winner_has_oos_summary` | winner=hgb_shallow15, summary_rows=1 |
| PASS | `artifact:rf_hyperparameter_validation_summary/v1` | rows=11 |
| PASS | `artifact:rf_hyperparameter_protocol/v1` | rows=1 |
| PASS | `artifact:model_benchmark_validation_grid/v1` | rows=16 |
| PASS | `rf_stability_excludes_historical_oos` | evaluation_period=validation_2016_2020_only |
| PASS | `rf_stability_protocol_arithmetic` | neighbors=10, pass_fraction=70.0% |
| PASS | `rf_stability_reproduces_frozen_validation_baseline` | RankIC=0.098745, annual_return=66.20% |
| PASS | `rf_stability_does_not_promote_posthoc_winner` | local_robustness_pass=False |
| PASS | `artifact:rf_forensic_summary/v1` | rows=2 |
| PASS | `artifact:rf_forensic_holdings_summary/v1` | rows=1 |
| PASS | `artifact:rf_forensic_label_audit/v1` | rows=250 |
| PASS | `artifact:rf_oos_execution_summary/v1` | rows=2 |
| PASS | `artifact:rf_oos_execution_diagnostics/v1` | rows=1 |
| PASS | `rf_forensic_sampled_labels_rebuild_exactly` | rows=250, max_abs_error=0.000e+00 |
| PASS | `rf_validation_execution_bias_detected` | ideal=66.20%, constrained=19.49% |
| PASS | `rf_oos_execution_comparison_uses_common_periods` | ideal=293, constrained=293 |
| PASS | `rf_oos_execution_bias_detected` | ideal=28.17%, constrained=24.74% |
| PASS | `artifact:rf_daily_retry_daily/validation_v1` | rows=1,210 |
| PASS | `artifact:rf_daily_ideal_daily/validation_v1` | rows=1,210 |
| PASS | `artifact:rf_daily_retry_summary/validation_v1` | rows=1 |
| PASS | `artifact:rf_daily_ideal_summary/validation_v1` | rows=1 |
| PASS | `artifact:rf_daily_timing_audit/validation_v1` | rows=1 |
| PASS | `rf_daily_validation_accounting_identity` | max_abs_error=1.033e-14 |
| PASS | `rf_daily_validation_return_identity` | max_abs_error=7.565e-15 |
| PASS | `rf_daily_validation_no_hidden_daily_rebalance` | ideal_non_rebalance_max_turnover=0.000e+00 |
| PASS | `rf_daily_validation_execution_haircut_detected` | ideal=53.93%, constrained=10.83% |
| PASS | `rf_daily_validation_overlap_removed` | duplicated_horizon_days=61 |
| PASS | `artifact:rf_daily_retry_daily/historical_oos_v1` | rows=1,387 |
| PASS | `artifact:rf_daily_ideal_daily/historical_oos_v1` | rows=1,387 |
| PASS | `artifact:rf_daily_retry_summary/historical_oos_v1` | rows=1 |
| PASS | `artifact:rf_daily_ideal_summary/historical_oos_v1` | rows=1 |
| PASS | `artifact:rf_daily_timing_audit/historical_oos_v1` | rows=1 |
| PASS | `rf_daily_historical_oos_accounting_identity` | max_abs_error=9.992e-15 |
| PASS | `rf_daily_historical_oos_return_identity` | max_abs_error=6.502e-15 |
| PASS | `rf_daily_historical_oos_no_hidden_daily_rebalance` | ideal_non_rebalance_max_turnover=0.000e+00 |
| PASS | `rf_daily_historical_oos_execution_haircut_detected` | ideal=25.74%, constrained=22.02% |
| PASS | `rf_daily_historical_oos_overlap_removed` | duplicated_horizon_days=79 |
| PASS | `artifact:factor_registry/v1` | rows=15 |
| PASS | `artifact:factor_distribution_summary/v1` | rows=15 |
| PASS | `artifact:factor_rolling_correlation_stability/v1` | rows=105 |
| PASS | `artifact:hgb_hyperparameter_protocol/v1` | rows=1 |
| PASS | `artifact:hgb_hyperparameter_summary/v1` | rows=9 |
| PASS | `artifact:portfolio_sensitivity_protocol/v1` | rows=1 |
| PASS | `artifact:portfolio_sensitivity_summary/v1` | rows=16 |
| PASS | `artifact:execution_cost_stress/v1` | rows=16 |
| PASS | `factor_registry_complete_and_versioned` | factors=15, version=1.1.0 |
| PASS | `factor_visual_diagnostics_complete` | distributions=15, rolling_pairs=105, expected_pairs=105 |
| PASS | `hgb_surface_validation_only_and_complete` | surface_points=9, evaluation=2016_2020_validation_only |
| PASS | `portfolio_sensitivity_common_date_and_no_reselection` | variants={'historical_oos': 8, 'validation': 8}, performance_period_counts={'historical_oos': 1, 'validation': 1} |
| PASS | `execution_cost_stress_complete_and_monotone` | costs=[0.0, 20.0, 50.0, 100.0], monotone=True |
| PASS | `artifact:research_conclusion_decisions/v2` | rows=6 |
| PASS | `artifact:research_factor_multiplicity/v2` | rows=15 |
| PASS | `artifact:research_lifecycle_audit/v2` | rows=16 |
| PASS | `artifact:factor_registry/v1` | rows=15 |
| PASS | `conclusion_decision_hierarchy_complete` | verdicts={'neutralized_factor_signal_exists': 'SUPPORTED_AS_DIAGNOSTIC_EVIDENCE', 'hgb_adds_stable_increment_over_ridge': 'NOT_ESTABLISHED', 'ml_adds_value_over_style_matched_baseline': 'MIXED_LOCAL_EVIDENCE', 'executable_backtest_survives_costs': 'SUPPORTED_HISTORICALLY', 'pure_stock_selection_alpha_is_significant': 'NOT_SUPPORTED', 'ready_for_live_capital': 'NO'} |
| PASS | `single_factor_multiple_testing_controlled` | factors=15, bh_fdr_5pct_pass=14, maximum_q_error=0.000e+00 |
| PASS | `research_and_deployment_scope_separated` | historical research chain is complete; paper/live evidence remains a separate gate |
| PASS | `price_and_return_convention_explicit` | price_convention=baostock_unadjusted_ohlc_and_ex_right_reference_pct_chg |
| PASS | `final_report_states_decision_boundaries` | missing_phrases=[] |
| PASS | `artifact:research_governance_protocol/v1` | rows=1 |
| PASS | `artifact:research_evidence_registry/v1` | rows=5 |
| PASS | `artifact:research_candidate_freeze/v1` | rows=1 |
| PASS | `artifact:research_lockbox_registry/v1` | rows=1 |
| PASS | `artifact:research_paper_trading_registry/v1` | rows=1 |
| PASS | `artifact:research_hypothesis_registry/v1` | rows=4 |
| PASS | `artifact:research_governance_checks/v1` | rows=10 |
| PASS | `artifact:research_access_log_snapshot/v1` | rows=4 |
| PASS | `evidence_classes_complete` | classes=['future_lockbox', 'historical_oos', 'paper_trading', 'training', 'validation_selection'] |
| PASS | `historical_oos_consumed_and_not_selectable` | consumed=True, selection_allowed=False |
| PASS | `future_lockbox_is_sealed_and_unconsumed` | state=SEALED_EMPTY, consumed=False |
| PASS | `paper_tracking_is_forward_only` | state=FROZEN_NOT_STARTED, observations=0 |
| PASS | `candidate_freeze_is_commit_anchored` | commit=978aedf9615482e79cbca9670c458e8663b5ebcc, dirty=False |
| PASS | `governance_checks_all_pass` | checks=10, failures=0 |
| PASS | `future_hypothesis_family_frozen` | hypotheses=4 |
| PASS | `access_log_hash_chain_extends_frozen_snapshot` | live_events=6, frozen_prefix_events=4 |
| WARN | `known_scope_limitations` | Industry neutralization uses monthly-sampled historical CSRC broad sections rather than Shenwan membership; there is no official historical CSI membership and no broker-calibrated market-impact model. |

## Headline findings

- `registered_frozen_experiments`: 51
- `invalidated_experiments`: 23
- `common_oos_weeks`: 291
- `ridge_annualized_return_20bp`: 0.2005894880405501
- `challenger_annualized_return_20bp`: 0.2507876712519381
- `ridge_excess_vs_csi500_ci_low`: 0.05321262787587622
- `ridge_excess_vs_csi500_ci_high`: 0.3423690925447747
- `challenger_minus_ridge_ci_low`: 0.01190661886988064
- `challenger_minus_ridge_ci_high`: 0.0919165453739704
- `ridge_rank_ic_positive_every_oos_year`: True
- `label_validation_selected_target`: h20_raw_return
- `label_selected_oos_increment_ci_low`: -0.047140589968736234
- `label_selected_oos_increment_ci_high`: 0.05875119904009788
- `size_neutralized_oos_annualized_return`: 0.16231035204956568
- `size_neutralized_oos_mean_rank_ic`: 0.0944149822519997
- `size_neutralized_minus_raw_return_ci_low`: -0.08605184753945505
- `size_neutralized_minus_raw_return_ci_high`: -0.006663469172683896
- `pit_industry_minimum_coverage`: 0.9854873144982776
- `factor_industry_size_neutral_significantly_strengthened`:
- `factor_industry_size_neutral_significantly_weakened`: momentum_120d,momentum_20d,momentum_20d_skip_5d,momentum_60d
- `factor_tear_sheet_oos_weeks`: 293
- `factor_tear_sheet_significant_neutral_spreads`: momentum_20d,momentum_60d,momentum_120d,momentum_20d_skip_5d,reversal_5d,volatility_20d,amihud_20d,turnover_20d,turnover_surprise_20d,volume_surprise_20d,amplitude_20d,return_volume_corr_20d,max_return_20d,book_to_price
- `factor_tear_sheet_down_high_nonpositive_ic`: volatility_20d,amplitude_20d,earnings_yield,book_to_price
- `capm_csi500_significant_executable_strategies`: hgb_shallow15_locked/industry_size_neutralized,hgb_shallow15_locked/raw,ridge_locked/industry_size_neutralized,ridge_locked/raw
- `capm_csi500_executable_alpha_min`: 0.13596796819067866
- `capm_csi500_executable_alpha_max`: 0.17919035509947603
- `capm_negative_rolling_alpha_windows`: 10
- `style_factor_significant_residual_alpha_strategies`:
- `style_factor_executable_alpha_min`: -0.025461077104721716
- `style_factor_executable_alpha_max`: 0.010696115035796726
- `style_factor_executable_r_squared_min`: 0.894457709698638
- `style_factor_executable_r_squared_max`: 0.8976358104033699
- `ridge_combined_neutral_oos_rank_ic`: 0.08995183218412618
- `ridge_combined_neutral_oos_annualized_return`: 0.15539048523839227
- `model_neutralization_combined_intervals`: ridge_locked/rank_ic=[-0.014296,-0.002910]; ridge_locked/net_return_annualized_arithmetic=[-0.095274,-0.010141]; ridge_locked/industry_active_weight_l1=[0.003849,0.030062]; ridge_locked/abs_score_log_size_correlation=[-0.129148,-0.083662]; ridge_locked/score_industry_max_abs_mean=[-0.113562,-0.095326]; ridge_locked/portfolio_industry_hhi=[-0.008638,0.114750]; hgb_shallow15_locked/rank_ic=[-0.011770,-0.001434]; hgb_shallow15_locked/net_return_annualized_arithmetic=[-0.042655,0.030413]; hgb_shallow15_locked/industry_active_weight_l1=[-0.000598,0.027242]; hgb_shallow15_locked/abs_score_log_size_correlation=[-0.045098,-0.001908]; hgb_shallow15_locked/score_industry_max_abs_mean=[-0.118173,-0.090128]; hgb_shallow15_locked/portfolio_industry_hhi=[0.004733,0.104182]
- `constrained_hgb_raw_oos_annualized_return`: 0.24679778053823487
- `constrained_hgb_raw_oos_sharpe`: 0.9242068728336268
- `constrained_hgb_raw_oos_maximum_drawdown`: -0.328971034742949
- `constrained_hgb_raw_mean_industry_active_weight_l1`: 0.0949679158474287
- `constrained_portfolio_turnover_relaxation_fraction`: 0.0
- `constrained_daily_hgb_raw_annualized_return`: 0.20811666153329655
- `constrained_daily_hgb_raw_sharpe`: 0.9371647586448054
- `constrained_daily_hgb_raw_maximum_drawdown`: -0.3332492493261945
- `constrained_daily_hgb_raw_mean_blocked_buy_weight`: 0.004518262280589924
- `institutional_execution_hgb_raw_annualized_return`: 0.20664712035609334
- `institutional_execution_hgb_raw_sharpe`: 0.9322853955205596
- `institutional_execution_hgb_raw_maximum_drawdown`: -0.3330956293053998
- `institutional_execution_hgb_raw_fill_ratio`: 0.9883369584835313
- `unified_account_daily_rows`: 1387
- `unified_account_security_rows`: 296714
- `unified_account_order_exceptions`: 17495
- `unified_account_max_cash_error_cny`: 1.1851079761981964e-07
- `unified_account_max_nav_error_cny`: 1.1920928955078125e-07
- `unified_account_max_share_error`: 0.0
- `statistical_inference_hypotheses`: 37
- `statistical_trial_registry_rows`: 172
- `hgb_raw_incremental_return_point_estimate`: 0.02870154626718977
- `hgb_raw_incremental_return_ci_low`: -0.004378561383624404
- `hgb_raw_incremental_return_ci_high`: 0.06472945903992583
- `hgb_raw_incremental_return_fdr_q`: 0.10294852573713144
- `incremental_alpha_individual_intervals_above_zero`: 1
- `incremental_alpha_bh_fdr_survivors`: 0
- `hgb_neutral_incremental_return_annualized_arithmetic`: 0.03646786854268133
- `hgb_neutral_incremental_return_ci_low`: 0.0024438912915467014
- `hgb_neutral_incremental_return_ci_high`: 0.07164682883394921
- `ridge_raw_turnover_value_survives_multiplicity`: True
- `alpha_evidence_phase_status`: COMPLETE_SHARE_LEVEL_FILL_COMPARISON
- `alpha_evidence_regression_specifications`: 96
- `alpha_evidence_risk_adjusted_fdr_survivors`: 0
- `hgb_raw_risk_adjusted_incremental_alpha`: 0.0016893221338765405
- `hgb_neutral_risk_adjusted_incremental_alpha`: 0.01220379692104986
- `engineering_canonical_execution_market_module`: src/quant_research/execution_market.py
- `engineering_test_layers`: 5
- `engineering_open_p0_gaps`: 0
- `pit_beta_minimum_coverage`: 0.914180582483791
- `hgb_raw_annualized_style_contribution_arithmetic`: 0.13613172549903055
- `hgb_raw_annualized_specific_selection_contribution_arithmetic`: -0.004344077537185444
- `hgb_raw_specific_selection_ci_low`: -0.03518133583467788
- `hgb_raw_specific_selection_ci_high`: 0.030758079727737763
- `hgb_raw_mean_forecast_tracking_error`: 0.08072122788209943
- `risk_aware_v1_policy_adopted`: False
- `risk_aware_hgb_raw_te_before`: 0.08072122788209943
- `risk_aware_hgb_raw_te_after`: 0.04433561121082952
- `risk_aware_hgb_raw_annualized_return`: 0.1868838087174196
- `risk_aware_hgb_raw_return_delta_ci_high`: -0.00804427238394979
- `model_benchmark_validation_winner`: hgb_shallow15
- `model_winner_oos_mean_rank_ic`: 0.09993460606343071
- `model_winner_oos_annualized_return`: 0.2435850654086391
- `model_winner_minus_ridge_return_ci_low`: -0.008577511954826257
- `model_winner_minus_ridge_return_ci_high`: 0.07257600687652796
- `rf_neighbor_pass_fraction`: 0.7
- `rf_local_robustness_pass`: False
- `rf_stability_minimum_neighbor_rank_ic`: 0.0922194831499425
- `rf_stability_minimum_neighbor_sharpe`: 0.7901006347612033
- `rf_validation_idealized_annualized_return`: 0.6619536230943661
- `rf_validation_execution_constrained_annualized_return`: 0.19491845920942996
- `rf_oos_idealized_annualized_return`: 0.281734851058123
- `rf_oos_execution_constrained_annualized_return`: 0.2473717128945987
- `rf_oos_execution_constrained_sharpe`: 0.8883099540628371
- `rf_validation_mean_size_percentile`: 0.28736277013002354
- `rf_validation_mean_blocked_buy_weight`: 0.02451857583058863
- `rf_daily_validation_ideal_annualized_return`: 0.5392643588460468
- `rf_daily_validation_constrained_annualized_return`: 0.10830339578458403
- `rf_daily_oos_ideal_annualized_return`: 0.25741190201164055
- `rf_daily_oos_constrained_annualized_return`: 0.22019142129987523
- `rf_daily_oos_constrained_sharpe`: 0.8848300774679845
- `rf_daily_oos_constrained_maximum_drawdown`: -0.4298437083608838
- `hgb_validation_surface_rank_ic_min`: 0.09713976299008095
- `hgb_validation_surface_rank_ic_max`: 0.09980789470340727
- `factor_registry_count`: 15
- `neutral_factor_spreads_surviving_bh_fdr_5pct`: 14
- `research_backtest_chain_complete`: True
- `ready_for_live_capital`: False
- `future_lockbox_state`: SEALED_EMPTY
- `future_lockbox_consumed`: False
- `paper_trading_state`: FROZEN_NOT_STARTED
