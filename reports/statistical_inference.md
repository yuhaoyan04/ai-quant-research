# Phase 8：统一统计推断与多重检验

## 证据边界

本阶段在已查看的 2021–2026 historical OOS 上统一推断口径，属于 post-historical-OOS adaptation；不选择新模型、因子、块长或组合方法，也不提供未来盈利概率。

## 检验族

| family_id                        |   hypotheses |   individually_positive |   survives_family_bh_fdr |   economically_material_positive |   minimum_q_value |
|:---------------------------------|-------------:|------------------------:|-------------------------:|---------------------------------:|------------------:|
| factor_gross_spread              |           15 |                      14 |                       14 |                               15 |          0.000937 |
| factor_rank_ic                   |           15 |                      15 |                       15 |                               15 |          0.000500 |
| model_incremental_alpha          |            4 |                       1 |                        0 |                                4 |          0.071964 |
| portfolio_construction_increment |            3 |                       0 |                        0 |                                0 |          1.000000 |

BH-FDR 只在同一研究问题内部执行，因子 RankIC、因子多空毛收益、模型增量和组合方法增量绝不混池。

## 主模型增量 Alpha

raw HGB 相对风格复制基准的年化算术收益差为 2.87%，主 20 日块长 95% 区间 [-0.44%, 6.47%]，族内 BH-FDR q=0.1029。结论：positive_value_not_established_after_family_fdr。

## Block length 敏感性

| family_id                        | hypothesis_id                                   |   point_estimate |   minimum_ci_low |   maximum_ci_low |   minimum_ci_high |   maximum_ci_high |   minimum_p_value |   maximum_p_value | all_block_intervals_positive   | all_block_intervals_negative   |
|:---------------------------------|:------------------------------------------------|-----------------:|-----------------:|-----------------:|------------------:|------------------:|------------------:|------------------:|:-------------------------------|:-------------------------------|
| factor_gross_spread              | amihud_20d                                      |         0.218797 |         0.107086 |         0.133527 |          0.297141 |          0.325663 |          0.000500 |          0.000500 | True                           | False                          |
| factor_gross_spread              | amplitude_20d                                   |         0.233230 |        -0.012447 |         0.037499 |          0.410253 |          0.495552 |          0.008496 |          0.035982 | False                          | False                          |
| factor_gross_spread              | book_to_price                                   |         0.139782 |        -0.005822 |         0.033152 |          0.250201 |          0.289706 |          0.005997 |          0.032484 | False                          | False                          |
| factor_gross_spread              | earnings_yield                                  |         0.056333 |        -0.106216 |        -0.069843 |          0.189542 |          0.214490 |          0.198401 |          0.258871 | False                          | False                          |
| factor_gross_spread              | max_return_20d                                  |         0.269462 |         0.091400 |         0.123880 |          0.385460 |          0.453864 |          0.000500 |          0.002499 | True                           | False                          |
| factor_gross_spread              | momentum_120d                                   |         0.283298 |         0.072156 |         0.106174 |          0.436567 |          0.518174 |          0.002499 |          0.004998 | True                           | False                          |
| factor_gross_spread              | momentum_20d                                    |         0.447265 |         0.223058 |         0.243221 |          0.638456 |          0.672292 |          0.000500 |          0.000500 | True                           | False                          |
| factor_gross_spread              | momentum_20d_skip_5d                            |         0.213274 |         0.013673 |         0.052416 |          0.383971 |          0.414554 |          0.008496 |          0.019490 | True                           | False                          |
| factor_gross_spread              | momentum_60d                                    |         0.378001 |         0.156872 |         0.212879 |          0.553443 |          0.608254 |          0.000500 |          0.001499 | True                           | False                          |
| factor_gross_spread              | return_volume_corr_20d                          |         0.167281 |         0.068636 |         0.080194 |          0.254812 |          0.267001 |          0.000500 |          0.001000 | True                           | False                          |
| factor_gross_spread              | reversal_5d                                     |         0.378651 |         0.235566 |         0.254238 |          0.516240 |          0.537708 |          0.000500 |          0.000500 | True                           | False                          |
| factor_gross_spread              | turnover_20d                                    |         0.398261 |         0.203756 |         0.247214 |          0.538753 |          0.612718 |          0.000500 |          0.000500 | True                           | False                          |
| factor_gross_spread              | turnover_surprise_20d                           |         0.265054 |         0.158262 |         0.175969 |          0.345656 |          0.375073 |          0.000500 |          0.000500 | True                           | False                          |
| factor_gross_spread              | volatility_20d                                  |         0.271293 |         0.051460 |         0.116678 |          0.408235 |          0.509784 |          0.000500 |          0.009995 | True                           | False                          |
| factor_gross_spread              | volume_surprise_20d                             |         0.264373 |         0.156851 |         0.182006 |          0.349963 |          0.373465 |          0.000500 |          0.000500 | True                           | False                          |
| factor_rank_ic                   | amihud_20d                                      |         0.038067 |         0.026584 |         0.029273 |          0.046303 |          0.049489 |          0.000500 |          0.000500 | True                           | False                          |
| factor_rank_ic                   | amplitude_20d                                   |         0.072623 |         0.051140 |         0.055615 |          0.087657 |          0.093132 |          0.000500 |          0.000500 | True                           | False                          |
| factor_rank_ic                   | book_to_price                                   |         0.048028 |         0.030056 |         0.034604 |          0.060396 |          0.065772 |          0.000500 |          0.000500 | True                           | False                          |
| factor_rank_ic                   | earnings_yield                                  |         0.028815 |         0.014540 |         0.017817 |          0.039708 |          0.043234 |          0.000500 |          0.000500 | True                           | False                          |
| factor_rank_ic                   | max_return_20d                                  |         0.070027 |         0.054692 |         0.057085 |          0.081895 |          0.085814 |          0.000500 |          0.000500 | True                           | False                          |
| factor_rank_ic                   | momentum_120d                                   |         0.038822 |         0.023470 |         0.025184 |          0.052511 |          0.054791 |          0.000500 |          0.000500 | True                           | False                          |
| factor_rank_ic                   | momentum_20d                                    |         0.053029 |         0.037090 |         0.038287 |          0.067778 |          0.069337 |          0.000500 |          0.000500 | True                           | False                          |
| factor_rank_ic                   | momentum_20d_skip_5d                            |         0.034252 |         0.020220 |         0.020868 |          0.046423 |          0.048873 |          0.000500 |          0.000500 | True                           | False                          |
| factor_rank_ic                   | momentum_60d                                    |         0.050491 |         0.035275 |         0.039953 |          0.061672 |          0.065972 |          0.000500 |          0.000500 | True                           | False                          |
| factor_rank_ic                   | return_volume_corr_20d                          |         0.044749 |         0.035203 |         0.036875 |          0.052816 |          0.054808 |          0.000500 |          0.000500 | True                           | False                          |
| factor_rank_ic                   | reversal_5d                                     |         0.039102 |         0.027010 |         0.027810 |          0.051541 |          0.053310 |          0.000500 |          0.000500 | True                           | False                          |
| factor_rank_ic                   | turnover_20d                                    |         0.072457 |         0.055161 |         0.057771 |          0.086557 |          0.089491 |          0.000500 |          0.000500 | True                           | False                          |
| factor_rank_ic                   | turnover_surprise_20d                           |         0.030942 |         0.021330 |         0.022986 |          0.038842 |          0.041084 |          0.000500 |          0.000500 | True                           | False                          |
| factor_rank_ic                   | volatility_20d                                  |         0.071271 |         0.050905 |         0.054987 |          0.084075 |          0.090692 |          0.000500 |          0.000500 | True                           | False                          |
| factor_rank_ic                   | volume_surprise_20d                             |         0.031285 |         0.021784 |         0.023547 |          0.039446 |          0.041431 |          0.000500 |          0.000500 | True                           | False                          |
| model_incremental_alpha          | hgb_shallow15_locked__industry_size_neutralized |         0.036468 |        -0.000395 |         0.003329 |          0.070281 |          0.073806 |          0.014493 |          0.028486 | False                          | False                          |
| model_incremental_alpha          | hgb_shallow15_locked__raw                       |         0.028702 |        -0.009003 |        -0.004379 |          0.063686 |          0.064978 |          0.047976 |          0.067466 | False                          | False                          |
| model_incremental_alpha          | ridge_locked__industry_size_neutralized         |         0.020542 |        -0.021856 |        -0.015747 |          0.057633 |          0.062376 |          0.145427 |          0.171414 | False                          | False                          |
| model_incremental_alpha          | ridge_locked__raw                               |         0.014905 |        -0.019666 |        -0.017733 |          0.046885 |          0.052504 |          0.186907 |          0.204898 | False                          | False                          |
| portfolio_construction_increment | P1_inverse_volatility                           |        -0.004540 |        -0.014676 |        -0.012900 |          0.004154 |          0.006332 |          0.807596 |          0.844078 | False                          | False                          |
| portfolio_construction_increment | P2_score_weighted                               |        -0.020829 |        -0.031658 |        -0.030805 |         -0.010838 |         -0.009896 |          0.999500 |          1.000000 | False                          | True                           |
| portfolio_construction_increment | P4_constrained                                  |        -0.038897 |        -0.095063 |        -0.088924 |          0.016097 |          0.021213 |          0.900050 |          0.920540 | False                          | False                          |

## 完整试验台账

| trial_category          |   trials |
|:------------------------|---------:|
| factor_family           |       15 |
| hgb_hyperparameter_grid |        9 |
| incremental_alpha       |        4 |
| label_grid              |       30 |
| model_family_grid       |       16 |
| portfolio_construction  |        4 |
| portfolio_sensitivity   |       16 |
| preprocessing_grid      |       45 |
| rf_hyperparameter_grid  |       11 |
| risk_aware_portfolio    |       12 |
| transaction_cost        |       10 |

## 机器验收

| check_name                                | status   | observed                                                                                                                                                                                                                                                                                                                                                         |
|:------------------------------------------|:---------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| raw_series_keys_unique                    | PASS     | 18499                                                                                                                                                                                                                                                                                                                                                            |
| paired_daily_clocks_complete              | PASS     | 1387                                                                                                                                                                                                                                                                                                                                                             |
| families_separate_and_complete            | PASS     | {'factor_gross_spread': 15, 'factor_rank_ic': 15, 'model_incremental_alpha': 4, 'portfolio_construction_increment': 3}                                                                                                                                                                                                                                           |
| block_length_sensitivity_complete         | PASS     | [np.int64(4), np.int64(5), np.int64(8), np.int64(10), np.int64(13), np.int64(20), np.int64(26), np.int64(40), np.int64(60)]                                                                                                                                                                                                                                      |
| hac_lag_sensitivity_complete              | PASS     | [np.int64(1), np.int64(4), np.int64(5), np.int64(8), np.int64(10), np.int64(13), np.int64(20), np.int64(40)]                                                                                                                                                                                                                                                     |
| p_values_and_q_values_valid               | PASS     | 37                                                                                                                                                                                                                                                                                                                                                               |
| point_estimates_recompute_from_raw        | PASS     | 37                                                                                                                                                                                                                                                                                                                                                               |
| trial_registry_covers_major_search_space  | PASS     | {'factor_family': 15, 'hgb_hyperparameter_grid': 9, 'incremental_alpha': 4, 'label_grid': 30, 'model_family_grid': 16, 'portfolio_construction': 4, 'portfolio_sensitivity': 16, 'preprocessing_grid': 45, 'rf_hyperparameter_grid': 11, 'risk_aware_portfolio': 12, 'transaction_cost': 10}                                                                     |
| primary_incremental_alpha_not_overclaimed | PASS     | [{'point_estimate': 0.02870154626718977, 'ci_low': -0.004378561383624404, 'ci_high': 0.06472945903992583, 'bh_fdr_adjusted_p': 0.10294852573713144, 'claim_status': 'positive_value_not_established_after_family_fdr'}]                                                                                                                                          |
| historical_oos_not_claimed_independent    | PASS     | Phase 8 harmonizes inference on already observed historical-OOS results. It does not select a new model, factor, block length or portfolio method. HAC and circular-block results are diagnostics under explicit dependence assumptions, not probabilities of future profitability. Families are kept separate because they answer different research questions. |

> 统计显著不自动等于可交易 Alpha；经济阈值、成本、换手、容量与未来 lockbox/paper evidence 仍是独立门禁。