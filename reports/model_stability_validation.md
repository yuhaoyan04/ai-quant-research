# Phase 3：模型稳定性、随机种子与时间感知验证

## 结论边界

本实验不重新选模、不晋升参数。2016–2020 仍是选择/验证区间；2021–2026 仅作已被反复查看的 historical OOS 描述，不是 pristine lockbox。M5 允许测试年份之后的数据进入训练，只用于跨时期泛化诊断，不能解释为可交易历史业绩。

## M0–M6 完成度

| experiment   | status                    | evidence                                                        | interpretation                                                                  |
|:-------------|:--------------------------|:----------------------------------------------------------------|:--------------------------------------------------------------------------------|
| M0           | COMPLETE                  | model_benchmark_v1                                              | validation-selected frozen family baselines                                     |
| M1           | COMPLETE_WITH_INSTABILITY | hgb_hyperparameter_stability_v1; rf_hyperparameter_stability_v1 | RF neighbor pass fraction 0.700 vs floor 0.750; HGB RankIC range 0.0971..0.0998 |
| M2           | COMPLETE                  | three preregistered RF seeds                                    | mean/std/range reported; no best-seed selection                                 |
| M3           | COMPLETE                  | expanding vs rolling_10y                                        | paired validation and descriptive historical-OOS comparison                     |
| M4           | COMPLETE                  | annual RankIC and portfolio metrics                             | nonpositive RankIC/return years explicitly flagged                              |
| M5           | COMPLETE_SUPPLEMENTAL     | five label-interval-purged validation-year folds                | noncausal generalization diagnostic; not trading performance                    |
| M6           | COMPLETE_MIXED            | incremental_alpha_test_v1                                       | positive significant raw intervals=1/4; no model promotion                      |

## Expanding 与 10 年 Rolling

| split          | window_policy   | candidate            |   mean_rank_ic |   portfolio_annualized_return |   portfolio_sharpe_zero_rate |   portfolio_maximum_drawdown |   portfolio_mean_turnover |   portfolio_annualized_transaction_cost_arithmetic |
|:---------------|:----------------|:---------------------|---------------:|------------------------------:|-----------------------------:|-----------------------------:|--------------------------:|---------------------------------------------------:|
| historical_oos | expanding       | elastic_a0.01_l10.5  |       0.097461 |                      0.241284 |                     0.851428 |                    -0.383726 |                  0.535319 |                                           0.055673 |
| historical_oos | expanding       | hgb_shallow15        |       0.099493 |                      0.245421 |                     0.856885 |                    -0.406734 |                  0.660288 |                                           0.068670 |
| historical_oos | expanding       | naive_directional    |       0.097313 |                      0.178014 |                     0.756163 |                    -0.312127 |                  0.424562 |                                           0.044154 |
| historical_oos | expanding       | random_forest_depth6 |       0.100139 |                      0.281917 |                     0.955738 |                    -0.395570 |                  0.509783 |                                           0.053017 |
| historical_oos | expanding       | ridge_1              |       0.097752 |                      0.210174 |                     0.792536 |                    -0.368401 |                  0.560186 |                                           0.058259 |
| historical_oos | rolling_10y     | elastic_a0.01_l10.5  |       0.093891 |                      0.195015 |                     0.738259 |                    -0.384124 |                  0.506283 |                                           0.052653 |
| historical_oos | rolling_10y     | hgb_shallow15        |       0.094742 |                      0.170901 |                     0.678879 |                    -0.405624 |                  0.633749 |                                           0.065910 |
| historical_oos | rolling_10y     | naive_directional    |       0.097313 |                      0.178014 |                     0.756163 |                    -0.312127 |                  0.424562 |                                           0.044154 |
| historical_oos | rolling_10y     | random_forest_depth6 |       0.099636 |                      0.257316 |                     0.913973 |                    -0.377635 |                  0.510747 |                                           0.053118 |
| historical_oos | rolling_10y     | ridge_1              |       0.095586 |                      0.188859 |                     0.739942 |                    -0.368575 |                  0.551074 |                                           0.057312 |
| validation     | expanding       | elastic_a0.01_l10.5  |       0.091259 |                      0.272778 |                     1.016018 |                    -0.289781 |                  0.565031 |                                           0.058763 |
| validation     | expanding       | hgb_shallow15        |       0.104158 |                      1.243624 |                     2.914553 |                    -0.218233 |                  0.651855 |                                           0.067793 |
| validation     | expanding       | naive_directional    |       0.086252 |                      0.113462 |                     0.567208 |                    -0.314122 |                  0.436852 |                                           0.045433 |
| validation     | expanding       | random_forest_depth6 |       0.099900 |                      0.680865 |                     1.987001 |                    -0.260532 |                  0.580190 |                                           0.060340 |
| validation     | expanding       | ridge_1              |       0.092448 |                      0.590918 |                     1.860782 |                    -0.169960 |                  0.566201 |                                           0.058885 |
| validation     | rolling_10y     | elastic_a0.01_l10.5  |       0.091882 |                      0.345368 |                     1.217434 |                    -0.219706 |                  0.518342 |                                           0.053908 |
| validation     | rolling_10y     | hgb_shallow15        |       0.103717 |                      1.269618 |                     2.955436 |                    -0.202099 |                  0.615529 |                                           0.064015 |
| validation     | rolling_10y     | naive_directional    |       0.086252 |                      0.113462 |                     0.567208 |                    -0.314122 |                  0.436852 |                                           0.045433 |
| validation     | rolling_10y     | random_forest_depth6 |       0.099992 |                      0.784943 |                     2.196600 |                    -0.254190 |                  0.568915 |                                           0.059167 |
| validation     | rolling_10y     | ridge_1              |       0.092562 |                      0.633608 |                     1.946145 |                    -0.177889 |                  0.545751 |                                           0.056758 |

### 配对差异（rolling - expanding）

| split          | candidate            | metric                           | contrast                    |   observations |   point_estimate |   ci_2_5pct |   ci_97_5pct |   one_sided_probability_nonpositive |   block_length_weeks |
|:---------------|:---------------------|:---------------------------------|:----------------------------|---------------:|-----------------:|------------:|-------------:|------------------------------------:|---------------------:|
| validation     | elastic_a0.01_l10.5  | rank_ic                          | rolling_10y_minus_expanding |            254 |         0.000622 |   -0.000726 |     0.002129 |                            0.189405 |                   13 |
| validation     | elastic_a0.01_l10.5  | net_return_annualized_arithmetic | rolling_10y_minus_expanding |            254 |         0.055767 |    0.016990 |     0.103446 |                            0.002999 |                   13 |
| validation     | hgb_shallow15        | rank_ic                          | rolling_10y_minus_expanding |            254 |        -0.000441 |   -0.002563 |     0.001621 |                            0.663168 |                   13 |
| validation     | hgb_shallow15        | net_return_annualized_arithmetic | rolling_10y_minus_expanding |            254 |         0.011700 |   -0.014969 |     0.036148 |                            0.173913 |                   13 |
| validation     | naive_directional    | rank_ic                          | rolling_10y_minus_expanding |            254 |         0.000000 |    0.000000 |     0.000000 |                            1.000000 |                   13 |
| validation     | naive_directional    | net_return_annualized_arithmetic | rolling_10y_minus_expanding |            254 |         0.000000 |    0.000000 |     0.000000 |                            1.000000 |                   13 |
| validation     | random_forest_depth6 | rank_ic                          | rolling_10y_minus_expanding |            254 |         0.000092 |   -0.001247 |     0.001606 |                            0.440780 |                   13 |
| validation     | random_forest_depth6 | net_return_annualized_arithmetic | rolling_10y_minus_expanding |            254 |         0.060915 |    0.018045 |     0.107872 |                            0.001999 |                   13 |
| validation     | ridge_1              | rank_ic                          | rolling_10y_minus_expanding |            254 |         0.000114 |   -0.000636 |     0.000898 |                            0.367316 |                   13 |
| validation     | ridge_1              | net_return_annualized_arithmetic | rolling_10y_minus_expanding |            254 |         0.027274 |    0.003895 |     0.049796 |                            0.012494 |                   13 |
| historical_oos | elastic_a0.01_l10.5  | rank_ic                          | rolling_10y_minus_expanding |            293 |        -0.003571 |   -0.006758 |    -0.000816 |                            0.993003 |                   13 |
| historical_oos | elastic_a0.01_l10.5  | net_return_annualized_arithmetic | rolling_10y_minus_expanding |            291 |        -0.054763 |   -0.086090 |    -0.025400 |                            1.000000 |                   13 |
| historical_oos | hgb_shallow15        | rank_ic                          | rolling_10y_minus_expanding |            293 |        -0.004751 |   -0.008770 |    -0.000649 |                            0.989505 |                   13 |
| historical_oos | hgb_shallow15        | net_return_annualized_arithmetic | rolling_10y_minus_expanding |            292 |        -0.062243 |   -0.111731 |    -0.019250 |                            0.999000 |                   13 |
| historical_oos | naive_directional    | rank_ic                          | rolling_10y_minus_expanding |            293 |         0.000000 |    0.000000 |     0.000000 |                            1.000000 |                   13 |
| historical_oos | naive_directional    | net_return_annualized_arithmetic | rolling_10y_minus_expanding |            293 |         0.000000 |    0.000000 |     0.000000 |                            1.000000 |                   13 |
| historical_oos | random_forest_depth6 | rank_ic                          | rolling_10y_minus_expanding |            293 |        -0.000503 |   -0.003479 |     0.002269 |                            0.653673 |                   13 |
| historical_oos | random_forest_depth6 | net_return_annualized_arithmetic | rolling_10y_minus_expanding |            293 |        -0.022939 |   -0.049797 |     0.001631 |                            0.961019 |                   13 |
| historical_oos | ridge_1              | rank_ic                          | rolling_10y_minus_expanding |            293 |        -0.002166 |   -0.003859 |    -0.000711 |                            0.998501 |                   13 |
| historical_oos | ridge_1              | net_return_annualized_arithmetic | rolling_10y_minus_expanding |            293 |        -0.019261 |   -0.037577 |     0.000406 |                            0.972514 |                   13 |

### 失败年份计数

| split          | window_policy   | candidate            |   years |   rank_ic_failure_years |   net_return_failure_years |
|:---------------|:----------------|:---------------------|--------:|------------------------:|---------------------------:|
| historical_oos | expanding       | elastic_a0.01_l10.5  |       6 |                       0 |                          0 |
| historical_oos | expanding       | hgb_shallow15        |       6 |                       0 |                          1 |
| historical_oos | expanding       | naive_directional    |       6 |                       0 |                          0 |
| historical_oos | expanding       | random_forest_depth6 |       6 |                       0 |                          1 |
| historical_oos | expanding       | ridge_1              |       6 |                       0 |                          1 |
| historical_oos | rolling_10y     | elastic_a0.01_l10.5  |       6 |                       0 |                          1 |
| historical_oos | rolling_10y     | hgb_shallow15        |       6 |                       0 |                          1 |
| historical_oos | rolling_10y     | naive_directional    |       6 |                       0 |                          0 |
| historical_oos | rolling_10y     | random_forest_depth6 |       6 |                       0 |                          1 |
| historical_oos | rolling_10y     | ridge_1              |       6 |                       0 |                          1 |
| validation     | expanding       | elastic_a0.01_l10.5  |       5 |                       0 |                          2 |
| validation     | expanding       | hgb_shallow15        |       5 |                       0 |                          0 |
| validation     | expanding       | naive_directional    |       5 |                       0 |                          1 |
| validation     | expanding       | random_forest_depth6 |       5 |                       0 |                          0 |
| validation     | expanding       | ridge_1              |       5 |                       0 |                          0 |
| validation     | rolling_10y     | elastic_a0.01_l10.5  |       5 |                       0 |                          1 |
| validation     | rolling_10y     | hgb_shallow15        |       5 |                       0 |                          0 |
| validation     | rolling_10y     | naive_directional    |       5 |                       0 |                          1 |
| validation     | rolling_10y     | random_forest_depth6 |       5 |                       0 |                          0 |
| validation     | rolling_10y     | ridge_1              |       5 |                       0 |                          0 |

## RF 多随机种子

| model                | metric                      |   seed_count |      mean |   standard_deviation |   minimum |   maximum | best_seed_selected   |
|:---------------------|:----------------------------|-------------:|----------:|---------------------:|----------:|----------:|:---------------------|
| random_forest_depth6 | mean_rank_ic                |            3 |  0.098785 |             0.000168 |  0.098640 |  0.098968 | False                |
| random_forest_depth6 | portfolio_annualized_return |            3 |  0.664512 |             0.003618 |  0.661954 |  0.668652 | False                |
| random_forest_depth6 | portfolio_sharpe_zero_rate  |            3 |  1.931502 |             0.005950 |  1.925207 |  1.937034 | False                |
| random_forest_depth6 | portfolio_maximum_drawdown  |            3 | -0.256469 |             0.005784 | -0.262690 | -0.251254 | False                |

## 参数邻域

| family        |   candidate_count |   mean_rank_ic_minimum |   mean_rank_ic_maximum |   mean_rank_ic_range |   portfolio_sharpe_minimum |   portfolio_sharpe_maximum | local_robustness_pass   | interpretation                                                  |
|:--------------|------------------:|-----------------------:|-----------------------:|---------------------:|---------------------------:|---------------------------:|:------------------------|:----------------------------------------------------------------|
| hgb           |                 9 |               0.097140 |               0.099808 |             0.002668 |                   1.827321 |                   2.796013 |                         | RankIC surface narrow; portfolio outcomes materially wider      |
| random_forest |                11 |               0.092219 |               0.100011 |             0.007791 |                   0.790101 |                   2.118576 | False                   | predeclared neighbor robustness floor failed; no point promoted |

## Purged CV（补充、非交易模拟）

| candidate            |   mean_rank_ic |   portfolio_annualized_return |   portfolio_sharpe_zero_rate |   portfolio_maximum_drawdown |   portfolio_mean_turnover |   portfolio_annualized_transaction_cost_arithmetic |
|:---------------------|---------------:|------------------------------:|-----------------------------:|-----------------------------:|--------------------------:|---------------------------------------------------:|
| elastic_a0.01_l10.5  |       0.091357 |                      0.230523 |                     0.904535 |                    -0.334214 |                  0.586832 |                                           0.061030 |
| hgb_shallow15        |       0.104550 |                      1.483224 |                     3.203715 |                    -0.224752 |                  0.654712 |                                           0.068090 |
| naive_directional    |       0.086252 |                      0.112402 |                     0.563160 |                    -0.314796 |                  0.445856 |                                           0.046369 |
| random_forest_depth6 |       0.101442 |                      0.642893 |                     1.903210 |                    -0.266300 |                  0.579698 |                                           0.060289 |
| ridge_1              |       0.093579 |                      0.706965 |                     2.158025 |                    -0.225753 |                  0.600910 |                                           0.062495 |

## 日期与横截面对齐审计

| split                  | window_policy   |   test_year |   candidate_count |   evaluation_dates | all_candidates_share_dates   | all_candidates_share_cross_section_size   | all_candidates_share_portfolio_dates   |
|:-----------------------|:----------------|------------:|------------------:|-------------------:|:-----------------------------|:------------------------------------------|:---------------------------------------|
| historical_oos         | expanding       |        2021 |                 5 |                 52 | True                         | True                                      | True                                   |
| historical_oos         | expanding       |        2022 |                 5 |                 50 | True                         | True                                      | True                                   |
| historical_oos         | expanding       |        2023 |                 5 |                 50 | True                         | True                                      | True                                   |
| historical_oos         | expanding       |        2024 |                 5 |                 52 | True                         | True                                      | True                                   |
| historical_oos         | expanding       |        2025 |                 5 |                 53 | True                         | True                                      | True                                   |
| historical_oos         | expanding       |        2026 |                 5 |                 36 | True                         | True                                      | True                                   |
| historical_oos         | rolling_10y     |        2021 |                 5 |                 52 | True                         | True                                      | True                                   |
| historical_oos         | rolling_10y     |        2022 |                 5 |                 50 | True                         | True                                      | True                                   |
| historical_oos         | rolling_10y     |        2023 |                 5 |                 50 | True                         | True                                      | True                                   |
| historical_oos         | rolling_10y     |        2024 |                 5 |                 52 | True                         | True                                      | True                                   |
| historical_oos         | rolling_10y     |        2025 |                 5 |                 53 | True                         | True                                      | True                                   |
| historical_oos         | rolling_10y     |        2026 |                 5 |                 36 | True                         | True                                      | True                                   |
| validation             | expanding       |        2016 |                 5 |                 50 | True                         | True                                      | True                                   |
| validation             | expanding       |        2017 |                 5 |                 51 | True                         | True                                      | True                                   |
| validation             | expanding       |        2018 |                 5 |                 51 | True                         | True                                      | True                                   |
| validation             | expanding       |        2019 |                 5 |                 52 | True                         | True                                      | True                                   |
| validation             | expanding       |        2020 |                 5 |                 50 | True                         | True                                      | True                                   |
| validation             | rolling_10y     |        2016 |                 5 |                 50 | True                         | True                                      | True                                   |
| validation             | rolling_10y     |        2017 |                 5 |                 51 | True                         | True                                      | True                                   |
| validation             | rolling_10y     |        2018 |                 5 |                 51 | True                         | True                                      | True                                   |
| validation             | rolling_10y     |        2019 |                 5 |                 52 | True                         | True                                      | True                                   |
| validation             | rolling_10y     |        2020 |                 5 |                 50 | True                         | True                                      | True                                   |
| purged_cv_supplemental | nan             |        2016 |                 5 |                 50 | True                         | True                                      | True                                   |
| purged_cv_supplemental | nan             |        2017 |                 5 |                 51 | True                         | True                                      | True                                   |
| purged_cv_supplemental | nan             |        2018 |                 5 |                 51 | True                         | True                                      | True                                   |
| purged_cv_supplemental | nan             |        2019 |                 5 |                 52 | True                         | True                                      | True                                   |
| purged_cv_supplemental | nan             |        2020 |                 5 |                 50 | True                         | True                                      | True                                   |

## 审计说明

每次 fit 的训练区间、标签成熟日期、参数、seed、依赖版本及预测输出 SHA-256 均写入 fit audit。逐日预测分数分布、逐年 RankIC/净收益/Sharpe/回撤/换手和成本均保留在数据湖。任何不稳定结果均原样报告，未以新结果改写冻结候选。