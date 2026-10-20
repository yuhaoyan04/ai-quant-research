# Phase 2：人民币无风险利率与收益归因口径升级

## 研究边界

本实验只修正无风险利率与归因口径；信号、模型、组合、20bp基线成本、逐日成交规则和2021–2026 historical OOS 区间均未改变，也未据此重新选择模型或参数。R0旧结果保留，R1/R2作为描述性重算，不能被表述为全新独立样本证据。

## 数据选择与非等价候选

- R1：China Central Depository & Clearing Co., Ltd. (CCDC) 发布的中债国债收益率曲线3个月期限；这是主有来源利率基线。
- R2：同一曲线1年期限，仅作期限敏感性，避免跨数据源差异混入比较。
- 国债曲线是估值/到期收益率曲线，不是可直接按该收益率逐日无摩擦投资的现金账户。
- Shibor 是报价银行的人民币无担保批发性同业拆出利率，带银行信用和融资市场含义；央行逆回购/MLF 是政策工具利率。二者都不能与短期国债收益率直接视为同一对象。

## PIT 对齐与换算

- 原始单位：年度百分比报价；换算：`(1+annual_rate_decimal)^(calendar_days/365)-1`，日计数为 ACT/365，没有使用年利率/252近似。
- 历史接口未提供可靠的逐条盘中发布时间。本实验保守规定观察日后1个日历日才可用；每个收益区间在上一交易日仅向后查找当时已可用的最近观察，绝不未来填充。
- 风格因子是零净投资多空收益，不再减无风险利率；组合左侧和价格指数市场因子均减同一rf。
- CSI300/500/1000仍是价格指数收益、不含股息；因此相对可投资总收益基准的Alpha可能被高估。

## 覆盖审计

| scenario_id    |   dates | start_date   | end_date   |   missing_dates |   missing_rate |   interpolated_dates |   stale_dates |   maximum_staleness_calendar_days |   minimum_annual_rate_percent |   maximum_annual_rate_percent |
|:---------------|--------:|:-------------|:-----------|----------------:|---------------:|---------------------:|--------------:|----------------------------------:|------------------------------:|------------------------------:|
| R0_zero_rate   |    1387 | 2021-01-11   | 2026-09-29 |               0 |     0.00000000 |                    0 |             0 |                      nan          |                    0.00000000 |                    0.00000000 |
| R1_ccdc_cgb_3m |    1387 | 2021-01-11   | 2026-09-29 |               0 |     0.00000000 |                    0 |             0 |                       10.00000000 |                    0.78190000 |                    2.69990000 |
| R2_ccdc_cgb_1y |    1387 | 2021-01-11   | 2026-09-29 |               0 |     0.00000000 |                    0 |             0 |                       10.00000000 |                    0.93070000 |                    2.71550000 |

## Sharpe 与收益口径

`annualized_portfolio_return_geometric`是组合复合年化；`annualized_excess_return_arithmetic`是日超额收益均值乘252；Sharpe用组合超额收益计算。最大回撤来自原组合净收益，rf变化不会改变它。

| rate_scenario   | model                | alpha_pipeline            |   observations |   annualized_portfolio_return_geometric |   annualized_risk_free_return_geometric |   annualized_excess_return_arithmetic |   sharpe_excess_return |   maximum_drawdown |
|:----------------|:---------------------|:--------------------------|---------------:|----------------------------------------:|----------------------------------------:|--------------------------------------:|-----------------------:|-------------------:|
| R0_zero_rate    | hgb_shallow15_locked | industry_size_neutralized |           1387 |                                0.202598 |                                0.000000 |                              0.208682 |               0.952482 |          -0.315308 |
| R0_zero_rate    | hgb_shallow15_locked | raw                       |           1387 |                                0.208117 |                                0.000000 |                              0.215740 |               0.937165 |          -0.333249 |
| R0_zero_rate    | ridge_locked         | industry_size_neutralized |           1387 |                                0.158653 |                                0.000000 |                              0.170723 |               0.790291 |          -0.317149 |
| R0_zero_rate    | ridge_locked         | raw                       |           1387 |                                0.177959 |                                0.000000 |                              0.188076 |               0.855759 |          -0.329912 |
| R1_ccdc_cgb_3m  | hgb_shallow15_locked | industry_size_neutralized |           1387 |                                0.202598 |                                0.017210 |                              0.191618 |               0.874772 |          -0.315308 |
| R1_ccdc_cgb_3m  | hgb_shallow15_locked | raw                       |           1387 |                                0.208117 |                                0.017210 |                              0.198675 |               0.863182 |          -0.333249 |
| R1_ccdc_cgb_3m  | ridge_locked         | industry_size_neutralized |           1387 |                                0.158653 |                                0.017210 |                              0.153659 |               0.711437 |          -0.317149 |
| R1_ccdc_cgb_3m  | ridge_locked         | raw                       |           1387 |                                0.177959 |                                0.017210 |                              0.171012 |               0.778240 |          -0.329912 |
| R2_ccdc_cgb_1y  | hgb_shallow15_locked | industry_size_neutralized |           1387 |                                0.202598 |                                0.018776 |                              0.190080 |               0.867761 |          -0.315308 |
| R2_ccdc_cgb_1y  | hgb_shallow15_locked | raw                       |           1387 |                                0.208117 |                                0.018776 |                              0.197137 |               0.856505 |          -0.333249 |
| R2_ccdc_cgb_1y  | ridge_locked         | industry_size_neutralized |           1387 |                                0.158653 |                                0.018776 |                              0.152120 |               0.704318 |          -0.317149 |
| R2_ccdc_cgb_1y  | ridge_locked         | raw                       |           1387 |                                0.177959 |                                0.018776 |                              0.169473 |               0.771241 |          -0.329912 |

## CAPM（相对 CSI500）

日频含常数OLS，Newey–West(5) Bartlett HAC，Alpha按252日算术年化；每行明确报告样本数和95%置信区间。

| rate_scenario   | model                | alpha_pipeline            |   observations |   annualized_alpha_arithmetic |   annualized_alpha_ci_low |   annualized_alpha_ci_high |   alpha_hac_t_stat |     beta |   r_squared |
|:----------------|:---------------------|:--------------------------|---------------:|------------------------------:|--------------------------:|---------------------------:|-------------------:|---------:|------------:|
| R0_zero_rate    | hgb_shallow15_locked | industry_size_neutralized |           1387 |                      0.173638 |                  0.041211 |                   0.306065 |           2.569902 | 0.748846 |    0.557871 |
| R0_zero_rate    | hgb_shallow15_locked | raw                       |           1387 |                      0.179190 |                  0.039920 |                   0.318461 |           2.521760 | 0.781001 |    0.549645 |
| R0_zero_rate    | ridge_locked         | industry_size_neutralized |           1387 |                      0.135968 |                  0.003994 |                   0.267942 |           2.019275 | 0.742661 |    0.564389 |
| R0_zero_rate    | ridge_locked         | raw                       |           1387 |                      0.153175 |                  0.018378 |                   0.287973 |           2.227180 | 0.745778 |    0.549871 |
| R1_ccdc_cgb_3m  | hgb_shallow15_locked | industry_size_neutralized |           1387 |                      0.169355 |                  0.037125 |                   0.301586 |           2.510237 | 0.748746 |    0.557718 |
| R1_ccdc_cgb_3m  | hgb_shallow15_locked | raw                       |           1387 |                      0.175455 |                  0.036297 |                   0.314613 |           2.471193 | 0.780931 |    0.549505 |
| R1_ccdc_cgb_3m  | ridge_locked         | industry_size_neutralized |           1387 |                      0.131579 |                 -0.000261 |                   0.263419 |           1.956091 | 0.742571 |    0.564240 |
| R1_ccdc_cgb_3m  | ridge_locked         | raw                       |           1387 |                      0.148839 |                  0.014181 |                   0.283497 |           2.166375 | 0.745717 |    0.549732 |
| R2_ccdc_cgb_1y  | hgb_shallow15_locked | industry_size_neutralized |           1387 |                      0.168969 |                  0.036757 |                   0.301181 |           2.504870 | 0.748730 |    0.557712 |
| R2_ccdc_cgb_1y  | hgb_shallow15_locked | raw                       |           1387 |                      0.175119 |                  0.035971 |                   0.314266 |           2.466635 | 0.780921 |    0.549501 |
| R2_ccdc_cgb_1y  | ridge_locked         | industry_size_neutralized |           1387 |                      0.131183 |                 -0.000645 |                   0.263012 |           1.950372 | 0.742564 |    0.564238 |
| R2_ccdc_cgb_1y  | ridge_locked         | raw                       |           1387 |                      0.148448 |                  0.013802 |                   0.283094 |           2.160874 | 0.745714 |    0.549731 |

## 本地六风格归因

日频含常数回归；右侧为CSI500超额收益以及六个PIT自融资风格因子，推断和年化口径与CAPM一致。

| rate_scenario   | model                | alpha_pipeline            |   observations |   annualized_alpha_arithmetic |   annualized_alpha_ci_low |   annualized_alpha_ci_high |   alpha_hac_t_stat |   r_squared |
|:----------------|:---------------------|:--------------------------|---------------:|------------------------------:|--------------------------:|---------------------------:|-------------------:|------------:|
| R0_zero_rate    | hgb_shallow15_locked | industry_size_neutralized |           1387 |                      0.010696 |                 -0.056165 |                   0.077557 |           0.313546 |    0.897636 |
| R0_zero_rate    | hgb_shallow15_locked | raw                       |           1387 |                      0.010471 |                 -0.058252 |                   0.079193 |           0.298625 |    0.895399 |
| R0_zero_rate    | ridge_locked         | industry_size_neutralized |           1387 |                     -0.025461 |                 -0.095848 |                   0.044926 |          -0.708974 |    0.894458 |
| R0_zero_rate    | ridge_locked         | raw                       |           1387 |                     -0.017398 |                 -0.085627 |                   0.050830 |          -0.499792 |    0.896041 |
| R1_ccdc_cgb_3m  | hgb_shallow15_locked | industry_size_neutralized |           1387 |                      0.008522 |                 -0.058247 |                   0.075290 |           0.250152 |    0.897588 |
| R1_ccdc_cgb_3m  | hgb_shallow15_locked | raw                       |           1387 |                      0.008558 |                 -0.060092 |                   0.077208 |           0.244333 |    0.895355 |
| R1_ccdc_cgb_3m  | ridge_locked         | industry_size_neutralized |           1387 |                     -0.027691 |                 -0.098004 |                   0.042622 |          -0.771869 |    0.894411 |
| R1_ccdc_cgb_3m  | ridge_locked         | raw                       |           1387 |                     -0.019726 |                 -0.087869 |                   0.048417 |          -0.567362 |    0.895999 |
| R2_ccdc_cgb_1y  | hgb_shallow15_locked | industry_size_neutralized |           1387 |                      0.008329 |                 -0.058428 |                   0.075085 |           0.244531 |    0.897586 |
| R2_ccdc_cgb_1y  | hgb_shallow15_locked | raw                       |           1387 |                      0.008388 |                 -0.060253 |                   0.077030 |           0.239520 |    0.895354 |
| R2_ccdc_cgb_1y  | ridge_locked         | industry_size_neutralized |           1387 |                     -0.027889 |                 -0.098193 |                   0.042416 |          -0.777487 |    0.894410 |
| R2_ccdc_cgb_1y  | ridge_locked         | raw                       |           1387 |                     -0.019932 |                 -0.088065 |                   0.048200 |          -0.573393 |    0.895998 |

## R0回归核对

| check                              |   legacy_rows |   recomputed_rows |   numeric_columns_checked |   maximum_absolute_error |   tolerance | passed   |
|:-----------------------------------|--------------:|------------------:|--------------------------:|-------------------------:|------------:|:---------|
| capm_summary_v1                    |            24 |                24 |                        17 |                0.000e+00 |   1.000e-12 | True     |
| style_factor_regression_summary_v1 |             8 |                 8 |                        44 |                0.000e+00 |   1.000e-12 | True     |

## 结论与限制

更换rf只改变超额收益、Sharpe和回归截距定义，不改变组合财富或最大回撤，也不能据此宣称经济Alpha得到改善。主结论应比较R0/R1/R2的量级与统计区间稳定性。公开曲线缺少可交易现金工具的费用、滚动和再投资实现；指数又不含股息，故本结果是口径改进后的研究归因，不是实盘可复制的无风险套利账本。
