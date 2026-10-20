# 价格指数与全收益指数双口径归因

## 研究边界

本实验仅替换市场基准收益定义。策略、持仓、成交、成本、无风险利率、风格因子和日期均不变；2021–2026仍是已查看的historical OOS，没有据此重新选择模型。旧BaoStock价格指数结果完整保留。

## 数据与可比性设计

- `legacy_baostock_price`：原价格指数序列，只用于历史结果零漂移核对。
- `official_csindex_price`：中证指数有限公司官网价格指数收盘点位计算的收益。
- `official_csindex_total_return`：同一官网全收益指数收盘点位计算的收益；按中证规则计入税前现金红利再投资。
- 股息效应只比较后两个同源序列；不能把BaoStock与CSIndex之间的供应商/精度差异解释成股息。
- 官方接口的`changePct`仅保留两位百分数。本实验用收盘点位重算收益，并把该字段作为舍入容差内的交叉核对。

## 覆盖

| benchmark   | variant_id                    | return_type   | index_code   | source_provider                        |   dates | start_date   | end_date   |   missing_returns |   interpolated_dates |
|:------------|:------------------------------|:--------------|:-------------|:---------------------------------------|--------:|:-------------|:-----------|------------------:|---------------------:|
| CSI1000     | legacy_baostock_price         | price_return  | sh.000852    | BaoStock                               |    1387 | 2021-01-11   | 2026-09-29 |                 0 |                    0 |
| CSI1000     | official_csindex_price        | price_return  | 000852       | China Securities Index Co., Ltd. (CSI) |    1387 | 2021-01-11   | 2026-09-29 |                 0 |                    0 |
| CSI1000     | official_csindex_total_return | total_return  | H00852       | China Securities Index Co., Ltd. (CSI) |    1387 | 2021-01-11   | 2026-09-29 |                 0 |                    0 |
| CSI300      | legacy_baostock_price         | price_return  | sh.000300    | BaoStock                               |    1387 | 2021-01-11   | 2026-09-29 |                 0 |                    0 |
| CSI300      | official_csindex_price        | price_return  | 000300       | China Securities Index Co., Ltd. (CSI) |    1387 | 2021-01-11   | 2026-09-29 |                 0 |                    0 |
| CSI300      | official_csindex_total_return | total_return  | H00300       | China Securities Index Co., Ltd. (CSI) |    1387 | 2021-01-11   | 2026-09-29 |                 0 |                    0 |
| CSI500      | legacy_baostock_price         | price_return  | sh.000905    | BaoStock                               |    1387 | 2021-01-11   | 2026-09-29 |                 0 |                    0 |
| CSI500      | official_csindex_price        | price_return  | 000905       | China Securities Index Co., Ltd. (CSI) |    1387 | 2021-01-11   | 2026-09-29 |                 0 |                    0 |
| CSI500      | official_csindex_total_return | total_return  | H00905       | China Securities Index Co., Ltd. (CSI) |    1387 | 2021-01-11   | 2026-09-29 |                 0 |                    0 |

## 官方价格与全收益差异

| benchmark   |   observations |   official_price_annualized_return_geometric |   official_total_return_annualized_return_geometric |   annualized_arithmetic_total_minus_price |   annualized_geometric_total_relative_to_price |   maximum_abs_daily_total_minus_price |   material_daily_difference_threshold |   days_abs_total_minus_price_above_threshold |
|:------------|---------------:|---------------------------------------------:|----------------------------------------------------:|------------------------------------------:|-----------------------------------------------:|--------------------------------------:|--------------------------------------:|---------------------------------------------:|
| CSI1000     |           1387 |                                     0.019537 |                                            0.031751 |                                  0.011916 |                                       0.011981 |                              0.000692 |                              0.000100 |                                          228 |
| CSI300      |           1387 |                                    -0.041770 |                                           -0.016669 |                                  0.025852 |                                       0.026195 |                              0.002361 |                              0.000100 |                                          324 |
| CSI500      |           1387 |                                     0.023193 |                                            0.041041 |                                  0.017299 |                                       0.017444 |                              0.000987 |                              0.000100 |                                          280 |

## 供应商与收益字段核对

| benchmark   |   observations |   official_price_vs_legacy_correlation |   official_price_vs_legacy_mean_error |   official_price_vs_legacy_mean_abs_error |   official_price_vs_legacy_max_abs_error |   official_close_return_vs_published_change_pct_max_abs_error |   published_change_pct_rounding_tolerance | published_change_pct_crosscheck_passed   |
|:------------|---------------:|---------------------------------------:|--------------------------------------:|------------------------------------------:|-----------------------------------------:|--------------------------------------------------------------:|------------------------------------------:|:-----------------------------------------|
| CSI1000     |           1387 |                             1.00000000 |                           -0.00000000 |                                0.00000055 |                               0.00000196 |                                                    0.00005119 |                                0.00006000 | True                                     |
| CSI300      |           1387 |                             1.00000000 |                            0.00000000 |                                0.00000081 |                               0.00000277 |                                                    0.00005157 |                                0.00006000 | True                                     |
| CSI500      |           1387 |                             1.00000000 |                           -0.00000000 |                                0.00000054 |                               0.00000182 |                                                    0.00005105 |                                0.00006000 | True                                     |

## 相对 CSI500 的收益效率（R1 3M国债rf）

算术超额收益为策略减基准的日收益均值乘252；几何相对收益按两条财富曲线之比年化，二者不混用。

| variant_id                    | return_type   | model                | alpha_pipeline            |   observations |   benchmark_annualized_return_geometric |   strategy_annualized_return_geometric |   annualized_arithmetic_excess |   annualized_geometric_relative_return |   information_ratio |
|:------------------------------|:--------------|:---------------------|:--------------------------|---------------:|----------------------------------------:|---------------------------------------:|-------------------------------:|---------------------------------------:|--------------------:|
| legacy_baostock_price         | price_return  | hgb_shallow15_locked | industry_size_neutralized |           1387 |                                0.023193 |                               0.202598 |                       0.161884 |                               0.175338 |            1.039877 |
| official_csindex_price        | price_return  | hgb_shallow15_locked | industry_size_neutralized |           1387 |                                0.023193 |                               0.202598 |                       0.161885 |                               0.175338 |            1.039877 |
| official_csindex_total_return | total_return  | hgb_shallow15_locked | industry_size_neutralized |           1387 |                                0.041041 |                               0.202598 |                       0.144586 |                               0.155187 |            0.928756 |
| legacy_baostock_price         | price_return  | hgb_shallow15_locked | raw                       |           1387 |                                0.023193 |                               0.208117 |                       0.168942 |                               0.180732 |            1.044594 |
| official_csindex_price        | price_return  | hgb_shallow15_locked | raw                       |           1387 |                                0.023193 |                               0.208117 |                       0.168942 |                               0.180732 |            1.044594 |
| official_csindex_total_return | total_return  | hgb_shallow15_locked | raw                       |           1387 |                                0.041041 |                               0.208117 |                       0.151643 |                               0.160489 |            0.937712 |
| legacy_baostock_price         | price_return  | ridge_locked         | industry_size_neutralized |           1387 |                                0.023193 |                               0.158653 |                       0.123925 |                               0.132389 |            0.808552 |
| official_csindex_price        | price_return  | ridge_locked         | industry_size_neutralized |           1387 |                                0.023193 |                               0.158653 |                       0.123925 |                               0.132389 |            0.808553 |
| official_csindex_total_return | total_return  | ridge_locked         | industry_size_neutralized |           1387 |                                0.041041 |                               0.158653 |                       0.106626 |                               0.112975 |            0.695564 |
| legacy_baostock_price         | price_return  | ridge_locked         | raw                       |           1387 |                                0.023193 |                               0.177959 |                       0.141278 |                               0.151257 |            0.896606 |
| official_csindex_price        | price_return  | ridge_locked         | raw                       |           1387 |                                0.023193 |                               0.177959 |                       0.141279 |                               0.151257 |            0.896607 |
| official_csindex_total_return | total_return  | ridge_locked         | raw                       |           1387 |                                0.041041 |                               0.177959 |                       0.123980 |                               0.131519 |            0.786799 |

## CAPM（R1 3M国债rf，CSI500）

日频含常数OLS；左右两侧均减同一rf；Newey–West(5) Bartlett HAC；Alpha按252日算术年化。

| variant_id                    | return_type   | model                | alpha_pipeline            |   observations |   annualized_alpha_arithmetic |   annualized_alpha_ci_low |   annualized_alpha_ci_high |   alpha_hac_t_stat |     beta |   r_squared |
|:------------------------------|:--------------|:---------------------|:--------------------------|---------------:|------------------------------:|--------------------------:|---------------------------:|-------------------:|---------:|------------:|
| legacy_baostock_price         | price_return  | hgb_shallow15_locked | industry_size_neutralized |           1387 |                      0.169355 |                  0.037125 |                   0.301586 |           2.510237 | 0.748746 |    0.557718 |
| official_csindex_price        | price_return  | hgb_shallow15_locked | industry_size_neutralized |           1387 |                      0.169355 |                  0.037125 |                   0.301586 |           2.510237 | 0.748747 |    0.557717 |
| official_csindex_total_return | total_return  | hgb_shallow15_locked | industry_size_neutralized |           1387 |                      0.156405 |                  0.023913 |                   0.288897 |           2.313710 | 0.748700 |    0.557752 |
| legacy_baostock_price         | price_return  | hgb_shallow15_locked | raw                       |           1387 |                      0.175455 |                  0.036297 |                   0.314613 |           2.471193 | 0.780931 |    0.549505 |
| official_csindex_price        | price_return  | hgb_shallow15_locked | raw                       |           1387 |                      0.175455 |                  0.036297 |                   0.314613 |           2.471193 | 0.780932 |    0.549504 |
| official_csindex_total_return | total_return  | hgb_shallow15_locked | raw                       |           1387 |                      0.161946 |                  0.022608 |                   0.301285 |           2.277968 | 0.780926 |    0.549598 |
| legacy_baostock_price         | price_return  | ridge_locked         | industry_size_neutralized |           1387 |                      0.131579 |                 -0.000261 |                   0.263419 |           1.956091 | 0.742571 |    0.564240 |
| official_csindex_price        | price_return  | ridge_locked         | industry_size_neutralized |           1387 |                      0.131579 |                 -0.000260 |                   0.263419 |           1.956091 | 0.742572 |    0.564239 |
| official_csindex_total_return | total_return  | ridge_locked         | industry_size_neutralized |           1387 |                      0.118740 |                 -0.013360 |                   0.250840 |           1.761739 | 0.742440 |    0.564144 |
| legacy_baostock_price         | price_return  | ridge_locked         | raw                       |           1387 |                      0.148839 |                  0.014181 |                   0.283497 |           2.166375 | 0.745717 |    0.549732 |
| official_csindex_price        | price_return  | ridge_locked         | raw                       |           1387 |                      0.148839 |                  0.014181 |                   0.283497 |           2.166376 | 0.745718 |    0.549731 |
| official_csindex_total_return | total_return  | ridge_locked         | raw                       |           1387 |                      0.135942 |                  0.001072 |                   0.270812 |           1.975539 | 0.745658 |    0.549744 |

## CSI500 + 六风格归因

市场因子使用相应价格/全收益指数超额收益；六个零净投资风格因子不重复减rf。

| variant_id                    | return_type   | model                | alpha_pipeline            |   observations |   annualized_alpha_arithmetic |   annualized_alpha_ci_low |   annualized_alpha_ci_high |   alpha_hac_t_stat |   r_squared |
|:------------------------------|:--------------|:---------------------|:--------------------------|---------------:|------------------------------:|--------------------------:|---------------------------:|-------------------:|------------:|
| legacy_baostock_price         | price_return  | hgb_shallow15_locked | industry_size_neutralized |           1387 |                      0.008522 |                 -0.058247 |                   0.075290 |           0.250152 |    0.897588 |
| official_csindex_price        | price_return  | hgb_shallow15_locked | industry_size_neutralized |           1387 |                      0.008521 |                 -0.058247 |                   0.075290 |           0.250138 |    0.897585 |
| official_csindex_total_return | total_return  | hgb_shallow15_locked | industry_size_neutralized |           1387 |                     -0.006282 |                 -0.073015 |                   0.060450 |          -0.184520 |    0.897768 |
| legacy_baostock_price         | price_return  | hgb_shallow15_locked | raw                       |           1387 |                      0.008558 |                 -0.060092 |                   0.077208 |           0.244333 |    0.895355 |
| official_csindex_price        | price_return  | hgb_shallow15_locked | raw                       |           1387 |                      0.008558 |                 -0.060093 |                   0.077208 |           0.244320 |    0.895353 |
| official_csindex_total_return | total_return  | hgb_shallow15_locked | raw                       |           1387 |                     -0.006503 |                 -0.075223 |                   0.062217 |          -0.185480 |    0.895556 |
| legacy_baostock_price         | price_return  | ridge_locked         | industry_size_neutralized |           1387 |                     -0.027691 |                 -0.098004 |                   0.042622 |          -0.771869 |    0.894411 |
| official_csindex_price        | price_return  | ridge_locked         | industry_size_neutralized |           1387 |                     -0.027691 |                 -0.098004 |                   0.042622 |          -0.771882 |    0.894409 |
| official_csindex_total_return | total_return  | ridge_locked         | industry_size_neutralized |           1387 |                     -0.042436 |                 -0.112839 |                   0.027967 |          -1.181382 |    0.894460 |
| legacy_baostock_price         | price_return  | ridge_locked         | raw                       |           1387 |                     -0.019726 |                 -0.087869 |                   0.048417 |          -0.567362 |    0.895999 |
| official_csindex_price        | price_return  | ridge_locked         | raw                       |           1387 |                     -0.019726 |                 -0.087869 |                   0.048417 |          -0.567375 |    0.895997 |
| official_csindex_total_return | total_return  | ridge_locked         | raw                       |           1387 |                     -0.034380 |                 -0.102567 |                   0.033807 |          -0.988208 |    0.896128 |

## 历史价格指数回归核对

| check                           |   legacy_rows |   recomputed_rows |   numeric_columns_checked |   maximum_absolute_error |   tolerance | passed   |
|:--------------------------------|--------------:|------------------:|--------------------------:|-------------------------:|------------:|:---------|
| risk_free_rate_capm_summary_v1  |            72 |                72 |                        17 |                0.000e+00 |   1.000e-12 | True     |
| risk_free_rate_style_summary_v1 |            24 |                24 |                        44 |                0.000e+00 |   1.000e-12 | True     |

## 解释限制

全收益指数比价格指数更适合衡量持有股票基准的经济回报，但仍是理论指数：没有ETF费率、跟踪误差、税费和实际再投资摩擦。价格与全收益结果均应保留；前者用于和旧研究连续比较，后者作为主要经济归因口径。任何Alpha变化首先是基准定义变化，不是模型产生了新的Alpha。
