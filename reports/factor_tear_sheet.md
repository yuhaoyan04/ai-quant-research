# PIT 单因子 Tear Sheet

## 口径

- 样本：2021–2026 historical OOS，共用 PIT 可交易股票池与完整行业/规模暴露样本。
- 方向：仅按 2006–2015 raw 训练期平均 RankIC 的符号锁定；OOS 不选因子、不翻向。
- 组合：每周十分位等权，D10 为预测收益最高组；Top–Bottom 为 D10 减 D1。
- 成本：多空两腿各按目标权重换手乘 20bp，仅作敏感性估计。
- 市场状态：信号日可见的中证500过去60交易日趋势，并以历史扩张中位数区分高/低波动。
- 推断：Newey–West(4) 与 13周 circular block bootstrap 95% 区间。

## 核心发现

- 行业+规模中性化后，Top–Bottom 年化算术价差 95% 下界为正的因子：amplitude_20d, turnover_20d, volatility_20d, max_return_20d, momentum_20d, momentum_60d, book_to_price, return_volume_corr_20d, reversal_5d, momentum_120d, amihud_20d, momentum_20d_skip_5d, volume_surprise_20d, turnover_surprise_20d。
- 六个 OOS 年度定向 RankIC 全为正的因子：amplitude_20d, turnover_20d, volatility_20d, max_return_20d, momentum_20d, momentum_60d, book_to_price, return_volume_corr_20d, reversal_5d, momentum_120d, amihud_20d, momentum_20d_skip_5d, volume_surprise_20d, turnover_surprise_20d, earnings_yield。
- `annualized_net_top_minus_bottom` 对换手成本非常敏感，不等同于可直接成交的多空策略回测。

## 行业+规模中性化结果

| factor                 |   mean_oriented_rank_ic |   annualized_icir |   bucket_return_spearman |   annualized_gross_top_minus_bottom |   ci_2_5pct |   ci_97_5pct |   annualized_net_top_minus_bottom |   mean_long_short_turnover |   positive_ic_years |   positive_ic_regimes | spread_significant   |
|:-----------------------|------------------------:|------------------:|-------------------------:|------------------------------------:|------------:|-------------:|----------------------------------:|---------------------------:|--------------------:|----------------------:|:---------------------|
| amplitude_20d          |                  0.0726 |            2.9717 |                   0.6606 |                              0.2332 |      0.0119 |       0.4141 |                            0.1926 |                     0.3806 |                   6 |                     3 | yes                  |
| turnover_20d           |                  0.0725 |            3.7165 |                   1      |                              0.3983 |      0.2307 |       0.5419 |                            0.369  |                     0.2763 |                   6 |                     4 | yes                  |
| volatility_20d         |                  0.0713 |            3.1392 |                   0.8061 |                              0.2713 |      0.0876 |       0.4195 |                            0.2168 |                     0.5102 |                   6 |                     3 | yes                  |
| max_return_20d         |                  0.07   |            3.9504 |                   0.9152 |                              0.2695 |      0.097  |       0.4023 |                            0.2067 |                     0.5973 |                   6 |                     4 | yes                  |
| momentum_20d           |                  0.053  |            2.9022 |                   0.6242 |                              0.4473 |      0.2201 |       0.6683 |                            0.346  |                     0.9176 |                   6 |                     4 | yes                  |
| momentum_60d           |                  0.0505 |            2.8463 |                   0.9394 |                              0.378  |      0.1684 |       0.5789 |                            0.3147 |                     0.5701 |                   6 |                     4 | yes                  |
| book_to_price          |                  0.048  |            2.42   |                   0.9636 |                              0.1398 |      0.0025 |       0.2594 |                            0.1294 |                     0.0993 |                   6 |                     3 | yes                  |
| return_volume_corr_20d |                  0.0447 |            4.164  |                   0.9394 |                              0.1673 |      0.069  |       0.2674 |                            0.074  |                     0.853  |                   6 |                     4 | yes                  |
| reversal_5d            |                  0.0391 |            2.3056 |                   0.6606 |                              0.3787 |      0.2348 |       0.5226 |                            0.2034 |                     1.6585 |                   6 |                     4 | yes                  |
| momentum_120d          |                  0.0388 |            2.1841 |                   0.7333 |                              0.2833 |      0.0832 |       0.457  |                            0.2358 |                     0.412  |                   6 |                     4 | yes                  |
| amihud_20d             |                  0.0381 |            2.8187 |                   0.7091 |                              0.2188 |      0.1258 |       0.3012 |                            0.1887 |                     0.3042 |                   6 |                     4 | yes                  |
| momentum_20d_skip_5d   |                  0.0343 |            1.9018 |                   0.6364 |                              0.2133 |      0.0149 |       0.3978 |                            0.1133 |                     0.9201 |                   6 |                     4 | yes                  |
| volume_surprise_20d    |                  0.0313 |            3.1076 |                   0.5515 |                              0.2644 |      0.1693 |       0.3626 |                            0.1084 |                     1.5227 |                   6 |                     4 | yes                  |
| turnover_surprise_20d  |                  0.0309 |            3.0782 |                   0.4545 |                              0.2651 |      0.1701 |       0.3622 |                            0.1095 |                     1.5173 |                   6 |                     4 | yes                  |
| earnings_yield         |                  0.0288 |            1.8978 |                   0.7818 |                              0.0563 |     -0.0959 |       0.1902 |                            0.0431 |                     0.1086 |                   6 |                     3 | no                   |

## 限制

分位数组合未施加涨跌停、停牌、容量和借券约束；因此它是因子诊断而不是独立策略业绩。市场状态仅用于事后分层描述，但其分类变量在每个信号日均可获得。
