# 因子分布与相关性稳定性

本页只做描述性展示，不用于重新筛选因子或模型。分布使用 2021–2026 周频信号截面；热图使用验证期，滚动相关使用 52 周历史 OOS 窗口。

![因子相关性热图](figures/factor_correlation_heatmap.svg)

![因子分布](figures/factor_distributions.svg)

## 滚动相关变化最大的因子对

| factor_a       | factor_b       |   rolling_windows |   mean_rolling_correlation |   minimum_rolling_correlation |   maximum_rolling_correlation |   rolling_correlation_std |   range |
|:---------------|:---------------|------------------:|---------------------------:|------------------------------:|------------------------------:|--------------------------:|--------:|
| momentum_120d  | amplitude_20d  |               244 |                     0.3602 |                       -0.0384 |                        0.6135 |                    0.1875 |  0.6519 |
| momentum_120d  | volatility_20d |               244 |                     0.3130 |                       -0.0663 |                        0.5558 |                    0.1764 |  0.6222 |
| momentum_60d   | amplitude_20d  |               244 |                     0.3492 |                        0.0312 |                        0.5848 |                    0.1523 |  0.5536 |
| momentum_60d   | volatility_20d |               244 |                     0.3099 |                        0.0027 |                        0.5337 |                    0.1453 |  0.5310 |
| momentum_120d  | max_return_20d |               244 |                     0.3203 |                        0.0176 |                        0.5197 |                    0.1416 |  0.5020 |
| momentum_60d   | max_return_20d |               244 |                     0.3596 |                        0.1168 |                        0.5316 |                    0.1128 |  0.4148 |
| amihud_20d     | amplitude_20d  |               244 |                    -0.0595 |                       -0.2730 |                        0.1377 |                    0.1170 |  0.4108 |
| momentum_120d  | turnover_20d   |               244 |                     0.2662 |                        0.0320 |                        0.4307 |                    0.1127 |  0.3987 |
| volatility_20d | amihud_20d     |               244 |                    -0.0661 |                       -0.2467 |                        0.1430 |                    0.1089 |  0.3897 |
| momentum_120d  | earnings_yield |               244 |                    -0.0999 |                       -0.2667 |                        0.0891 |                    0.0872 |  0.3558 |
