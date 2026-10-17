# 风险约束组合的日频可成交审计

## 设计

- 使用冻结的风险约束目标权重，不重新选择模型、alpha 或组合参数。
- 信号后首个交易日收盘提交股数订单；停牌、涨停买入和跌停卖出订单逐日重试。
- 新周目标取消旧未成交订单；自然价格漂移不会触发隐藏的日频再平衡。
- 理想成交上界与受限场景共享唯一非重叠日频净值时钟和 20bp 成本。

## 2021–2026 historical OOS

| model                | alpha_pipeline            | strategy                                    | scenario                                     |   periods |   terminal_wealth |   annualized_return |   annualized_volatility |   sharpe_zero_rate |   maximum_drawdown |   mean_turnover |   mean_blocked_buy_weight |   mean_blocked_sell_weight |   mean_target_tracking_l1 |   mean_cash_weight |
|:---------------------|:--------------------------|:--------------------------------------------|:---------------------------------------------|----------:|------------------:|--------------------:|------------------------:|-------------------:|-------------------:|----------------:|--------------------------:|---------------------------:|--------------------------:|-------------------:|
| hgb_shallow15_locked | industry_size_neutralized | industry_size_neutralized__risk_constrained | daily_retry_limit_and_suspension_constrained |      1387 |          2.760448 |            0.202598 |                0.219093 |           0.952482 |          -0.315308 |        0.063716 |                  0.004323 |                   0.003532 |                  0.020032 |           0.000298 |
| hgb_shallow15_locked | industry_size_neutralized | industry_size_neutralized__risk_constrained | nonoverlap_all_fills_upper_bound             |      1387 |          2.860602 |            0.210410 |                0.220011 |           0.978878 |          -0.316936 |        0.063929 |                  0.000000 |                   0.000000 |                  0.013465 |           0.000000 |
| hgb_shallow15_locked | raw                       | raw__risk_constrained                       | daily_retry_limit_and_suspension_constrained |      1387 |          2.830901 |            0.208117 |                0.230205 |           0.937165 |          -0.333249 |        0.063826 |                  0.004518 |                   0.003477 |                  0.021275 |           0.000493 |
| hgb_shallow15_locked | raw                       | raw__risk_constrained                       | nonoverlap_all_fills_upper_bound             |      1387 |          3.011378 |            0.221759 |                0.231146 |           0.982903 |          -0.334557 |        0.063956 |                  0.000000 |                   0.000000 |                  0.014242 |           0.000000 |
| ridge_locked         | industry_size_neutralized | industry_size_neutralized__risk_constrained | daily_retry_limit_and_suspension_constrained |      1387 |          2.249042 |            0.158653 |                0.216025 |           0.790291 |          -0.317149 |        0.063539 |                  0.005535 |                   0.004788 |                  0.021437 |           0.000183 |
| ridge_locked         | industry_size_neutralized | industry_size_neutralized__risk_constrained | nonoverlap_all_fills_upper_bound             |      1387 |          2.318224 |            0.165049 |                0.217242 |           0.812434 |          -0.315989 |        0.063929 |                  0.000000 |                   0.000000 |                  0.012898 |           0.000000 |
| ridge_locked         | raw                       | raw__risk_constrained                       | daily_retry_limit_and_suspension_constrained |      1387 |          2.463186 |            0.177959 |                0.219777 |           0.855759 |          -0.329912 |        0.063199 |                  0.008575 |                   0.007840 |                  0.026837 |           0.000094 |
| ridge_locked         | raw                       | raw__risk_constrained                       | nonoverlap_all_fills_upper_bound             |      1387 |          2.549081 |            0.185317 |                0.221614 |           0.878624 |          -0.328717 |        0.063930 |                  0.000000 |                   0.000000 |                  0.013068 |           0.000000 |

## 受限成交减理想成交的配对区间

| model                | alpha_pipeline            | strategy                                    | metric                                                   |   observations |   point_estimate |   ci_2_5pct |   ci_97_5pct |   probability_delta_nonpositive |
|:---------------------|:--------------------------|:--------------------------------------------|:---------------------------------------------------------|---------------:|-----------------:|------------:|-------------:|--------------------------------:|
| hgb_shallow15_locked | industry_size_neutralized | industry_size_neutralized__risk_constrained | constrained_minus_ideal_net_return_annualized_arithmetic |           1387 |        -0.006682 |   -0.010075 |    -0.003822 |                        1.000000 |
| hgb_shallow15_locked | raw                       | raw__risk_constrained                       | constrained_minus_ideal_net_return_annualized_arithmetic |           1387 |        -0.011455 |   -0.016628 |    -0.006840 |                        1.000000 |
| ridge_locked         | industry_size_neutralized | industry_size_neutralized__risk_constrained | constrained_minus_ideal_net_return_annualized_arithmetic |           1387 |        -0.005772 |   -0.009364 |    -0.002747 |                        1.000000 |
| ridge_locked         | raw                       | raw__risk_constrained                       | constrained_minus_ideal_net_return_annualized_arithmetic |           1387 |        -0.006639 |   -0.010302 |    -0.003427 |                        1.000000 |
