# Phase 5：机构级日频执行引擎

## 证据定位

本实验仅改变执行层，候选信号、目标权重和 20bp 成本保持冻结。
2021–2026 已被研究者查看，因此结果属于 historical OOS 诊断，不是新的独立证据。

## 执行约定

- 每日先按未复权收盘价计价，再在收盘执行；收益只计入一次。
- 新目标将旧未成交订单标为 REPLACED；订单可部分成交、受阻重试，并在 5 个交易日后过期。
- 受限场景执行停牌/涨跌停、10% 日成交量参与率、现金、T+1 与申报数量规则。
- 公司行动使用供应商涨跌幅与原始收盘价推导的价值保持因子；这是代理，不是官方公司行动记录。
- 日线无法识别盘中排队、bid/ask、盘口深度或精确成交时刻。

## 绩效与成交结果

|   periods |   terminal_wealth |   annualized_return |   annualized_volatility |   sharpe_zero_rate |   maximum_drawdown |   mean_turnover |   mean_blocked_buy_weight |   mean_blocked_sell_weight |   mean_target_tracking_l1 |   mean_cash_weight | scenario                               |   total_orders |   total_fills |   fill_ratio_by_shares |   total_cost_cny |
|----------:|------------------:|--------------------:|------------------------:|-------------------:|-------------------:|----------------:|--------------------------:|---------------------------:|--------------------------:|-------------------:|:---------------------------------------|---------------:|--------------:|-----------------------:|-----------------:|
|      1387 |          2.812000 |            0.206647 |                0.230070 |           0.932285 |          -0.333096 |        0.064085 |                  0.004097 |                   0.003184 |                  0.023068 |           0.001331 | daily_close_institutional_constraints  |          68262 |         81650 |               0.988337 |  31122909.225687 |
|      1387 |          2.998563 |            0.220813 |                0.230243 |           0.982483 |          -0.332839 |        0.063746 |                  0.002293 |                   0.002019 |                  0.019099 |           0.000683 | market_constraints_removed_upper_bound |          68233 |         79565 |               0.990464 |  32373665.798722 |

## 会计不变量

| scenario                               |   max_abs_nav_reconciliation_error |    minimum_cash |   max_abs_accounting_weight_error |   negative_position_count |   negative_cash_day_count |   t_plus_one_violation_count |   max_abs_order_fill_reconciliation_error |   fee_reconciliation_error |   unreconciled_corporate_action_count | all_invariants_pass   |
|:---------------------------------------|-----------------------------------:|----------------:|----------------------------------:|--------------------------:|--------------------------:|-----------------------------:|------------------------------------------:|---------------------------:|--------------------------------------:|:----------------------|
| daily_close_institutional_constraints  |                       0.0000001788 |    0.6862432245 |                      0.0000000000 |                         0 |                         0 |                            0 |                              0.0000000000 |               0.0000000000 |                                     0 | True                  |
| market_constraints_removed_upper_bound |                       0.0000002086 | 2805.7423940931 |                      0.0000000000 |                         0 |                         0 |                            0 |                              0.0000000000 |               0.0000000000 |                                     0 | True                  |

## 订单终态

| scenario                               | status   | terminal_reason          |   orders |   original_shares |   filled_shares |   remaining_shares |
|:---------------------------------------|:---------|:-------------------------|---------:|------------------:|----------------:|-------------------:|
| daily_close_institutional_constraints  | EXPIRED  | max_pending_trading_days |    14241 |    785210666.9654 |  751637482.0000 |      33286031.7327 |
| daily_close_institutional_constraints  | FILLED   | nan                      |    50767 |   2589267370.1576 | 2589265904.5663 |             0.0000 |
| daily_close_institutional_constraints  | REPLACED | new_target               |     3254 |    185014204.2317 |  177074349.0000 |       7908117.3567 |
| market_constraints_removed_upper_bound | EXPIRED  | max_pending_trading_days |    20679 |   1121303673.3799 | 1091477463.0000 |      29424457.9006 |
| market_constraints_removed_upper_bound | FILLED   | nan                      |    44012 |   2377351325.8661 | 2377351325.8661 |             0.0000 |
| market_constraints_removed_upper_bound | REPLACED | new_target               |     3542 |    216131984.9406 |  210532993.0000 |       5559536.4276 |

## 诊断结论

受限场景年化收益 20.66%，理想上界 22.08%；成交股数比例 98.83%。
该差异衡量日频近似执行摩擦，不应解释为真实券商成交损失。