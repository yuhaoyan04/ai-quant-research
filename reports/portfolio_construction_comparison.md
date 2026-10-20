# Phase 6：组合权重与风险优化比较

## 证据边界

本实验是已经被反复查看的 2021–2026 historical OOS 回顾性诊断，不是新的独立证据，也不用于选择或晋升组合方法。P4 支持集与回退处理曾依据本段历史中的结构性可行性失败改进，因此治理分类为 `post_historical_oos_adaptation`；没有按收益调参。
四种方法固定使用同一 raw HGB 分数、调仓时钟、PIT 股票池、20bp 成本和 Phase 5 日频执行器。

## 目标权重构造审计

| method                |   dates |   mean_candidates |   mean_optimizer_candidates |   mean_target_turnover |   optimizer_success_fraction |   fallback_fraction |   mean_industry_active_l1 |   mean_abs_size_z |
|:----------------------|--------:|------------------:|----------------------------:|-----------------------:|-----------------------------:|--------------------:|--------------------------:|------------------:|
| P0_equal              |     295 |        240.979661 |                  240.979661 |               0.503436 |                     1.000000 |            0.000000 |                  0.153583 |          0.806817 |
| P1_inverse_volatility |     295 |        240.979661 |                  240.979661 |               0.512151 |                     1.000000 |            0.000000 |                  0.159656 |          0.792447 |
| P2_score_weighted     |     295 |        240.979661 |                  240.979661 |               0.630127 |                     1.000000 |            0.000000 |                  0.160572 |          0.820949 |
| P4_constrained        |     295 |        240.979661 |                  382.494915 |               0.338940 |                     0.454237 |            0.545763 |                  0.115024 |          0.112692 |

P4 对 Top-200/400 普通候选以外的行业/规模可行性支持股票单独标记。严格 LP 不可行时使用显式 carry-forward capacity repair；该日期不计为优化成功，且从不放宽约束。

## 1 亿元实盘近似执行

| method                |   annualized_return |   annualized_volatility |   sharpe_zero_rate |   sharpe_reliable_risk_free |   maximum_drawdown |   mean_one_way_turnover |   mean_gross_two_way_turnover |   fill_ratio_by_shares |   mean_cash_weight |   mean_target_tracking_l1 |   total_cost_cny |
|:----------------------|--------------------:|------------------------:|-------------------:|----------------------------:|-------------------:|------------------------:|------------------------------:|-----------------------:|-------------------:|--------------------------:|-----------------:|
| P0_equal              |            0.221440 |                0.262494 |           0.894388 |                    0.829509 |          -0.429221 |                0.105944 |                      0.210175 |               0.941280 |           0.000906 |                  0.028497 |  51156475.551388 |
| P1_inverse_volatility |            0.217737 |                0.256746 |           0.896729 |                    0.830394 |          -0.423962 |                0.107854 |                      0.213806 |               0.943108 |           0.000945 |                  0.029881 |  52062993.378140 |
| P2_score_weighted     |            0.195592 |                0.264638 |           0.808434 |                    0.744055 |          -0.434934 |                0.133286 |                      0.263892 |               0.939073 |           0.001441 |                  0.030467 |  60656497.127843 |
| P4_constrained        |            0.188064 |                0.216281 |           0.905647 |                    0.826913 |          -0.322592 |                0.069839 |                      0.137238 |               0.587949 |           0.002188 |                  0.076017 |  31108178.687616 |

## 相对 P0 的配对块区间

| method                | reference   |   observations |   annualized_arithmetic_return_delta |   ci_2_5pct |   ci_97_5pct |   probability_delta_nonpositive |
|:----------------------|:------------|---------------:|-------------------------------------:|------------:|-------------:|--------------------------------:|
| P1_inverse_volatility | P0_equal    |           1387 |                            -0.004540 |   -0.014424 |     0.005903 |                        0.800600 |
| P2_score_weighted     | P0_equal    |           1387 |                            -0.020829 |   -0.031622 |    -0.009755 |                        1.000000 |
| P4_constrained        | P0_equal    |           1387 |                            -0.038897 |   -0.098312 |     0.021703 |                        0.903548 |

## 容量敏感性

| method                |   initial_cash_cny |   annualized_return |   fill_ratio_by_shares |   mean_cash_weight |   mean_target_tracking_l1 | all_invariants_pass   |
|:----------------------|-------------------:|--------------------:|-----------------------:|-------------------:|--------------------------:|:----------------------|
| P0_equal              |   100000000.000000 |            0.221440 |               0.941280 |           0.000906 |                  0.028497 | True                  |
| P0_equal              |    25000000.000000 |            0.219674 |               0.942463 |           0.002955 |                  0.031038 | True                  |
| P0_equal              |   500000000.000000 |            0.215641 |               0.939371 |           0.010646 |                  0.110910 | True                  |
| P1_inverse_volatility |   100000000.000000 |            0.217737 |               0.943108 |           0.000945 |                  0.029881 | True                  |
| P1_inverse_volatility |    25000000.000000 |            0.216877 |               0.944332 |           0.002756 |                  0.030772 | True                  |
| P1_inverse_volatility |   500000000.000000 |            0.214215 |               0.937130 |           0.011064 |                  0.129899 | True                  |
| P2_score_weighted     |   100000000.000000 |            0.195592 |               0.939073 |           0.001441 |                  0.030467 | True                  |
| P2_score_weighted     |    25000000.000000 |            0.195705 |               0.939854 |           0.003022 |                  0.030007 | True                  |
| P2_score_weighted     |   500000000.000000 |            0.189022 |               0.932807 |           0.017837 |                  0.164829 | True                  |
| P4_constrained        |   100000000.000000 |            0.188064 |               0.587949 |           0.002188 |                  0.076017 | True                  |
| P4_constrained        |    25000000.000000 |            0.188895 |               0.589878 |           0.007798 |                  0.083908 | True                  |
| P4_constrained        |   500000000.000000 |            0.195845 |               0.626522 |           0.016211 |                  0.166981 | True                  |

## 会计不变量

| method                | scenario                                                     |   max_abs_nav_reconciliation_error |   minimum_cash |   max_abs_accounting_weight_error |   negative_position_count |   negative_cash_day_count |   t_plus_one_violation_count |   max_abs_order_fill_reconciliation_error |   fee_reconciliation_error |   unreconciled_corporate_action_count | all_invariants_pass   |
|:----------------------|:-------------------------------------------------------------|-----------------------------------:|---------------:|----------------------------------:|--------------------------:|--------------------------:|-----------------------------:|------------------------------------------:|---------------------------:|--------------------------------------:|:----------------------|
| P0_equal              | P0_equal__daily_close_institutional_constraints              |                       0.0000002384 |   9.5800031823 |                      0.0000000000 |                         0 |                         0 |                            0 |                              0.0000000000 |               0.0000000000 |                                     0 | True                  |
| P1_inverse_volatility | P1_inverse_volatility__daily_close_institutional_constraints |                       0.0000002384 |   7.5041305098 |                      0.0000000000 |                         0 |                         0 |                            0 |                              0.0000000000 |               0.0000000000 |                                     0 | True                  |
| P2_score_weighted     | P2_score_weighted__daily_close_institutional_constraints     |                       0.0000003278 |   0.2107504061 |                      0.0000000000 |                         0 |                         0 |                            0 |                              0.0000000000 |               0.0000000000 |                                     0 | True                  |
| P4_constrained        | P4_constrained__daily_close_institutional_constraints        |                       0.0000002086 |  53.5027724001 |                      0.0000000000 |                         0 |                         0 |                            0 |                              0.0000000000 |               0.0000000037 |                                     0 | True                  |

## 验收检查

| check                           | status   | detail                                                                                                       |
|:--------------------------------|:---------|:-------------------------------------------------------------------------------------------------------------|
| long_only_targets               | PASS     | minimum=4.17462408336e-06                                                                                    |
| fully_invested_targets          | PASS     | maximum_abs_error=7.77156117238e-16                                                                          |
| common_target_dates             | PASS     | date_counts={'P0_equal': 294, 'P1_inverse_volatility': 294, 'P2_score_weighted': 294, 'P4_constrained': 294} |
| no_silent_constraint_relaxation | PASS     | relaxed_dates=0                                                                                              |
| optimizer_failures_explicit     | PASS     | fallback_dates=161                                                                                           |
| accounting_invariants           | PASS     | passing_methods=4/4                                                                                          |
| common_execution_clock          | PASS     | periods={'P0_equal': 1387, 'P1_inverse_volatility': 1387, 'P2_score_weighted': 1387, 'P4_constrained': 1387} |
| reliable_risk_free_reported     | PASS     | scenario=['R1_ccdc_cgb_3m']                                                                                  |

## 结论与决策

- 保留 P0 为透明基线：年化收益 22.14%、Sharpe 0.894、最大回撤 -42.92%。
- P1 未显示增量收益，但有轻微风险效率改善：年化收益 21.77%，波动 25.67%，Sharpe 0.897；相对 P0 收益差区间跨零，不能声称优于基线。
- P2 不予采用：相对 P0 年化算术收益差 95% 区间 [-3.16%, -0.98%] 完全低于零。
- P4 将波动降至 21.63%、最大回撤改善至 -32.26%，但严格优化仅 45.4% 日期成功，成交股数比例 58.79%、平均目标偏离 7.60%；这是风险/换手价值与可实现性不足并存的研究原型，不可晋升。
复杂方法只有在未来 lockbox 或前瞻 paper trading 中稳定改善收益、风险或交易效率时才可替换 P0；本表不得用于 OOS 后择优。
日线执行仍无法重建盘口、排队、bid/ask 或精确成交概率。