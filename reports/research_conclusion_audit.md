# 研究结论与全链路审计

本页把研究结果拆成六个独立决策，防止把预测力、回测收益、模型增量、风险调整 Alpha 和上线资格混为一谈。

## 决策矩阵

| question                                  | verdict                          | evidence                                                                                                                                                                     | scope_limit                                                          |
|:------------------------------------------|:---------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------|
| neutralized_factor_signal_exists          | SUPPORTED_AS_DIAGNOSTIC_EVIDENCE | 14/15 neutralized gross factor spreads survive BH-FDR 5%                                                                                                                     | diagnostic long-short portfolios omit borrow and full execution      |
| hgb_adds_stable_increment_over_ridge      | NOT_ESTABLISHED                  | RankIC delta 95% CI [-0.0023, 0.0071]; annual return delta 95% CI [-0.86%, 7.26%]                                                                                            | historical OOS has been inspected                                    |
| ml_adds_value_over_style_matched_baseline | MIXED_LOCAL_EVIDENCE             | incremental return intervals above zero 1/4; BH-FDR survivors 0/4; HGB neutral delta 3.65% with 95% CI [0.24%, 7.16%]; multiplicity-controlled risk/efficiency survivors 1/4 | matched peers use local six-style proxies and monthly broad industry |
| executable_backtest_survives_costs        | SUPPORTED_HISTORICALLY           | 20bp annualized return 20.81%, Sharpe 0.94, max drawdown -33.32%; fixed-path 100bp annualized return 6.23%                                                                   | close execution, approximate limits, no broker-calibrated impact     |
| pure_stock_selection_alpha_is_significant | NOT_SUPPORTED                    | style-adjusted alpha range -2.55% to 1.07%; all_95pct_intervals_cross_zero=True; R2 range 0.894-0.898                                                                        | local factors are research proxies, not a commercial risk model      |
| ready_for_live_capital                    | NO                               | no unseen forward sample, paper track record, or broker fill calibration                                                                                                     | next gate is frozen-parameter paper trading on future data           |

## Quant Research 生命周期

| stage                            | status              | evidence                                                           | boundary                                              |
|:---------------------------------|:--------------------|:-------------------------------------------------------------------|:------------------------------------------------------|
| research_mandate                 | COMPLETE            | question hierarchy and falsifiable hypotheses                      | final report sections 1 and 10                        |
| data_acquisition                 | COMPLETE_WITH_LIMIT | free-source PIT lake and manifests                                 | BaoStock revisions prevent byte-identical redownloads |
| data_quality                     | COMPLETE            | schema, coverage, truncation, and key gates                        | publication audit                                     |
| point_in_time_universe           | COMPLETE_WITH_LIMIT | historical A-share snapshots delayed one day                       | not official historical CSI membership                |
| factor_definition                | COMPLETE            | versioned registry, formulas, missingness, direction               | 15-factor registry                                    |
| label_and_timing                 | COMPLETE            | t signal, t+1 close execution, t+2..t+6 label                      | purged split boundaries                               |
| preprocessing_and_neutralization | COMPLETE_WITH_LIMIT | same-date transforms and PIT industry/size residualization         | monthly CSRC broad industry                           |
| signal_validation                | COMPLETE            | IC, deciles, regimes, decay, LOFO, BH-FDR                          | historical OOS is not pristine                        |
| model_selection                  | COMPLETE            | validation-only benchmark and parameter stability                  | HGB increment over Ridge not established              |
| portfolio_construction           | COMPLETE            | single-name, industry, size, and turnover constraints              | long-only research policy                             |
| execution_simulation             | COMPLETE_WITH_LIMIT | share orders, cash, suspensions, limits, retry clock               | no intraday queue or IPO no-limit window              |
| cost_capacity_stress             | COMPLETE_WITH_LIMIT | 0-100bp and ADV/impact scenarios                                   | impact not calibrated to broker orders                |
| risk_and_attribution             | COMPLETE_WITH_LIMIT | CAPM, six-style and cross-sectional risk attribution               | price index and local proxy factors                   |
| robustness_and_inference         | COMPLETE            | paired block bootstrap, HAC, common-date tests, failed experiments | no untouched production lockbox                       |
| reproducibility_and_governance   | COMPLETE            | tests, immutable experiments, hashes, invalidations                | full data lake intentionally excluded from Git        |
| paper_live_monitoring            | NEXT_STAGE          | frozen-parameter forward evidence required                         | not part of a completed historical research backtest  |

结论：历史研究链路已经闭环；`paper_live_monitoring` 是下一阶段，不能用回测替代。

## 单因子多重检验

在 15 个行业+规模中性因子上，按 BH-FDR 5% 控制后，14 个毛 Top–Bottom 价差仍通过。
这加强了因子诊断证据，但不把未建模借券和完整成交的多空组合称为可交易策略。

| factor                 |   point_estimate |   ci_2_5pct |   ci_97_5pct |   one_sided_probability_nonpositive |   bh_fdr_adjusted_p | survives_bh_fdr   |
|:-----------------------|-----------------:|------------:|-------------:|------------------------------------:|--------------------:|:------------------|
| amihud_20d             |        0.218797  |  0.125813   |     0.30117  |                          0.00049975 |          0.00107089 | True              |
| momentum_20d           |        0.447265  |  0.220147   |     0.668277 |                          0.00049975 |          0.00107089 | True              |
| return_volume_corr_20d |        0.167281  |  0.0689816  |     0.267385 |                          0.00049975 |          0.00107089 | True              |
| reversal_5d            |        0.378651  |  0.234813   |     0.522591 |                          0.00049975 |          0.00107089 | True              |
| turnover_20d           |        0.398261  |  0.230714   |     0.541921 |                          0.00049975 |          0.00107089 | True              |
| turnover_surprise_20d  |        0.265054  |  0.170095   |     0.362244 |                          0.00049975 |          0.00107089 | True              |
| volume_surprise_20d    |        0.264373  |  0.169341   |     0.362555 |                          0.00049975 |          0.00107089 | True              |
| momentum_60d           |        0.378001  |  0.168423   |     0.578949 |                          0.0009995  |          0.00187406 | True              |
| max_return_20d         |        0.269462  |  0.0969776  |     0.40233  |                          0.001999   |          0.00333167 | True              |
| momentum_120d          |        0.283298  |  0.083215   |     0.457041 |                          0.00449775 |          0.0061333  | True              |
| volatility_20d         |        0.271293  |  0.0875819  |     0.419531 |                          0.00449775 |          0.0061333  | True              |
| amplitude_20d          |        0.23323   |  0.0118815  |     0.414075 |                          0.01999    |          0.0230654  | True              |
| momentum_20d_skip_5d   |        0.213274  |  0.0148821  |     0.39778  |                          0.0189905  |          0.0230654  | True              |
| book_to_price          |        0.139782  |  0.00253231 |     0.259414 |                          0.0224888  |          0.0240951  | True              |
| earnings_yield         |        0.0563334 | -0.0959368  |     0.190236 |                          0.224388   |          0.224388   | False             |

## 年化指标口径

- 最终组合年化收益是日频净值的几何复合年化。
- CAPM/风格 Alpha 是日回归截距的算术年化。
- 风险贡献是周频贡献的算术年化；它与几何收益不能逐项直接相加比较。
