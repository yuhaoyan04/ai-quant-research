# Phase 0：当前指标核对报告

**核对日期：** 2026-10-20 UTC  
**权威基线：** `final_research_baseline_v4`  
**代码锚点：** `db5cd282d05f0da61047d2ebf2bd9c09e9165032`

本报告只核对现有代码、curated tables、冻结快照和发布文档，不重新选择模型、不调整参数、不覆盖任何历史产物。表内数值来自当前 live tables，并与权威冻结实验中的同名表逐项核对；冻结实验逐文件完整性另见 `current_baseline_inventory.csv`。

## 1. 口径字典

| 指标 | 当前实际口径 | 容易误读之处 |
|---|---|---|
| `annualized_return` | 日频/周频净收益的几何年化；日频乘数 252，周频乘数 52 | 不等于日均收益乘年化因子 |
| `annualized_*_arithmetic` | 均值乘年化因子；用于回归 Alpha、配对收益差和归因可加性 | 不应与几何 CAGR 直接相减替代 |
| `sharpe_zero_rate` | 日/周均净收益除波动率后年化，`rf=0` | 不是人民币无风险利率调整后的 Sharpe |
| weekly `turnover` | 首期 1；以后为新目标权重相对漂移后旧权重的 `0.5 * sum(abs(delta))` | 单边权重换手 |
| daily `turnover` | `max(buy_notional, sell_notional) / pretrade_nav` | 不是买卖成交额之和除 NAV |
| daily `transaction_cost` | `max(buy_notional, sell_notional) * 20bp / pretrade_nav` | 20bp 被视为一次完整换仓的 round-trip 情景成本，不在买卖两边各扣 20bp |
| maximum drawdown | NAV 相对历史峰值的最小跌幅 | 为负值，越接近零越好 |

## 2. 数据层声明核对

| 声明 | 当前复核值 | 状态 |
|---|---:|---|
| 历史证券 union | 7,939 个证券记录，其中 A 股 5,521 只 | 一致 |
| raw/validated/factor/label A 股覆盖 | 四层均为 5,521 / 5,521；另含 3 个指数 | 一致 |
| preprocessing staging 行数 | 14,862,783 | 与“约 1,479 万”一致 |
| security snapshots | 1,059 个分区 | 一致 |
| 行业快照 | 研究期调度并使用 250 个日期；raw 目录实际 288 个文件 | 主研究口径一致；另有 38 个未使用的早期冗余文件 |
| PIT 行业覆盖率最低值 | 98.548731% | 一致 |

### 简历关键数字的可追溯定位

| 简历/发布数字 | experiment / 表 / 字段 | 约束或成本配置 | 核对值 |
|---|---|---|---:|
| 5,521 只历史 A 股、覆盖率 100% | `research_coverage/a_share`: `historical_a_share_union`, `download_coverage` | `config/free_sources.yaml` | 5,521；1.0000 |
| 约 1,479 万行模型特征 | `model_features/2006..2026` 的 Parquet metadata `num_rows` 求和 | `config/preprocessing.yaml`, `config/model_features.yaml` | 14,862,783 |
| 37 个不可变实验 | `reports/experiments/index.json` 与 37 个 `FROZEN.json` | `config/frozen_baselines.yaml` | 37 |
| 单股≤0.5%、行业±3%、`|size-z|≤0.10`、周换手≤30% | `constrained_portfolio_oos_weekly/v1`: `maximum_stock_weight`, `maximum_abs_industry_active_weight`, `portfolio_size_z`, `effective_turnover_limit` 及 violation fields | `config/constrained_portfolio_neutralization.yaml` → `optimizer.*` | 0.005 / 0.03 / 0.10 / 0.30 |
| HGB raw 20.81%、Sharpe 0.94、MDD -33.32% | `final_research_baseline_v4` → `constrained_daily_execution_summary/v1`, row `hgb_shallow15_locked/raw/daily_retry...`; fields `annualized_return`, `sharpe_zero_rate`, `maximum_drawdown` | `config/constrained_portfolio_daily_execution.yaml`: 20bp / retry execution | 20.8117% / 0.9372 / -33.3249% |
| 六风格 Alpha -2.55% 至 +1.07%，CI 均跨零 | `final_research_baseline_v4` → `style_factor_regression_summary/v1`; fields `annualized_alpha_arithmetic`, `annualized_alpha_ci_low/high` | `config/style_factor_attribution.yaml`: zero rf, HAC lag 5 | -2.5461% 至 +1.0696% |
| HGB neutral 增量 3.65%，CI [0.24%, 7.16%]，0/4 收益通过 FDR | `final_research_baseline_v4` → `incremental_alpha_comparison/v1`; incremental return estimate/CI/q/survival fields | `config/incremental_alpha_test.yaml`: common execution, 20-day paired blocks, BH-FDR | 3.6468%；[0.2444%, 7.1647%]；q=7.5962% |

上述实验中的 2021–2026 均为 historical OOS；这些定位不把该区间提升为 pristine lockbox。

## 3. 最终可执行日频结果

场景均为 `daily_retry_limit_and_suspension_constrained`，1,387 个共同交易日，成本 20bp，Sharpe 使用零无风险利率。

| 模型 | pipeline | 几何年化收益 | 年化波动 | Sharpe | 最大回撤 | 平均单边执行换手 |
|---|---|---:|---:|---:|---:|---:|
| HGB | neutralized | 20.2598% | 21.9093% | 0.9525 | -31.5308% | 6.3716% |
| HGB | raw | 20.8117% | 23.0205% | 0.9372 | -33.3249% | 6.3826% |
| Ridge | neutralized | 15.8653% | 21.6025% | 0.7903 | -31.7149% | 6.3539% |
| Ridge | raw | 17.7959% | 21.9777% | 0.8558 | -32.9912% | 6.3199% |

核对结论：README/最终报告中 HGB raw 的 20.81%、0.94、-33.32% 与当前表和冻结表一致。

### 理想成交与受限成交差异

以下为受限成交减理想全成交的日频净收益差，采用 20 日 block bootstrap，数值为算术年化：

| 模型 | pipeline | 点估计 | 95% CI |
|---|---|---:|---:|
| HGB | neutralized | -0.6682% | [-1.0075%, -0.3822%] |
| HGB | raw | -1.1455% | [-1.6628%, -0.6840%] |
| Ridge | neutralized | -0.5772% | [-0.9364%, -0.2747%] |
| Ridge | raw | -0.6639% | [-1.0302%, -0.3427%] |

四组受限成交损失区间都低于零；该结论只反映当前停牌/收盘涨跌停和 retry 规则，不包含整手、T+1、成交量参与率、盘口或真实滑点。

## 4. 交易成本、现金和 NAV 恒等式

对 `constrained_daily_execution_daily/v1` 的 11,096 行（8 组 × 1,387 日）重新计算：

| 核对项 | 最大绝对误差 / 结果 |
|---|---:|
| group/date 重复键 | 0 |
| 每组交易日数 | 1,387 |
| `net_return - (gross_return - transaction_cost)` | 5.84e-15 |
| 终值 NAV vs `product(1 + net_return)` | 2.49e-14 |
| `transaction_cost - 0.002 * turnover * (1 + gross_return)` | 4.34e-18 |
| `cash_weight + sum(actual_weights) - 1` | 8.33e-15 |
| 负费用记录 | 0 |
| 负现金权重记录 | 0 |
| turnover vs `max(buy_weight, sell_weight)` | 0 |

这些恒等式证明当前聚合模拟在自身假设下守恒，但不能替代 fill-level 对账：当前没有持久化 order、fill、position 表；`shares` 是以 1.0 合成价格初始化并允许小数的归一化份额。

## 5. 周频组合约束

当前周频目标共有 293 个日期、2,344 个组合日期记录，其中 1,172 个风险约束组合记录。

| 约束核对 | 结果 |
|---|---:|
| turnover relaxations | 0 |
| 单股权重上限最大违反 | 0 |
| 规模约束最大违反 | 2.50e-16 |
| 行业约束最大违反 | 4.72e-16 |
| 非首期最大单边换手 | 0.30000005 |
| 最大单股权重 | 0.00500000 |
| 权重和最大误差 | 7.77e-16 |

HGB raw 风险约束周频组合的几何年化收益为 24.6798%、Sharpe 0.9242、最大回撤 -32.8971%、平均换手 30.2389%。它是 5 日标签/理想周频目标口径，不能与上节的日频可执行结果混称。

## 6. CAPM、六风格和风险归因

### CAPM

四组 CAPM 均为 1,387 个日频观测、Newey–West lag=5、`rf=0`，并使用 CSI 价格指数而非含股息总收益指数。当前 CAPM 年化算术 Alpha 范围为 13.5968% 至 17.9190%，四组 95% 区间均高于零。该结果在加入本地六风格解释变量后不再稳健，因此不能单独作为“纯 Alpha”结论。

### 六风格回归

| 模型 | pipeline | 年化算术 Alpha | 95% CI | HAC t |
|---|---|---:|---:|---:|
| HGB | neutralized | 1.0696% | [-5.6165%, 7.7557%] | 0.314 |
| HGB | raw | 1.0471% | [-5.8252%, 7.9193%] | 0.299 |
| Ridge | neutralized | -2.5461% | [-9.5848%, 4.4926%] | -0.709 |
| Ridge | raw | -1.7398% | [-8.5627%, 5.0830%] | -0.500 |

四组区间全部跨零，与最终报告“纯选股 Alpha 未获支持”一致。风格组合为零成本解释因子，不是独立可交易策略业绩。

### 周频多因子风险归因

1,172 条组合日期记录的收益归因恒等式最大误差为 `2.22e-16`，reported net 最大误差为 `5.55e-17`。HGB raw 的年化算术 style contribution 为 13.6132%，specific selection contribution 为 -0.4344%。该归因对应理想周频目标组合，不对应日频实际成交持仓和现金，因此不能用来解释 blocked fills 的逐日风险贡献。

## 7. Incremental Alpha Test 核对

模型与匹配风格基准使用相同 1,387 日历史区间、相同受限成交器和 20bp 成本。收益差为模型减风格基准的算术年化，并采用 20 日配对 block bootstrap。

| 模型 | pipeline | 模型 / 基准几何年化 | 增量收益 | 95% CI | 收益差 BH q | Sharpe 改善 | 回撤改善 | 换手改善 | 总体多重校正证据 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| HGB | neutralized | 20.2598% / 15.9625% | 3.6468% | [0.2444%, 7.1647%] | 7.5962% | +0.1652 | -0.1217% | +0.0062% | 否 |
| HGB | raw | 20.8117% / 17.5770% | 2.8702% | [-0.6198%, 6.5532%] | 10.8946% | +0.0993 | -0.3210% | -0.0134% | 否 |
| Ridge | neutralized | 15.8653% / 13.4465% | 2.0542% | [-1.9353%, 5.6890%] | 20.5897% | +0.1025 | -0.1366% | -0.0019% | 否 |
| Ridge | raw | 17.7959% / 16.0881% | 1.4905% | [-2.0586%, 5.0567%] | 20.8396% | +0.0621 | -1.7607% | +0.0853% | 是，仅来自换手改善 |

HGB neutralized 的单项收益区间排除零，但在四策略同指标 BH-FDR 后不通过 5% 阈值。四组中没有任何收益增量通过 FDR；Ridge raw 的总体标记为真，仅因为换手改善的校正后概率为 0.1999%，而非因为收益 Alpha 显著。最终文档中的“mixed local evidence”判定与表一致。

## 8. 模型验证和可重现性

| 项目 | 核对结果 |
|---|---|
| 时间切分 | train 2006–2015；validation 2016–2020；historical OOS 2021–2026 |
| 标签边界 | 年度训练边界执行 5 个交易日 purge |
| 重训 | OOS 按年 expanding-window 重训 |
| 模型选择 | validation-only；frozen metadata 均声明 `historical_oos_used_for_selection=false` |
| HGB | sklearn 内部 early stopping 关闭；日期/证券哈希抽样确定性 |
| 参数面 | 3×3 HGB stability surface 只看 validation |
| fit audit 缺口 | 未逐 fit 保存标签成熟截止日、完整参数、seed、模型文件 hash、预测 hash |
| 最终测试地位 | 2021–2026 已反复查看，只是 historical OOS，不是 pristine lockbox |

## 9. 冻结产物核对

- 37 个冻结实验，`FROZEN.json` 管理 415 个文件，共 86,703,820 bytes。
- 逐文件重算后快照完整性 37/37 PASS，声明 content hash 与重算 hash 37/37 一致。
- 19 个实验当前 live inputs 完全匹配，18 个有 drift；18 个 drift 实验全部在 `experiment_invalidations.yaml` 中有失效或替代记录。
- 21 个实验总体处于 invalidated/superseded 生命周期状态；其中 3 个快照的 live inputs 仍匹配。
- `final_research_baseline_v4` 与 `incremental_alpha_test_v1` 当前 live inputs 均匹配。
- 24/37 个 frozen metadata 没有 Git commit；另 13/37 记录了 commit，但均记录 `dirty=true`。所有 37 个实验都有 `source_tree_sha256`。

## 10. 最终核对结论

当前文档中的主数值、风险限制、historical OOS 定位和“不宣称显著纯 Alpha”的结论，与当前实现及冻结产物一致。需要降格或限定的措辞集中在“股数订单/交易账本/可成交/真实成本”：现有实现是有现金和合成份额状态的日频研究模拟，不是可由逐笔 fills 独立重放的机构级账本。Phase 1 应首先标准化成本和换手协议；不得用同一 2021–2026 历史区间重新宣称独立发现。
