# Phase 6：组合权重与约束优化

## 研究边界

本阶段固定 raw `hgb_shallow15_locked` 分数、PIT 截面、调仓时钟、20bp 成本和 Phase 5 账户执行器，只比较目标权重构造。2021–2026 已被查看；P4 的约束支持与显式回退处理又依据这段历史中的结构性可行性失败完善，因此实验属于 `post_historical_oos_adaptation`，不是 pristine OOS。

## 方法

- P0：Top-200 新进、Top-400 旧候选保留后的等权；
- P1：只使用信号日已知的 20 日波动率，5%/95% 截尾后逆波动率加权；
- P2：用截面 score percentile 的正值变换加权，不把 HGB 分数当作校准预期收益；
- P4：最大化 score rank，long-only、满仓、单票不超过 1%，行业主动权重 ±3%、`|size-z|≤0.10`、单边换手预算 30%。

普通候选严格遵守 Top-200/400。P4 如需承载行业下限或规模可行性，可加入单独标记的 `constraint_support` 股票；这不是普通入选。LP 不可行时不放松约束，而是记录失败并采用显式 carry-forward capacity repair，供执行路径连续性诊断。

## 执行与输出

四条路径使用人民币股数、未复权收盘价、停牌/涨跌停、10% 日成交量参与率、T+1、主板/科创板申报数量规则及实际 fill 费用。主结果为 1 亿元；2500 万与 5 亿元完整重跑容量敏感性。

主要数据表：

| 表 | 内容 |
|---|---|
| `portfolio_construction_signal_panel/v1` | 逐行复现的 raw HGB 全截面分数与 PIT 风险输入 |
| `portfolio_construction_targets/v1` | 四种方法的目标权重、缓冲和支持标记 |
| `portfolio_construction_audit/v1` | 候选数、目标换手、LP 状态、回退与风险暴露 |
| `portfolio_construction_execution_*/v1` | 日账本、订单、fill、持仓、公司行动与不变量 |
| `portfolio_construction_capacity/v1` | 三档资金规模的真实执行重跑 |
| `portfolio_construction_bootstrap/v1` | 相对 P0 的日频配对循环块区间 |
| `portfolio_construction_acceptance_checks/v1` | long-only、满仓、共同日期、无静默放宽和账本门禁 |

数值结论见[Phase 6 报告](../reports/portfolio_construction_comparison.md)。当前决策是保留 P0；P1 仅有轻微风险效率改善，P2 明显较差，P4 风险下降但可行性与执行跟踪不足。
