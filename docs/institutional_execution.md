# Phase 5：机构级日频执行近似

## 目的

本阶段把冻结的 raw HGB 风险约束目标权重转成可审计的人民币股数订单、事件、成交、持仓、现金和
NAV。它只改变执行层，不重新训练模型、不修改目标组合，也不根据 2021–2026 结果调参。

## 日内时序

每个交易日依次执行：T+1 库存转为可卖、应用公司行动代理、按收盘价计价得到交易前 NAV、如有新
目标则替换旧订单、生成股数订单、先卖后买、按成交量与交易状态产生 fill、计提实际 fill 费用、
更新现金和持仓、核对交易后 NAV。收益不会在收盘价格和供应商涨跌幅上重复计入。

## 订单与账本

- 状态事件：`NEW`、`BLOCKED`、`PARTIALLY_FILLED`、`FILLED`、`CANCELLED`、`EXPIRED`、`REPLACED`；
- 每个 fill 保存订单 ID、日期、方向、下单/成交股数、收盘成交价、名义金额、费用分项、限制原因、
  成交前持仓与可卖股数；
- 主板买入采用 100 股整手；科创板最低 200 股、超过部分按 1 股递增；零股只允许随清仓卖出；
- 普通股票执行 T+1，买入股数当日记入 `unsettled_shares`，下一交易日才进入 `available_shares`；
- 新目标把旧未完成订单记为 `REPLACED`，其余订单最多重试 5 个交易日后记为 `EXPIRED`。

上交所交易规则规定主板竞价买入通常为 100 股或其整数倍，零股余额卖出需一次处理；科创板官方
投教材料说明申报最低 200 股、超过部分可按 1 股递增。普通 A 股在交收前不得卖出。参考：
[上交所交易规则](https://www.sse.com.cn/lawandrules/sselawsrules2025/fund/trading/c/c_20260424_10817739.shtml)、
[科创板交易规则说明](https://edu.sse.com.cn/tib/qa/c/4761956.shtml)、
[深交所回转交易说明](https://investor.szse.cn/knowledge/t20171010_538497.html)。

## 输出表

| 表 | 内容 |
|---|---|
| `institutional_execution_orders/v1` | 每张订单的原始、已成交、剩余股数与终态 |
| `institutional_execution_events/v1` | 不可静默丢失的状态转换和受阻原因 |
| `institutional_execution_fills/v1` | 成交及 fill 级费用、T+1 审计字段 |
| `institutional_execution_positions/v1` | 每日实际、可卖与未交收股数 |
| `institutional_execution_daily/v1` | 现金、市值、NAV、换手、费用和目标偏离 |
| `institutional_execution_invariants/v1` | 现金、持仓、T+1、费用与 NAV 验收结果 |

## 限制

这不是交易所或券商撮合器。没有可靠分钟、逐笔、盘口和券商订单数据时，不估计 bid/ask、排队位置、
真实成交概率或精确冲击。公司行动使用价值保持代理而不是官方权益分派表；IPO 无涨跌幅限制窗口也
尚未完整建模。所有历史数值属于已查看的 historical OOS 诊断，不是 pristine lockbox 或实盘业绩。
