# Phase 7：统一账户时序账本

Phase 7 将冻结的主候选账户规范化为可逐键查询的统一账本。它不重新生成信号、不改变成交，也不重算或优化收益；2021–2026 结果仍仅属于已查看的 historical OOS 诊断。

## 账户与时序

账户固定为 `candidate_bundle_v1_primary`，即 raw HGB 风险约束目标在 `daily_close_institutional_constraints` 场景下的 Phase 5 执行结果。

逐日顺序固定为：处理公司行动并按未复权收盘价盯市 → 计算交易前 NAV → 新目标替换旧订单 → 检查交易资格、现金、可卖股数、T+1、成交量和申报数量 → 收盘成交并扣费 → 更新现金与实际股数 → 计算交易后 NAV、实际权重与现金权重。收益只在“先盯市、后收盘成交”的约定下计入一次。

## 数据语义

| 字段 | 唯一含义 |
|---|---|
| `target_weight` | 冻结目标希望持有的权重 |
| `actual_shares` | 模拟账户真实持有股数 |
| `actual_weight` | `actual_shares × mark_price / NAV` |
| `cash_weight` | 现金除以 NAV |
| `weight_gap` | `actual_weight - target_weight` |

未成交订单只存在于订单和事件日志，绝不计入实际持仓。证券日主账以目标证券与实际持仓证券的并集为键，因此完全卖出、买入受阻和目标为零的情况仍可审计。

## 主要产物

- `unified_account_daily_ledger/v1`：账户日级现金、NAV、成交、费用和跟踪误差；
- `unified_account_security_ledger/v1`：逐证券的目标权重、实际股数、实际权重和差异；
- `unified_account_cash_journal/v1`：前日现金、买卖成交、费用、公司行动现金流与期末现金；
- `unified_account_orders/events/fills/v1`：带账户键的订单状态机明细；
- `unified_account_daily_reconciliation/v1` 与 `position_reconciliation/v1`：独立现金/NAV及股数迁移回放；
- `unified_account_order_exceptions/v1` 与 `failed_assertions/v1`：异常订单和失败断言。

## 已知历史缺口

Phase 5 将两个执行场景的公司行动表合并时遗漏了 `scenario` 字段。历史冻结产物不覆盖；Phase 7 改用所选账户的前日实际持仓和冻结市场行动字段，重建 1,034 条账户专属公司行动。该修复只恢复谱系身份，不改变原持仓或净值。

完整数值见[统一账本报告](../reports/unified_account_ledger.md)与[Phase 7 审计](audit/PHASE7_UNIFIED_ACCOUNT_LEDGER_AUDIT.md)。
