# Phase 7：现金、持仓、NAV 与目标权重统一时序账本

## 定位

本阶段只读规范化已冻结的 Phase 5 主账户，不改变目标、成交、费用、持仓或收益。
2021–2026 仍是已查看的 historical OOS；本阶段是会计与谱系诊断，不构成新 Alpha 证据。

## 时序约定

`mark_existing_positions_then_replace_orders_then_close_execute_then_posttrade_mark`：先公司行动与盯市，得到交易前 NAV；新目标替换旧订单；随后在收盘执行、扣费并更新现金/持仓；最后计算交易后 NAV、实际权重和现金权重。

## 统一对象

- `target_weight` 只来自冻结目标；`actual_shares` 只来自账户持仓；`actual_weight` 由实际股数、收盘价与 NAV 计算；`cash_weight` 单列。
- 未成交订单仅在订单/事件表出现，不进入实际持仓。
- 公司行动从所选账户前日持仓与冻结市场字段重建，因为 Phase 5 旧公司行动合并表遗漏 `scenario`。旧表不被覆盖。

## 机器对账

| check_name                              | status   | observed                                                        |
|:----------------------------------------|:---------|:----------------------------------------------------------------|
| daily_cash_reconciles                   | PASS     | 1.1851079761981964e-07                                          |
| daily_nav_reconciles                    | PASS     | 1.1920928955078125e-07                                          |
| daily_market_value_reconciles           | PASS     | 1.1920928955078125e-07                                          |
| daily_fill_notional_reconciles          | PASS     | 0.0                                                             |
| daily_fees_reconcile                    | PASS     | 2.9103830456733704e-11                                          |
| cash_plus_actual_weights_equal_one      | PASS     | 6.661338147750939e-16                                           |
| target_actual_tracking_reconciles       | PASS     | 3.8781911709806494e-16                                          |
| position_share_transitions_reconcile    | PASS     | 0.0                                                             |
| cash_never_negative                     | PASS     | 0.6862432244688534                                              |
| positions_never_negative                | PASS     | 4.0                                                             |
| t_plus_one_sell_inventory               | PASS     | 0                                                               |
| order_fill_shares_reconcile             | PASS     | 0.0                                                             |
| orders_are_terminal                     | PASS     | ['EXPIRED', 'FILLED', 'REPLACED']                               |
| one_new_event_per_order                 | PASS     | 68262                                                           |
| corporate_actions_value_reconcile       | PASS     | 4.656612873077393e-10                                           |
| legacy_action_scenario_identity_present | WARN     | legacy table omitted scenario during two-scenario concatenation |

## 规模

- 账户日：1,387；证券日：296,714；订单：68,262；成交：81,650；公司行动：1,034。
- 异常/未完全成交订单：17,495；硬失败：0；警告：1。

## 结论

若所有硬门禁通过，则 Phase 7 证明已有收益序列可由现金、成交、费用、公司行动和实际持仓逐日重建；该结论只证明会计一致性，不证明策略未来有效。