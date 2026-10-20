# 研究证据、Future Lockbox 与 Paper Trading 治理

## 证据分层

- `training`：2006–2015，可用于拟合；
- `validation_selection`：2016–2020，可用于参数和模型选择；
- `historical_oos`：2021–2026-09-30，拟合时未使用，但研发中已被反复查看，永久标记为 consumed；
- `future_lockbox`：2026-10-01 之后按冻结规则自动积累、在门槛满足前不查看绩效；
- `paper_trading`：候选提交后真实时间到达的信号、订单、成交和账户记录，禁止历史回填。

Future lockbox 与 paper trading 可能覆盖相同日历区间，但研究角色不同：前者用于一次性封存评估，后者用于可观察的运营与执行验证。查看 paper-trading 结果后仍可修改模型，但必须生成新候选版本，不能把修改后的结果继续归于旧版本。

## 当前冻结候选

主假设固定为验证期选出的 `hgb_shallow15_locked / raw / risk_constrained` 相对行业及六风格匹配组合的增量净收益。HGB industry+size neutral、Ridge raw、Ridge industry+size neutral 是三个次级假设。四项策略检验使用共同日频时钟、20 日 circular block bootstrap 和 BH-FDR 5%。

执行主口径为 C0 固定 20 bps 往返成本与日频 blocked-order retry；C2 公共税费加显式佣金假设是次级成本场景。无盘口历史、无券商冲击校准和不完整盘中涨跌停队列继续作为限制披露。

## 操作命令

候选版本只初始化一次：

```bash
quant-research-governance initialize
quant-research-governance status
```

访问任何研究证据时追加事件：

```bash
quant-research-governance record-access \
  --actor researcher \
  --evidence-class historical_oos \
  --action DIAGNOSTIC_REVIEW \
  --reason "documented diagnostic purpose" \
  --result-accessed
```

访问 future lockbox 结果时必须提供其 `lockbox_id`。任何 `VIEW_RESULTS`、`EXPORT_RESULTS` 或 `USE_FOR_CHANGE` 事件都会被视为结果访问；future lockbox 状态随即永久变为 `OPENED_CONSUMED`。日志逐行使用前序哈希构成 SHA-256 链，能检测修改或删改，但本地实现不是独立托管或密码学保密系统。

## 开箱门槛

`future_lockbox_2026q4_2027q3_v1` 只有同时满足以下条件才计划开箱：

- 到达 2027-09-30；
- 至少 252 个日频观测；
- 至少 52 个周频信号；
- 至少 12 个日历月；
- 数据覆盖率至少 95%。

开箱前允许查看数据下载、任务状态、schema、缺失率和账本对账等运营健康信息，不允许查看收益、Sharpe、回撤、IC 或相对基准绩效。
