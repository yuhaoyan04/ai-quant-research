# 标签期限与目标变换实验

## 1. 研究问题

固定股票池、15 个因子、预处理、模型和交易成本，只改变两个对象：预测未来多少天，以及
回归原始收益还是同日截面排名。实验共六组：5/10/20 日 × raw-return/rank regression。

这一步不是寻找最大的回测数字，而是回答：标签是否和周频交易匹配；原始收益中的尺度信息
是否带来稳定增量；更长预测期限的信号是否只是由重叠标签制造的虚假显著性。

## 2. 可执行标签

设信号日为 $t$，执行滞后为一个交易日，日收益为 $r_{i,s}$。$h$ 日标签为：

$$
y^{(h)}_{i,t}=\prod_{k=1}^{h}(1+r_{i,t+1+k})-1,
\qquad h\in\{5,10,20\}.
$$

因此标签从 $t+2$ 开始，不使用已经进入 $t$ 日收盘信号的收益，也不给缺失未来价格伪造零收益。

raw-return 模型直接令 $\tilde y_{i,t}=y^{(h)}_{i,t}$。rank 模型先在每个日期独立计算：

$$
\tilde y_{i,t}=
\frac{\operatorname{rank}_t(y^{(h)}_{i,t})-
\overline{\operatorname{rank}_t(y^{(h)}_t)}}
{s_t(\operatorname{rank}_t(y^{(h)}_t))}.
$$

前者保留收益幅度，因而高波动日期和极端收益对拟合影响更大；后者只学习股票相对次序，对
横截面极端收益更鲁棒。

## 3. 防泄露与可比性

- 2006–2015 训练，2016–2020 验证，2021–2026 historical OOS。
- 2021–2026 已在项目开发中查看，不是 pristine lockbox。
- 每个期限在训练年末分别 purge 5/10/20 个交易日；验证末端统一采用最保守的 20 日边界。
- 因子只用同日截面 1%/99% 缩尾、中位数填补、z-score 和缺失指示，不跨日期估计均值方差。
- 每种标签先按 own-horizon 验证 RankIC 选择 Ridge $\lambda$，再以统一的下一周 5 日
  Top-200 组合 Sharpe 选择标签。所有组合使用 20 bp round-trip cost。
- 10/20 日标签可以用于 RankIC，但不能把相互重叠的 10/20 日收益当作独立周收益连续复利；
  因而六组模型的交易评价始终使用同一个 5 日收益。
- OOS 每年年初扩展窗口重估；测试年及其未来从不进入拟合或选择。

模型为：

$$
\hat\beta=\arg\min_\beta\sum_{t,i}(\tilde y_{i,t}-x_{i,t}'\beta)^2
+\lambda N\lVert\beta\rVert_2^2.
$$

## 4. 验证期选择

六种标签的 own-horizon RankIC 最优惩罚均为 $\lambda=1$。两阶段预注册规则最终选择
`h20_raw_return`：验证平均 RankIC 为 0.0923，统一 5 日交易组合 Sharpe 为 3.20。

这个验证 Sharpe 远高于 rank 模型，并不是可以直接相信的“新 alpha”。raw-return 回归按收益幅度
加权训练样本，会让少数高波动时期对系数影响更大；因此必须依靠后续固定规则 OOS 检查稳定性。

## 5. Historical OOS 结果

| 标签 | Mean RankIC | HAC t | 20bp 后年化收益 | Sharpe | 最大回撤 |
|---|---:|---:|---:|---:|---:|
| 5d raw | 0.0895 | 10.45 | 15.62% | 0.61 | -40.00% |
| 5d rank | 0.0977 | 10.60 | 21.22% | 0.80 | -36.96% |
| 10d raw | 0.1083 | 9.44 | 17.18% | 0.65 | -40.07% |
| 10d rank | 0.1160 | 9.49 | 23.71% | 0.86 | -36.80% |
| 20d raw（验证优胜） | 0.1325 | 8.82 | 18.10% | 0.67 | -39.82% |
| 20d rank | 0.1376 | 8.35 | 26.50% | 0.94 | -35.87% |

HAC t-stat 按 5/10/20 日分别采用 1/2/4 周 Newey–West lag，修正重叠标签导致的序列相关。
更长期限的 RankIC 更高是合理现象，因为累计窗口信噪比更高；它不能直接与 5 日 RankIC
解释为同一种预测难度，也不能替代实际周频组合评价。

验证优胜 `h20_raw_return` 相对 5 日 rank 基线的 OOS 年化算术净收益差为 +0.03 pct，13 周
block bootstrap 95% 区间为 [-5.09, +5.03] pct。它没有证明可交易优越性，并且在 2022、
2024、2026 三个年度产生负收益。验证期的异常高表现没有泛化。

## 6. 事后发现与研究纪律

事后看，`h20_cross_sectional_rank` 相对 5 日 rank 基线的年化算术净收益差为 +4.38 pct，
区间 [+1.50, +7.01] pct；10 日 rank 的差为 +2.13 pct，区间 [+0.35, +3.92] pct。

但这两个结果是在 historical OOS 打开后才知道的，不可据此替换正式选择。正确处理是：

1. 保留预注册赢家失败的原始结论；
2. 将“20 日 rank 标签可能更适合低换手周频选股”登记为新假设；
3. 未来用新增时间数据、nested walk-forward 或独立市场验证，而不是在当前 OOS 再选一次。

这比把最好的测试结果包装成最终模型更接近真实机构研究流程。

## 7. 可复现资产

- 配置：`config/label_horizon.yaml`
- 面板构建：`src/quant_research/label_panel.py`
- 实验实现：`src/quant_research/label_horizon.py`
- 数据审计：`label_horizon_panel_audit/v1`
- 验证网格与选择：`label_horizon_validation_grid/v1`、`label_horizon_selection/v1`
- OOS 汇总、年度结果与区块检验：`label_horizon_oos_summary/v1`、
  `label_horizon_oos_annual/v1`、`label_horizon_oos_bootstrap/v1`
