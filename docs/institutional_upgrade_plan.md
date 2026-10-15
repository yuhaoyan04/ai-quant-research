# 从简历项目到机构级 Quant Research：缺口审计与升级路线

## 1. 审计结论

`quant_prompt.md` 给出的 19 项实验覆盖了因子、标签、模型、组合、风险和归因，是合理的研究框架。但当前项目不是“只完成了一个模型”：数据时序、PIT 股票池、严格标签、Ridge/HGB、持仓缓冲、成本、动态容量、规模压力、涨跌停和 block bootstrap 已经明显超过普通 toy case。

真正的差距集中在四个方面：

1. **实验治理不完整**：缺少统一 `experiment_id`、数据版本、配置快照和实验目录；
2. **因子研究没有形成完整 tear sheet**：缺 preprocessing ablation、IC decay、分组收益、因子换手和增量性检验；
3. **风险模型与收益归因不足**：缺可靠 PIT 行业中性化、组合风格暴露、波动控制和统计回归归因；
4. **模型与组合横向基准不足**：已有 Ridge/HGB，但缺 ElasticNet/LightGBM、参数稳定面和权重方案对照。

此外，2021–2026 已在开发中被多次查看，只能称为 historical out-of-sample，不能再称为从未打开的 pristine lockbox。后续所有报告必须继续保留这一披露。

## 2. 19 项实验逐项对账

| 编号 | `quant_prompt.md` 实验 | 当前状态 | 现有证据 | 仍需补齐 |
|---:|---|---|---|---|
| 1 | Factor Dictionary | 部分完成 | `docs/factor_catalog.md` 已有 15 因子定义、逻辑和 caveat | 补单位、更新频率、原始字段、可用日、缺失机制、方向假设和版本 |
| 2 | Winsorization | 未完成实验 | 已实现每日 median/MAD 缩尾 | 缺 raw / quantile / MAD 的同样本消融对照 |
| 3 | Cross-sectional Normalization | 未完成实验 | 已实现同日 z-score 与缺失指示 | 缺 raw / rank / z-score 的对照分布、IC 和模型结果 |
| 4 | Industry/Size Neutralization | 关键缺失 | 已做规模剔除压力，没有中性化 | 需要可靠 PIT 行业分类；比较 raw、industry、industry+size residual |
| 5 | Factor Correlation | 已完成 | `factor_correlation`、冗余簇与高相关因子对 | 补可视化 heatmap 和滚动相关稳定性 |
| 6 | Single Factor IC | 部分完成 | 单因子 RankIC、ICIR、训练/验证方向稳定性 | 缺 IC decay、分位数组合、因子换手、年度/市场状态 tear sheet |
| 7 | Prediction Target | 未完成，且需重构 | 已有 5/10/20 日 lag1 标签，正式模型只用 5 日 rank target | 比较预测期限与 raw/rank regression；同日减统一 benchmark 不改变排名，应避免伪实验 |
| 8 | Model Benchmark | 部分完成 | Ridge 与 HistGradientBoosting，walk-forward | 补 naive factor ensemble、OLS/ElasticNet、LightGBM；RF 可作为可选而非硬性要求 |
| 9 | Hyperparameter Robustness | 部分完成 | Ridge penalty grid、3 个 HGB 复杂度 | 补二维稳定面、参数邻域、年度选择稳定性和误差区间 |
| 10 | Top-N Sensitivity | 部分完成 | 验证比较 Top50/100/200 | 补 Top500、共同日期测试曲线和容量/换手联动 |
| 11 | Weighting Method | 未完成 | 当前为等权 | 比较等权、rank-score、inverse-vol、约束优化；个股 risk parity 不宜机械使用 |
| 12 | Buffer Mechanism | 已完成 | no buffer 与 250/300/400 exit 对照 | 已得到换手下降约 24.5% 的样本外证据 |
| 13 | Cost Sensitivity | 基本完成 | 0/10/20/40bp | 扩到 50/100bp，并按成交规模联动冲击而非只用线性费率 |
| 14 | ADV Capacity | 已完成且更深入 | 静态 ADV + 动态部分成交、现金、冲击、路径偏离 | 未来用真实订单校准冲击系数 |
| 15 | Factor Exposure | 部分完成 | 有规模与流动性压力 | 缺组合对 market beta、size、value、momentum、volatility 的时序暴露 |
| 16 | Industry Exposure | 未完成 | 无可靠 PIT 行业表 | 数据源是前置依赖；需要行业权重、主动权重和集中度 |
| 17 | Volatility Control | 未完成 | 只报告 realized volatility | 补 target-vol、杠杆/现金上限和危机期表现 |
| 18 | CAPM Alpha | 未完成 | 有 CSI 300/500/1000 基准超额 | 补 OLS + Newey–West alpha、beta、置信区间和滚动 beta |
| 19 | Fama–French Attribution | 未完成且需本地化 | 无 | 不能直接套美国因子；应构建 PIT 中国 size/value/profitability/investment/momentum 风格组合 |

最终报告方面，当前已有 Markdown、HTML、SVG/CSV 和自动审计，但没有 `final_report.pdf`。PDF 应在实验矩阵稳定后生成，而不是先排版再补研究。

## 3. 不应机械照做的地方

### 3.1 “未来收益 vs 超额收益”可能是同一个排名

若某日对所有股票减去同一个市场收益 $r_{m,t}$：

$$
y^{excess}_{i,t}=y_{i,t}-r_{m,t},
$$

则当天任意两只股票 $i,j$ 满足：

$$
y^{excess}_{i,t}>y^{excess}_{j,t}
\iff y_{i,t}>y_{j,t}.
$$

因此横截面 rank label 和 RankIC 完全不变。只有当目标是原始数值回归、使用股票特异 benchmark，或先做行业/风格残差化时，该对比才有信息量。

### 3.2 模型数量不等于研究质量

Linear Regression、Lasso、ElasticNet、RandomForest、HGB、LightGBM 全跑一遍，不会自动变成机构研究。更重要的是：统一数据、统一时间切分、相同候选池、相同成本、配对日期和参数选择只发生在验证期。

建议核心 benchmark 为：

1. 单因子等权或 RankIC 加权的 naive ensemble；
2. OLS / Ridge / ElasticNet；
3. HGB；
4. LightGBM。

RandomForest 对百万级横截面数据训练慢、外推弱，可作为补充，不应阻塞主线。

### 3.3 A 股归因不能直接使用美国 FF5

如果使用美国市场下载的 SMB/HML/RMW/CMA 回归 A 股收益，经济对象、交易时区、币种和投资者结构都不匹配。正确做法是用本项目 PIT A 股池构建中国本地风格因子收益，完整记录分组断点、权重、再平衡频率和交易成本；否则只能称为近似风格归因，不能称为严格 Fama–French 5 因子检验。

### 3.4 个股 Risk Parity 未必是合适的 alpha 权重基准

200 只股票的协方差矩阵噪声很大。优先级应是：等权 → score/rank 权重 → inverse-vol → 带 shrinkage covariance、行业/个股/换手约束的优化器。只有定义清楚风险预算后，才讨论 risk parity。

## 4. 推荐升级顺序

### Sprint 0：冻结基准与实验注册系统

这是所有新增实验的前置条件。每次实验自动保存：

```text
reports/experiments/<experiment_id>/
├── config.yaml
├── metadata.json
├── metrics.json
├── tables/
├── figures/
└── notes.md
```

`metadata.json` 至少包含代码版本或源码哈希、dataset manifest hash、Python 与依赖版本、样本日期、输入表、随机种子、运行状态，以及是否使用 historical OOS 做选择。

验收：同一配置重复运行得到相同关键指标；现有 Ridge/HGB 结果注册为 frozen baseline，不能被新实验覆盖。

### Sprint 1：Factor Lab

1. 将 `factor_catalog.md` 升级为机器可读 factor registry；
2. raw / quantile winsor / MAD winsor 对照；
3. raw / rank / z-score 对照；
4. 为 15 因子生成 coverage、分布、年度 RankIC、ICIR、decay、quantile return、turnover tear sheet；
5. 相关性、聚类和 leave-one-factor-out 增量检验。

选择只使用训练与验证期；historical OOS 只作固定规则报告。

### Sprint 2：标签与预测期限

优先比较可执行的 5/10/20 日标签：raw-return regression、cross-sectional rank regression，以及等待中性化数据后的行业/规模 residual return。

评价 RankIC、IC decay、Top-N 成本后收益、标签重叠、换手和年度稳定性。最终选择不能只看一个最佳点，应说明预测期限与调仓频率是否匹配。

### Sprint 3：PIT 行业与风险暴露

1. 获得带生效日期的历史行业分类；
2. 建立 daily cross-sectional residualizer；
3. 比较 raw、industry-neutral、industry+log-size-neutral；
4. 计算组合行业主动权重、HHI 和风格暴露；
5. 验证 alpha 是否在中性化后仍存在。

这是当前研究可信度最高优先级的缺口。不能用“今天的行业分类”回填历史。

### Sprint 4：模型 benchmark 与参数稳定性

在同一特征与切分下运行 naive / OLS / Ridge / ElasticNet / HGB / LightGBM。输出验证与 historical OOS RankIC/ICIR、Top-N 收益、Sharpe、IR、MDD、换手、训练预测耗时、年度稳定性和参数邻域热图。

模型选择规则必须在运行 historical OOS 前写入 config。

### Sprint 5：组合构建与风险控制

固定 alpha 后比较 equal weight、positive rank-score weight、inverse-vol weight 和 constrained optimizer。优化器至少包含个股最大权重、行业主动权重、风格暴露、换手惩罚、ADV 交易上限；随后加入 target-vol overlay，明确现金和杠杆上限。

### Sprint 6：绩效归因

1. 相对 CSI 指数的 CAPM alpha/beta，使用 Newey–West 标准误；
2. 自建中国风格因子组合；
3. 全期与滚动窗口回归；
4. 收益拆分为 market、style、industry、stock-specific、cost；
5. 对 alpha 的 t 值、区间和状态稳定性进行解释。

### Sprint 7：最终求职交付

- 自动生成 `final_report.pdf`；
- 一页 executive summary；
- 完整实验索引和失败实验记录；
- 5 分钟项目介绍稿、15 分钟深挖稿；
- 可现场运行的小规模 reproducibility demo。

## 5. 优先级与求职价值

| 优先级 | 工作 | 为什么最值钱 |
|---|---|---|
| P0 | 实验注册、数据版本、冻结 baseline | 证明研究可复现，防止结果覆盖和隐性挑选 |
| P0 | PIT 行业/规模中性化 | 回答收益是否只是行业和小盘暴露 |
| P0 | 因子 tear sheet + IC decay | 展示真正的 alpha research 能力 |
| P1 | 标签期限与目标对照 | 说明预测目标和交易频率为何匹配 |
| P1 | 统一模型 benchmark + 参数稳定面 | 证明不是只挑一个幸运模型 |
| P1 | 组合权重、风险约束、target vol | 从预测研究走向 portfolio research |
| P1 | CAPM/本地风格归因 | 解释收益来源，而不只展示净值 |
| P2 | 100bp 成本、更多容量场景 | 完善压力边界 |
| P2 | PDF 与面试材料 | 最终包装，必须建立在研究完成之后 |

## 6. 下一步的具体实施

> 实施状态（2026-10-14）：Sprint 0 已完成。`baseline_ridge_v1` 与
> `challenger_hgb_blend_v1` 已写入不可覆盖实验注册表，记录完整配置、输入与数据集哈希、
> 代码树、运行环境、指标和图表；`quant-audit` 会校验快照完整性和当前输入是否漂移。
> 下一项正式实验为 `factor_preprocessing_ablation_v1`。

> Sprint 1 进度：3×3 预处理消融的验证期选择已完成并锁定。选择结果为
> `quantile(1%,99%) + cross-sectional z-score + Ridge λ=1.0`；完整解释见
> [因子预处理消融实验](factor_preprocessing_ablation.md)。下一步是在不重新选参的前提下运行
> 2021–2026 历史 OOS，并继续完成 IC decay 与 leave-one-factor-out。

> Sprint 1 OOS 状态：已完成。锁定的 Quantile+Z-score 方案在 2021–2026 的平均
> RankIC 为 0.09772，20bp 后年化收益为 21.22%；相对 MAD+Z-score 的成本后年化
> 算术增量为 +0.98 pct，13 周块 bootstrap 95% 区间为 [+0.02, +1.91] pct。
> RankIC 增量区间仍跨 0，因此结论限定为交易效率改善。Sprint 1 剩余 IC decay 与
> leave-one-factor-out。

> IC decay 已完成：模型在首个 5 日窗口 RankIC 为 0.0977，第 6–10 日与第
> 11–20 日独立窗口仍有 0.0744 和 0.0768。信号衰减但没有在一周后消失；下一项为
> leave-one-factor-out，检验这些单因子方向能否转化为独立模型增量。

> Leave-one-factor-out 已完成：`amihud_20d` 是唯一在验证与 historical OOS 都有明确
> 正边际贡献的因子；波动率/振幅/最大收益簇具有强单因子 IC 但边际贡献接近零；多个
> 验证期信号在 OOS 方向反转。Sprint 1 因子预处理、IC decay 与 LOFO 已闭环，下一阶段
> 进入标签期限对照与 PIT 风险中性化。

> Sprint 2 标签期限对照已完成：在统一 PIT 股票池和周频调仓下比较 5/10/20 日
> raw-return 与 cross-sectional-rank 回归。验证期预注册规则选择 `h20_raw_return`，但其
> historical OOS 相对原 5 日 rank 基线的年化算术净收益增量仅 +0.03 pct，13 周块
> bootstrap 95% 区间为 [-5.09, +5.03] pct，未证明优越。事后诊断中 20 日 rank 表现较好，
> 但不得用已查看的 OOS 反向改选，只登记为未来数据上的待确认假设。完整结果见
> [标签期限与目标变换实验](label_horizon.md)。

> Sprint 3 第一阶段已完成 PIT 规模中性化。原模型 OOS 分数与 log 规模平均相关 -0.069，
> Top-200 平均位于规模第 34.4 百分位；逐日规模残差化后相关为数值零、持仓提升到第 46.5
> 百分位。20bp 后年化收益由 21.22% 降至 16.30%，中性化减原始模型的年化算术净收益差
> 为 -4.85 pct，13 周块区间 [-8.76, -0.54] pct，说明原收益显著包含小盘暴露；但中性化后
> RankIC 仍为 0.0944、Sharpe 0.69，并未完全消失。PIT 行业中性化仍等待带生效日期的历史
> 分类数据，不能用当前行业回填。详见 [PIT 规模中性化实验](size_neutralization.md)。

> Sprint 4 统一模型基准已完成：naive、OLS、Ridge、ElasticNet、RandomForest 与 HGB 在
> 同一量化预处理、标签、股票池、时间切分和成本口径下比较。验证期正式选择深度 6 RF；其
> historical OOS Mean RankIC 0.1002；理想完整成交口径下 20bp 后年化收益 27.95%、Sharpe 0.95。相对 Ridge 的
> 年化算术净收益增量为 +5.94 pct，区间 [+1.30, +10.98] pct；但 RankIC 增量区间跨零，且
> RF 相对 HGB 的收益差区间也跨零。六次年度重训 RF 约耗时 511 秒，HGB 约 33 秒、Ridge
> 约 0.12 秒。LightGBM/XGBoost 因当前环境无依赖而明确登记为未运行。详见
> [统一模型基准](model_benchmark.md)。

> Sprint 4 的 RF 参数稳健性补充实验已完成：仅使用 2006–2015 训练和 2016–2020 验证，
> 对深度、叶节点样本数、特征比例、树数量及随机种子做 one-factor-at-a-time 扰动。10 个邻居
> 中 7 个通过预设 IC/Sharpe 下限，70% 低于 75% 总门槛，因此正式结论为“局部稳健性未通过”。
> 随机种子与树数量稳定，但降低模型容量的三个配置显著损害 Top-200 表现；不据此事后改选
> 参数。详见 [RF 超参数局部稳健性](rf_hyperparameter_stability.md)。

> RF 收益取证已完成：复利公式、收益恒等式与 250 个原始日线标签复算均无误，但组合层存在
> 节假日短周持有期重叠，模型也显著偏向小盘和次日难成交股票。最终引擎用唯一日频净值时钟、
> 股数订单和失败订单逐日重试；验证期同口径理想上界 57.46%、约束后 10.77%，historical OOS
> 从 25.38% 降至 21.62%、Sharpe 0.87、最大回撤 -43.27%。原模型 RankIC 与选模结论保留，
> 69.98%、19.09% 和 24.19% 组合数字均已登记为被新实验取代。
> 详见 [RF 收益取证与成交约束审计](rf_return_forensics.md)。

第一步应实现 **Sprint 0：Experiment Registry + Frozen Baseline**，而不是立刻再训练一个模型。原因是后面每个 winsorization、label、model 或 weighting 实验都需要相同记录协议；如果先做实验再补治理，会再次产生散落的表和不可比较的口径。

完成 Sprint 0 后，第一个正式新增实验是：

> `factor_preprocessing_ablation_v1`：在完全相同的股票池、标签、周频日期和 Ridge 模型下，对比 raw、quantile winsor、MAD winsor，以及 rank/z-score 变体。

验收指标：

- 每个版本的 coverage、截面分布和异常率；
- 15 因子的平均 RankIC、IC 波动和 ICIR；
- Ridge 验证 RankIC/ICIR；
- 固定 Top-200 组合的收益、换手和成本后表现；
- 参数和结果全部写入独立 experiment artifact；
- 不用 historical OOS 选择预处理版本。
