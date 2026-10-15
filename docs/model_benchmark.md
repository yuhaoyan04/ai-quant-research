# 统一模型基准实验

## 1. 研究问题

复杂模型是否真的比简单线性模型更有价值？这个问题不能通过各自使用不同特征、样本或回测
代码来回答。本实验固定：

- PIT 股票池与 5 日可执行 rank 标签；
- 15 个因子、1%/99% 截尾、同日 z-score 和缺失指示；
- 2006–2015 训练、2016–2020 验证、2021–2026 historical OOS；
- 每年扩展窗口重训和 5 交易日 purge；
- Top-200 等权、周频调仓、漂移后换手率和 20 bp round-trip cost。

比较六个模型家族：方向等权 naive ensemble、OLS、Ridge、ElasticNet、RandomForest、HGB。
LightGBM 与 XGBoost 在当前环境未安装，明确登记为未执行，而不是用其他模型冒名替代。

## 2. 模型与公平性

OLS/Ridge 使用全量训练横截面的充分统计量：

$$
\hat\beta_{ridge}=(X'X+\lambda N I)^{-1}X'y.
$$

naive 模型只使用训练期单因子方向，15 个因子等权，缺失指示权重为零。ElasticNet 为：

$$
\frac{1}{2N}\lVert y-X\beta\rVert_2^2+
\alpha\left(\rho\lVert\beta\rVert_1+
\frac{1-\rho}{2}\lVert\beta\rVert_2^2\right).
$$

ElasticNet、RF 与 HGB 每个训练日期使用最多 300 条样本。样本由股票—日期哈希确定，与收益无关；
这控制内存和后期上市公司数量增长造成的年代权重变化。它们与全量 Ridge 的训练样本量不同，
因此报告中同时披露 fit observations 与耗时，不能将差异全部归因于算法结构。

Ridge、ElasticNet 和 HGB 的参数只按验证 Mean RankIC 在各自家族内选择。各家族代表再以同一
规则比较；historical OOS 不参与选择。

## 3. 验证期选择

| 家族 | 验证期代表 | Mean RankIC | 年化收益 | Sharpe |
|---|---|---:|---:|---:|
| Naive | directional equal weight | 0.0860 | 11.34% | 0.57 |
| OLS | OLS | 0.0868 | 19.92% | 0.79 |
| Ridge | $\lambda=1$ | 0.0889 | 19.11% | 0.77 |
| ElasticNet | $\alpha=0.01,\rho=0.5$ | 0.0885 | 17.79% | 0.73 |
| RandomForest | depth=6, 100 trees | 0.0991 | 69.98% | 1.99 |
| HGB | 15 leaves | 0.0985 | 66.95% | 1.91 |

验证规则正式选择 `random_forest_depth6`。HGB 的 31-leaf 配置虽然组合收益更高，但 Mean
RankIC 略低，因此没有被事后改选；这体现“先写规则、再看结果”。

## 4. Historical OOS

| 模型 | Mean RankIC | ICIR | 20bp 后年化收益 | Sharpe | 最大回撤 | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 0.0973 | 4.05 | 17.84% | 0.76 | -31.18% | 42.61% |
| OLS | 0.0980 | 4.74 | 23.10% | 0.83 | -38.15% | 49.40% |
| Ridge | 0.0977 | 4.53 | 21.22% | 0.80 | -36.96% | 56.30% |
| ElasticNet | 0.0975 | 4.75 | 23.98% | 0.85 | -38.37% | 53.66% |
| RandomForest | 0.1002 | 5.21 | 27.95% | 0.95 | -39.56% | 52.52% |
| HGB | 0.1000 | 5.05 | 25.24% | 0.87 | -40.06% | 68.42% |

以下组合收益来自“次日完整成交”的原始统一口径，现已确认对 RF 存在不可成交偏差；RankIC
结论仍有效，收益数字必须结合[RF 收益取证与成交约束审计](rf_return_forensics.md)阅读。

RF 相对 Ridge：

- RankIC 差 +0.00244，13 周 block-bootstrap 95% 区间 [-0.00272, +0.00785]；
- 年化算术净收益差 +5.94 pct，区间 [+1.30, +10.98] pct。

因此原实验只能说明 RF 在理想成交口径中产生组合收益增量，不能直接称为可实现收益，也不能
说全横截面预测相关显著提高。最终非重叠日频执行审计还修复了节假日短周的收益重叠，并把
失败订单改为按股数逐日重试；RF historical OOS 的同口径理想上界为 25.38%，执行约束后
21.62%、Sharpe 0.87、最大回撤 -43.27%。旧 24.19% 周频压力数字已被取代。

RF 相对 HGB 的年化算术净收益差为 +2.02 pct，区间 [-1.54, +5.75] pct；RankIC 差区间也
跨零，不能宣称 RF 显著优于 HGB。

RF 在六个 OOS 年份 RankIC 均为正，但 2024 年组合收益为 -6.16%，全期最大回撤接近 40%。
模型改进没有消除状态风险。

## 5. 计算成本

六次 OOS 年度重训总耗时：

| 家族 | 总拟合耗时 |
|---|---:|
| Naive | <0.01 s |
| OLS | <0.01 s |
| Ridge | 0.12 s |
| ElasticNet | 2.10 s |
| HGB | 33.38 s |
| RandomForest | 511.03 s |

RF 的收益优势伴随约 15 倍于 HGB、4,000 倍于 Ridge 的拟合时间。在只有 30 个设计列时仍可
接受，但在更大因子库、更多参数和日频重训场景中会成为实质研究成本。

## 6. 研究结论与限制

1. naive ensemble 已能获得正 IC，说明因子方向本身有信息，而非全部依赖复杂优化。
2. OLS、Ridge、ElasticNet 的 RankIC 非常接近；正则化主要改变尾部排序和换手，不是巨大预测跃迁。
3. RF/HGB 的组合收益高于线性模型，支持存在非线性或交互，但 RF 与 HGB 之间没有显著优劣。
4. RF 参数邻域已作为独立、仅验证集的 robustness study 完成。10 个邻居中 7 个通过预设
   门槛，低于 75% 要求，不能把单点高收益表述成宽阔稳定的平台；详见
   [RF 超参数局部稳健性](rf_hyperparameter_stability.md)。
5. 2021–2026 已被项目开发反复查看，只能称 historical OOS，而不是 pristine lockbox。
6. LightGBM/XGBoost 尚未执行；安装依赖后必须使用相同数据与选择协议，不能单独优化口径。

## 7. 可复现资产

- 配置：`config/model_benchmark.yaml`
- 实现：`src/quant_research/model_benchmark.py`
- 验证网格：`model_benchmark_validation_grid/v1`
- 家族选择：`model_benchmark_family_selection/v1`
- OOS 汇总与年度结果：`model_benchmark_oos_summary/v1`、`model_benchmark_oos_annual/v1`
- 配对检验：`model_benchmark_oos_bootstrap/v1`、`model_benchmark_winner_pairwise/v1`
- 训练耗时：`model_benchmark_fit_audit/v1`
