# 机构级研究缺口：最终对账与后续路线

## 当前结论

截至 2026-10-17，现有免费数据范围内的研究主链已闭环：PIT 数据库、因子 registry、预处理与标签消融、行业/规模中性化、单因子 tear sheet、模型 benchmark、HGB 二维参数面、组合风险约束、日频成交、容量/成本压力、风险模型、CAPM 与本地六风格归因、失败实验治理、最终报告和求职材料均已完成。

项目不再把“继续在 2021–2026 上调参数”列为研究任务。该区间已经被查看，继续优化只会增加事后选择。下一阶段应以新数据、授权数据或 paper trading 为信息增量。

## 19 项实验最终状态

| 编号 | 模块 | 状态 | 当前证据 | 真正剩余项 |
|---:|---|---|---|---|
| 1 | Factor Dictionary | 完成 | 机器可读 registry：公式、单位、字段、窗口、缺失、方向、版本 | 外部数据新增因子时扩展版本 |
| 2 | Winsorization | 完成 | raw / quantile / MAD 同样本消融与分布图 | 无 |
| 3 | Normalization | 完成 | raw / rank / z-score 对照 | 无 |
| 4 | Industry/Size Neutralization | 完成 | PIT CSRC 宽行业、规模、因子/模型/组合三层实验 | 申万细分行业需授权历史生效日数据 |
| 5 | Factor Correlation | 完成 | 热图、冗余簇、52 周滚动相关 | 无 |
| 6 | Single Factor IC | 完成 | RankIC/ICIR、decile、turnover、年度、状态、decay、LOFO | 未来数据验证 |
| 7 | Prediction Target | 完成 | 5/10/20 日 raw/rank，预注册选择 | 未来数据验证 20 日假设 |
| 8 | Model Benchmark | 完成 | naive/OLS/Ridge/ElasticNet/RF/HGB | LightGBM/XGBoost 只是可选工程扩展 |
| 9 | Hyperparameter Robustness | 完成 | RF OFAT + HGB 3×3 二维面与年度最优坐标 | 无 |
| 10 | Top-N Sensitivity | 完成 | Top50/100/200/500 共同日期 | 不据 historical OOS 重选 |
| 11 | Weighting Method | 完成 | 等权、inverse-vol、score/约束、风险感知 | 风险感知首版失败并保留 |
| 12 | Buffer Mechanism | 完成 | 200/400 缓冲与换手对照 | 无 |
| 13 | Cost Sensitivity | 完成到免费数据边界 | 0/20/50/100bp 固定成交路径 | 券商校准的非线性冲击 |
| 14 | ADV Capacity | 完成到情景边界 | 动态部分成交、现金、冲击与路径偏离 | 真实订单校准 |
| 15 | Factor Exposure | 完成 | 六风格 PIT 时序暴露与贡献 | 商业风险模型交叉验证 |
| 16 | Industry Exposure | 完成 | 宽行业主动权重、HHI、约束与归因 | 申万细分行业 |
| 17 | Volatility Control | 完成且拒绝首版 | lagged risk optimizer + target-vol；收益显著受损 | 新验证期前不调预算 |
| 18 | CAPM Alpha | 完成 | 三个 CSI 指数、NW 区间、年度/滚动 | 总收益指数与真实无风险利率 |
| 19 | Local Style Attribution | 完成 | 六个 PIT 自融资风格、HAC 回归、VIF | 公告日 profitability/investment 数据 |

## 已交付

- `reports/final_research_report.md` 与 `reports/final_report.pdf`；
- 一页 executive summary；
- 简历 bullet、30 秒介绍、5 分钟与 15 分钟面试稿；
- 不可变实验索引、invalidation 记录和 0 FAIL publication audit；
- Windows 一键全量重建脚本、跨机器复现指南和离线 dashboard。

## 下一阶段优先级

### P0：真正的新证据

1. 冻结当前参数，按周运行 paper portfolio；
2. 记录信号生成时间、目标订单、实际可成交价、拒单/部分成交和费用；
3. 至少积累 3–6 个月未参与历史调参的数据，再评价模型与风格暴露漂移。

### P1：授权数据升级

1. 接入带生效日期的申万二/三级行业；
2. 接入公告日可用的盈利、投资、质量基本面；
3. 使用含股息总收益指数、真实无风险利率和交易所/券商费用表；
4. 用真实订单校准冲击与容量参数。

### P2：生产化

加入定时增量数据、数据修订检测、模型漂移监控、订单风控、Paper/Live 对账和告警。该部分是交易系统工程，不应与当前研究回测混为一谈。

## 不再做的事

- 不因 2021–2026 表现继续切换模型、Top-N、权重或风险预算；
- 不把诊断多空价差当作可融券实盘收益；
- 不把 CAPM 截距称作纯 Alpha；
- 不用当前行业分类回填历史；
- 不覆盖失败或旧版本实验来获得“干净”的结果目录。
