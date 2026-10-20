# Phase 0：项目架构审计与基线冻结

**审计日期：** 2026-10-20 UTC  
**审计范围：** `src/quant_research/`、`config/`、`tests/`、`docs/`、`reports/`、37 个冻结实验及其当前数据湖输入  
**执行边界：** 本轮只做只读核验和新增审计材料；未修改标签、因子、模型、组合、成本、成交或报告生成逻辑，未覆盖任何历史实验目录。

## 1. 审计结论

当前仓库已经形成一条完整的历史研究链路：PIT 股票池与因子、时间切分、Ridge/HGB、组合约束、日频受限成交、统计推断、风险归因和不可变实验登记都已有实现。当前公开结论——“可成交历史回测为正，但六风格后的纯选股 Alpha 未显著，Incremental Alpha 只提供局部证据”——与当前主基线 `final_research_baseline_v4` 的冻结指标一致。

但“历史研究链路完成”不等于已经达到机构级交易账本和前瞻验证标准。最重要的缺口是：

1. 日频执行器只有内存中的现金、持仓和待成交股数状态，以及按日汇总输出；没有持久化的 order/fill/position ledger，因而无法从逐笔成交独立重建费用和 NAV。
2. 所谓“股数订单”是以初始合成价格 1.0、允许小数股的归一化份额，不是按真实未复权收盘价、A 股整手和 T+1 约束执行的券商式账本；公司行动也未显式入账。
3. 20bp 是“往返费率 × 单边换手”的统一研究假设，不是按买卖方向、日期和市场分项计算的历史真实费率。
4. Sharpe、CAPM 和六风格回归均以 `rf=0` 为当前基线；市场指数为价格收益，不含股息。
5. 2021–2026 已被反复查看，只能称 historical OOS。当前不存在 pristine lockbox 或 paper-trading 证据。

因此，当前结果适合被描述为“可审计的历史研究基线”，不适合被描述为“完整机构交易系统”“历史真实成交”或“前瞻验证 Alpha”。

## 2. 本次冻结锚点

| 项目 | Phase 0 锚点 |
|---|---|
| Git HEAD | `db5cd282d05f0da61047d2ebf2bd9c09e9165032` |
| Git tree | `10ca49c33a18a9259c4dbca48bcf3b811885a625` |
| 提交时间 / 标题 | `2026-10-18T07:19:15+00:00` / `Add executable incremental alpha test` |
| 审计开始时 tracked worktree | 无已跟踪文件修改 |
| 审计开始时 untracked | `.ipynb_checkpoints/`、`QUANT_RESEARCH_INSTITUTIONAL_UPGRADE_PLAN.md` |
| 规格文件 SHA-256 | `8fbae3d332d239777c089f9520151bd03baa2052f788cb7dd7772ea40c0966ed` |
| `src/**/*.py + pyproject.toml` 组合哈希 | `6621a1334f61454b131d1038911cad8c9db777386d6fef996da14610a810e66f` |
| `config/*.yaml` 组合哈希 | `1d139e10fa61aa63a024ae75404cbce5ac9941863e5d7242f32d0897517912e4` |
| `tests/test_*.py` 组合哈希 | `f7698c1b2950f44b12e7b4fed7c7e6e56af5a7a672433d7628c3a0fdd3bf5e75` |
| 当前权威冻结实验 | `final_research_baseline_v4` |
| 当前结论审计 | `research_conclusion_audit_v2` |

组合哈希按相对路径排序，以 `relative_path + NUL + file_bytes` 逐文件输入 SHA-256。它们是本次审计锚点，不替代各冻结实验自己的 `FROZEN.json`。

## 3. 实际研究架构

```text
BaoStock raw bars / security snapshots / industry snapshots
  -> quality validation and historical A-share union
  -> PIT universe + raw factors + executable labels
  -> same-date cross-sectional preprocessing
  -> factor diagnostics / PIT industry-size residualization
  -> validation-only model selection and annual expanding refit
  -> frozen weekly target weights
  -> risk-constrained LP portfolios
  -> daily synthetic-share retry execution
  -> daily performance / CAPM / six-style attribution
  -> matched-peer incremental-alpha test
  -> immutable experiment registry / conclusion audit / publication reports
```

该架构已经分离周频标签组合与日频非重叠执行结果，但尚未完全分离“执行价格”“真实 fill”“逐证券持仓账本”“公司行动现金流”这四层对象。

## 4. 主要实验、入口与证据映射

| 研究层 | 入口与关键函数 | 主要配置 | 主要输入 | 主要输出 / 冻结实验 | 直接测试 |
|---|---|---|---|---|---|
| 原始数据与断点续传 | `free_ingest.py`、provider adapters | `free_sources.yaml`, `quality.yaml` | BaoStock / AKShare | calendar、security snapshots、bars、manifest | `test_free_ingest.py`, `test_baostock_provider.py`, `test_ingest.py` |
| PIT 股票池与标签 | `panel.py`, `label_panel.py` | `research_panel.yaml`, `label_horizon.yaml` | security snapshots、validated bars | eligible panels、multi-horizon labels | `test_panel.py`, `test_pit_universe.py`, `test_label_panel.py`, `test_label_horizon.py` |
| 因子与预处理 | `factors.py:build_raw_factors`；`preprocessing.py:preprocess_cross_section` | `factors.yaml`, `preprocessing.yaml` | validated bars、labels | factors、model features、audit | `test_quality_factors.py`, `test_preprocessing.py` |
| 因子中性化 | `factor_neutralization.py`；`risk_exposure.py` | `factor_neutralization.yaml`, `risk_exposure.yaml` | factor panel、PIT industry/size | factor summaries / bootstrap | `factor_neutralization_v2`; related unit tests |
| 单因子评价 | `factor_tear_sheet.py` | `factor_tear_sheet.yaml` | raw/neutralized factor panel | deciles、regimes、NW/bootstrap、BH-FDR | `factor_tear_sheet_v1`; `test_factor_tear_sheet.py` |
| Ridge | `model_baseline.py:purged_end_date,fit_ridge,evaluate` | `model_baseline.yaml`, `model_features.yaml` | yearly model features | validation grid、OOS predictions | `baseline_ridge_v1`（历史，已替代）；`test_model_baseline.py` |
| HGB challenger | `model_challenger.py:build_training_sample,make_model,blend_predictions` | `model_challenger.yaml` | model features、Ridge predictions | validation grid、OOS predictions | `challenger_hgb_blend_v1`（历史，已替代）；`test_model_challenger.py` |
| 统一模型比较 | `model_benchmark.py` | `model_benchmark.yaml` | multi-horizon panel | common-date Ridge/ElasticNet/RF/HGB metrics | `model_benchmark_v1`（组合收益有执行边界）；`test_model_benchmark.py` |
| 参数稳定性 | `hgb_hyperparameter_stability.py` | `hgb_hyperparameter_stability.yaml` | validation panel | 3×3 HGB validation surface | `hgb_hyperparameter_stability_v1`; related test |
| 基础组合 | `portfolio.py:simulate_long_only,performance_summary` | `portfolio.yaml` | model predictions | weekly Top-N metrics | `test_portfolio.py` |
| 组合风险约束 | `constrained_portfolio_neutralization.py:solve_risk_constrained_weights,portfolio_step` | `constrained_portfolio_neutralization.yaml` | PIT model scores/exposures | targets、weekly metrics、constraint audit | `constrained_portfolio_neutralization_v1`; 3 tests |
| 日频执行 | `daily_execution.py:_execute_share_orders,simulate_daily_retry`；`constrained_portfolio_daily_execution.py` | `constrained_portfolio_daily_execution.yaml` | weekly targets、daily returns/status | daily NAV aggregates、summary、paired execution loss | `constrained_portfolio_daily_execution_v1`; 7 direct tests |
| 周频风险归因 | `multi_factor_risk_attribution.py` | `multi_factor_risk_attribution.yaml` | constrained weekly portfolio、PIT exposures | factor returns、risk forecasts、exact return attribution | `multi_factor_risk_attribution_v1`; 3 direct tests |
| 日频 CAPM | `capm_attribution.py:newey_west_ols,capm_record` | `capm_attribution.yaml` | executable daily return、CSI price returns | CAPM summary/annual/rolling | `capm_attribution_v1`; 2 direct tests |
| 日频六风格归因 | `style_factor_attribution.py` | `style_factor_attribution.yaml` | PIT style targets、daily returns、CAPM panel | self-financing style factors、HAC regressions | `style_factor_attribution_v1`; 2 direct tests |
| Incremental Alpha | `incremental_alpha_test.py:match_industry_peers,paired_bootstrap` | `incremental_alpha_test.yaml` | frozen ML targets、PIT six styles、daily market | matched style targets、paired daily return、BH-FDR | `incremental_alpha_test_v1`; 5 direct tests |
| 治理与发布 | `experiment_registry.py`, `audit.py`, `research_conclusion_audit.py` | `frozen_baselines.yaml`, `experiment_invalidations.yaml`, `research_conclusion_audit.yaml` | all selected metrics and configs | 37 immutable snapshots、audit、final report | registry/audit/conclusion/dashboard/report tests |

37 个实验的逐项 title、stage、数据版本、快照哈希、live-input 状态和替代关系见 `reports/audit/current_baseline_inventory.csv`。

## 5. 重点专项审计

### 5.1 交易成本和换手率

**实现事实**

- 周频组合的 `turnover` 是漂移后权重与新目标权重的 `0.5 * sum(abs(delta_weight))`；首期记为 1。
- 日频执行的 `turnover` 是 `max(buy_notional, sell_notional) / pretrade_nav`。
- 日频费用是 `max(buy_notional, sell_notional) * round_trip_cost_bps / 10_000`。因此 20bp 被当作“完整换仓的往返成本”，没有在买卖两边再次各乘一次。
- 受限与理想执行使用同一费用函数。Incremental Alpha 的模型与匹配基准也共用相同函数和 20bp 配置。
- 单因子多空 tear sheet 使用两条腿各自换手乘 20bp，是另一种明确标注的诊断口径；风格解释因子则为 0 成本毛收益。

**缺口**

- 没有统一输出绝对 `buy_notional`、`sell_notional`、双边成交额和权重式单边换手；当前日频表只输出归一化的 `buy_weight`、`sell_weight` 和 `turnover`。
- 没有买卖佣金、最低收费、印花税、过户费、spread 和实际 fill 级 impact 分项。
- 0/20/50/100bp 压力测试固定既有成交路径重定价，不反馈到股数、现金或成交路径。

结论：20bp 内部计算一致，但它是统一情景成本，不是历史真实费率。

### 5.2 现金、持仓与 NAV 对账

**已验证**

- 8 个日频策略/场景组、每组 1,387 日，共 11,096 行；组内日期键无重复。
- `net_return = gross_return - transaction_cost` 最大误差 `5.84e-15`。
- 终值 NAV 与逐日净收益复利最大误差 `2.49e-14`。
- `cash_weight + sum(actual_weights) = 1` 最大误差 `8.33e-15`；费用无负值；现金权重未出现负值。
- 当前 20bp 费用公式重算最大误差 `4.34e-18`。

**尚不能验证**

- 输出没有逐笔 order ID、订单状态、成交股数、成交价和费用分项，也没有逐证券每日 position ledger。因此无法从持久化 fill 重新计算 cash/NAV。
- 内部 `prices` 从 1.0 初始化并用 `pct_chg` 演化，`shares` 允许小数；它是归一化份额模拟，不是基于真实未复权收盘价和整手约束的股数账本。
- 没有 T+1 可卖股数、公司行动股数/现金调整、部分成交、成交量参与率和真实订单过期状态。

结论：现有聚合账本在自己的简化假设下守恒，但未满足机构规格的独立逐笔复核能力。

### 5.3 模型训练与验证

**已验证**

- 配置为 2006–2015 train、2016–2020 validation、2021–2026 historical OOS。
- Ridge 和 HGB 在年度边界按 5 个交易日标签跨度 purge；OOS 按年扩展窗口重训。
- HGB 关闭 sklearn 内部随机 early stopping；每日期训练抽样由证券代码和日期哈希确定。
- 统一 benchmark 的模型与候选参数由 validation 选择；当前选择记录为 `hgb_shallow15`，`historical_oos_used_for_selection=false`。
- HGB 3×3 参数面只使用 validation，不据此在 OOS 重选。

**缺口**

- `model_benchmark_fit_audit/v1` 只有 test year、candidate、family、fit time 和 observation count；没有每次 fit 的标签成熟截止日、完整参数、seed、模型序列化哈希和预测输出哈希。
- 随机森林只有有限的种子/树数局部诊断，尚未形成按预注册 seed 集合报告均值、标准差和范围的正式比较。
- historical OOS 已被多次用于诊断模型、组合和执行，不能再作为 pristine lockbox。

### 5.4 无风险利率和风险归因

**实现事实**

- 最终 Sharpe 全部是 `sharpe_zero_rate`。
- CAPM 回归直接使用策略净收益和 CSI 价格收益，等价于 `rf=0`；配置和报告均明确记录。
- 六风格时间序列回归同样使用 `rf=0`，风格组合不扣交易成本，仅用于解释。
- CAPM/风格回归使用共同的 1,387 日日频时钟，CSI300/500/1000 在策略日期无缺失，Newey–West lag=5。
- 周频风险归因使用 5 日标签与理想目标组合，不是日频实际成交持仓归因；报告已经披露这个边界。
- 周频归因恒等式最大误差 `2.22e-16`，reported net 误差最大 `5.55e-17`。

**缺口**

- 没有可追溯的人民币无风险利率数据产品，也没有 risk-free-adjusted Sharpe/CAPM/风格回归。
- CSI 基准不含股息；本地风格组合是研究代理，不是商业风险模型。
- 风险归因未使用日频实际持仓、现金和实际 fill，因此不能称为最终执行账本的逐日归因。

### 5.5 冻结实验和治理

- 37 个实验目录、415 个 `FROZEN.json` 所管文件、86,703,820 bytes；逐文件重算后 37/37 快照完整，0 个篡改。
- 当前 live inputs 与 19 个实验完全匹配；18 个历史实验存在 input drift。这 18 个都已在 `experiment_invalidations.yaml` 中记录替代或失效关系，没有通过覆盖旧快照消除差异。
- 当前 publication baseline `final_research_baseline_v4` 和 `incremental_alpha_test_v1` 的 live inputs 均匹配。
- 24/37 个实验的 frozen metadata 未捕获 Git commit，13/37 捕获到的实验又记录为 dirty worktree。所有实验仍有 `source_tree_sha256`，但不完全满足“明确 Git commit”的目标规格。
- 现有自动 publication audit 文件记录 337 PASS / 43 WARN / 0 FAIL；本轮没有重写该历史报告，而是独立重算快照与 live-input 哈希，得到相同的 37/37、19/18 分布。

## 6. 文档声明核对

| 文档声明 | 代码/产物事实 | 判定 |
|---|---|---|
| 5,521 只历史 A 股、下载覆盖率 100% | 历史 union 中按 `instrument_type` 识别 5,521 只；raw/validated/factor/label 四层各覆盖 5,521/5,521，另有 3 个指数 | 一致 |
| 约 1,479 万行特征 | staging index 合计 14,862,783 行 | 一致 |
| 1,059 个证券快照 | `security_snapshot` 有 1,059 个分区 | 一致 |
| 250 个历史行业快照 | 当前调度函数在研究期产生并实际使用 250 个日期；raw 目录另保留 38 个未被调度使用的早期冗余快照，共 288 个文件 | 口径需补充，但主声明可复核 |
| “股数订单、现金、失败订单逐日重试” | 内存状态和日汇总存在；份额为合成小数份额，无持久化 order/fill/position ledger | 部分一致，文案容易高估真实度 |
| 最终 HGB raw 20.81% / 0.94 / -33.32% | 日频受限执行表为 20.8117% / 0.9372 / -33.3249% | 一致 |
| 六风格 Alpha -2.55% 至 +1.07%，区间均跨零 | 主受限场景为 -2.5461% 至 +1.0696%，四条区间均跨零 | 一致 |
| HGB neutral 增量 3.65%，CI [0.24%, 7.16%]，FDR 未通过 | frozen incremental comparison 对应 3.6468%，[0.2444%, 7.1647%]，q=7.5962% | 一致 |
| 37 个不可变实验 | index 和目录均为 37；快照完整性 37/37 | 一致 |
| 2021–2026 是 historical OOS 而非 pristine lockbox | README、最终报告、冻结 metadata 均披露 | 一致 |
| `docs/research_report.md` 中 RF 22.02% 等数字 | 文件首段明确声明为早期冻结轨迹，最终权威结果另见 final report；正文和旧简历段仍可被脱离上下文误引 | 已披露但建议后续归档标识强化 |

## 7. 信息状态分类

### 已实现并由本轮重算验证

- PIT 时间切分、边界 purge、年度扩展窗口。
- 周频单边权重换手、统一 20bp 情景成本。
- LP 单股/行业/规模/换手约束；293 周无放宽。
- 日频停牌/涨跌停阻塞和剩余订单重试；聚合 NAV 恒等式。
- CAPM、六风格回归、周频多因子归因及其当前报告数值。
- Incremental Alpha 的共同日历、共同成本、配对 block bootstrap 和按指标的四策略 BH-FDR。
- 冻结目录不可覆盖、快照逐文件完整性和 invalidation 关系。

### 文档有声明，但实现只满足简化版本

- “股数订单”：实际为归一化、允许小数的合成份额。
- “现金与持仓账本”：实际为函数内状态和日汇总，不是可重放逐笔账本。
- “可成交”：只近似停牌和收盘涨跌停；无整手、T+1、部分成交、量约束、盘口队列和 IPO 无涨跌幅窗口。
- “20bp 成本”：是固定往返情景，不是历史真实费率。
- “风险模型”：是本地研究代理，周频目标组合口径；不等同商业风险模型或日频实际持仓归因。

### 尚未实现或尚未验证

- 日期生效的佣金、税费、过户费、spread 和 impact 组件。
- 可追溯人民币无风险利率数据产品和调整后指标。
- 完整 order/fill/position/cash/corporate-action ledger。
- 正式的多 seed 随机模型稳定性与 block-length 敏感性。
- future lockbox、paper trading 和结果访问/消耗日志。
- 券商订单、盘口或分钟级成交校准。

## 8. 当前实现 vs 目标规格差异（按优先级）

| 优先级 | 差异 | 当前证据 | 风险 / 验收门槛 | 建议阶段 |
|---|---|---|---|---|
| P0 | 没有持久化 order/fill/position ledger，费用不能从 fills 独立重建 | 只有日汇总和函数内字典状态 | 影响现金/NAV/费用的外部可审计性；需逐 fill 对账 | Phase 5/7；在 Phase 1 先定义兼容接口 |
| P0 | 合成小数份额代替真实价格、整手与证券可卖数量 | `prices` 初始为 1.0，目标可产生小数 shares | 影响账户真实性及 T+1/公司行动正确性 | Phase 5/7 |
| P0 | 公司行动没有显式现金和股数事件 | 依赖供应商 `pct_chg` 演化合成价格 | 极端事件下不能证明股份/NAV 守恒 | Phase 5/7 |
| P0 | 没有 pristine future lockbox / paper record | 2021–2026 已被查看 | 新改动不能用同一区间声称独立证据 | Phase 4/11 |
| P1 | 20bp 固定成本没有历史分项费率 | `max(buy,sell) * 20bp` | 影响净收益与策略公平比较 | **Phase 1 首要** |
| P1 | 换手字段未统一覆盖买、卖、双边、单边及权重式定义 | 各模块只有 `turnover` 或两腿特有字段 | 容易在报告中混淆口径 | **Phase 1 首要** |
| P1 | 风险自由利率为 0，指数不含股息 | CAPM/style config 明示 zero rate | Sharpe 和 Alpha 可能偏高 | Phase 2 |
| P1 | 模型 fit audit 缺标签成熟日、完整参数、seed、模型/预测 hash | 当前只有年份、family、fit rows/time | 无法逐 fit 完整重建 | Phase 3/10 |
| P1 | 随机模型未按正式预注册 seed 集合报告分布 | 只有局部 RF 稳定性研究 | 随机性与模型选择偏差未完全量化 | Phase 3 |
| P1 | 统计推断只有单一 block length | 周频 13、日频 20 | 区间对依赖长度的敏感性未知 | Phase 8 |
| P1 | 周频风险归因不是实际日频成交持仓归因 | config 明示 weekly label attribution | 无法解释 cash、blocked fills 对风险贡献的影响 | Phase 7/9 |
| P1 | 24/37 frozen metadata 缺 Git commit；13 个已记录 commit 的快照来自 dirty tree | registry 调用 Git 时受 safe-directory 影响 | source hash 可保真，但代码版本语义不完整 | Phase 10 |
| P1 | 18 个历史实验与 live inputs 漂移 | 全部快照完整且已有 invalidation | 当前已治理；新实验必须禁止模糊引用旧 live baseline | Phase 10 |
| P2 | raw 行业目录含 38 个调度外冗余快照 | 使用调度 250，目录总数 288 | 不影响当前 as-of join，但规模描述容易混淆 | 文档/数据治理 |
| P2 | 配置注释仍保留早期“行业/规模约束未实现”措辞 | 基础 `portfolio.yaml` / `preprocessing.yaml` 与后续专用模块并存 | 容易误解模块层级 | Phase 10 文档整理 |
| P2 | 日线涨跌停规则未覆盖完整 IPO/特殊规则窗口 | scope note 已披露 | 更高保真执行扩展 | Phase 5 |
| P2 | 盘口、spread、冲击模型未用券商订单校准 | 容量结果仅情景分析 | 不能用于资金容量承诺 | Phase 5 |

P0 条目不是指当前数值已经被证明错误，而是现有产物不足以在目标机构规格下证明其账户和结论正确性。

## 9. Phase 1 详细实施方案（仅方案，未执行）

### 9.1 目标和不变量

Phase 1 只改变成本表示与报告，不改变信号、标签、模型、目标权重、交易资格或成交数量规则。所有成本实验必须复用同一 frozen target 和同一 fill path；任何需要改变 fill path 的现金反馈实验单独升 experiment ID，不能和纯重定价混在一起。

必须保持：

- C0 完全复现当前 20bp 结果；
- 0 成本严格为 0；
- 每日费用等于当日 fill 费用分项之和；
- 固定 round-trip 成本与逐项费用模型互斥，禁止双重扣费；
- 不因费用升级而重新选择模型或组合。

### 9.2 拟新增配置

建议新增 `config/transaction_cost_model.yaml`：

```yaml
experiment_id: transaction_cost_model_v1
mode: fixed_round_trip_baseline
turnover_definitions:
  weight_based_one_way: half_l1_target_vs_drifted
  execution_one_way: max_buy_sell_over_pretrade_nav
fixed_scenarios_bps: [0, 10, 20, 40, 60, 100]
component_scenarios:
  - name: documented_fee_scenario_v1
    evidence_status: scenario_not_historical_truth
    effective_date_rules: []
commission: {}
stamp_duty: {}
transfer_fee: {}
spread: {}
market_impact: {}
```

在没有可靠费率来源前，组件参数保持为空并标记 `scenario_not_historical_truth`，不得编造券商费率。

### 9.3 拟新增/调整模块

1. 在 `src/quant_research/` 增加单一可复用成本组件（名称在实施时按现有风格确定），输入 fills、trade date、side、notional、venue、instrument 和可选市场数据，输出费用分项。
2. 为现有日频执行结果增加标准换手字段：
   - `buy_notional`
   - `sell_notional`
   - `buy_turnover`
   - `sell_turnover`
   - `gross_two_way_turnover`
   - `one_way_turnover`
   - `weight_based_one_way_turnover`（适用时）
3. 保留现有 `turnover` 作为兼容字段，但在 schema 中明确它等于哪一种定义，后续再决定是否弃用。
4. 为费用输出增加：commission、stamp duty、transfer fee、spread、impact、total cost 和 assumption/evidence 标记。
5. C0 使用原函数的同一成交路径做回归；C1–C5 生成新表和新 experiment ID，不覆盖 `v1`。

### 9.4 测试矩阵

- Unit：0 成本、仅买、仅卖、等额轮换、买卖不等额、卖方税、最低佣金、费率生效日边界、无成交零费用、负费用拒绝。
- Definition：验证五种换手字段的手算例；证明 20bp round-trip 不会被买卖两边重复乘 2。
- Integration：同一 fills 在固定成本与分项成本下各自对账；每日费用总和等于 fill 费用总和；NAV 只因费用下降。
- Regression：C0 的 8 组日频 annualized return、volatility、Sharpe、MDD、terminal wealth 与本 Phase 0 基线在严格容差内一致。
- Data contract：费用规则含 effective date、单位、来源/evidence status；缺失规则不得静默回填。

### 9.5 新产物规划

- 新 lake 表：cost protocol、standardized turnover、fill-cost detail（若 Phase 1 先基于可重建的聚合 fills，则必须明确为 synthetic）、daily cost reconciliation、scenario summary。
- 新报告：固定成本基线、费用分项、毛/净收益、单边/双边换手、费率证据与未知项。
- 新冻结实验：使用全新 experiment ID；配置快照、当前 Git commit、source hash、输入 hash、测试结果一并登记。

### 9.6 Phase 1 实施顺序

1. 先冻结术语和 schema；
2. 写手算测试，再实现统一成本接口；
3. 将现有 20bp 适配为 C0，验证零差异；
4. 增加 0/10/20/40/60/100bp C1；
5. 只有取得可引用的历史费率证据后才实现 C2；
6. C3–C5 作为明确情景分析，等 spread/ADV/impact 数据满足合同后再运行；
7. 生成新报告和冻结实验，保留所有旧产物；
8. Phase 1 验收后停止，另行申请进入 Phase 2。

## 10. 停止条件

Phase 0 已完成。本文不授权也未执行 Phase 1 的代码、配置或研究结果变更。下一步应先由用户审阅 P0/P1/P2 差异表和 Phase 1 方案。
