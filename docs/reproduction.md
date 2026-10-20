# 跨机器从零复现指南

## 1. 复现边界

GitHub 仓库保存代码、配置、测试、文档、图表、结果表和 46 个冻结实验版本，**不保存完整
`data/` 数据湖**。新机器从 BaoStock 下载股票主数据；AKShare 承担股票抽样核对和中债官方
历史接口适配，市场归因从中证指数公司官方接口取得价格/全收益序列。不同来源不静默互相覆盖。

这提供的是“研究流程可复现”，不是“供应商字节级快照可复现”。免费数据源可能修订历史记录、
改变接口或停止提供退市证券，因此重新下载后的行数、哈希和末位指标可能与 2026-09-30 冻结
结果不同。仓库中的旧实验目录始终保留原始配置、指标和输入哈希，不能用新运行覆盖。

## 2. 机器和环境

建议准备：

- Windows 10/11、Git、64 位 Python 3.11；Conda 可选；
- 至少 16 GB 内存，完整模型 benchmark 建议 32 GB；
- 至少 30 GB 可用磁盘空间；当前已构建数据湖约 11.1 GiB，中间表和日志还会继续增长；
- 可稳定访问 BaoStock 与 AKShare 上游站点的网络。

```powershell
git clone https://github.com/yuhaoyan04/ai-quant-research.git
cd ai-quant-research

conda create -n ai-quant python=3.11 -y
conda activate ai-quant
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m pytest -q
```

也可以使用 `python -m venv .venv`。项目不需要 Tushare Token；主流程仅依赖 BaoStock 与
AKShare。GitHub Actions 会在无数据环境运行单元/回归测试。

## 3. 先做隔离的小样本连通性测试

`--limit 5` 会故意生成不完整行情，**不要在准备完整重建的正式 checkout 中运行**，否则这些
已存在的小样本文件会被断点续传逻辑视为已完成。请在单独的临时 clone 中执行：

```powershell
git clone https://github.com/yuhaoyan04/ai-quant-research.git ai-quant-smoke
cd ai-quant-smoke
pip install -e ".[dev]"
quant-free-lake bootstrap --start 2024-01-01 --limit 5
quant-free-lake crosscheck --start 2024-01-01
```

验收：`data/lake/curated/` 下生成 calendar、security snapshot 和 5 个证券的 Parquet；
`data/lake/manifests/ingestion.jsonl` 记录请求、行数、字段和内容哈希；交叉核对目录非空。

## 4. 正式全量下载

在干净的正式 checkout 中，若希望尽可能贴近本项目冻结样本，固定结束日：

```powershell
quant-free-lake bootstrap --start 2006-01-01 --end 2026-09-30 --isolate-bars
quant-free-lake verify --start 2006-01-01 --end 2026-09-30
quant-free-lake crosscheck --start 2006-01-01 --end 2026-09-30
```

BaoStock [现行访问规则](https://www.baostock.com/blacklist)禁止并发连接。下载器因此只允许一个持久会话；不要通过多进程、多终端或
多个 checkout 同时请求 BaoStock，否则会触发供应商黑名单。`--isolate-bars` 仍是串行恢复模式，
不能与并发参数组合。

若要接力研究并下载运行当天可获得的最新历史数据，省略 `--end`：

```powershell
quant-free-lake bootstrap --start 2006-01-01 --isolate-bars
quant-free-lake verify --start 2006-01-01
quant-free-lake crosscheck --start 2006-01-01
```

下载器按周保存 PIT 股票快照、按证券原子写入日线 Parquet，并额外下载 robustness/dashboard
所需的 CSI 300、CSI 500、CSI 1000 三个基准指数。它在
`data/lake/checkpoints/free_ingest.json` 记录最近完成项。网络中断后重复同一条命令即可跳过
已经完整落盘的证券。`--isolate-bars` 会把单只证券请求放入可终止子进程，速度略慢，但更适合
无人值守长任务。`verify` 会检查预期 A 股、周度快照、行情文件以及每只证券 manifest 中的请求
日期范围；有下载失败或混入短历史 smoke 文件时会返回非零状态，阻止下游研究在残缺数据上运行。

行业研究另需下载历史证监会行业快照。为遵守访问规则并控制请求数，项目查询首个周频信号日和
之后每月最后一个周频信号日，再对研究日期做严格的 backward as-of join：

```powershell
quant-free-lake industries --start 2006-01-01 --end 2026-09-30
quant-free-lake verify-industries --start 2006-01-01 --end 2026-09-30
quant-risk-exposure build-industry
quant-factor-neutralization
quant-factor-tear-sheet
quant-capm-attribution
quant-style-factor-attribution
quant-risk-free-attribution
quant-market-benchmark-attribution
quant-model-stability-validation
quant-research-governance status
quant-model-portfolio-neutralization
quant-constrained-portfolio-neutralization
quant-constrained-daily-execution all
quant-institutional-execution all
quant-portfolio-construction-comparison all
quant-unified-account-ledger run
quant-transaction-costs run
quant-risk-attribution all
quant-incremental-alpha all
quant-risk-aware-portfolio
```

新上市股票在下一次行业快照前保持缺失，不允许用后续分类回填。原始行业字符串、分类版本、查询
日期、上游更新日期和宽行业映射全部保留，以便审计跨年代的证监会分类变化。下载器会拒绝停在
BaoStock 2,000 行满页上的疑似截断响应；`verify-industries` 还要求每个行业快照覆盖至少 95%
的同日 PIT A 股证券快照，防止网络中断后把可读但不完整的 Parquet 当作成功结果。
四条因子管线始终使用收益、行业和规模均完整的同一截面；因子方向只由 2006–2015 年 raw
RankIC 的符号确定，2021–2026 历史 OOS 不参与翻向或选择。除原始有符号 RankIC 外，结果同时
保存定向 RankIC、定向多空收益及配对 13 周 circular block bootstrap。

重要限制：现有下载器适合“同一开始日/结束日范围内的中断续跑”，不负责把一个较短的既有行情
文件自动延长到更早或更晚日期。变更日期范围时应使用新的干净 checkout/数据目录，避免混合
覆盖范围。

## 5. 从数据湖重建研究结果

推荐使用带失败即停、日志和正确依赖顺序的一键脚本：

```powershell
# 冻结日期口径：下载 + 全部研究实验
powershell -ExecutionPolicy Bypass -File scripts/rebuild_all.ps1 `
  -StartDate 2006-01-01 -EndDate 2026-09-30 -IsolateBars

# 如果 data/lake 已完整下载，只重跑处理与研究
powershell -ExecutionPolicy Bypass -File scripts/rebuild_all.ps1 -SkipDownload

# 只跑基础数据、Ridge/HGB、组合与常规压力测试
powershell -ExecutionPolicy Bypass -File scripts/rebuild_all.ps1 -SkipDownload -CoreOnly
```

日志写入 `run_logs/rebuild_*.log`。完整脚本依次完成：

1. 数据质量、因子、可执行标签与 PIT 股票池；
2. 无泄漏截面预处理，训练/验证/OOS 切分和 Ridge 基线；
3. 组合、成本、流动性、容量、缓冲、规模与涨跌停压力测试；
4. HGB challenger、解释与 robustness；
5. 预处理消融、IC decay、LOFO、5/10/20 日标签实验；
6. PIT 规模/行业暴露，以及 raw、size、industry、industry+size 因子和锁定模型/组合中性化；
7. 显式单股、行业主动权重、规模和换手约束组合、日频可成交审计及多因子收益归因；
8. 统一模型 benchmark、RF/HGB 参数稳定性、收益取证和日频成交重试；
9. Top-N、inverse-vol、fill 级 C0/C1/C2 成本协议、因子分布与滚动相关诊断；
10. 同一 HGB 信号下的 P0/P1/P2/P4 权重比较、账户执行与 2500 万/1 亿/5 亿容量重跑；
11. 冻结主账户的目标/实际权重、现金、订单、成交、公司行动与 NAV 统一账本及独立回放；
12. 结论证据矩阵、BH-FDR 多重检验、研究生命周期审计；
13. 离线 dashboard、publication audit 与最终 PDF。

如果需要手工定位失败步骤，可打开 [rebuild_all.ps1](../scripts/rebuild_all.ps1) 查看完全展开、
可逐条复制的命令序列。脚本默认不删除、不覆盖 `data/`；多数上游阶段通过 `--missing` 或文件
存在检查支持断点续跑。

## 6. 关键目录与数据血缘

| 目录 | 内容 | 是否进 Git |
|---|---|---:|
| `data/lake/curated/daily_bars_baostock/` | 按证券分区的未复权日线与估值字段 | 否 |
| `data/lake/curated/security_snapshot/` | 周频历史可交易证券快照 | 否 |
| `data/lake/curated/industry_snapshot_baostock/` | 月频抽样的历史证监会行业原始快照 | 否 |
| `data/lake/curated/` 其他表 | 质量表、因子、标签、模型特征、预测、持仓、成交 | 否 |
| `data/lake/manifests/` | 请求、schema、行数、内容哈希 | 否 |
| `config/` | 数据、因子、模型、组合和实验配置 | 是 |
| `reports/experiments/` | 46 个不可变实验版本与输入哈希 | 是 |
| `reports/tables/`、`reports/figures/` | 求职展示所需结果 | 是 |
| `reports/research_dashboard.html` | 离线交互面板 | 是 |

## 7. 验收与结果解释

```powershell
python -m pytest -q
quant-experiment list
quant-experiment verify rf_daily_retry_execution_v1
quant-conclusion-audit
quant-audit
quant-research-dashboard
```

当前验收记录为 193 tests passed；研究审计为 409 PASS / 46 WARN / 0 FAIL（WARN 保留历史版本漂移与已知边界）。新下载数据应至少
满足：测试通过、schema/唯一键/日期门禁无 FAIL、模型比较使用共同日期、持仓权重和收益恒等式
通过。

`quant-experiment verify` 同时检查“冻结快照自身”与“当前数据湖输入”。前者应通过；后者在供应商
修订后出现哈希不一致是预期的数据版本漂移，不应通过修改旧快照来消除。正确做法是保留旧版本，
为新数据生成新的 experiment ID，并在报告中并列说明差异。

`quant-research-governance initialize` 不是普通重建步骤：它只在形成新候选 Git commit 后执行一次，创建空 future lockbox 和零观测 paper record。已有候选使用 `status` 验证访问日志哈希链；不得为了让重建脚本通过而删除日志或重复初始化。

`quant-institutional-execution all` 首次运行会从未复权日线构建人民币价格/成交量执行面板，再运行受限场景和移除市场约束上界。它不会改动冻结模型或目标权重。日线模拟不包含盘口队列、bid/ask、精确成交时刻或真实券商订单；公司行动字段是价值保持代理，不能替代官方权益分派记录。

`quant-portfolio-construction-comparison all` 会重建冻结 raw HGB 全截面分数，生成 Top-200/400 缓冲下的四种目标权重，扩展执行市场并真实重跑三个资金规模。该步骤计算量和产物体积都较大；其 2021–2026 结果属于 post-OOS adaptation/诊断，不得用于宣称新的独立样本外最优方法。

`quant-unified-account-ledger run` 不重跑信号或策略，只读 Phase 5 的冻结主账户输出，生成目标/实际权重、现金、订单、成交、公司行动与 NAV 的统一账本并执行独立对账。它要求上游 institutional execution 表已经存在；任何硬对账失败都会以非零状态退出。

## 8. 常见故障

- BaoStock 单只证券长时间无响应：加 `--isolate-bars --request-timeout 180`，然后原命令续跑。
- AKShare 抽样失败：先确认版本和网络；它是独立质量核对，不能拿它静默替换 BaoStock 主表。
- ChinaBond 利率下载失败：重跑 `quant-risk-free-attribution`；若已有经 manifest 固定的完整官方快照，可用 `--reuse-observations` 重算，但不得把缺口前向填充。历史接口要求不足一年的分段请求，配置已固定到 2026-09-30。
- CSIndex 基准下载失败：重跑 `quant-market-benchmark-attribution`；已有完整官方快照时可用 `--reuse-observations`。价格/全收益比较必须保持同一官方来源，不能拿供应商差异冒充股息收益。
- 内存不足：先用 `-CoreOnly`；关闭其他进程，再单独运行模型 benchmark 与 RF 系列。
- 命令不存在：确认已激活正确环境，并重新执行 `pip install -e ".[dev]"`。
- 审计哈希 FAIL：先区分 frozen artifact 自身损坏与 live input 漂移；不要覆盖冻结实验目录。
- 需要更新到最新日期：使用新的数据目录从 2006-01-01 完整拉取，再以新实验版本接力；当前工具
  不是增量行情维护器。

全量下载通常是最长阶段，免费源响应速度会显著波动；普通笔记本应按数十小时到数天规划，而不应
依赖固定完成时间。每个证券独立落盘，因此中断不会让已经完成的大部分工作失效。
