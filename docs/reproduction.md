# 跨机器从零复现指南

## 1. 复现边界

GitHub 仓库保存代码、配置、测试、文档、图表、结果表和 13 个冻结实验快照，**不保存完整
`data/` 数据湖**。新机器从 BaoStock 下载主数据，AKShare 只做独立抽样核对，不参与拼接或
静默覆盖。

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

若要接力研究并下载运行当天可获得的最新历史数据，省略 `--end`：

```powershell
quant-free-lake bootstrap --start 2006-01-01 --isolate-bars
quant-free-lake verify --start 2006-01-01
quant-free-lake crosscheck --start 2006-01-01
```

下载器按周保存 PIT 股票快照、按证券原子写入日线 Parquet，并在
`data/lake/checkpoints/free_ingest.json` 记录最近完成项。网络中断后重复同一条命令即可跳过
已经完整落盘的证券。`--isolate-bars` 会把单只证券请求放入可终止子进程，速度略慢，但更适合
无人值守长任务。`verify` 会检查预期 A 股、周度快照、行情文件以及每只证券 manifest 中的请求
日期范围；有下载失败或混入短历史 smoke 文件时会返回非零状态，阻止下游研究在残缺数据上运行。

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
6. PIT 规模暴露和规模中性化；
7. 统一模型 benchmark、RF 参数稳定性、收益取证和日频成交重试；
8. 离线 dashboard 与 publication audit。

如果需要手工定位失败步骤，可打开 [rebuild_all.ps1](../scripts/rebuild_all.ps1) 查看完全展开、
可逐条复制的命令序列。脚本默认不删除、不覆盖 `data/`；多数上游阶段通过 `--missing` 或文件
存在检查支持断点续跑。

## 6. 关键目录与数据血缘

| 目录 | 内容 | 是否进 Git |
|---|---|---:|
| `data/lake/curated/daily_bars_baostock/` | 按证券分区的未复权日线与估值字段 | 否 |
| `data/lake/curated/security_snapshot/` | 周频历史可交易证券快照 | 否 |
| `data/lake/curated/` 其他表 | 质量表、因子、标签、模型特征、预测、持仓、成交 | 否 |
| `data/lake/manifests/` | 请求、schema、行数、内容哈希 | 否 |
| `config/` | 数据、因子、模型、组合和实验配置 | 是 |
| `reports/experiments/` | 13 个不可变实验快照与输入哈希 | 是 |
| `reports/tables/`、`reports/figures/` | 求职展示所需结果 | 是 |
| `reports/research_dashboard.html` | 离线交互面板 | 是 |

## 7. 验收与结果解释

```powershell
python -m pytest -q
quant-experiment list
quant-experiment verify rf_daily_retry_execution_v1
quant-audit
quant-research-dashboard
```

原冻结版本的验收记录为 73 tests passed、研究审计 173 PASS / 6 WARN / 0 FAIL。新下载数据应至少
满足：测试通过、schema/唯一键/日期门禁无 FAIL、模型比较使用共同日期、持仓权重和收益恒等式
通过。

`quant-experiment verify` 同时检查“冻结快照自身”与“当前数据湖输入”。前者应通过；后者在供应商
修订后出现哈希不一致是预期的数据版本漂移，不应通过修改旧快照来消除。正确做法是保留旧版本，
为新数据生成新的 experiment ID，并在报告中并列说明差异。

## 8. 常见故障

- BaoStock 单只证券长时间无响应：加 `--isolate-bars --request-timeout 180`，然后原命令续跑。
- AKShare 抽样失败：先确认版本和网络；它是独立质量核对，不能拿它静默替换 BaoStock 主表。
- 内存不足：先用 `-CoreOnly`；关闭其他进程，再单独运行模型 benchmark 与 RF 系列。
- 命令不存在：确认已激活正确环境，并重新执行 `pip install -e ".[dev]"`。
- 审计哈希 FAIL：先区分 frozen artifact 自身损坏与 live input 漂移；不要覆盖冻结实验目录。
- 需要更新到最新日期：使用新的数据目录从 2006-01-01 完整拉取，再以新实验版本接力；当前工具
  不是增量行情维护器。

全量下载通常是最长阶段，免费源响应速度会显著波动；普通笔记本应按数十小时到数天规划，而不应
依赖固定完成时间。每个证券独立落盘，因此中断不会让已经完成的大部分工作失效。
