# 跨机器复现与接力运行

## 1. 推荐路径：恢复冻结数据快照

仓库代码、配置、文档、小型结果和冻结实验快照直接保存在 Git 中。约 11.12 GiB 的完整
Parquet 数据湖保存在 GitHub Release `data-v1`，以小于 2 GiB 的 tar 分卷发布；每个分卷均有
SHA-256 和字节数记录。Parquet 已使用 Zstandard 压缩，因此归档不再二次压缩。

在新机器安装 Git、GitHub CLI、Python 3.11 和 Conda，然后执行：

```powershell
git clone https://github.com/yuhaoyan04/ai-quant-research.git
cd ai-quant-research

conda create -n ai-quant python=3.11 -y
conda activate ai-quant
python -m pip install --upgrade pip
pip install -e ".[dev]"

gh auth login
powershell -ExecutionPolicy Bypass -File scripts/download_data_release.ps1
```

下载脚本会完成四件事：下载 `data-v1` 全部分卷、核对文件大小、逐文件计算 SHA-256、流式恢复
到 `data/lake`。恢复过程不会在本地生成第二份 11 GiB 的完整 tar 文件。

恢复后运行：

```powershell
python -m pytest -q
quant-experiment verify rf_daily_retry_execution_v1
quant-audit
quant-research-dashboard
```

预期门禁是 73 个测试通过、冻结输入哈希一致、研究审计 0 FAIL。历史 OOS 已经被查看过，不应
重新称为 pristine lockbox。

## 2. 从免费数据源完全重建

如不恢复 Release，可从 BaoStock 主源重新下载。AKShare 只用于交叉核对，不会静默覆盖主源。
首次先做小样本 smoke test：

```powershell
quant-free-lake bootstrap --start 2024-01-01 --limit 5
quant-free-lake crosscheck --start 2024-01-01
```

确认网络、字段和 Parquet 正常后执行全量流水线：

```powershell
quant-free-lake bootstrap --start 2006-01-01
quant-materialize --missing
quant-research factor-research
quant-preprocess --stage
quant-preprocess --all
quant-feature-analysis
quant-model-baseline
quant-portfolio
quant-liquidity-stress
quant-turnover-buffer
quant-capacity build-exposures
quant-capacity analyze
quant-capacity-execution
quant-model-challenger
quant-challenger-portfolio
quant-model-interpretation
quant-size-stress build-exposures
quant-size-stress analyze
quant-execution-constraints build-flags
quant-execution-constraints analyze
quant-robustness
quant-label-panel build
quant-label-horizon
quant-risk-exposure build
quant-size-neutralization
quant-model-benchmark
quant-rf-hyperparameter-stability
quant-rf-target-holdings
quant-rf-daily-retry all --split both
quant-research-dashboard
quant-audit
```

下载器按证券独立落盘并维护 manifest，可中断后继续。由于供应商修订、退市证券可访问性和
下载日期不同，从源站重建不保证逐字节等同于 `data-v1`；需要复核结论时优先使用冻结快照。

## 3. 数据和结果目录

- `data/lake/curated/`：行情、标签、因子、模型预测、持仓与执行结果。
- `data/lake/manifests/`：采集请求、行数、字段和内容哈希。
- `reports/experiments/`：13 个不可变实验快照及输入哈希。
- `reports/research_audit.md`：全项目自动审计。
- `reports/public_results.json`：适合程序读取的最终结论。
- `reports/research_dashboard.html`：离线可视化面板。

## 4. 数据发布维护

维护者生成某一分组的分卷：

```powershell
python scripts/data_release.py pack --group factors_raw --output release_staging
```

可用分组为 `factors_raw`、`eligible_panel`、`multi_horizon_panel`、`validated_bars`、
`model_features`、`baostock_bars`、`labels_raw`、`multi_horizon_labels` 和 `remaining`。
`data_release_manifest.json` 会在每组完成后更新。上传确认后即可删除该组本地分卷，再生成下一组，
避免额外占用一整份数据湖空间。
