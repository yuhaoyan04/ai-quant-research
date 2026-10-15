# A股 Point-in-Time 多因子机器学习 Alpha Research Platform

# Research Upgrade Plan

## 0. 项目目标

本项目目标不是简单优化一个回测策略，而是将当前 A 股多因子机器学习选股系统升级为接近真实量化机构 Research Pipeline 的完整研究平台。

最终目标：

构建一个完整的：

> Cross-sectional Equity Alpha Research Framework

研究链路：

```
Data Infrastructure
        |
        ↓
Point-in-Time Dataset
        |
        ↓
Factor Engineering
        |
        ↓
Factor Evaluation
        |
        ↓
ML Alpha Model
        |
        ↓
Walk-forward Validation
        |
        ↓
Portfolio Construction
        |
        ↓
Risk Management
        |
        ↓
Performance Attribution
        |
        ↓
Research Report
```

---

# 1. 总体开发原则

## 1.1 不允许破坏已有研究结果

任何新增实验：

必须：

- 保留已有 baseline
- 新增实验分支
- 支持横向比较


禁止：

- 直接修改旧代码导致历史结果无法复现
- 为单个实验写一次性脚本


---

## 1.2 代码必须工程化

最终代码结构要求：

```
ai-quant-research/

├── configs/
│   ├── data.yaml
│   ├── factor.yaml
│   ├── model.yaml
│   └── backtest.yaml
│
├── src/
│
│   ├── data/
│   │   ├── loader.py
│   │   ├── pit.py
│   │   └── validator.py
│   │
│   ├── factors/
│   │   ├── momentum.py
│   │   ├── value.py
│   │   ├── liquidity.py
│   │   └── processor.py
│   │
│   ├── models/
│   │   ├── ridge.py
│   │   ├── hgb.py
│   │   └── trainer.py
│   │
│   ├── portfolio/
│   │   ├── construction.py
│   │   ├── optimizer.py
│   │   └── cost.py
│   │
│   ├── evaluation/
│   │   ├── ic.py
│   │   ├── attribution.py
│   │   └── report.py
│
├── tests/
│
├── reports/
│
└── README.md
```

---

# 1.3 实验必须配置化

禁止：

在代码中硬编码：

```python
top_n = 200
lambda = 1
window = 20
```


必须：

通过 config 控制：

```yaml
portfolio:
    top_n: 200

model:
    alpha: 1.0

feature:
    winsor_quantile: 0.01
```

---

# 1.4 每个实验必须产生实验记录

每次运行保存：

```
experiment_id

config

dataset_version

model_version

metrics

plots

results
```


例如：

```
reports/
    exp_001_ridge_baseline/
        config.yaml
        metrics.json
        ic_curve.png
        equity_curve.png
```

---

# 2. Phase 1：Factor Research Upgrade

目标：

将当前：

> 15个feature输入模型

升级为：

> 经过严格研究验证的alpha factor library


---

# Experiment 1：Factor Dictionary 建立


## 目标

明确每个factor：

- 数学定义
- 数据来源
- 更新频率
- 经济含义


---

## 实验设置

建立：

`docs/factor_dictionary.md`

包含：

|Factor|Formula|Source|Economic Logic|
|-|-|-|-|
|Momentum|12M return|price|underreaction|
|Value|PE/PB|fundamental|mispricing|
|Liquidity|ADV|volume|liquidity premium|


---

## 验收标准

必须：

- 所有15个factor有定义
- 有经济解释
- 有数据时间规则


---

# Experiment 2：Factor Winsorization


## 背景

金融因子存在大量极端值。

例如：

PE:

-10000


会影响：

模型训练。


---

## 实验设置


比较：

### Version A

Raw factor


### Version B

Winsorization


方法：

quantile clipping:

```
1% - 99%
```


---

## 输出指标

比较：

- Mean IC
- IC volatility
- Portfolio return
- Turnover


---

## 验收标准

生成：

```
reports/factor/winsorization/
```

包含：

实验表格和图。


---

## 预期结果

合理结果：

- IC更加稳定
- 极端收益减少
- 模型泛化改善


---

# Experiment 3：Cross-sectional Normalization


## 目标

消除不同factor尺度差异。


---

## 方法


每日：

z-score:


$$
z_i=
\frac{x_i-\mu}{\sigma}
$$


---

## 实验


比较：

Raw

vs

Normalized


---

## 验收标准


输出：

- factor distribution
- IC comparison


---

# Experiment 4：Industry Neutralization


## 目标

确认alpha不是行业暴露。


---

## 方法


每日回归：

$$
factor_i
=
industry+\epsilon
$$


使用：

残差作为neutral factor。


---

## 实验


比较：

1. Raw factor

2. Industry neutral

3. Industry + Size neutral


---

## 输出


表：

|Method|IC|Return|Sharpe|
|-|-|-|-|


---

## 验收标准

必须证明：

模型收益不是简单行业轮动。


---

# Experiment 5：Factor Correlation Analysis


## 目标

分析factor redundancy。


---

## 方法

计算：

每日factor correlation。


输出：

heatmap。


---

## 验收标准

生成：

```
factor_correlation.png
```


并说明：

哪些factor高度相关。

---

# Experiment 6：Single Factor IC Research


## 目标

研究单因子有效性。


---

## 实验设置


每个factor：

计算：

- RankIC
- ICIR
- IC decay


---

## 输出


Factor Tear Sheet。


格式：

```
Factor:

Mean IC:
ICIR:
Annual Return:
Turnover:
Decay:
```


---

# 3. Phase 2：Label Research


目标：

研究最佳预测目标。


---

# Experiment 7：Prediction Target Comparison


比较：

## Target A

Future Return


## Target B

Excess Return


## Target C

Rank Label


---

## 指标

比较：

- RankIC
- Portfolio return
- Stability


---

## 验收标准

说明：

为什么最终选择当前label。


---

# 4. Phase 3：Model Research


目标：

建立完整model benchmark。


---

# Experiment 8：Model Benchmark


必须包含：


## Linear

- Linear Regression
- Ridge
- Lasso
- ElasticNet


## Tree

- RandomForest
- HGB
- LightGBM


---

## 指标


禁止只看收益。


必须：

- RankIC
- ICIR
- Sharpe
- Drawdown
- Turnover


---

# Experiment 9：Hyperparameter Robustness


## 目标

避免参数过拟合。


---

## Ridge示例


测试：

```
alpha:

0.01
0.1
1
10
100
```


---

## 验收标准


性能应该：

平滑变化。


如果：

单点最好：

警惕overfitting。


---

# 5. Phase 4：Portfolio Construction


---

# Experiment 10：Top-N Sensitivity


测试：

```
Top50
Top100
Top200
Top500
```


---

指标：

- Return
- Sharpe
- Turnover


---

# Experiment 11：Weighting Method


比较：


## Equal Weight


## Score Weight


## Volatility Scaling


## Risk Parity


---

验收：

输出比较表。


---

# Experiment 12：Buffer Mechanism


比较：

No buffer

vs

Buffer


例如：

进入：

Top150


退出：

Bottom250


---

指标：

- turnover reduction
- cost reduction


---

# 6. Phase 5：Transaction Cost & Capacity


---

# Experiment 13：Cost Sensitivity


测试：

```
10bp
20bp
50bp
100bp
```


---

输出：

Return decay curve。


---

# Experiment 14：ADV Capacity


测试资金规模：

```
10 million

100 million

1 billion
```


限制：

```
order_size < x% ADV
```


---

验收：

输出：

capacity report。


---

# 7. Phase 6：Risk Management


---

# Experiment 15：Factor Exposure Analysis


分析：

组合暴露：

- Market
- Size
- Value
- Momentum


---

# Experiment 16：Industry Exposure


输出：

行业权重变化。


---

# Experiment 17：Volatility Control


比较：

Baseline

vs

Target Volatility Portfolio


---

# 8. Phase 7：Performance Attribution


---

# Experiment 18：CAPM Alpha Test


回归：

$$
R_p=\alpha+\beta R_m
$$


输出：

alpha。


---

# Experiment 19：Fama-French Attribution


加入：

5 factor model。


分析：

收益来源。


---

# 9. 最终Research Report要求


最终生成：

```
reports/final_report.pdf
```


包含：

## 1. Strategy Overview

## 2. Data Pipeline

## 3. Factor Research

## 4. Model Comparison

## 5. Portfolio Construction

## 6. Backtest

## 7. Risk Analysis

## 8. Attribution

## 9. Limitations

## 10. Future Improvement


---

# 10. Codex执行规则


执行过程中必须：

1. 每完成一个experiment：

更新README。

2. 每新增模块：

写对应unit test。

3. 不允许：

复制粘贴大量重复代码。

4. 不允许：

创建：

experiment1.py

experiment2.py

experiment3.py

导致代码失控。


应该：

抽象：

```
factor_pipeline.py

backtest_engine.py

evaluation.py
```


5. 所有结果必须可复现。


---

# 最终目标

完成后，该项目应该达到：

不是：

“我训练了一个模型预测股票”。

而是：

“我构建了一套完整的A股cross-sectional alpha research platform，从PIT数据、因子研究、机器学习预测、组合构建、交易成本、风险控制到收益归因进行了系统验证。”

这才符合真实 Quant Research 工作流。
