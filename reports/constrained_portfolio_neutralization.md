# 组合层行业/规模/换手约束实验

## 设计

- 固定 raw 与 industry+size neutralized 两套 alpha、Ridge/HGB 参数和年度扩展窗口。
- 线性规划目标为最大化截面 score rank；单股权重不超过 0.5%。
- 相对 PIT 等权股票池的行业主动权重逐行业不超过 ±3%，组合 |size-z| 不超过 0.10。
- 周度换手目标上限 30%；仅为保持可行性按预设阶梯放宽，并完整披露。
- 收益仍是理想完整成交、20bp round-trip 的研究口径，不替代日频涨跌停执行审计。

## 2021–2026 历史样本外结果

| model                | strategy                                    |   mean_rank_ic |   annualized_return |   sharpe_zero_rate |   maximum_drawdown |   mean_turnover |   mean_abs_portfolio_size_z |   mean_industry_active_weight_l1 |   mean_maximum_abs_industry_active_weight |   turnover_relaxation_fraction |
|:---------------------|:--------------------------------------------|---------------:|--------------------:|-------------------:|-------------------:|----------------:|----------------------------:|---------------------------------:|------------------------------------------:|-------------------------------:|
| ridge_locked         | raw__top200_equal                           |       0.098584 |            0.212414 |           0.797896 |          -0.364231 |        0.557927 |                    0.528089 |                         0.176710 |                                  0.104754 |                       0.000000 |
| ridge_locked         | raw__risk_constrained                       |       0.098584 |            0.213586 |           0.840348 |          -0.327426 |        0.302389 |                    0.094258 |                         0.107749 |                                  0.029918 |                       0.000000 |
| hgb_shallow15_locked | raw__top200_equal                           |       0.100648 |            0.246362 |           0.855978 |          -0.396991 |        0.652341 |                    0.807166 |                         0.158549 |                                  0.097077 |                       0.000000 |
| hgb_shallow15_locked | raw__risk_constrained                       |       0.100648 |            0.246798 |           0.924207 |          -0.328971 |        0.302389 |                    0.099914 |                         0.094968 |                                  0.029628 |                       0.000000 |
| ridge_locked         | industry_size_neutralized__top200_equal     |       0.089952 |            0.155390 |           0.652911 |          -0.346145 |        0.580320 |                    0.337038 |                         0.193889 |                                  0.126212 |                       0.000000 |
| ridge_locked         | industry_size_neutralized__risk_constrained |       0.089952 |            0.188355 |           0.773480 |          -0.314225 |        0.302389 |                    0.089277 |                         0.115769 |                                  0.029847 |                       0.000000 |
| hgb_shallow15_locked | industry_size_neutralized__top200_equal     |       0.094067 |            0.242558 |           0.868618 |          -0.385737 |        0.548019 |                    0.757810 |                         0.171442 |                                  0.119934 |                       0.000000 |
| hgb_shallow15_locked | industry_size_neutralized__risk_constrained |       0.094067 |            0.229308 |           0.911207 |          -0.312818 |        0.302389 |                    0.100000 |                         0.105076 |                                  0.029767 |                       0.000000 |

## 完整中性化栈相对 raw Top-200 的配对区间

| model                | metric                           |   point_estimate |   ci_2_5pct |   ci_97_5pct |
|:---------------------|:---------------------------------|-----------------:|------------:|-------------:|
| ridge_locked         | net_return_annualized_arithmetic |        -0.028235 |   -0.074906 |     0.023121 |
| ridge_locked         | industry_active_weight_l1        |        -0.060941 |   -0.071471 |    -0.051066 |
| ridge_locked         | abs_portfolio_size_z             |        -0.438812 |   -0.502712 |    -0.376460 |
| ridge_locked         | turnover                         |        -0.255538 |   -0.269627 |    -0.241961 |
| hgb_shallow15_locked | net_return_annualized_arithmetic |        -0.028705 |   -0.082297 |     0.031424 |
| hgb_shallow15_locked | industry_active_weight_l1        |        -0.053473 |   -0.062851 |    -0.044438 |
| hgb_shallow15_locked | abs_portfolio_size_z             |        -0.707166 |   -0.738328 |    -0.676912 |
| hgb_shallow15_locked | turnover                         |        -0.349952 |   -0.370024 |    -0.329546 |
