# 日频可成交组合 CAPM Alpha 与滚动 Beta

## 研究口径

- 收益：约束组合的唯一非重叠日频净值时钟，包含20bp换手成本。
- 主结果：停牌/涨跌停订单逐日重试情景，相对 CSI500 日收益。
- 稳健性：同样回归 CSI300、CSI1000，并保留理想全成交上界。
- 推断：OLS + Newey–West(5)；Alpha 按252日算术年化。
- 无风险利率按0处理；指数为价格收益、不含股息，因此 Alpha 可能被上估。
- 2021–2026 是已查看的 historical OOS，不用于本实验重新选择模型或参数。

## 主结果

- 相对 CSI500、95% Alpha 区间下界为正：hgb_shallow15_locked/industry_size_neutralized, hgb_shallow15_locked/raw, ridge_locked/industry_size_neutralized, ridge_locked/raw。
- 年度正 Alpha 数量和滚动 Beta 区间用于识别时间稳定性，不用于事后择时。

| model                | alpha_pipeline            |   observations |   annualized_alpha_arithmetic |   annualized_alpha_ci_low |   annualized_alpha_ci_high |   alpha_hac_t_stat |   beta |   beta_ci_low |   beta_ci_high |   r_squared |   positive_alpha_years |   rolling_beta_min |   rolling_beta_max |
|:---------------------|:--------------------------|---------------:|------------------------------:|--------------------------:|---------------------------:|-------------------:|-------:|--------------:|---------------:|------------:|-----------------------:|-------------------:|-------------------:|
| hgb_shallow15_locked | industry_size_neutralized |           1387 |                        0.1736 |                    0.0412 |                     0.3061 |             2.5699 | 0.7488 |        0.6619 |         0.8358 |      0.5579 |                      6 |             0.3563 |             0.9983 |
| hgb_shallow15_locked | raw                       |           1387 |                        0.1792 |                    0.0399 |                     0.3185 |             2.5218 | 0.781  |        0.6842 |         0.8778 |      0.5496 |                      6 |             0.3669 |             1.0321 |
| ridge_locked         | industry_size_neutralized |           1387 |                        0.136  |                    0.004  |                     0.2679 |             2.0193 | 0.7427 |        0.647  |         0.8383 |      0.5644 |                      6 |             0.3216 |             0.9906 |
| ridge_locked         | raw                       |           1387 |                        0.1532 |                    0.0184 |                     0.288  |             2.2272 | 0.7458 |        0.6486 |         0.843  |      0.5499 |                      6 |             0.3197 |             0.9935 |

## 限制

单因子 CAPM 不能吸收规模、价值、动量、流动性等本地风格溢价。这里的 Alpha 是相对价格指数的一因子截距，不等同于纯个股选股能力；应与本地多因子归因联合解释。
