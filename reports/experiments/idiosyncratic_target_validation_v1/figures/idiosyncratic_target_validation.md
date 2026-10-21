# Idiosyncratic Target Validation v1

## 研究结论

通过预注册门槛的模型：['hgb_shallow15_locked', 'ridge_locked']。这只允许进入下一阶段组合研究，不等于已经识别可交易纯 Alpha。

本实验只使用 2008–2015 训练和 2016–2020 验证。252 日 PIT Beta 从 2008 年开始；两条标签路径使用完全相同的样本，没有对 2006–2007 缺失 Beta 做回填。没有读取 2021–2026 结果，也没有访问或修改 future lockbox 与已冻结 paper candidate。

## 验证汇总

| model                | training_target             |   observations |   mean_raw_rank_ic |   mean_residual_rank_ic |   residual_rank_ic_std |   positive_residual_rank_ic_fraction |   mean_max_abs_score_style_correlation |
|:---------------------|:----------------------------|---------------:|-------------------:|------------------------:|-----------------------:|-------------------------------------:|---------------------------------------:|
| hgb_shallow15_locked | fwd_return_5d_lag1          |            254 |           0.089681 |                0.053271 |               0.061789 |                             0.830709 |                               0.501103 |
| hgb_shallow15_locked | fwd_specific_return_5d_lag1 |            254 |           0.088175 |                0.083660 |               0.068345 |                             0.909449 |                               0.438292 |
| ridge_locked         | fwd_return_5d_lag1          |            254 |           0.091703 |                0.056128 |               0.062045 |                             0.822835 |                               0.606253 |
| ridge_locked         | fwd_specific_return_5d_lag1 |            254 |           0.084987 |                0.077758 |               0.064471 |                             0.901575 |                               0.509317 |

## 配对推断

| model                | metric           | comparison                       |   observations |   point_estimate |   ci_2_5pct |   ci_97_5pct |   one_sided_probability_nonpositive |   bh_fdr_adjusted_p | survives_bh_fdr_5pct   |
|:---------------------|:-----------------|:---------------------------------|---------------:|-----------------:|------------:|-------------:|------------------------------------:|--------------------:|:-----------------------|
| hgb_shallow15_locked | residual_rank_ic | residual_target_minus_raw_target |            254 |         0.030389 |    0.025682 |     0.035496 |                            0.000500 |            0.000500 | True                   |
| ridge_locked         | residual_rank_ic | residual_target_minus_raw_target |            254 |         0.021630 |    0.018295 |     0.025273 |                            0.000500 |            0.000500 | True                   |

## 预先指定的晋级门槛

| model                |   positive_validation_years |   required_positive_validation_years | positive_point_estimate   | ci_excludes_zero   | bh_fdr_5pct   | positive_year_breadth   | positive_residual_rank_ic   | promote_to_portfolio_research   |
|:---------------------|----------------------------:|-------------------------------------:|:--------------------------|:-------------------|:--------------|:------------------------|:----------------------------|:--------------------------------|
| hgb_shallow15_locked |                           5 |                                    4 | True                      | True               | True          | True                    | True                        | True                            |
| ridge_locked         |                           5 |                                    4 | True                      | True               | True          | True                    | True                        | True                            |

## 机器验收

| check                            | status   | detail                |
|:---------------------------------|:---------|:----------------------|
| development_period_ends_2020     | PASS     | 2020                  |
| historical_oos_not_evaluated     | PASS     | False                 |
| future_lockbox_not_accessed      | PASS     | False                 |
| residual_normal_equations_close  | PASS     | 6.681667715134163e-16 |
| paired_model_dates_complete      | PASS     | [254, 254]            |
| external_pit_data_not_fabricated | PASS     | BLOCKED_EXTERNAL_DATA |

## 解释边界

残差标签使用信号日可知的 PIT 行业与风格暴露，但标签本身仍是未来收益，只能用于训练和事后评价。信号层改善必须经过相同成本、成交和风格匹配基准的组合检验，随后从新日期重新开始 forward evidence。公告日基本面数据仍处于外部数据阻塞状态，未被伪造。
晋级规则在成功运行前已写入本地配置，但没有独立第三方时间戳，不能冒充外部预注册。
