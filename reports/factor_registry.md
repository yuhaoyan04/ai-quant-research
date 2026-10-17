# 因子注册表

版本：`1.0.0`；生效日：`2026-10-17`。
所有因子只使用 `asof_date` 收盘前已知数据，并在下一交易日才可用。理论方向是事前经济假设；训练期方向仅由 2006–2015 原始 RankIC 锁定，两者不一致不作事后改写。

| factor                 | family              | unit                                |   lookback_observations | raw_fields         |   theoretical_direction |   locked_training_direction | direction_agrees   |
|:-----------------------|:--------------------|:------------------------------------|------------------------:|:-------------------|------------------------:|----------------------------:|:-------------------|
| amihud_20d             | liquidity           | inverse_baostock_amount             |                      20 | pct_chg,amount     |                       1 |                           1 | True               |
| amplitude_20d          | risk                | decimal_price_range                 |                      20 | high,low,pre_close |                      -1 |                          -1 | True               |
| book_to_price          | value               | inverse_pb                          |                       1 | pb_mrq             |                       1 |                           1 | True               |
| earnings_yield         | value               | inverse_pe                          |                       1 | pe_ttm             |                       1 |                           1 | True               |
| max_return_20d         | lottery_attention   | decimal_return                      |                      20 | pct_chg            |                      -1 |                          -1 | True               |
| momentum_120d          | momentum            | decimal_return                      |                     120 | pct_chg            |                       1 |                          -1 | False              |
| momentum_20d           | momentum            | decimal_return                      |                      20 | pct_chg            |                       1 |                          -1 | False              |
| momentum_20d_skip_5d   | momentum            | decimal_return                      |                      25 | pct_chg            |                       1 |                          -1 | False              |
| momentum_60d           | momentum            | decimal_return                      |                      60 | pct_chg            |                       1 |                          -1 | False              |
| return_volume_corr_20d | price_volume        | correlation                         |                      21 | pct_chg,volume     |                     nan |                          -1 |                    |
| reversal_5d            | reversal            | decimal_return                      |                       5 | pct_chg            |                       1 |                           1 | True               |
| turnover_20d           | liquidity_attention | baostock_turnover_percentage_points |                      20 | turn               |                      -1 |                          -1 | True               |
| turnover_surprise_20d  | liquidity_attention | ratio_minus_one                     |                      21 | turn               |                      -1 |                          -1 | True               |
| volatility_20d         | risk                | daily_decimal_volatility            |                      20 | pct_chg            |                      -1 |                          -1 | True               |
| volume_surprise_20d    | liquidity_attention | ratio_minus_one                     |                      21 | volume             |                      -1 |                          -1 | True               |

完整公式、缺失机制与研究假设以 `config/factor_registry.yaml` 为唯一机器可读来源。