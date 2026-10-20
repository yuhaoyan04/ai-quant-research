# Phase 1：交易成本与换手口径升级

## 研究边界

- 信号、模型、目标权重、成交资格和 C0 成交股数路径全部冻结；不同费用只对同一批 fills 重定价。
- fill notional 是归一化组合 NAV 单位，不是人民币券商账单；因此 C2 不启用最低佣金。
- C0/C1 固定往返成本按 portfolio-day 的 `max(buy, sell)` 计算一次，再按成交额比例分配到 fills 以便对账。
- C2 的印花税和过户费按日期生效；3bp 双边佣金是明确情景，不是历史券商真实费率。
- C3–C5 因缺少盘口、ADV 和券商校准冲击数据而保持 blocked，不以虚构参数补齐。

## 成本和换手定义

- `buy_turnover = buy_notional / pretrade_nav`
- `sell_turnover = sell_notional / pretrade_nav`
- `gross_two_way_turnover = (buy_notional + sell_notional) / pretrade_nav`
- `one_way_turnover = max(buy_notional, sell_notional) / pretrade_nav`
- `weight_based_one_way_turnover = 0.5 * sum(abs(target - drifted))`，只在新目标日定义。

## 情景协议

| scenario_id                          | scenario_group   | implementation_status   | mode             |   round_trip_cost_bps | evidence_status                                                     | fixed_c0_fill_path   | fee_feedback_to_shares_or_cash   | notional_unit                    |
|:-------------------------------------|:-----------------|:------------------------|:-----------------|----------------------:|:--------------------------------------------------------------------|:---------------------|:---------------------------------|:---------------------------------|
| C1_fixed_000bp                       | C1               | IMPLEMENTED             | fixed_round_trip |                     0 | research_scenario                                                   | True                 | False                            | normalized_portfolio_nav_not_cny |
| C1_fixed_010bp                       | C1               | IMPLEMENTED             | fixed_round_trip |                    10 | research_scenario                                                   | True                 | False                            | normalized_portfolio_nav_not_cny |
| C0_fixed_020bp                       | C0_C1            | IMPLEMENTED             | fixed_round_trip |                    20 | existing_research_baseline                                          | True                 | False                            | normalized_portfolio_nav_not_cny |
| C1_fixed_040bp                       | C1               | IMPLEMENTED             | fixed_round_trip |                    40 | research_scenario                                                   | True                 | False                            | normalized_portfolio_nav_not_cny |
| C1_fixed_060bp                       | C1               | IMPLEMENTED             | fixed_round_trip |                    60 | research_scenario                                                   | True                 | False                            | normalized_portfolio_nav_not_cny |
| C1_fixed_100bp                       | C1               | IMPLEMENTED             | fixed_round_trip |                   100 | research_scenario                                                   | True                 | False                            | normalized_portfolio_nav_not_cny |
| C2_public_levies_commission_scenario | C2               | IMPLEMENTED             | components       |                   nan | documented_public_levies_plus_explicit_broker_commission_assumption | True                 | False                            | normalized_portfolio_nav_not_cny |
| C3                                   | C3               | BLOCKED_BY_DATA         | nan              |                   nan | blocked: no reproducible bid-ask quote history                      | True                 | False                            | normalized_portfolio_nav_not_cny |
| C4                                   | C4               | BLOCKED_BY_DATA         | nan              |                   nan | blocked: no broker-calibrated market-impact data                    | True                 | False                            | normalized_portfolio_nav_not_cny |
| C5                                   | C5               | BLOCKED_BY_DATA         | nan              |                   nan | blocked: C4 impact model unavailable for ADV segmentation           | True                 | False                            | normalized_portfolio_nav_not_cny |

## C0 回归

| metric                |   observations_expected |   observations_actual |   key_mismatches |   maximum_absolute_error |   tolerance | passed   |
|:----------------------|------------------------:|----------------------:|-----------------:|-------------------------:|------------:|:---------|
| gross_return          |                    5548 |                  5548 |                0 |                8.660e-15 |   1.000e-12 | True     |
| transaction_cost      |                    5548 |                  5548 |                0 |                4.012e-18 |   1.000e-12 | True     |
| net_return            |                    5548 |                  5548 |                0 |                8.660e-15 |   1.000e-12 | True     |
| nav                   |                    5548 |                  5548 |                0 |                1.130e-13 |   1.000e-12 | True     |
| buy_weight            |                    5548 |                  5548 |                0 |                1.776e-15 |   1.000e-12 | True     |
| sell_weight           |                    5548 |                  5548 |                0 |                1.499e-15 |   1.000e-12 | True     |
| turnover              |                    5548 |                  5548 |                0 |                1.610e-15 |   1.000e-12 | True     |
| blocked_buy_weight    |                    5548 |                  5548 |                0 |                2.051e-15 |   1.000e-12 | True     |
| blocked_sell_weight   |                    5548 |                  5548 |                0 |                1.043e-15 |   1.000e-12 | True     |
| target_tracking_l1    |                    5548 |                  5548 |                0 |                7.858e-15 |   1.000e-12 | True     |
| cash_weight           |                    5548 |                  5548 |                0 |                1.611e-15 |   1.000e-12 | True     |
| accounting_weight_sum |                    5548 |                  5548 |                0 |                1.354e-14 |   1.000e-12 | True     |

## Fill / daily 对账

| scenario_id                          | cost_mode        |   daily_rows |   fill_rows |   maximum_daily_minus_fill_cost_error |   maximum_fill_component_sum_error |   negative_fill_cost_rows |   no_fill_nonzero_cost_days | passed   |
|:-------------------------------------|:-----------------|-------------:|------------:|--------------------------------------:|-----------------------------------:|--------------------------:|----------------------------:|:---------|
| C1_fixed_000bp                       | fixed_round_trip |         5548 |      440126 |                             0.000e+00 |                          0.000e+00 |                         0 |                           0 | True     |
| C1_fixed_010bp                       | fixed_round_trip |         5548 |      440126 |                             0.000e+00 |                          0.000e+00 |                         0 |                           0 | True     |
| C0_fixed_020bp                       | fixed_round_trip |         5548 |      440126 |                             0.000e+00 |                          0.000e+00 |                         0 |                           0 | True     |
| C1_fixed_040bp                       | fixed_round_trip |         5548 |      440126 |                             0.000e+00 |                          0.000e+00 |                         0 |                           0 | True     |
| C1_fixed_060bp                       | fixed_round_trip |         5548 |      440126 |                             0.000e+00 |                          0.000e+00 |                         0 |                           0 | True     |
| C1_fixed_100bp                       | fixed_round_trip |         5548 |      440126 |                             0.000e+00 |                          0.000e+00 |                         0 |                           0 | True     |
| C2_public_levies_commission_scenario | components       |         5548 |      440126 |                             0.000e+00 |                          0.000e+00 |                         0 |                           0 | True     |

## 主要发现

- HGB raw 的 C0 20bp 年化为 20.81%，严格复现原执行基线。
- 同一 fill path 在 100bp 下年化为 6.20%；仍为正，但显著低于 C0。
- C2 公共税费 + 3bp 双边佣金情景年化为 22.11%；它不是券商历史账单。
- 旧 `execution_cost_stress/v1` 没有把 pretrade NAV / previous NAV 的口径差异纳入重定价，因此 exact stress values 已在 invalidation registry 中标记由本实验替代。

## 2021–2026 historical OOS

| scenario_id                          | model                | alpha_pipeline            |   annualized_return |   annualized_volatility |   sharpe_zero_rate |   maximum_drawdown |   mean_buy_turnover |   mean_sell_turnover |   mean_gross_two_way_turnover |   mean_one_way_turnover |   total_cost_return_arithmetic |
|:-------------------------------------|:---------------------|:--------------------------|--------------------:|------------------------:|-------------------:|-------------------:|--------------------:|---------------------:|------------------------------:|------------------------:|-------------------------------:|
| C1_fixed_000bp                       | hgb_shallow15_locked | industry_size_neutralized |            0.241855 |                0.219293 |           1.098387 |          -0.310855 |            0.063406 |             0.062814 |                      0.126220 |                0.063716 |                       0.000000 |
| C1_fixed_000bp                       | hgb_shallow15_locked | raw                       |            0.247624 |                0.230398 |           1.076318 |          -0.328898 |            0.063422 |             0.062828 |                      0.126250 |                0.063826 |                       0.000000 |
| C1_fixed_000bp                       | ridge_locked         | industry_size_neutralized |            0.196370 |                0.216213 |           0.938019 |          -0.306512 |            0.063273 |             0.062679 |                      0.125952 |                0.063539 |                       0.000000 |
| C1_fixed_000bp                       | ridge_locked         | raw                       |            0.216095 |                0.219947 |           1.000204 |          -0.319501 |            0.062990 |             0.062396 |                      0.125386 |                0.063199 |                       0.000000 |
| C1_fixed_010bp                       | hgb_shallow15_locked | industry_size_neutralized |            0.222072 |                0.219184 |           1.025509 |          -0.313085 |            0.063406 |             0.062814 |                      0.126220 |                0.063716 |                       0.088576 |
| C1_fixed_010bp                       | hgb_shallow15_locked | raw                       |            0.227714 |                0.230293 |           1.006807 |          -0.331077 |            0.063422 |             0.062828 |                      0.126250 |                0.063826 |                       0.088731 |
| C1_fixed_010bp                       | ridge_locked         | industry_size_neutralized |            0.177363 |                0.216110 |           0.864223 |          -0.311850 |            0.063273 |             0.062679 |                      0.125952 |                0.063539 |                       0.088308 |
| C1_fixed_010bp                       | ridge_locked         | raw                       |            0.196878 |                0.219854 |           0.928046 |          -0.324726 |            0.062990 |             0.062396 |                      0.125386 |                0.063199 |                       0.087831 |
| C0_fixed_020bp                       | hgb_shallow15_locked | industry_size_neutralized |            0.202598 |                0.219093 |           0.952482 |          -0.315308 |            0.063406 |             0.062814 |                      0.126220 |                0.063716 |                       0.177151 |
| C0_fixed_020bp                       | hgb_shallow15_locked | raw                       |            0.208117 |                0.230205 |           0.937165 |          -0.333249 |            0.063422 |             0.062828 |                      0.126250 |                0.063826 |                       0.177462 |
| C0_fixed_020bp                       | ridge_locked         | industry_size_neutralized |            0.158653 |                0.216025 |           0.790291 |          -0.317149 |            0.063273 |             0.062679 |                      0.125952 |                0.063539 |                       0.176615 |
| C0_fixed_020bp                       | ridge_locked         | raw                       |            0.177959 |                0.219777 |           0.855759 |          -0.329912 |            0.062990 |             0.062396 |                      0.125386 |                0.063199 |                       0.175663 |
| C1_fixed_040bp                       | hgb_shallow15_locked | industry_size_neutralized |            0.164558 |                0.218964 |           0.806052 |          -0.321687 |            0.063406 |             0.062814 |                      0.126220 |                0.063716 |                       0.354302 |
| C1_fixed_040bp                       | hgb_shallow15_locked | raw                       |            0.169838 |                0.230077 |           0.797545 |          -0.337575 |            0.063422 |             0.062828 |                      0.126250 |                0.063826 |                       0.354924 |
| C1_fixed_040bp                       | ridge_locked         | industry_size_neutralized |            0.122104 |                0.215909 |           0.642096 |          -0.330255 |            0.063273 |             0.062679 |                      0.125952 |                0.063539 |                       0.353230 |
| C1_fixed_040bp                       | ridge_locked         | raw                       |            0.140996 |                0.219676 |           0.710867 |          -0.340170 |            0.062990 |             0.062396 |                      0.125386 |                0.063199 |                       0.351326 |
| C1_fixed_060bp                       | hgb_shallow15_locked | industry_size_neutralized |            0.127700 |                0.218905 |           0.659237 |          -0.332077 |            0.063406 |             0.062814 |                      0.126220 |                0.063716 |                       0.531453 |
| C1_fixed_060bp                       | hgb_shallow15_locked | raw                       |            0.132750 |                0.230017 |           0.657580 |          -0.352492 |            0.063422 |             0.062828 |                      0.126250 |                0.063826 |                       0.532386 |
| C1_fixed_060bp                       | ridge_locked         | industry_size_neutralized |            0.086686 |                0.215864 |           0.493578 |          -0.349127 |            0.063273 |             0.062679 |                      0.125952 |                0.063539 |                       0.529845 |
| C1_fixed_060bp                       | ridge_locked         | raw                       |            0.105172 |                0.219645 |           0.565663 |          -0.352650 |            0.062990 |             0.062396 |                      0.125386 |                0.063199 |                       0.526988 |
| C1_fixed_100bp                       | hgb_shallow15_locked | industry_size_neutralized |            0.057385 |                0.218999 |           0.365017 |          -0.364093 |            0.063406 |             0.062814 |                      0.126220 |                0.063716 |                       0.885755 |
| C1_fixed_100bp                       | hgb_shallow15_locked | raw                       |            0.062002 |                0.230097 |           0.377099 |          -0.388498 |            0.063422 |             0.062828 |                      0.126250 |                0.063826 |                       0.887309 |
| C1_fixed_100bp                       | ridge_locked         | industry_size_neutralized |            0.019110 |                0.215987 |           0.196161 |          -0.401911 |            0.063273 |             0.062679 |                      0.125952 |                0.063539 |                       0.883075 |
| C1_fixed_100bp                       | ridge_locked         | raw                       |            0.036803 |                0.219791 |           0.274869 |          -0.399250 |            0.062990 |             0.062396 |                      0.125386 |                0.063199 |                       0.878314 |
| C2_public_levies_commission_scenario | hgb_shallow15_locked | industry_size_neutralized |            0.215475 |                0.219148 |           1.000927 |          -0.313351 |            0.063406 |             0.062814 |                      0.126220 |                0.063716 |                       0.118428 |
| C2_public_levies_commission_scenario | hgb_shallow15_locked | raw                       |            0.221116 |                0.230261 |           0.983495 |          -0.331337 |            0.063422 |             0.062828 |                      0.126250 |                0.063826 |                       0.118454 |
| C2_public_levies_commission_scenario | ridge_locked         | industry_size_neutralized |            0.171011 |                0.216075 |           0.839276 |          -0.312786 |            0.063273 |             0.062679 |                      0.125952 |                0.063539 |                       0.118142 |
| C2_public_levies_commission_scenario | ridge_locked         | raw                       |            0.190428 |                0.219825 |           0.903543 |          -0.325639 |            0.062990 |             0.062396 |                      0.125386 |                0.063199 |                       0.117624 |

## 费率证据

- 2008-09-19 起证券交易印花税改为卖方 1‰：[财政部](https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/200809/t20080919_76432.htm)。
- 2023-08-28 起证券交易印花税减半：[国家税务总局政策法规库](https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html)。
- 2015-08-01 起沪深 A 股过户费统一为成交额 0.02‰ 双向：[上交所](https://big5.sse.com.cn/site/cht/www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20150912_3988866.shtml)。
- 2022-04-29 起过户费为成交额 0.01‰ 双向：[中国结算收费表](https://www.chinaclear.cn/zdjs/editor_file/20220701154723234.pdf)。

> 所有 2021–2026 结果仍是 historical OOS，不是 pristine lockbox。