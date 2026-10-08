# Factor catalogue v1

Every factor is calculated from data available by the close of `asof_date` and
is assigned `available_date = next trading day`. The final row of an observed
history has no next-trading-day mapping and is not tradable.

| Factor | Formula / construction | Research hypothesis | Caveat |
|---|---|---|---|
| `momentum_20d`, `60d`, `120d` | trailing compounded daily return | Under-reaction and trend persistence | Must be industry/size neutralized before conclusion |
| `momentum_20d_skip_5d` | 20-day return ending five days ago | Medium-term trend, excluding short-term reversal | Requires 25 days of valid returns |
| `reversal_5d` | negative trailing 5-day compounded return | Short-term liquidity provision and overreaction | Strongly cost and liquidity sensitive |
| `volatility_20d` | trailing standard deviation of daily return | Risk/disagreement proxy | Not an alpha claim by itself |
| `amihud_20d` | mean(abs(return) / amount) | Illiquidity is priced or reflects limits to arbitrage | Units depend on the vendor's amount convention |
| `turnover_20d` | trailing mean turnover | Attention and speculative-trading proxy | Non-linear; high and low tails can behave differently |
| `turnover_surprise_20d` | current turnover / prior 20-day average − 1 | Unusual investor attention | May be event-driven and transient |
| `volume_surprise_20d` | current volume / prior 20-day average − 1 | Volume confirms or contradicts price move | Volume units must be stable across history |
| `amplitude_20d` | mean((high − low) / pre-close) | Intraday disagreement/risk proxy | Distinct from close-to-close volatility |
| `return_volume_corr_20d` | rolling corr(return, log-volume change) | Whether price moves receive volume confirmation | Sensitive to suspensions/zero volume |
| `max_return_20d` | largest daily return in last 20 days | Lottery-like payoff and attention proxy | Can be dominated by limit-up events |
| `earnings_yield` | 1 / positive PE(TTM) | Cheap firms earn value premium | Requires later announcement-date verification |
| `book_to_price` | 1 / positive PB(MRQ) | Cheap firms earn value premium | Requires later announcement-date verification |

## Discovery protocol

A factor enters the candidate library only after all of the following:

1. its economic story is recorded before examining final test performance;
2. units, lookback, winsorization and availability lag are fixed in config;
3. it shows incremental RankIC after industry and size neutralization;
4. it survives walk-forward tests in multiple market regimes and after costs;
5. it is not a duplicate of an existing factor (correlation and incremental-model tests);
6. its outcome is reported whether positive, negative, or inconclusive.

`cross_sectional_rank_zscore` performs per-date 1%/99% clipping and z-scoring.
It is intentionally a later stage: apply it only after all eligible securities
for that date are assembled.
