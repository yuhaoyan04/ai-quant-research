"""Fair portfolio comparison using policy fixed by the ridge research path."""

from __future__ import annotations

import pandas as pd

from quant_research.ingest import Lake
from quant_research.liquidity_stress import enrich_liquidity
from quant_research.portfolio import performance_summary
from quant_research.turnover_buffer import _screen_parameters, simulate_buffered_long_only


ANNUALIZATION_PERIODS = 52
MINIMUM_REALIZED_WEIGHT_FRACTION = 0.98
ROUND_TRIP_COST_BPS = 20.0


def _annual(weekly: pd.DataFrame) -> pd.DataFrame:
    included = weekly.loc[weekly["included_in_performance"]].copy()
    included["year"] = included["asof_date"].str.slice(0, 4).astype(int)
    return pd.DataFrame([
        {"year": int(year), **performance_summary(group, ANNUALIZATION_PERIODS)}
        for year, group in included.groupby("year")
    ])


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    ridge_policy = lake.read("turnover_buffer_selection", "v1")
    challenger_selection = lake.read("model_challenger_selection", "hgb_v1")
    if ridge_policy.empty or challenger_selection.empty:
        raise RuntimeError("ridge portfolio policy or challenger model selection missing")
    entry_n = int(ridge_policy.iloc[0]["entry_n"])
    exit_n = int(ridge_policy.iloc[0]["selected_exit_n"])
    target = str(challenger_selection.iloc[0]["target"])
    screen_name, max_amihud, min_turnover = _screen_parameters(lake)

    validation = enrich_liquidity(
        lake.read("model_challenger_validation_predictions", "hgb_v1"), lake)
    test = enrich_liquidity(lake.read("model_challenger_oos_predictions", "hgb_v1"), lake)
    validation_weekly, _ = simulate_buffered_long_only(
        validation, entry_n, exit_n, target, ROUND_TRIP_COST_BPS,
        MINIMUM_REALIZED_WEIGHT_FRACTION, max_amihud, min_turnover)
    test_weekly, holdings = simulate_buffered_long_only(
        test, entry_n, exit_n, target, ROUND_TRIP_COST_BPS,
        MINIMUM_REALIZED_WEIGHT_FRACTION, max_amihud, min_turnover)
    validation_summary = pd.DataFrame([{
        "model": "hgb_ridge_blend", "period": "validation", "entry_n": entry_n,
        "exit_n": exit_n, "liquidity_screen": screen_name,
        **performance_summary(validation_weekly, ANNUALIZATION_PERIODS),
    }])
    test_summary = pd.DataFrame([{
        "model": "hgb_ridge_blend", "period": "test_lockbox", "entry_n": entry_n,
        "exit_n": exit_n, "liquidity_screen": screen_name,
        **performance_summary(test_weekly, ANNUALIZATION_PERIODS),
    }])
    audit = {
        "policy_source": "ridge_validation_selected_policy_frozen",
        "entry_n": entry_n, "exit_n": exit_n, "liquidity_screen": screen_name,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "challenger_portfolio_hyperparameters_retuned": False,
    }
    lake.write("challenger_portfolio_validation_weekly", validation_weekly, key="v1",
               source="challenger_portfolio.v1", request=audit)
    lake.write("challenger_portfolio_validation_summary", validation_summary, key="v1",
               source="challenger_portfolio.v1", request=audit)
    lake.write("challenger_portfolio_oos_weekly", test_weekly, key="v1",
               source="challenger_portfolio.v1", request=audit)
    lake.write("challenger_portfolio_oos_summary", test_summary, key="v1",
               source="challenger_portfolio.v1", request=audit)
    lake.write("challenger_portfolio_oos_annual", _annual(test_weekly), key="v1",
               source="challenger_portfolio.v1", request=audit)
    lake.write("challenger_portfolio_oos_holdings", holdings, key="v1",
               source="challenger_portfolio.v1", request=audit)
    print(validation_summary.to_string(index=False))
    print(test_summary.to_string(index=False))
    print(_annual(test_weekly).to_string(index=False))


if __name__ == "__main__":
    run()
