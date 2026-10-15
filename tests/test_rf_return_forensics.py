import pandas as pd

from quant_research.rf_return_forensics import return_concentration


def test_return_concentration_drops_only_best_weeks() -> None:
    weekly = pd.DataFrame({
        "net_return": [0.10, -0.02, 0.03],
        "included_in_performance": [True, True, True],
    })
    result = return_concentration(weekly, 52)
    original = result.loc[result["dropped_best_weeks"].eq(0)].iloc[0]
    dropped = result.loc[result["dropped_best_weeks"].eq(1)].iloc[0]
    assert abs(original["terminal_wealth"] - 1.1 * 0.98 * 1.03) < 1e-12
    assert abs(dropped["terminal_wealth"] - 0.98 * 1.03) < 1e-12
    assert dropped["remaining_periods"] == 2
