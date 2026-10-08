import pandas as pd
import pytest

from quant_research.execution_constraints import execute_with_daily_constraints, price_limit_pct


def test_board_limit_rules() -> None:
    rules = {"st_limit_pct": 5, "main_board_limit_pct": 10, "star_limit_pct": 20,
             "chinext_limit_pct_before_2020_08_24": 10,
             "chinext_limit_pct_after_2020_08_24": 20}
    assert price_limit_pct("sh.600000", "2024-01-01", False, rules) == 10
    assert price_limit_pct("sh.688001", "2024-01-01", False, rules) == 20
    assert price_limit_pct("sz.300001", "2019-01-01", False, rules) == 10
    assert price_limit_pct("sz.300001", "2024-01-01", False, rules) == 20
    assert price_limit_pct("sh.600000", "2024-01-01", True, rules) == 5


def test_blocked_sell_is_retained_and_blocked_buy_not_entered() -> None:
    post, cash, trades = execute_with_daily_constraints(
        {"old": 1.0}, 0.0, {"new": 1.0},
        {"new": False}, {"old": False})
    assert post == {"old": pytest.approx(1.0)}
    assert cash == pytest.approx(0.0)
    assert trades == {}


def test_allowed_rotation_preserves_budget() -> None:
    post, cash, _ = execute_with_daily_constraints(
        {"old": 1.0}, 0.0, {"new": 1.0},
        {"new": True}, {"old": True})
    assert post == {"new": pytest.approx(1.0)}
    assert cash == pytest.approx(0.0)
