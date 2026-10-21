from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from quant_research.idiosyncratic_target_validation import (
    InsufficientCrossSectionError,
    load_config,
    residualize_cross_section,
    validate_config,
    validate_external_pit_events,
)


def _cross_section(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(20261021)
    size = rng.normal(size=rows)
    beta = rng.normal(size=rows)
    momentum = rng.normal(size=rows)
    industries = np.array(["A", "B", "C", "D"])[np.arange(rows) % 4]
    industry_return = pd.Series(industries).map({"A": 0.02, "B": 0.01, "C": -0.01, "D": 0.0})
    return pd.DataFrame({
        "instrument_id": [f"sh.{600000 + index:06d}" for index in range(rows)],
        "asof_date": ["2020-12-18"] * rows,
        "fwd_return_5d_lag1": (
            industry_return.to_numpy() + 0.01 * size + 0.008 * beta
            + 0.006 * momentum + rng.normal(scale=0.004, size=rows)
        ),
        "industry_broad_code": industries,
        "float_cap_proxy_cny": np.exp(22 + size),
        "beta_252d": beta,
        "volatility_20d": np.exp(-4 + rng.normal(scale=0.2, size=rows)),
        "amihud_20d": np.exp(-10 + rng.normal(scale=0.4, size=rows)),
        "turnover_20d": np.exp(-3 + rng.normal(scale=0.3, size=rows)),
        "earnings_yield": rng.normal(0.05, 0.02, rows),
        "book_to_price": rng.normal(0.6, 0.2, rows),
        "momentum_120d": momentum,
        "momentum_20d_skip_5d": momentum + rng.normal(scale=0.2, size=rows),
    })


def test_residual_target_is_orthogonal_to_declared_controls() -> None:
    cfg = load_config()
    result = residualize_cross_section(_cross_section(), cfg)
    assert len(result.frame) == 240
    assert result.maximum_abs_normal_equation_error < 1e-12
    assert result.r_squared > 0.5
    assert abs(result.frame[cfg["residual_target"]].mean()) < 1e-12


def test_residual_target_rejects_duplicate_instruments() -> None:
    cfg = load_config()
    frame = _cross_section()
    frame.loc[1, "instrument_id"] = frame.loc[0, "instrument_id"]
    with pytest.raises(ValueError, match="duplicate instrument"):
        residualize_cross_section(frame, cfg)


def test_small_cross_section_has_specific_non_contract_exception() -> None:
    cfg = load_config()
    with pytest.raises(InsufficientCrossSectionError):
        residualize_cross_section(_cross_section(50), cfg)


def test_external_pit_contract_rejects_period_end_backfill_and_early_availability() -> None:
    cfg = load_config()
    event = pd.DataFrame([{
        "instrument_id": "sh.600000", "field": "net_profit", "value": 1.0,
        "fiscal_period_end": "2020-12-31", "announced_at": "2021-03-30T08:00:00Z",
        "available_date": "2021-03-30", "revision_sequence": 0,
        "source": "licensed_vendor", "source_version": "snapshot-1",
        "ingested_at": "2021-03-30T09:00:00Z",
    }])
    result = validate_external_pit_events(event, cfg)
    assert result.iloc[0]["announcement_time_contract_passed"]
    event.loc[0, "available_date"] = "2020-12-31"
    with pytest.raises(ValueError, match="precedes"):
        validate_external_pit_events(event, cfg)


def test_config_prohibits_consumed_oos_and_future_lockbox() -> None:
    cfg = load_config()
    validate_config(cfg)
    changed = copy.deepcopy(cfg)
    changed["historical_oos_evaluation_enabled"] = True
    with pytest.raises(ValueError, match="historical OOS"):
        validate_config(changed)
    changed = copy.deepcopy(cfg)
    changed["future_lockbox_access_allowed"] = True
    with pytest.raises(ValueError, match="future lockbox"):
        validate_config(changed)
