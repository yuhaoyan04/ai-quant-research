"""Config-driven transaction-cost pricing for synthetic execution fills."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


COMPONENTS = (
    "commission", "stamp_duty", "transfer_fee", "spread", "market_impact",
)
COST_COLUMNS = tuple(f"{name}_cost" for name in COMPONENTS) + (
    "fixed_round_trip_cost", "total_cost",
)


def infer_venue(instrument_id: str) -> str:
    """Map the repository's BaoStock instrument identifier to a venue."""
    value = str(instrument_id).lower()
    if value.startswith("sh."):
        return "SSE"
    if value.startswith("sz."):
        return "SZSE"
    return "UNKNOWN"


def validate_scenario(scenario: dict[str, Any]) -> None:
    required = {"scenario_id", "mode", "evidence_status"}
    if missing := required - set(scenario):
        raise KeyError(f"cost scenario fields missing: {sorted(missing)}")
    mode = str(scenario["mode"])
    if mode not in {"fixed_round_trip", "components"}:
        raise ValueError(f"unsupported cost mode: {mode}")
    if mode == "fixed_round_trip":
        rate = float(scenario.get("round_trip_cost_bps", np.nan))
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("fixed round-trip cost must be finite and non-negative")
        if scenario.get("components"):
            raise ValueError("fixed round-trip and component fees are mutually exclusive")
        return
    if "round_trip_cost_bps" in scenario:
        raise ValueError("component scenario cannot also define round-trip cost")
    components = scenario.get("components")
    if not isinstance(components, dict):
        raise KeyError("component scenario requires a components mapping")
    unknown = set(components) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown cost components: {sorted(unknown)}")
    for name in COMPONENTS:
        spec = components.get(name, {})
        minimum = float(spec.get("minimum_fee_per_fill", 0.0))
        if not np.isfinite(minimum) or minimum < 0:
            raise ValueError(f"{name}: minimum fee must be finite and non-negative")
        rules = spec.get("rules", [])
        if not isinstance(rules, list):
            raise TypeError(f"{name}: rules must be a list")
        for rule in rules:
            for side in ("buy", "sell"):
                rate = float(rule.get(f"{side}_bps", 0.0))
                if not np.isfinite(rate) or rate < 0:
                    raise ValueError(f"{name}: {side} rate must be non-negative")


def _rule_matches(rule: dict[str, Any], *, trade_date: str, venue: str) -> bool:
    date = pd.Timestamp(trade_date)
    start = pd.Timestamp(rule.get("start_date", "1900-01-01"))
    raw_end = rule.get("end_date")
    end = pd.Timestamp(raw_end) if raw_end else pd.Timestamp.max.normalize()
    venues = rule.get("venues")
    return start <= date <= end and (not venues or venue in {str(v) for v in venues})


def _component_rate_bps(
    spec: dict[str, Any], *, trade_date: str, side: str, venue: str, component: str,
) -> float:
    matches = [
        rule for rule in spec.get("rules", [])
        if _rule_matches(rule, trade_date=trade_date, venue=venue)
    ]
    if len(matches) > 1:
        raise ValueError(
            f"{component}: overlapping fee rules for {trade_date}/{venue}"
        )
    if not matches:
        if bool(spec.get("require_full_coverage", True)):
            raise ValueError(
                f"{component}: no fee rule for {trade_date}/{venue}"
            )
        return 0.0
    return float(matches[0].get(f"{side}_bps", 0.0))


def _validate_fills(fills: pd.DataFrame) -> pd.DataFrame:
    required = {"fill_id", "trade_date", "instrument_id", "side", "notional"}
    if missing := required - set(fills):
        raise KeyError(f"fill fields missing: {sorted(missing)}")
    result = fills.copy()
    result["side"] = result["side"].astype(str).str.lower()
    if not result["side"].isin(["buy", "sell"]).all():
        raise ValueError("fill side must be buy or sell")
    result["notional"] = pd.to_numeric(result["notional"], errors="raise")
    if (~np.isfinite(result["notional"]) | result["notional"].le(0)).any():
        raise ValueError("fill notional must be finite and positive")
    if "venue" not in result:
        result["venue"] = result["instrument_id"].map(infer_venue)
    return result


def _apply_fixed_round_trip(
    fills: pd.DataFrame, scenario: dict[str, Any], batch_columns: list[str],
) -> pd.DataFrame:
    result = fills.copy()
    for column in COST_COLUMNS:
        result[column] = 0.0
    rate = float(scenario["round_trip_cost_bps"]) / 10_000.0
    grouping = [*batch_columns, "trade_date"]
    if missing := set(grouping) - set(result):
        raise KeyError(f"fixed-cost batch fields missing: {sorted(missing)}")
    for _, index in result.groupby(grouping, sort=False, dropna=False).groups.items():
        rows = result.loc[index]
        buy = float(rows.loc[rows["side"].eq("buy"), "notional"].sum())
        sell = float(rows.loc[rows["side"].eq("sell"), "notional"].sum())
        total = max(buy, sell) * rate
        gross = float(rows["notional"].sum())
        if gross > 0 and total > 0:
            result.loc[index, "fixed_round_trip_cost"] = (
                rows["notional"] / gross * total
            )
    result["total_cost"] = result["fixed_round_trip_cost"]
    return result


def _apply_components(fills: pd.DataFrame, scenario: dict[str, Any]) -> pd.DataFrame:
    result = fills.copy()
    result["fixed_round_trip_cost"] = 0.0
    components = scenario["components"]
    for component in COMPONENTS:
        spec = components.get(component, {"rules": [], "require_full_coverage": False})
        rates = []
        for row in result.itertuples(index=False):
            rates.append(_component_rate_bps(
                spec, trade_date=str(row.trade_date), side=str(row.side),
                venue=str(row.venue), component=component,
            ))
        variable = result["notional"].to_numpy(dtype=float) * np.asarray(rates) / 10_000.0
        minimum = float(spec.get("minimum_fee_per_fill", 0.0))
        result[f"{component}_cost"] = np.maximum(variable, minimum)
    result["total_cost"] = result[[f"{name}_cost" for name in COMPONENTS]].sum(axis=1)
    return result


def price_fills(
    fills: pd.DataFrame, scenario: dict[str, Any], *,
    batch_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Price each fill and return an auditable component-level cost ledger.

    Fixed round-trip cost is a portfolio-day convention rather than a literal
    fill tariff. It is allocated pro rata over that day's gross fill notional
    solely so its aggregate can be reconciled from the fill ledger.
    """
    validate_scenario(scenario)
    validated = _validate_fills(fills)
    if str(scenario["mode"]) == "fixed_round_trip":
        result = _apply_fixed_round_trip(validated, scenario, list(batch_columns))
    else:
        result = _apply_components(validated, scenario)
    if (result[list(COST_COLUMNS)] < -1e-15).any().any():
        raise RuntimeError("transaction cost model produced a negative fee")
    component_total = result[[
        "commission_cost", "stamp_duty_cost", "transfer_fee_cost",
        "spread_cost", "market_impact_cost", "fixed_round_trip_cost",
    ]].sum(axis=1)
    if not np.allclose(component_total, result["total_cost"], atol=1e-15, rtol=1e-12):
        raise RuntimeError("fill total cost does not equal component sum")
    result.insert(0, "scenario_id", str(scenario["scenario_id"]))
    result.insert(1, "scenario_group", str(scenario.get("scenario_group", "")))
    result.insert(2, "cost_mode", str(scenario["mode"]))
    result.insert(3, "evidence_status", str(scenario["evidence_status"]))
    return result


def scenario_inventory(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ordered, validated C0/C1/C2 scenario list."""
    scenarios = [*config.get("fixed_scenarios", []), *config.get("component_scenarios", [])]
    identifiers = [str(item.get("scenario_id")) for item in scenarios]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("transaction-cost scenario identifiers must be unique")
    for scenario in scenarios:
        validate_scenario(scenario)
    return scenarios
