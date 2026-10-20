"""Phase 1 transaction-cost protocol on the frozen C0 executed-share path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_research.daily_execution import (
    daily_performance_summary,
    simulate_daily_retry_with_fills,
)
from quant_research.ingest import ROOT, Lake
from quant_research.transaction_costs import COMPONENTS, COST_COLUMNS, price_fills, scenario_inventory


CONFIG_PATH = ROOT / "config" / "transaction_cost_model.yaml"
REPORT_PATH = ROOT / "reports" / "transaction_cost_model.md"
TABLE_ROOT = ROOT / "reports" / "tables"
OUTPUT_KEY = "v1"
GROUP_COLUMNS = ["model", "alpha_pipeline", "strategy"]
FILL_BATCH_COLUMNS = [*GROUP_COLUMNS, "source_execution_scenario"]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    scenario_inventory(result)
    return result


def build_c0_path(
    targets: pd.DataFrame, market: pd.DataFrame, *, cost_bps: float,
    source_execution_scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run the locked targets once and persist the C0 synthetic fills."""
    daily_frames: list[pd.DataFrame] = []
    fill_frames: list[pd.DataFrame] = []
    for keys, group in targets.groupby(GROUP_COLUMNS, sort=False):
        target = group[["execution_date", "instrument_id", "target_weight"]]
        daily, fills = simulate_daily_retry_with_fills(market, target, cost_bps)
        for column, value in reversed(list(zip(GROUP_COLUMNS, keys, strict=True))):
            daily.insert(0, column, value)
            fills.insert(0, column, value)
        daily.insert(3, "source_execution_scenario", source_execution_scenario)
        fills.insert(3, "source_execution_scenario", source_execution_scenario)
        daily_frames.append(daily)
        fill_frames.append(fills)
    return (
        pd.concat(daily_frames, ignore_index=True),
        pd.concat(fill_frames, ignore_index=True),
    )


def reprice_daily_path(
    c0_daily: pd.DataFrame, priced_fills: pd.DataFrame, scenario: dict[str, Any],
) -> pd.DataFrame:
    """Reprice one fixed fill path without feeding alternative fees into shares."""
    scenario_id = str(scenario["scenario_id"])
    selected = priced_fills.loc[priced_fills["scenario_id"].eq(scenario_id)]
    fee_columns = list(COST_COLUMNS)
    aggregate = (
        selected.groupby([*FILL_BATCH_COLUMNS, "trade_date"], as_index=False)[fee_columns]
        .sum()
    )
    result = c0_daily.merge(
        aggregate, on=[*FILL_BATCH_COLUMNS, "trade_date"], how="left",
        validate="one_to_one", suffixes=("_c0", ""),
    )
    for column in fee_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    result.insert(0, "scenario_id", scenario_id)
    result.insert(1, "scenario_group", str(scenario.get("scenario_group", "")))
    result.insert(2, "cost_mode", str(scenario["mode"]))
    result.insert(3, "evidence_status", str(scenario["evidence_status"]))
    result["transaction_cost"] = result["total_cost"] / result["previous_nav"]
    result["net_return"] = result["gross_return"] - result["transaction_cost"]
    result["nav"] = result.groupby(GROUP_COLUMNS, sort=False)["net_return"].transform(
        lambda values: (1.0 + values).cumprod()
    )
    result["posttrade_nav_repriced_no_fee_feedback"] = result["nav"]
    result["cash_path_is_c0_reference"] = True
    for component in [*COMPONENTS, "fixed_round_trip"]:
        result[f"{component}_cost_return"] = (
            result[f"{component}_cost"] / result["previous_nav"]
        )
    result = result.rename(columns={
        "fixed_round_trip_cost_c0": "c0_reference_fixed_round_trip_cost",
        "total_cost_c0": "c0_reference_total_cost",
    })
    return result.drop(columns=[column for column in result if column.endswith("_c0")])


def c0_regression_audit(
    repriced_daily: pd.DataFrame, source_daily: pd.DataFrame, *,
    c0_scenario_id: str, source_execution_scenario: str, tolerance: float,
) -> pd.DataFrame:
    actual = repriced_daily.loc[repriced_daily["scenario_id"].eq(c0_scenario_id)].copy()
    expected = source_daily.loc[source_daily["scenario"].eq(source_execution_scenario)].copy()
    keys = [*GROUP_COLUMNS, "trade_date"]
    columns = [
        "gross_return", "transaction_cost", "net_return", "nav", "buy_weight",
        "sell_weight", "turnover", "blocked_buy_weight", "blocked_sell_weight",
        "target_tracking_l1", "cash_weight", "accounting_weight_sum",
    ]
    merged = expected[keys + columns].merge(
        actual[keys + columns], on=keys, how="outer", validate="one_to_one",
        suffixes=("_expected", "_actual"), indicator=True,
    )
    records = []
    key_mismatches = int(merged["_merge"].ne("both").sum())
    for column in columns:
        difference = (
            pd.to_numeric(merged[f"{column}_actual"], errors="coerce")
            - pd.to_numeric(merged[f"{column}_expected"], errors="coerce")
        ).abs()
        maximum = float(difference.max()) if difference.notna().any() else np.inf
        records.append({
            "metric": column,
            "observations_expected": len(expected),
            "observations_actual": len(actual),
            "key_mismatches": key_mismatches,
            "maximum_absolute_error": maximum,
            "tolerance": tolerance,
            "passed": bool(key_mismatches == 0 and maximum <= tolerance),
        })
    result = pd.DataFrame(records)
    if not result["passed"].all():
        raise RuntimeError("C0 did not reproduce the frozen 20bp execution baseline")
    return result


def reconciliation_audit(
    daily: pd.DataFrame, fills: pd.DataFrame, scenarios: list[dict[str, Any]],
) -> pd.DataFrame:
    records = []
    for scenario in scenarios:
        scenario_id = str(scenario["scenario_id"])
        selected_daily = daily.loc[daily["scenario_id"].eq(scenario_id)]
        selected_fills = fills.loc[fills["scenario_id"].eq(scenario_id)]
        fill_by_day = selected_fills.groupby(
            [*FILL_BATCH_COLUMNS, "trade_date"], as_index=False
        )["total_cost"].sum().rename(columns={"total_cost": "fill_total_cost"})
        compare = selected_daily.merge(
            fill_by_day, on=[*FILL_BATCH_COLUMNS, "trade_date"], how="left",
            validate="one_to_one",
        )
        compare["fill_total_cost"] = compare["fill_total_cost"].fillna(0.0)
        component_sum = selected_fills[[
            "commission_cost", "stamp_duty_cost", "transfer_fee_cost",
            "spread_cost", "market_impact_cost", "fixed_round_trip_cost",
        ]].sum(axis=1)
        records.append({
            "scenario_id": scenario_id,
            "cost_mode": str(scenario["mode"]),
            "daily_rows": len(selected_daily),
            "fill_rows": len(selected_fills),
            "maximum_daily_minus_fill_cost_error": float(
                (compare["total_cost"] - compare["fill_total_cost"]).abs().max()
            ),
            "maximum_fill_component_sum_error": float(
                (selected_fills["total_cost"] - component_sum).abs().max()
            ),
            "negative_fill_cost_rows": int(selected_fills["total_cost"].lt(0).sum()),
            "no_fill_nonzero_cost_days": int(compare.loc[
                compare["gross_two_way_turnover"].eq(0), "total_cost"
            ].ne(0).sum()),
            "passed": bool(
                np.allclose(compare["total_cost"], compare["fill_total_cost"], atol=1e-15)
                and np.allclose(selected_fills["total_cost"], component_sum, atol=1e-15)
                and not selected_fills["total_cost"].lt(0).any()
                and not compare.loc[
                    compare["gross_two_way_turnover"].eq(0), "total_cost"
                ].ne(0).any()
            ),
        })
    result = pd.DataFrame(records)
    if not result["passed"].all():
        raise RuntimeError("transaction-cost reconciliation failed")
    return result


def summarize(daily: pd.DataFrame, annualization: int) -> pd.DataFrame:
    records = []
    keys = ["scenario_id", "scenario_group", "cost_mode", "evidence_status", *GROUP_COLUMNS]
    for values, group in daily.groupby(keys, sort=False):
        record = dict(zip(keys, values, strict=True))
        record.update(daily_performance_summary(group, annualization))
        record.update({
            "mean_buy_turnover": float(group["buy_turnover"].mean()),
            "mean_sell_turnover": float(group["sell_turnover"].mean()),
            "mean_gross_two_way_turnover": float(group["gross_two_way_turnover"].mean()),
            "mean_one_way_turnover": float(group["one_way_turnover"].mean()),
            "mean_weight_based_one_way_turnover_on_target_days": float(
                group["weight_based_one_way_turnover"].dropna().mean()
            ),
            "total_cost_return_arithmetic": float(group["transaction_cost"].sum()),
        })
        for component in [*COMPONENTS, "fixed_round_trip"]:
            record[f"total_{component}_cost_return_arithmetic"] = float(
                group[f"{component}_cost_return"].sum()
            )
        records.append(record)
    return pd.DataFrame(records)


def protocol_frame(config: dict[str, Any], scenarios: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for scenario in scenarios:
        records.append({
            "experiment_id": config["experiment_id"],
            "scenario_id": scenario["scenario_id"],
            "scenario_group": scenario.get("scenario_group"),
            "implementation_status": "IMPLEMENTED",
            "mode": scenario["mode"],
            "round_trip_cost_bps": scenario.get("round_trip_cost_bps"),
            "evidence_status": scenario["evidence_status"],
            "description": scenario.get("description"),
            "components_json": json.dumps(
                scenario.get("components", {}), ensure_ascii=False, sort_keys=True
            ),
            "fixed_c0_fill_path": True,
            "fee_feedback_to_shares_or_cash": False,
            "notional_unit": config["notional_unit"],
            "historical_oos_used_for_selection": False,
        })
    for group, reason in (
        ("C3", "blocked: no reproducible bid-ask quote history"),
        ("C4", "blocked: no broker-calibrated market-impact data"),
        ("C5", "blocked: C4 impact model unavailable for ADV segmentation"),
    ):
        records.append({
            "experiment_id": config["experiment_id"], "scenario_id": group,
            "scenario_group": group, "implementation_status": "BLOCKED_BY_DATA",
            "mode": None, "round_trip_cost_bps": None,
            "evidence_status": reason, "description": reason,
            "components_json": "{}", "fixed_c0_fill_path": True,
            "fee_feedback_to_shares_or_cash": False,
            "notional_unit": config["notional_unit"],
            "historical_oos_used_for_selection": False,
        })
    return pd.DataFrame(records)


def _write_report(
    summary: pd.DataFrame, reconciliation: pd.DataFrame,
    c0_audit: pd.DataFrame, protocol: pd.DataFrame,
) -> None:
    display = summary[[
        "scenario_id", "model", "alpha_pipeline", "annualized_return",
        "annualized_volatility", "sharpe_zero_rate", "maximum_drawdown",
        "mean_buy_turnover", "mean_sell_turnover", "mean_gross_two_way_turnover",
        "mean_one_way_turnover", "total_cost_return_arithmetic",
    ]]
    protocol_display = protocol[[
        "scenario_id", "scenario_group", "implementation_status", "mode",
        "round_trip_cost_bps", "evidence_status", "fixed_c0_fill_path",
        "fee_feedback_to_shares_or_cash", "notional_unit",
    ]]
    primary = summary.loc[
        summary["model"].eq("hgb_shallow15_locked")
        & summary["alpha_pipeline"].eq("raw")
    ].set_index("scenario_id")
    c0_return = float(primary.loc["C0_fixed_020bp", "annualized_return"])
    c1_100_return = float(primary.loc["C1_fixed_100bp", "annualized_return"])
    c2_return = float(
        primary.loc["C2_public_levies_commission_scenario", "annualized_return"]
    )
    lines = [
        "# Phase 1：交易成本与换手口径升级", "",
        "## 研究边界", "",
        "- 信号、模型、目标权重、成交资格和 C0 成交股数路径全部冻结；不同费用只对同一批 fills 重定价。",
        "- fill notional 是归一化组合 NAV 单位，不是人民币券商账单；因此 C2 不启用最低佣金。",
        "- C0/C1 固定往返成本按 portfolio-day 的 `max(buy, sell)` 计算一次，再按成交额比例分配到 fills 以便对账。",
        "- C2 的印花税和过户费按日期生效；3bp 双边佣金是明确情景，不是历史券商真实费率。",
        "- C3–C5 因缺少盘口、ADV 和券商校准冲击数据而保持 blocked，不以虚构参数补齐。", "",
        "## 成本和换手定义", "",
        "- `buy_turnover = buy_notional / pretrade_nav`",
        "- `sell_turnover = sell_notional / pretrade_nav`",
        "- `gross_two_way_turnover = (buy_notional + sell_notional) / pretrade_nav`",
        "- `one_way_turnover = max(buy_notional, sell_notional) / pretrade_nav`",
        "- `weight_based_one_way_turnover = 0.5 * sum(abs(target - drifted))`，只在新目标日定义。", "",
        "## 情景协议", "", protocol_display.to_markdown(index=False), "",
        "## C0 回归", "", c0_audit.to_markdown(index=False, floatfmt=".3e"), "",
        "## Fill / daily 对账", "", reconciliation.to_markdown(index=False, floatfmt=".3e"), "",
        "## 主要发现", "",
        f"- HGB raw 的 C0 20bp 年化为 {c0_return:.2%}，严格复现原执行基线。",
        f"- 同一 fill path 在 100bp 下年化为 {c1_100_return:.2%}；仍为正，但显著低于 C0。",
        f"- C2 公共税费 + 3bp 双边佣金情景年化为 {c2_return:.2%}；它不是券商历史账单。",
        "- 旧 `execution_cost_stress/v1` 没有把 pretrade NAV / previous NAV 的口径差异纳入重定价，因此 exact stress values 已在 invalidation registry 中标记由本实验替代。", "",
        "## 2021–2026 historical OOS", "", display.to_markdown(index=False, floatfmt=".6f"), "",
        "## 费率证据", "",
        "- 2008-09-19 起证券交易印花税改为卖方 1‰：[财政部](https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/200809/t20080919_76432.htm)。",
        "- 2023-08-28 起证券交易印花税减半：[国家税务总局政策法规库](https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html)。",
        "- 2015-08-01 起沪深 A 股过户费统一为成交额 0.02‰ 双向：[上交所](https://big5.sse.com.cn/site/cht/www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20150912_3988866.shtml)。",
        "- 2022-04-29 起过户费为成交额 0.01‰ 双向：[中国结算收费表](https://www.chinaclear.cn/zdjs/editor_file/20220701154723234.pdf)。", "",
        "> 所有 2021–2026 结果仍是 historical OOS，不是 pristine lockbox。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run(lake: Lake | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    lake = lake or Lake()
    config = load_config()
    scenarios = scenario_inventory(config)
    targets_spec = config["source_targets"]
    market_spec = config["source_market"]
    baseline_spec = config["source_daily_baseline"]
    targets = lake.read(str(targets_spec["table"]), str(targets_spec["key"]))
    market = lake.read(str(market_spec["table"]), str(market_spec["key"]))
    source_daily = lake.read(str(baseline_spec["table"]), str(baseline_spec["key"]))
    source_scenario = str(config["primary_execution_scenario"])
    c0_daily, c0_fills = build_c0_path(
        targets, market, cost_bps=float(config["baseline_round_trip_cost_bps"]),
        source_execution_scenario=source_scenario,
    )
    priced_frames = [
        price_fills(c0_fills, scenario, batch_columns=FILL_BATCH_COLUMNS)
        for scenario in scenarios
    ]
    priced_fills = pd.concat(priced_frames, ignore_index=True)
    daily = pd.concat([
        reprice_daily_path(c0_daily, priced_fills, scenario)
        for scenario in scenarios
    ], ignore_index=True)
    c0 = next(item for item in scenarios if item["scenario_id"] == "C0_fixed_020bp")
    c0_audit = c0_regression_audit(
        daily, source_daily, c0_scenario_id=str(c0["scenario_id"]),
        source_execution_scenario=source_scenario,
        tolerance=float(config["c0_regression_tolerance"]),
    )
    reconciliation = reconciliation_audit(daily, priced_fills, scenarios)
    summary = summarize(daily, int(config["annualization_periods"]))
    protocol = protocol_frame(config, scenarios)
    outputs = {
        "transaction_cost_protocol": protocol,
        "transaction_cost_fill_detail": priced_fills,
        "transaction_cost_daily": daily,
        "transaction_cost_summary": summary,
        "transaction_cost_reconciliation": reconciliation,
        "transaction_cost_c0_regression": c0_audit,
    }
    request = {**config, "scenario_count": len(scenarios)}
    for table, frame in outputs.items():
        lake.write(table, frame, key=OUTPUT_KEY, source="transaction_cost_analysis.v1", request=request)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    for table in (
        "transaction_cost_protocol", "transaction_cost_summary",
        "transaction_cost_reconciliation", "transaction_cost_c0_regression",
    ):
        outputs[table].to_csv(TABLE_ROOT / f"{table}.csv", index=False)
    _write_report(summary, reconciliation, c0_audit, protocol)
    print(summary.to_string(index=False), flush=True)
    print(reconciliation.to_string(index=False), flush=True)
    return summary, reconciliation


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 transaction-cost analysis")
    parser.add_argument("command", choices=["run"])
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
