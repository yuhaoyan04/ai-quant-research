"""Coverage-gated, out-of-sample factor diagnostics.

No IC or bucket-return report is produced if the downloaded equity universe is
materially incomplete: a code-order download can otherwise make a fake alpha.
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake
from quant_research.panel import instrument_type


def config() -> dict:
    with (ROOT / "config" / "factor_research.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def downloaded_a_share_coverage(lake: Lake) -> pd.DataFrame:
    """Coverage of the historical eligible-security union, not today's universe."""
    snapshot_files = list((lake.curated / "security_snapshot").glob("key=*/data.parquet"))
    expected: set[str] = set()
    for path in snapshot_files:
        ids = pd.read_parquet(path, columns=["instrument_id"])["instrument_id"].dropna().astype(str)
        expected.update(code for code in ids if instrument_type(code) == "a_share")
    downloaded = {
        path.parent.name.removeprefix("key=").replace("_", ".")
        for path in (lake.curated / "daily_bars_baostock").glob("key=*/data.parquet")
    }
    rows = [
        {"metric": "historical_a_share_union", "value": len(expected)},
        {"metric": "downloaded_a_share_files", "value": len(expected & downloaded)},
        {"metric": "missing_a_share_files", "value": len(expected - downloaded)},
        {"metric": "download_coverage", "value": len(expected & downloaded) / len(expected) if expected else 0.0},
    ]
    return pd.DataFrame(rows)


def require_coverage(lake: Lake, threshold: float) -> None:
    coverage = downloaded_a_share_coverage(lake)
    rate = float(coverage.loc[coverage["metric"].eq("download_coverage"), "value"].iloc[0])
    if rate < threshold:
        raise RuntimeError(
            f"Factor diagnostics blocked: A-share download coverage is {rate:.1%}, below required {threshold:.0%}. "
            "Run after the downloader finishes; partial code-order samples create selection bias."
        )


def pit_snapshot_membership(lake: Lake) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Return weekly membership and the last *available* snapshot per date.

    `query_all_stock(day=t)` is an end-of-day observation, so we do not let it
    govern a portfolio until t+1.  This is a conservative point-in-time proxy,
    not a substitute for licensed CSI constituent-history data.
    """
    calendar = lake.read("calendar", "SSE")
    open_days = pd.to_datetime(
        calendar.loc[calendar["is_open"].astype(str).eq("1"), "cal_date"]
    ).sort_values().drop_duplicates()
    if open_days.empty:
        raise RuntimeError("calendar is missing; cannot build point-in-time universe")
    next_day = {day.strftime("%Y-%m-%d"): following.strftime("%Y-%m-%d")
                for day, following in zip(open_days.iloc[:-1], open_days.iloc[1:])}
    memberships: dict[str, set[str]] = defaultdict(set)
    available_snapshots: dict[str, str] = {}
    for path in sorted((lake.curated / "security_snapshot").glob("key=*/data.parquet")):
        snapshot_date = path.parent.name.removeprefix("key=")
        available_date = next_day.get(snapshot_date)
        if available_date is None:
            continue
        ids = pd.read_parquet(path, columns=["instrument_id"])["instrument_id"].dropna().astype(str)
        for code in ids:
            if instrument_type(code) == "a_share":
                memberships[code].add(snapshot_date)
        available_snapshots[available_date] = snapshot_date

    state: dict[str, str] = {}
    current: str | None = None
    for day in open_days.dt.strftime("%Y-%m-%d"):
        current = available_snapshots.get(day, current)
        if current is not None:
            state[day] = current
    return dict(memberships), state


def eligible_research_rows(frame: pd.DataFrame, membership: set[str],
                           snapshot_state: dict[str, str]) -> pd.DataFrame:
    """Keep only valid, tradable rows in the then-known weekly universe."""
    snapshot_for_row = frame["available_date"].map(snapshot_state)
    in_universe = snapshot_for_row.isin(membership)
    return frame.loc[
        frame["is_tradable"].eq(True)
        & frame["is_tradable_at_asof"].eq(True)
        & frame["quality_status"].ne("fail")
        & in_universe
    ].copy()


def factor_statistics(panel: pd.DataFrame, factor_columns: list[str], target: str,
                      minimum_cross_section_size: int, bucket_count: int) -> pd.DataFrame:
    """Daily Spearman RankIC and top-minus-bottom bucket returns."""
    records: list[dict[str, float | str | int]] = []
    for factor in factor_columns:
        daily: list[dict[str, float]] = []
        for _, group in panel[["asof_date", factor, target]].dropna().groupby("asof_date"):
            if len(group) < minimum_cross_section_size:
                continue
            # Spearman correlation equals Pearson correlation of ranks.  Writing
            # it explicitly avoids a hidden SciPy runtime dependency.
            ic = group[factor].rank(method="average").corr(group[target].rank(method="average"))
            ranks = group[factor].rank(method="first", pct=True)
            bucket = (ranks * bucket_count).clip(upper=bucket_count - 1e-9).astype(int)
            bucket_returns = group.assign(bucket=bucket).groupby("bucket")[target].mean()
            daily.append({"ic": ic, "long_short": bucket_returns.get(bucket_count - 1, float("nan")) - bucket_returns.get(0, float("nan"))})
        result = pd.DataFrame(daily)
        if result.empty:
            records.append({"factor": factor, "observations": 0, "mean_rank_ic": float("nan"),
                            "icir": float("nan"), "ic_t_stat": float("nan"), "mean_top_minus_bottom": float("nan")})
            continue
        mean_ic, std_ic = result["ic"].mean(), result["ic"].std(ddof=1)
        records.append({
            "factor": factor, "observations": len(result), "mean_rank_ic": mean_ic,
            "icir": mean_ic / std_ic if std_ic and pd.notna(std_ic) else float("nan"),
            "ic_t_stat": mean_ic / (std_ic / len(result) ** 0.5) if std_ic and pd.notna(std_ic) else float("nan"),
            "mean_top_minus_bottom": result["long_short"].mean(),
        })
    return pd.DataFrame(records)


def split_factor_statistics(panel: pd.DataFrame, factor_columns: list[str], target: str,
                            minimum_cross_section_size: int, bucket_count: int,
                            splits: list[dict[str, str]]) -> pd.DataFrame:
    """Evaluate identical factors in chronological, non-overlapping periods."""
    reports: list[pd.DataFrame] = []
    # `panel` is tens of millions of rows.  ISO dates are lexicographically
    # ordered, so filtering strings avoids a full datetime-copy/consolidation.
    columns = ["asof_date", *factor_columns, target]
    for split in splits:
        mask = panel["asof_date"].between(split["start"], split["end"])
        report = factor_statistics(panel.loc[mask, columns], factor_columns, target,
                                   minimum_cross_section_size, bucket_count)
        report.insert(0, "sample_split", split["name"])
        report["split_start"] = split["start"]
        report["split_end"] = split["end"]
        reports.append(report)
    return pd.concat(reports, ignore_index=True)


def run(lake: Lake | None = None) -> None:
    lake = lake or Lake()
    cfg = config()
    coverage = downloaded_a_share_coverage(lake)
    lake.write("research_coverage", coverage, key="a_share", source="factor_research.coverage.v1", request={})
    require_coverage(lake, float(cfg["minimum_download_coverage"]))
    memberships, snapshot_state = pit_snapshot_membership(lake)
    factor_files = {p.parent.name.removeprefix("key="): p for p in (lake.curated / "factors_raw").glob("key=*/data.parquet")}
    label_files = {p.parent.name.removeprefix("key="): p for p in (lake.curated / "labels_raw").glob("key=*/data.parquet")}
    frames: list[pd.DataFrame] = []
    for key in sorted(set(factor_files) & set(label_files)):
        code = key.replace("_", ".")
        if instrument_type(code) != "a_share":
            continue
        factors = pd.read_parquet(factor_files[key])
        labels = pd.read_parquet(label_files[key])
        merged = factors.merge(
            labels[["instrument_id", "asof_date", "quality_status", "is_tradable_at_asof", cfg["target"]]],
            on=["instrument_id", "asof_date"],
        )
        filtered = eligible_research_rows(merged, memberships.get(code, set()), snapshot_state)
        if not filtered.empty:
            frames.append(filtered)
    if not frames:
        raise RuntimeError("No matching factor and label files. Run quality, factors, and labels first.")
    panel = pd.concat(frames, ignore_index=True)
    factor_columns = [c for c in panel if c in {
        "momentum_20d", "momentum_60d", "momentum_120d", "momentum_20d_skip_5d", "reversal_5d",
        "volatility_20d", "amihud_20d", "turnover_20d", "turnover_surprise_20d", "volume_surprise_20d",
        "amplitude_20d", "return_volume_corr_20d", "max_return_20d", "earnings_yield", "book_to_price",
    }]
    report = factor_statistics(panel, factor_columns, cfg["target"], int(cfg["minimum_cross_section_size"]), int(cfg["bucket_count"]))
    report["target"] = cfg["target"]
    report["neutralization_status"] = cfg["neutralization_status"]
    report["universe_status"] = "weekly_pit_proxy_t_plus_1"
    report["tradability_filter"] = "trade_status_eq_1_and_non_st"
    lake.write("factor_diagnostics", report, key="v2", source="factor_research.v2", request=cfg)
    split_report = split_factor_statistics(
        panel, factor_columns, cfg["target"], int(cfg["minimum_cross_section_size"]),
        int(cfg["bucket_count"]), list(cfg["sample_splits"]),
    )
    split_report["target"] = cfg["target"]
    split_report["universe_status"] = "weekly_pit_proxy_t_plus_1"
    split_report["selection_rule"] = "select_on_train_and_validation_only"
    lake.write("factor_diagnostics_splits", split_report, key="v1",
               source="factor_research.splits.v1", request=cfg)
    print(report.to_string(index=False))
