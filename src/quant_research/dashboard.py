"""Generate a self-contained, offline HTML data-quality dashboard."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from quant_research.factors import build_raw_factors, next_trading_day_map
from quant_research.ingest import ROOT, Lake
from quant_research.panel import instrument_type
from quant_research.quality import validate_daily_bars


REPORT = ROOT / "reports" / "data_panel.html"


def _svg_line(values: pd.Series, *, color: str = "#4f8cff", width: int = 720, height: int = 220) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2:
        return "<p>数据不足，无法绘图。</p>"
    minimum, maximum = float(values.min()), float(values.max())
    span = maximum - minimum or 1.0
    points = " ".join(
        f"{i/(len(values)-1)*width:.1f},{height-(float(v)-minimum)/span*(height-16)-8:.1f}"
        for i, v in enumerate(values)
    )
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="line chart">
      <line x1="0" y1="{height-8}" x2="{width}" y2="{height-8}" stroke="#cbd5e1"/>
      <polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>
      <text x="4" y="16" fill="#64748b" font-size="12">max {maximum:.2f}</text>
      <text x="4" y="{height-12}" fill="#64748b" font-size="12">min {minimum:.2f}</text>
    </svg>'''


def _histogram(values: pd.Series, *, bins: int = 30, width: int = 720, height: int = 220) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "<p>数据不足，无法绘图。</p>"
    counts, _ = pd.cut(values, bins=bins, include_lowest=True).value_counts(sort=False), None
    maximum = max(counts.max(), 1)
    bar_width = width / len(counts)
    bars = "".join(
        f'<rect x="{i*bar_width:.1f}" y="{height-count/maximum*(height-22)-8:.1f}" '
        f'width="{max(bar_width-1, 1):.1f}" height="{count/maximum*(height-22):.1f}" fill="#35b9a4"/>'
        for i, count in enumerate(counts)
    )
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="histogram">
      <line x1="0" y1="{height-8}" x2="{width}" y2="{height-8}" stroke="#cbd5e1"/>{bars}
      <text x="4" y="16" fill="#64748b" font-size="12">日收益率分布（%）</text>
    </svg>'''


def _choose_equity(files: list[Path]) -> Path:
    for path in files:
        code = path.parent.name.removeprefix("key=").replace("_", ".")
        if instrument_type(code) == "a_share":
            return path
    if not files:
        raise RuntimeError("未找到任何日线文件；请先运行下载器。")
    return files[0]


def generate(output: Path = REPORT) -> Path:
    lake = Lake()
    raw_files = sorted((lake.curated / "daily_bars_baostock").glob("key=*/data.parquet"))
    selected = _choose_equity(raw_files)
    raw = pd.read_parquet(selected)
    quality = validate_daily_bars(raw)
    calendar = lake.read("calendar", "SSE")
    factors = build_raw_factors(quality.frame, next_trading_day_map(calendar))
    recent = quality.frame.tail(252).copy()
    snapshot_dirs = list((lake.curated / "security_snapshot").glob("key=*"))
    code = str(recent["instrument_id"].iloc[-1])
    quality_counts = quality.frame["quality_status"].value_counts().to_dict()
    coverage = {column: int(factors[column].notna().sum()) for column in [
        "momentum_20d", "momentum_60d", "momentum_120d", "volatility_20d",
        "amihud_20d", "earnings_yield", "book_to_price",
    ]}
    metrics = {
        "snapshot_count": len(snapshot_dirs), "bar_file_count": len(raw_files),
        "sample_instrument": code, "sample_rows": len(raw),
        "sample_start": str(raw["trade_date"].min()), "sample_end": str(raw["trade_date"].max()),
        "quality": quality_counts, "factor_coverage": coverage,
    }
    cards = [
        ("历史股票池快照", f"{metrics['snapshot_count']:,}", "周度 PIT 快照"),
        ("已下载证券文件", f"{metrics['bar_file_count']:,}", "BaoStock 原始日线"),
        ("展示样本", html.escape(code), f"{metrics['sample_rows']:,} 条日频记录"),
        ("样本覆盖", f"{metrics['sample_start']} → {metrics['sample_end']}", "原始未复权行情"),
    ]
    cards_html = "".join(f'<section class="card"><div>{label}</div><strong>{value}</strong><small>{sub}</small></section>' for label, value, sub in cards)
    quality_html = "".join(f'<div class="quality"><span>{status}</span><b>{count:,}</b></div>' for status, count in quality_counts.items())
    coverage_html = "".join(f'<tr><td>{name}</td><td>{count:,}</td><td>{count / len(factors):.1%}</td></tr>' for name, count in coverage.items())
    returns_percent = pd.to_numeric(recent["pct_chg"], errors="coerce")
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>A股研究数据面板</title>
    <style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f5f7fb;color:#172033;margin:0}}main{{max-width:1180px;margin:auto;padding:30px}}h1{{margin-bottom:4px}}.muted,small{{color:#64748b}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}}.card,.panel{{background:white;border-radius:12px;padding:18px;box-shadow:0 1px 4px #d8dee955}}.card strong{{display:block;font-size:22px;margin:8px 0}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}}svg{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #e7edf5;text-align:left}}.quality{{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #e7edf5}}code{{background:#edf2f7;padding:2px 5px;border-radius:4px}}@media(max-width:800px){{.grid,.two{{grid-template-columns:1fr}}}}</style></head><body><main>
    <h1>A 股研究数据面板</h1><p class="muted">生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} ｜ 数据源：BaoStock ｜ 样本：{html.escape(code)}</p>
    <div class="grid">{cards_html}</div>
    <div class="two"><section class="panel"><h2>近 252 个交易日收盘价</h2>{_svg_line(recent['close'])}<p class="muted">未复权价格；用于观察行情，不直接等同于总收益曲线。</p></section>
    <section class="panel"><h2>近 252 个交易日日收益率</h2>{_histogram(returns_percent)}<p class="muted">由供应商 <code>pct_chg</code> 提供，数值单位为百分数。</p></section></div>
    <div class="two"><section class="panel"><h2>质量门禁</h2>{quality_html}<p class="muted">fail 行禁止进入因子计算；warn 行保留，由策略规则决定是否排除。</p></section>
    <section class="panel"><h2>原始因子可用率</h2><table><tr><th>因子</th><th>有效观测</th><th>覆盖率</th></tr>{coverage_html}</table><p class="muted">滚动因子在初始窗口不足时为空，这是正确行为，不应填零。</p></section></div>
    <section class="panel"><h2>如何读这份面板</h2><ol><li>数据文件数反映下载进度，不代表全部都是 A 股；研究面板会按证券类型过滤指数等代码。</li><li>价格图用于检查连续性；突跳需要结合除权、复牌和异常标记判断。</li><li>质量通过不等于可交易：停牌、ST、涨跌停和流动性约束仍需进入回测执行层。</li><li>因子覆盖率不足通常来自滚动窗口或估值缺失，而不是应当盲目填补的数据。</li></ol></section>
    </main></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    (output.with_suffix(".json")).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the offline research data dashboard")
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    print(generate(args.output))


if __name__ == "__main__":
    main()
