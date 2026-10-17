"""Validate and materialize the machine-readable factor dictionary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quant_research.ingest import ROOT, Lake


CONFIG_PATH = ROOT / "config" / "factor_registry.yaml"
PREPROCESSING_PATH = ROOT / "config" / "preprocessing.yaml"
REPORT_PATH = ROOT / "reports" / "factor_registry.md"
REQUIRED_FIELDS = {
    "name", "family", "formula", "unit", "lookback_observations", "raw_fields",
    "missingness", "theoretical_direction", "hypothesis",
}


def load_registry(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_registry(registry: dict[str, Any], expected_factors: list[str]) -> None:
    if not registry.get("registry_version"):
        raise ValueError("factor registry version is required")
    entries = registry.get("factors", [])
    if not entries:
        raise ValueError("factor registry is empty")
    names = [str(item.get("name")) for item in entries]
    if len(names) != len(set(names)):
        raise ValueError("factor registry contains duplicate names")
    if set(names) != set(expected_factors):
        missing = sorted(set(expected_factors) - set(names))
        extra = sorted(set(names) - set(expected_factors))
        raise ValueError(f"registry/model feature mismatch: missing={missing}, extra={extra}")
    for item in entries:
        absent = REQUIRED_FIELDS - set(item)
        if absent:
            raise ValueError(f"{item.get('name')}: missing fields {sorted(absent)}")
        direction = item["theoretical_direction"]
        if direction not in (-1, 1, None):
            raise ValueError(f"{item['name']}: theoretical_direction must be -1, 1, or null")
        if int(item["lookback_observations"]) < 1 or not item["raw_fields"]:
            raise ValueError(f"{item['name']}: lookback and raw_fields must be non-empty")


def materialize_registry(lake: Lake | None = None) -> pd.DataFrame:
    lake = lake or Lake()
    registry = load_registry()
    with PREPROCESSING_PATH.open(encoding="utf-8") as handle:
        expected = list(yaml.safe_load(handle)["factor_columns"])
    validate_registry(registry, expected)

    directions = lake.read("factor_neutralization_directions", "v1")
    locked = ({str(row.factor): int(row.direction) for row in directions.itertuples()}
              if not directions.empty else {})
    records = []
    for entry in registry["factors"]:
        theoretical = entry["theoretical_direction"]
        observed = locked.get(str(entry["name"]))
        records.append({
            "registry_version": str(registry["registry_version"]),
            "effective_date": str(registry["effective_date"]),
            "factor": str(entry["name"]),
            "family": str(entry["family"]),
            "frequency": str(registry["frequency"]),
            "availability": str(registry["availability"]),
            "unit": str(entry["unit"]),
            "lookback_observations": int(entry["lookback_observations"]),
            "raw_fields": ",".join(str(value) for value in entry["raw_fields"]),
            "formula": str(entry["formula"]),
            "missingness": str(entry["missingness"]),
            "hypothesis": str(entry["hypothesis"]),
            "theoretical_direction": theoretical,
            "locked_training_direction": observed,
            "direction_agrees": (
                theoretical == observed if theoretical is not None and observed is not None else None
            ),
            "direction_policy": str(registry["direction_policy"]),
        })
    frame = pd.DataFrame(records).sort_values("factor").reset_index(drop=True)
    lake.write(
        "factor_registry", frame, key="v1", source="factor_registry.v1",
        request={"config": str(CONFIG_PATH.relative_to(ROOT))},
    )
    _write_report(frame, registry)
    return frame


def _write_report(frame: pd.DataFrame, registry: dict[str, Any]) -> None:
    display = frame[[
        "factor", "family", "unit", "lookback_observations", "raw_fields",
        "theoretical_direction", "locked_training_direction", "direction_agrees",
    ]].copy()
    lines = [
        "# 因子注册表", "",
        f"版本：`{registry['registry_version']}`；生效日：`{registry['effective_date']}`。",
        "所有因子只使用 `asof_date` 收盘前已知数据，并在下一交易日才可用。理论方向是事前经济假设；训练期方向仅由 2006–2015 原始 RankIC 锁定，两者不一致不作事后改写。",
        "", display.to_markdown(index=False), "",
        "完整公式、缺失机制与研究假设以 `config/factor_registry.yaml` 为唯一机器可读来源。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    frame = materialize_registry()
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
