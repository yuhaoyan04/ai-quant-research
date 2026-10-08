import pandas as pd

from quant_research.factor_research import eligible_research_rows


def test_pit_universe_and_tradability_filter() -> None:
    frame = pd.DataFrame({
        "available_date": ["2024-01-08", "2024-01-08", "2024-01-15"],
        "is_tradable": [True, False, True],
        "is_tradable_at_asof": [True, True, True],
        "quality_status": ["pass", "pass", "fail"],
    })
    result = eligible_research_rows(
        frame, {"2024-01-05"}, {"2024-01-08": "2024-01-05", "2024-01-15": "2024-01-05"}
    )
    assert len(result) == 1
    assert result.iloc[0]["available_date"] == "2024-01-08"
