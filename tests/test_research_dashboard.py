from quant_research.research_dashboard import _increment_interpretation


def test_increment_interpretation_tracks_confidence_interval() -> None:
    assert "above zero" in _increment_interpretation(0.01, 0.09)
    assert "crosses zero" in _increment_interpretation(-0.01, 0.07)
    assert "below zero" in _increment_interpretation(-0.09, -0.01)
