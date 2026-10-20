import copy
import json
from pathlib import Path

import pytest

from quant_research.research_governance import (
    append_access_event,
    lockbox_state,
    load_config,
    validate_config,
    verify_access_log,
)


def test_phase4_config_has_nonoverlapping_evidence_periods() -> None:
    cfg = load_config()
    validate_config(cfg)
    changed = copy.deepcopy(cfg)
    changed["evidence_periods"]["future_lockbox"]["start_date"] = "2026-09-01"
    with pytest.raises(ValueError, match="non-overlapping"):
        validate_config(changed)


def test_paper_trading_cannot_be_retroactively_backfilled() -> None:
    changed = copy.deepcopy(load_config())
    changed["evidence_periods"]["paper_trading"]["retroactive_backfill_allowed"] = True
    with pytest.raises(ValueError, match="backfilled"):
        validate_config(changed)


def test_hash_chained_access_log_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "access.jsonl"
    first = append_access_event(
        path, actor="researcher", evidence_class="future_lockbox",
        action="LOCKBOX_SEALED", reason="freeze", result_accessed=False,
        used_for_change=False, lockbox_id="box",
        occurred_at_utc="2026-10-20T00:00:00+00:00",
    )
    second = append_access_event(
        path, actor="researcher", evidence_class="future_lockbox",
        action="LOCKBOX_DATA_ARRIVED", reason="automated collection",
        result_accessed=False, used_for_change=False, lockbox_id="box",
        occurred_at_utc="2026-10-21T00:00:00+00:00",
    )
    assert second["previous_hash"] == first["event_hash"]
    assert len(verify_access_log(path)) == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    event["reason"] = "tampered"
    lines[0] = json.dumps(event)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_access_log(path)


def test_result_access_consumes_future_lockbox(tmp_path: Path) -> None:
    path = tmp_path / "access.jsonl"
    append_access_event(
        path, actor="researcher", evidence_class="future_lockbox",
        action="LOCKBOX_SEALED", reason="freeze", result_accessed=False,
        used_for_change=False, lockbox_id="box",
    )
    append_access_event(
        path, actor="researcher", evidence_class="future_lockbox",
        action="VIEW_RESULTS", reason="scheduled final evaluation", result_accessed=True,
        used_for_change=False, lockbox_id="box",
    )
    state, consumed, first_access = lockbox_state(verify_access_log(path), "box")
    assert state == "OPENED_CONSUMED"
    assert consumed
    assert first_access is not None


def test_access_log_rejects_undisclosed_change_use(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot drive change"):
        append_access_event(
            tmp_path / "access.jsonl", actor="researcher",
            evidence_class="historical_oos", action="DIAGNOSTIC",
            reason="test", result_accessed=False, used_for_change=True,
        )
