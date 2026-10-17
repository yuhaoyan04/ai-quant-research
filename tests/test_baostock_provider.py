from quant_research.providers.baostock import BaoStockProvider


class _TruncatedResult:
    error_code = "0"
    error_msg = "success"
    fields = ["value"]
    per_page_count = "2000"
    data = [[str(index)] for index in range(2000)]

    def __init__(self) -> None:
        self.position = 0

    def next(self) -> bool:
        if self.position < len(self.data):
            return True
        return False

    def get_row_data(self) -> list[str]:
        row = self.data[self.position]
        self.position += 1
        return row


def test_blacklist_error_is_not_retried() -> None:
    provider = BaoStockProvider(max_attempts=4, retry_backoff_seconds=0)
    calls = 0

    def blocked_call() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("BaoStock query failed [10001011]: user blacklisted")

    try:
        provider._retry("query", blocked_call)
    except RuntimeError as exc:
        assert "10001011" in str(exc)
    else:
        raise AssertionError("blacklist error should be raised")

    assert calls == 1


def test_full_terminal_page_is_rejected_as_possible_truncation() -> None:
    provider = BaoStockProvider()

    try:
        provider._frame(_TruncatedResult(), "query")
    except RuntimeError as exc:
        assert "may be truncated" in str(exc)
    else:
        raise AssertionError("a retained full page must not be accepted")
