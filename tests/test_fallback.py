import pytest

from smolagents.fallback import ErrorClass, FallbackMetrics, classify_error, is_transient_error


def test_classify_rate_limit():
    for msg in ["429 Too Many Requests", "rate limit exceeded", "Rate_LIMIT", "too many requests"]:
        assert classify_error(Exception(msg)) == ErrorClass.RATE_LIMIT


def test_classify_transient():
    for msg in [
        "502 Bad Gateway",
        "503 Service Unavailable",
        "504 Gateway Timeout",
        "500 Internal Server Error",
        "Request timed out",
        "connection reset by peer",
    ]:
        assert classify_error(Exception(msg)) == ErrorClass.TRANSIENT


def test_classify_non_retryable():
    for msg in ["400 Bad Request", "401 Unauthorized", "invalid argument"]:
        assert classify_error(Exception(msg)) == ErrorClass.NON_RETRYABLE


def test_is_transient_error():
    assert is_transient_error(Exception("502 Bad Gateway")) is True
    assert is_transient_error(Exception("429 too many requests")) is False
    assert is_transient_error(Exception("400 Bad Request")) is False


def test_fallback_metrics_summary():
    m = FallbackMetrics()
    m.record_success("primary")
    m.record_failure("primary", Exception("502 Bad Gateway"))
    m.record_success("backup")

    summary = m.summary()
    assert summary["calls_by_model"] == {"primary": 1, "backup": 1}
    assert summary["failures_by_model"] == {"primary": 1}
    assert summary["fallback_count"] == 1
    assert summary["served_by"] == ["primary", "backup"]
    assert summary["fallbacks"][0]["from"] == "primary"
    assert summary["fallbacks"][0]["error_class"] == "transient"
