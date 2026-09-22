import pytest

from smolagents.fallback import ErrorClass, classify_error, is_transient_error


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
