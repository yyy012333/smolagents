import pytest

from smolagents.fallback import ErrorClass, FallbackMetrics, FallbackModel, classify_error, is_transient_error
from smolagents.models import ChatMessage, MessageRole, Model


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


class FakeModel(Model):
    """可配置成功/失败的模型替身。"""

    def __init__(self, model_id="fake", fail_with=None):
        super().__init__(model_id=model_id)
        self.fail_with = fail_with  # BaseException 实例或 None
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        return ChatMessage(role=MessageRole.ASSISTANT, content=f"ok-from-{self.model_id}")


def test_fallback_primary_succeeds():
    primary = FakeModel(model_id="primary")
    backup = FakeModel(model_id="backup")
    model = FallbackModel([primary, backup])

    result = model.generate([ChatMessage(role=MessageRole.USER, content="hi")])

    assert result.content == "ok-from-primary"
    assert primary.calls == 1
    assert backup.calls == 0
    assert model.metrics.summary()["served_by"] == ["primary"]


def test_fallback_on_failure():
    primary = FakeModel(model_id="primary", fail_with=Exception("401 Unauthorized"))
    backup = FakeModel(model_id="backup")
    model = FallbackModel([primary, backup])

    result = model.generate([ChatMessage(role=MessageRole.USER, content="hi")])

    assert result.content == "ok-from-backup"
    summary = model.metrics.summary()
    assert summary["served_by"] == ["backup"]
    assert summary["fallback_count"] == 1


def test_fallback_all_fail_raises_last():
    primary = FakeModel(model_id="primary", fail_with=Exception("400 bad"))
    backup = FakeModel(model_id="backup", fail_with=Exception("401 unauthorized"))
    model = FallbackModel([primary, backup])

    with pytest.raises(Exception, match="401 unauthorized"):
        model.generate([ChatMessage(role=MessageRole.USER, content="hi")])


def test_fallback_requires_models():
    with pytest.raises(ValueError, match="at least one model"):
        FallbackModel([])
